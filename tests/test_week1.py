"""Week 1 sanity checks, run on a synthetic phantom (no real data needed).

Run from the project folder:   python -m pytest tests -v
"""
import copy
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from src import data, perturb
from src.phantom import make_phantom_dataset, make_phantom_nnunet

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="session")
def cfg():
    c = data.load_config(ROOT / "configs" / "week1.yaml")
    c = copy.deepcopy(c)
    c["patch"]["size"] = [32, 32, 32]       # phantom volumes are 48x48x40
    c["patch"]["num_samples"] = 2
    return c


@pytest.fixture(scope="session")
def phantom_root(tmp_path_factory):
    return make_phantom_dataset(tmp_path_factory.mktemp("phantom"), n_patients=14, seed=0)


@pytest.fixture(scope="session")
def df(phantom_root, cfg):
    return data.index_dataset(phantom_root, cfg)


@pytest.fixture(scope="session")
def split(df, cfg):
    return data.make_split(df, cfg["split"], cfg["seed"])


@pytest.fixture(scope="session")
def specs(cfg):
    return perturb.specs_from_config(cfg)


def _positive_record(df, cfg):
    row = df[df["lesion_voxels"] > 0].iloc[0]
    return {"pet": row["pet"], "ct": row["ct"], "mask": row["mask"], "case_id": row["case_id"]}


# --------------------------------------------------------------------------------- index
def test_index_counts_and_lesion_flags(phantom_root, df):
    n_files = len(list(Path(phantom_root).glob("PETCT_*/*/SEG.nii.gz")))
    assert len(df) == n_files
    assert (df["lesion_voxels"] > 0).any() and (df["lesion_voxels"] == 0).any()
    assert df["patient_id"].nunique() < len(df), "phantom should contain multi-study patients"


# --------------------------------------------------------------------------------- split
def test_split_has_no_patient_leakage(df, split):
    data.assert_no_leakage(split)
    all_patients = set(df["patient_id"])
    assert set().union(*map(set, split.values())) == all_patients
    assert all(len(v) > 0 for v in split.values())
    # multi-study patients are never separated across splits
    for pid, grp in df.groupby("patient_id"):
        homes = [k for k, v in split.items() if pid in v]
        assert len(homes) == 1


def test_split_deterministic_and_roundtrip(df, cfg, split, tmp_path):
    again = data.make_split(df.sample(frac=1.0, random_state=3).reset_index(drop=True), cfg["split"], cfg["seed"])
    assert again == split
    other = data.make_split(df, cfg["split"], cfg["seed"] + 1)
    assert other != split
    path = tmp_path / "splits.json"
    data.save_split(split, df, cfg, path)
    assert data.load_split(path) == split


def test_leakage_detector_fires():
    with pytest.raises(AssertionError):
        data.assert_no_leakage({"train": ["a", "b"], "test": ["b"]})


def test_max_cases_keeps_whole_patients(df):
    sub = data.apply_max_cases(df, 6, seed=42)
    assert len(sub) >= 6
    for pid in sub["patient_id"].unique():
        assert (sub["patient_id"] == pid).sum() == (df["patient_id"] == pid).sum()


# --------------------------------------------------------------------------------- loader
def test_clean_load_shapes_affine_and_mask(df, cfg):
    rec = _positive_record(df, cfg)
    s = data.load_clean_case(rec, cfg)
    raw = nib.load(rec["pet"])
    assert tuple(s["image"].shape) == (2, *raw.shape)
    assert tuple(s["mask"].shape) == (1, *raw.shape)
    assert np.allclose(np.asarray(s["image"].affine), raw.affine)
    assert float(s["image"].min()) >= 0.0 and float(s["image"].max()) <= 1.0
    assert set(torch.unique(s["mask"]).tolist()) <= {0.0, 1.0}
    expected = int(df[df["case_id"] == rec["case_id"]]["lesion_voxels"].iloc[0])
    assert int(s["mask"].sum()) == expected


def test_grid_mismatch_is_rejected(phantom_root, df, cfg, tmp_path):
    rec = _positive_record(df, cfg)
    ct = nib.load(rec["ct"])
    bad = tmp_path / "ct_bad.nii.gz"
    nib.save(nib.Nifti1Image(np.asarray(ct.dataobj)[:-2], ct.affine), str(bad))
    # MONAI wraps transform errors in a RuntimeError; the real message is on __cause__
    with pytest.raises(RuntimeError) as exc:
        data.load_clean_case({**rec, "ct": str(bad)}, cfg)
    assert "CheckSameGridd" in str(exc.value)
    assert "grid mismatch" in str(exc.value.__cause__)


def test_patch_loader_batches(df, split, cfg):
    recs = data.records_for(df, split, "train")
    ds = data.build_dataset(recs, data.train_transforms(cfg), cache_rate=0.0)
    batch = next(iter(data.build_loader(ds, batch_size=1)))
    n = cfg["patch"]["num_samples"]
    assert tuple(batch["image"].shape) == (n, 2, 32, 32, 32)
    assert tuple(batch["mask"].shape) == (n, 1, 32, 32, 32)


def test_patch_mode_handles_lesion_free_case(df, cfg):
    row = df[df["lesion_voxels"] == 0].iloc[0]
    rec = {"pet": row["pet"], "ct": row["ct"], "mask": row["mask"], "case_id": row["case_id"]}
    out = data.train_transforms(cfg)(rec)
    assert len(out) == cfg["patch"]["num_samples"]


def test_patch_mode_prefers_lesions(df, cfg):
    rec = _positive_record(df, cfg)
    hits = 0
    for _ in range(10):
        for p in data.train_transforms(cfg)(rec):
            hits += int(p["mask"].sum() > 0)
    assert hits > 0


# ----------------------------------------------------------------------------- perturbations
def test_specs_frozen_levels_match_record(specs):
    rec = ROOT / "configs" / "perturbations_frozen.json"
    assert rec.exists(), "run: python -m src.freeze to (re)write the frozen record"
    frozen = json.loads(rec.read_text())
    assert frozen["fingerprint"] == perturb.fingerprint(specs), \
        "perturbation levels changed after freezing; this invalidates comparisons across weeks"


def test_spec_list_content(specs):
    names = [s.name for s in specs]
    assert names[0] == "clean" and len(set(names)) == len(names)
    assert "pet_removed" in names and "lowres_x2" in names and "noise_pet_s0.1" in names


def test_identity_levels_reproduce_clean(df, cfg):
    img = data.load_clean_case(_positive_record(df, cfg), cfg)["image"]
    base = perturb._as_float_tensor(img)
    assert torch.equal(perturb.low_resolution(img, 1), base)
    assert torch.equal(perturb.gaussian_noise(img, 0.0, "both", "c"), base)
    assert torch.equal(perturb.apply_spec(perturb.Spec("clean", "clean", "clean", 0.0), img, "c"), base)


def test_inputs_never_modified_and_shapes_kept(df, cfg, specs):
    s = data.load_clean_case(_positive_record(df, cfg), cfg)
    before = s["image"].as_tensor().clone()
    mask_before = s["mask"].as_tensor().clone()
    for spec in specs:
        out = perturb.apply_to_sample(spec, s, s["case_id"])
        assert tuple(out["image"].shape) == tuple(before.shape), spec.name
        assert out["mask"] is s["mask"], "mask must be passed through untouched"
        assert torch.equal(out["mask"].as_tensor(), mask_before)
        assert set(torch.unique(out["mask"].as_tensor()).tolist()) <= {0.0, 1.0}
        assert int(out["mask"].sum()) == int(mask_before.sum())
    assert torch.equal(s["image"].as_tensor(), before)


def test_noise_is_deterministic_per_seed(df, cfg):
    img = data.load_clean_case(_positive_record(df, cfg), cfg)["image"]
    a = perturb.gaussian_noise(img, 0.1, "both", "case1", "noise_both_s0.1")
    b = perturb.gaussian_noise(img, 0.1, "both", "case1", "noise_both_s0.1")
    c = perturb.gaussian_noise(img, 0.1, "both", "case2", "noise_both_s0.1")
    d = perturb.gaussian_noise(img, 0.1, "both", "case1", "noise_both_s0.2")
    assert torch.equal(a, b)
    assert not torch.equal(a, c) and not torch.equal(a, d)
    assert perturb.seed_from("x", "y") == perturb.seed_from("x", "y")


def test_noise_target_and_scale(df, cfg):
    img = perturb._as_float_tensor(data.load_clean_case(_positive_record(df, cfg), cfg)["image"])
    pet_only = perturb.gaussian_noise(img, 0.2, "pet", "c")
    ct_only = perturb.gaussian_noise(img, 0.2, "ct", "c")
    assert torch.equal(pet_only[1], img[1]) and not torch.equal(pet_only[0], img[0])
    assert torch.equal(ct_only[0], img[0]) and not torch.equal(ct_only[1], img[1])
    resid = (pet_only[0] - img[0])
    assert abs(float(resid.std()) - 0.2) < 0.01 and abs(float(resid.mean())) < 0.01


def test_lowres_degrades_monotonically(df, cfg):
    img = perturb._as_float_tensor(data.load_clean_case(_positive_record(df, cfg), cfg)["image"])
    errs = [float((perturb.low_resolution(img, f)[0] - img[0]).abs().mean()) for f in (1, 1.5, 2, 3, 4)]
    assert errs[0] == 0.0 and all(b > a for a, b in zip(errs, errs[1:])), errs
    only_pet = perturb.low_resolution(img, 3, channels=(0,))
    assert torch.equal(only_pet[1], img[1])


def test_pet_removal_keeps_ct_bit_identical(df, cfg):
    img = perturb._as_float_tensor(data.load_clean_case(_positive_record(df, cfg), cfg)["image"])
    out = perturb.remove_pet(img)
    assert torch.equal(out[1], img[1])
    assert float(out[0].abs().max()) == 0.0


# ------------------------------------------------------------------------------ nnU-Net layout
def test_nnunet_index(tmp_path, cfg):
    root = make_phantom_nnunet(tmp_path, n_patients=8)
    d = data.index_dataset(root, cfg)
    assert len(d) == 9                                  # 8 patients, patient 0 has two studies
    assert d["patient_id"].nunique() == 8
    assert not d["patient_id"].str.startswith("psma").any()      # psma_ files ignored
    assert d["study_id"].str.contains("Ganzkoerper  primaer").all()   # spaces preserved
    assert d["pet"].str.endswith("_0001.nii.gz").all() and d["ct"].str.endswith("_0000.nii.gz").all()
    # channel mapping by value range: CT has negative HU, PET is non-negative-ish SUV
    rec = data.load_clean_case(data.records_for(d, {"train": list(d["patient_id"].unique())}, "train")[0], cfg)
    assert rec["image"].shape[0] == 2 and float(rec["image"].min()) >= 0 and float(rec["image"].max()) <= 1


def test_nnunet_multistudy_patient_same_split(tmp_path, cfg):
    d = data.index_dataset(make_phantom_nnunet(tmp_path, n_patients=8), cfg)
    sp = data.make_split(d, cfg["split"], cfg["seed"])
    data.assert_no_leakage(sp)
    pid0 = d[d["patient_id"].duplicated(keep=False)]["patient_id"].iloc[0]
    assert sum(pid0 in sp[k] for k in sp) == 1


def test_empty_mask_with_flipped_header_is_adopted_but_lesion_mask_is_not(tmp_path, cfg):
    """Mirrors 4 real lesion-free autoPET labels with a y-flipped header."""
    root = make_phantom_nnunet(tmp_path, n_patients=6, seed=3)
    d = data.index_dataset(root, cfg)
    for flip_lesion in (False, True):
        row = d[(d["lesion_voxels"] > 0) == flip_lesion].iloc[0]
        img = nib.load(row["mask"])
        aff = img.affine.copy(); aff[1, 1] *= -1; aff[1, 3] += 100
        nib.save(nib.Nifti1Image(np.asanyarray(img.dataobj), aff), row["mask"])
        rec = {"pet": row["pet"], "ct": row["ct"], "mask": row["mask"], "case_id": row["case_id"]}
        if flip_lesion:
            with pytest.raises(RuntimeError):
                data.load_clean_case(rec, cfg)
        else:
            assert data.load_clean_case(rec, cfg)["mask"].sum() == 0
