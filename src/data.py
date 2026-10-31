"""Data indexing, patient-disjoint split and MONAI loaders for PETCT-RobustSeg (Week 1).

Expected layout (autoPET NIfTI, from the dataset's conversion scripts):
    <root>/PETCT_<patientid>/<study>/{SUV.nii.gz, CTres.nii.gz, SEG.nii.gz}
where SUV is the PET in SUV units, CTres is the CT resampled to the PET grid, SEG the lesion mask.
NOTE: this layout is written from the published conversion scripts and has not been checked
against your downloaded copy; if yours differs, supply a manifest CSV (config: manifest) with
columns patient_id,study_id,pet,ct,mask.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (AsDiscreted, CastToTyped, Compose, ConcatItemsd, EnsureChannelFirstd,
                              LoadImaged, MapTransform, Orientationd, RandCropByPosNegLabeld,
                              ScaleIntensityRanged)


# --------------------------------------------------------------------------- config
def load_config(path="configs/week1.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_root(cfg: dict):
    """PETCT_DATA_ROOT env var wins over the config; None means 'use the phantom'."""
    return os.environ.get("PETCT_DATA_ROOT") or cfg.get("data_root")


# --------------------------------------------------------------------------- index
def _lesion_voxels(mask_path: str) -> int:
    import nibabel as nib
    return int(np.count_nonzero(np.asanyarray(nib.load(mask_path).dataobj)))


_NNUNET_RE = re.compile(r"^(?P<tracer>[a-z]+)_(?P<pid>[0-9a-f]+)_(?P<study>.+)\.nii\.gz$")


def index_nnunet(root, cfg: dict) -> pd.DataFrame:
    """nnU-Net layout: imagesTr/<tracer>_<pid>_<study>_0000 (CT) and _0001 (PET), labelsTr/<tracer>_<pid>_<study>.
    Only the tracer in cfg['nnunet']['tracer_prefix'] (default 'fdg') is indexed; other tracers are ignored.
    Study names may contain spaces."""
    nn = {"tracer_prefix": "fdg", "ct_suffix": "_0000", "pet_suffix": "_0001", "images_dir": "imagesTr",
          "labels_dir": "labelsTr", "metadata_csv": "fdg_metadata.csv", **(cfg.get("nnunet") or {})}
    root = Path(root)
    img_dir, lab_dir = root / nn["images_dir"], root / nn["labels_dir"]
    rows, skipped = [], []
    for lab in sorted(lab_dir.glob(nn["tracer_prefix"] + "_*.nii.gz")):
        m = _NNUNET_RE.match(lab.name)
        if not m:
            skipped.append(lab.name)
            continue
        stem = lab.name[: -len(".nii.gz")]
        ct, pet = img_dir / (stem + nn["ct_suffix"] + ".nii.gz"), img_dir / (stem + nn["pet_suffix"] + ".nii.gz")
        if ct.exists() and pet.exists():
            rows.append({"patient_id": m["pid"], "study_id": m["study"], "pet": str(pet), "ct": str(ct), "mask": str(lab)})
        else:
            skipped.append(lab.name)
    if not rows:
        raise FileNotFoundError(f"no '{nn['tracer_prefix']}' cases found under {root}")
    if skipped:
        print(f"[index] skipped {len(skipped)} labels (unparsed or missing a channel)")
    df = pd.DataFrame(rows)
    meta = root / nn["metadata_csv"]
    if meta.exists():   # diagnosis is for reporting only
        m = pd.read_csv(meta, usecols=["Subject ID", "diagnosis"]).drop_duplicates()
        m["patient_id"] = m["Subject ID"].str.replace("PETCT_", "", regex=False)
        df = df.merge(m[["patient_id", "diagnosis"]].drop_duplicates("patient_id"), on="patient_id", how="left")
    return df


def load_index_cache(root, csv_path) -> pd.DataFrame:
    """Load results/week1/index.csv (paths stored relative to the data root) -> absolute paths.
    The cache holds lesion voxel counts so the 1,014 masks need not be re-read."""
    df = pd.read_csv(csv_path)
    for k in ("pet", "ct", "mask"):
        df[k] = [str(Path(root) / v) for v in df[k]]
    return df


def index_dataset(root, cfg: dict, compute_lesion: bool = True) -> pd.DataFrame:
    """Build the case table. One row per study: patient_id, study_id, case_id, pet, ct, mask, lesion_voxels."""
    if cfg.get("manifest"):
        df = pd.read_csv(cfg["manifest"])
    elif root is not None and (Path(root) / (cfg.get("nnunet") or {}).get("images_dir", "imagesTr")).is_dir():
        df = index_nnunet(root, cfg)
        missing = {"patient_id", "study_id", "pet", "ct", "mask"} - set(df.columns)
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    else:
        names = cfg["file_names"]
        rows, skipped = [], []
        for pdir in sorted(Path(root).glob("PETCT_*")):
            if not pdir.is_dir():
                continue
            for sdir in sorted(p for p in pdir.iterdir() if p.is_dir()):
                paths = {k: sdir / names[k] for k in ("pet", "ct", "mask")}
                if all(p.exists() for p in paths.values()):
                    rows.append({"patient_id": pdir.name, "study_id": sdir.name,
                                 **{k: str(v) for k, v in paths.items()}})
                else:
                    skipped.append(str(sdir))
        if not rows:
            raise FileNotFoundError(f"no cases found under {root} (expected PETCT_*/<study>/"
                                    f"{names['pet']}, {names['ct']}, {names['mask']})")
        if skipped:
            print(f"[index] skipped {len(skipped)} study folders with missing files")
        df = pd.DataFrame(rows)
    df["case_id"] = df["patient_id"].astype(str) + "/" + df["study_id"].astype(str)
    if compute_lesion:
        df["lesion_voxels"] = [_lesion_voxels(p) for p in df["mask"]]
    return df.sort_values("case_id").reset_index(drop=True)


def _rank(patient_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{patient_id}".encode()).hexdigest()


def apply_max_cases(df: pd.DataFrame, max_cases, seed: int) -> pd.DataFrame:
    """Subset by whole patients (deterministic) until at least max_cases studies are kept."""
    if not max_cases or len(df) <= max_cases:
        return df
    keep, n = [], 0
    for pid in sorted(df["patient_id"].unique(), key=lambda p: _rank(p, seed)):
        keep.append(pid)
        n += int((df["patient_id"] == pid).sum())
        if n >= max_cases:
            break
    return df[df["patient_id"].isin(keep)].reset_index(drop=True)


# --------------------------------------------------------------------------- split
def make_split(df: pd.DataFrame, ratios: dict, seed: int) -> dict:
    """Patient-disjoint split. Patients are ordered by SHA-256(seed|patient_id) (stable across
    numpy/Python versions) and cut by the ratios. Returns {'train':[patient ids], 'val':..., 'test':...}."""
    patients = sorted(df["patient_id"].unique(), key=lambda p: _rank(p, seed))
    n = len(patients)
    n_val = max(1, round(ratios["val"] * n)) if n >= 3 else 0
    n_test = max(1, round(ratios["test"] * n)) if n >= 3 else 0
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"too few patients ({n}) for the requested split")
    split = {"train": patients[:n_train],
             "val": patients[n_train:n_train + n_val],
             "test": patients[n_train + n_val:]}
    assert_no_leakage(split)
    return split


def assert_no_leakage(split: dict) -> None:
    names = list(split)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(split[a]) & set(split[b])
            if overlap:
                raise AssertionError(f"patient leakage between {a} and {b}: {sorted(overlap)[:5]}")


def save_split(split: dict, df: pd.DataFrame, cfg: dict, path="splits.json") -> None:
    payload = {
        "seed": cfg["seed"], "ratios": cfg["split"], "max_cases": cfg.get("max_cases"),
        "patients": split,
        "cases": {k: sorted(df[df["patient_id"].isin(v)]["case_id"]) for k, v in split.items()},
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_split(path="splits.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))["patients"]


def records_for(df: pd.DataFrame, split: dict, name: str) -> list[dict]:
    """MONAI-style dict records for one split."""
    sub = df[df["patient_id"].isin(split[name])]
    return [{"pet": r.pet, "ct": r.ct, "mask": r.mask, "case_id": r.case_id,
             "lesion_voxels": int(getattr(r, "lesion_voxels", -1))} for r in sub.itertuples()]


# --------------------------------------------------------------------------- transforms
class CheckSameGridd(MapTransform):
    """Fail loudly if PET, CT and mask do not share one voxel grid (shape + affine)."""

    def __init__(self, keys, atol=1e-2):
        super().__init__(keys)
        self.atol = atol

    def __call__(self, data):
        d = dict(data)
        ref_key = self.keys[0]
        ref = d[ref_key]
        for k in self.keys[1:]:
            same_shape = tuple(ref.shape) == tuple(d[k].shape)
            same_affine = np.allclose(np.asarray(ref.affine), np.asarray(d[k].affine), atol=self.atol)
            if same_shape and not same_affine and k == "mask" and float(d[k].sum()) == 0.0:
                # 4 real autoPET FDG lesion-free labels carry a y-flipped qform-only header. An EMPTY mask has
                # no geometry to get wrong, so adopt the image affine; a non-empty mask must still match.
                d[k].meta["affine"] = d[ref_key].meta["affine"].clone()
                continue
            if not (same_shape and same_affine):
                raise ValueError(f"grid mismatch for {d.get('case_id', '?')}: '{ref_key}' {tuple(ref.shape)} "
                                 f"vs '{k}' {tuple(d[k].shape)} (affine equal: {same_affine})")
        return d


def clean_transforms(cfg: dict) -> Compose:
    """Clean pipeline -> {'image': (2,X,Y,Z) [PET, CT] in [0,1], 'mask': (1,X,Y,Z) binary, ...}."""
    p = cfg["preprocess"]
    keys = ["pet", "ct", "mask"]
    return Compose([
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
        CheckSameGridd(keys=keys),
        Orientationd(keys=keys, axcodes=p["orientation"]),
        ScaleIntensityRanged(keys="pet", a_min=p["pet_clip"][0], a_max=p["pet_clip"][1],
                             b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRanged(keys="ct", a_min=p["ct_clip"][0], a_max=p["ct_clip"][1],
                             b_min=0.0, b_max=1.0, clip=True),
        ConcatItemsd(keys=["pet", "ct"], name="image", dim=0),       # channel 0 = PET, 1 = CT
        AsDiscreted(keys="mask", threshold=0.5),                    # strictly binary; never interpolated
        CastToTyped(keys=["image", "mask"], dtype=torch.float32),
    ])


def train_transforms(cfg: dict) -> Compose:
    """Patch mode: clean pipeline + positive/negative balanced random crops."""
    pc = cfg["patch"]
    return Compose([
        clean_transforms(cfg),
        RandCropByPosNegLabeld(keys=["image", "mask"], label_key="mask", spatial_size=pc["size"],
                               pos=pc["pos"], neg=pc["neg"], num_samples=pc["num_samples"],
                               image_key="image", allow_smaller=True),
    ])


def eval_transforms(cfg: dict) -> Compose:
    """Full-volume mode (sliding-window inference comes later): just the clean pipeline."""
    return clean_transforms(cfg)


# --------------------------------------------------------------------------- datasets / loaders
def build_dataset(records: list[dict], transform, cache_rate: float = 0.0, num_workers: int = 0):
    if cache_rate and cache_rate > 0:
        return CacheDataset(records, transform=transform, cache_rate=cache_rate, num_workers=num_workers)
    return Dataset(records, transform=transform)


def build_loader(dataset, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def load_clean_case(record: dict, cfg: dict) -> dict:
    """Load one case through the full-volume clean pipeline (used by figures and tests)."""
    return eval_transforms(cfg)(record)
