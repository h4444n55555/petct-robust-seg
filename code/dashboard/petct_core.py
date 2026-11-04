"""Inference core for the dashboard. The preprocessing, perturbations, model definitions and scoring below are copied
verbatim from petct_lung_robustness.ipynb, so the dashboard sees exactly what the training / evaluation notebook sees."""
import json, hashlib
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch, torch.nn.functional as F
from scipy import ndimage
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Orientationd, ScaleIntensityRanged, ConcatItemsd,
                              CopyItemsd, EnsureTyped, SelectItemsd, MapTransform, AsDiscreted)
from monai.networks.nets import DynUNet, SegResNet, SwinUNETR, BasicUNet
from monai.inferers import sliding_window_inference

PET_CLIP = (0.0, 20.0)            # SUV -> [0, 1]
CT_CLIP = (-100.0, 250.0)         # HU  -> [0, 1]
LESION_MIN_OVERLAP = 0.10
SW_BATCH, SW_OVERLAP = 4, 0.25
PATCH = (96, 96, 96)              # overwritten from the run's config.json
FILTERS = (24, 48, 96, 192, 320)  # overwritten from the run's config.json
N_IN, N_CLASSES = 2, 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


class CheckSameGridd(MapTransform):
    """Fail loudly if PET, CT and mask do not share one voxel grid (shape + affine).
    4 autoPET FDG lesion-free labels have a flipped qform-only header; an EMPTY mask has no geometry to get wrong, so adopt the image affine."""
    def __init__(self, keys, atol=1e-2):
        super().__init__(keys); self.atol = atol
    def __call__(self, data):
        d = dict(data); ref_key = self.keys[0]; ref = d[ref_key]
        for k in self.keys[1:]:
            same_shape = tuple(ref.shape) == tuple(d[k].shape)
            same_affine = np.allclose(np.asarray(ref.affine), np.asarray(d[k].affine), atol=self.atol)
            if same_shape and not same_affine and k == "mask" and float(d[k].sum()) == 0.0:
                d[k].meta["affine"] = d[ref_key].meta["affine"].clone(); continue
            if not (same_shape and same_affine):
                raise ValueError(f"grid mismatch for {d.get('case_id', '?')}: '{ref_key}' {tuple(ref.shape)} vs '{k}' {tuple(d[k].shape)}")
        return d

def build_tf(keep_raw=False):
    """Full-volume pipeline -> {'image': (2,X,Y,Z), 'mask': (1,X,Y,Z)} (+ 'pet_raw','ct_raw' in SUV / HU if keep_raw)."""
    keys = ["pet", "ct", "mask"]
    t = [LoadImaged(keys=keys, image_only=True, dtype=np.float32), EnsureChannelFirstd(keys=keys),
         CheckSameGridd(keys=keys), Orientationd(keys=keys, axcodes="RAS")]
    if keep_raw:
        t.append(CopyItemsd(keys=["pet", "ct"], names=["pet_raw", "ct_raw"]))
    t += [ScaleIntensityRanged(keys="pet", a_min=PET_CLIP[0], a_max=PET_CLIP[1], b_min=0.0, b_max=1.0, clip=True),
          ScaleIntensityRanged(keys="ct", a_min=CT_CLIP[0], a_max=CT_CLIP[1], b_min=0.0, b_max=1.0, clip=True),
          ConcatItemsd(keys=["pet", "ct"], name="image", dim=0), AsDiscreted(keys="mask", threshold=0.5)]
    keep = ["image", "mask"] + (["pet_raw", "ct_raw"] if keep_raw else [])
    t += [EnsureTyped(keys=keep, dtype=torch.float32, track_meta=False), SelectItemsd(keys=keep + ["case_id"])]
    return Compose(t)

PET, CT = 0, 1
_TARGET = {"pet": (PET,), "ct": (CT,), "both": (PET, CT)}

@dataclass(frozen=True)
class Spec:
    name: str; kind: str; family: str; level: float; target: str = ""
    def to_dict(self):
        return {"name": self.name, "kind": self.kind, "family": self.family, "level": self.level, "target": self.target}

def seed_from(case_id, name):
    return int.from_bytes(hashlib.sha256(f"{case_id}|{name}".encode()).digest()[:8], "little") & 0x7FFFFFFFFFFFFFFF

def low_resolution(image, factor, channels=(PET, CT)):
    out = image.float().clone()
    if factor == 1:
        return out
    spatial = tuple(out.shape[1:]); small = tuple(max(1, int(np.ceil(s / factor))) for s in spatial)
    for c in channels:
        down = F.interpolate(out[c][None, None], size=small, mode="area")
        out[c] = F.interpolate(down, size=spatial, mode="trilinear", align_corners=False)[0, 0]
    return out

def gaussian_noise(image, sigma, target, case_id, name):
    out = image.float().clone()
    if sigma == 0:
        return out
    gen = torch.Generator().manual_seed(seed_from(case_id, name))
    for c in _TARGET[target]:
        out[c] = out[c] + sigma * torch.randn(out[c].shape, generator=gen)
    return out

def remove_pet(image):
    out = image.float().clone(); out[PET] = 0.0
    return out

def apply_spec(spec, image, case_id):
    if spec.kind == "clean":
        return image.float().clone()
    if spec.kind == "lowres":
        return low_resolution(image, spec.level)
    if spec.kind == "noise":
        return gaussian_noise(image, spec.level, spec.target, case_id, spec.name)
    if spec.kind == "pet_removed":
        return remove_pet(image)
    raise ValueError(spec.kind)

def build_model(name):
    if name == "dynunet":
        strides = [[1, 1, 1]] + [[2, 2, 2]] * (len(FILTERS) - 1)
        m = DynUNet(spatial_dims=3, in_channels=N_IN, out_channels=N_CLASSES, kernel_size=[[3, 3, 3]] * len(FILTERS),
                    strides=strides, upsample_kernel_size=strides[1:], filters=list(FILTERS),
                    norm_name=("INSTANCE", {"affine": True}), act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                    deep_supervision=True, deep_supr_num=2, res_block=False)
        return m, True
    if name == "segresnet":
        return SegResNet(spatial_dims=3, init_filters=16, in_channels=N_IN, out_channels=N_CLASSES,
                         blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1)), False
    if name == "swinunetr":
        try:
            m = SwinUNETR(img_size=PATCH, in_channels=N_IN, out_channels=N_CLASSES, feature_size=24, use_checkpoint=True)
        except TypeError:   # newer MONAI removed img_size
            m = SwinUNETR(in_channels=N_IN, out_channels=N_CLASSES, feature_size=24, use_checkpoint=True)
        return m, False   # use_checkpoint: gradient checkpointing, same result with less GPU memory (6.9 -> 4.6 GB at patch 96, batch 2)
    if name == "basic_unet":
        return BasicUNet(spatial_dims=3, in_channels=N_IN, out_channels=N_CLASSES, features=(16, 32, 64, 128, 256, 16)), False
    raise ValueError(f"unknown MODEL_NAME {name}")

STRUCT26 = np.ones((3, 3, 3), dtype=bool)

def body_mask(ct_hu):
    """Patient body (air outside removed, lungs filled in): HU > -300 with holes filled slice by slice."""
    b = ct_hu > -300
    for z in range(b.shape[2]):
        if b[:, :, z].any():
            b[:, :, z] = ndimage.binary_fill_holes(b[:, :, z])
    return b

def bbox_slices(body, gt, margin=8):
    m = body | gt; sls = []
    for d in range(3):
        nz = np.flatnonzero(m.any(axis=tuple(a for a in range(3) if a != d)))
        lo = max(0, nz[0] - margin) if len(nz) else 0
        hi = min(m.shape[d], nz[-1] + margin + 1) if len(nz) else m.shape[d]
        sls.append(slice(lo, hi))
    return tuple(sls)

def make_gt_pack(gt, voxel_ml):
    lab, n = ndimage.label(gt, structure=STRUCT26)
    return {"gt": gt, "lab": lab, "n": n, "objs": ndimage.find_objects(lab) if n else [],
            "sizes": np.bincount(lab.ravel(), minlength=n + 1)[1:] if n else np.zeros(0, int), "voxel_ml": voxel_ml}

def score_prediction(pred, gp):
    gt = gp["gt"]
    inter, ps, gs = int(np.logical_and(pred, gt).sum()), int(pred.sum()), int(gt.sum())
    lesions = []
    for i, sl in enumerate(gp["objs"], 1):
        size = int(gp["sizes"][i - 1]); ov = int(np.logical_and(pred[sl], gp["lab"][sl] == i).sum())
        lesions.append({"lesion_id": i, "gt_voxels": size, "overlap_voxels": ov, "overlap_frac": ov / size,
                        "detected": int(ov / size >= LESION_MIN_OVERLAP)})
    n_fp, fp_vox = 0, 0
    if ps > 0:
        plab, pn = ndimage.label(pred, structure=STRUCT26)
        hit = np.unique(plab[gt & (plab > 0)]); sizes_p = np.bincount(plab.ravel(), minlength=pn + 1)
        fp_ids = np.setdiff1d(np.arange(1, pn + 1), hit); n_fp, fp_vox = len(fp_ids), int(sizes_p[fp_ids].sum())
    row = {"gt_voxels": gs, "pred_voxels": ps, "dice": 2 * inter / (ps + gs) if gs > 0 else np.nan,
           "voxel_recall": inter / gs if gs > 0 else np.nan, "voxel_precision": inter / ps if ps > 0 else np.nan,
           "n_gt_lesions": len(lesions), "n_detected": int(sum(l["detected"] for l in lesions)),
           "n_fp_components": n_fp, "fp_volume_ml": fp_vox * gp["voxel_ml"], "pred_volume_ml": ps * gp["voxel_ml"]}
    return row, lesions

def load_specs(path):
    frozen = json.load(open(path))
    specs = [Spec(**c) for c in frozen["conditions"]]
    fp = hashlib.sha256(json.dumps([s.to_dict() for s in specs], sort_keys=True).encode()).hexdigest()
    assert fp == frozen["fingerprint"], "perturbation file does not match its own fingerprint"
    return specs


def load_case(rec, keep_raw=True):
    return build_tf(keep_raw)({k: rec[k] for k in ("pet", "ct", "mask", "case_id")})


def load_model(run_dir):
    """run_dir contains config.json and best.pt (as written by the training notebook)."""
    global PATCH, FILTERS
    run_dir = Path(run_dir); cfg = json.load(open(run_dir / "config.json"))
    PATCH, FILTERS = tuple(cfg["patch"]), tuple(cfg["filters"])
    net, _ = build_model(cfg["model"])
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    net.load_state_dict(ck["model"]); net.to(DEVICE).eval()
    return net, cfg, {"epoch": ck.get("epoch"), "val_dice": ck.get("val_dice")}


@torch.no_grad()
def predict_prob(net, image_crop):
    """image_crop: (2,x,y,z) float32 CPU tensor -> lesion probability map (float16 numpy) of the same size."""
    net.eval()
    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
        logits = sliding_window_inference(image_crop[None], roi_size=PATCH, sw_batch_size=SW_BATCH, predictor=net,
                                          overlap=SW_OVERLAP, mode="gaussian", sw_device=DEVICE, device=torch.device("cpu"), progress=False)
    return torch.softmax(logits.float(), dim=1)[0, 1].numpy().astype(np.float16)


def run_case(net, rec, spec):
    """Full pipeline for one study under one condition. Returns dict with the displayed volumes (cropped to the body box)."""
    s = load_case(rec, keep_raw=True)
    ct_raw, pet_raw = s["ct_raw"][0].numpy(), s["pet_raw"][0].numpy()
    gt = s["mask"][0].numpy() > 0
    body = body_mask(ct_raw); bb = bbox_slices(body, gt)
    img = apply_spec(spec, s["image"], rec["case_id"])
    prob = predict_prob(net, img[(slice(None),) + bb])
    return {"pet_model": img[PET].numpy()[bb], "ct_model": img[CT].numpy()[bb], "gt": gt[bb], "prob": prob,
            "pet_suv": pet_raw[bb], "voxel_ml": rec["voxel_ml"], "aspect": rec["aspect"]}


def score(prob, gt, voxel_ml, threshold=0.5):
    pred = prob.astype(np.float32) > threshold
    row, lesions = score_prediction(pred, make_gt_pack(gt, voxel_ml))
    return pred, row, lesions
