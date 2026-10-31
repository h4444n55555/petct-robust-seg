"""Tiny synthetic PET/CT phantom dataset written in the autoPET NIfTI folder layout.

Used ONLY to test the pipeline without the real data. Nothing here is meant to look like
real patient images: an ellipsoid "body", a bone-like blob, physiological-uptake blobs
(not lesions) and 0-3 spherical lesions with a matching binary mask.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def _sphere(shape, centre, radius):
    grid = np.indices(shape).astype(np.float32)
    d2 = sum((g - c) ** 2 for g, c in zip(grid, centre))
    return d2 <= radius ** 2


def make_case(rng: np.random.Generator, shape=(48, 48, 40), lesion_free=False):
    X, Y, Z = shape
    grid = np.indices(shape).astype(np.float32)
    cx, cy, cz = X / 2, Y / 2, Z / 2
    body = ((grid[0] - cx) / (0.40 * X)) ** 2 + ((grid[1] - cy) / (0.32 * Y)) ** 2 + ((grid[2] - cz) / (0.45 * Z)) ** 2 <= 1

    ct = np.full(shape, -1000.0, np.float32)
    ct[body] = 40.0
    lung = _sphere(shape, (cx - 0.12 * X, cy, cz + 0.12 * Z), 0.14 * X) & body
    ct[lung] = -700.0
    bone = _sphere(shape, (cx, cy + 0.2 * Y, cz), 0.07 * X) & body
    ct[bone] = 300.0
    ct += rng.normal(0, 8, shape).astype(np.float32)

    pet = np.zeros(shape, np.float32)
    pet[body] = 1.0
    organ = _sphere(shape, (cx + 0.15 * X, cy - 0.1 * Y, cz - 0.1 * Z), 0.08 * X) & body   # physiological uptake
    pet[organ] = 5.0

    mask = np.zeros(shape, np.uint8)
    if not lesion_free:
        for _ in range(int(rng.integers(1, 4))):
            for _try in range(20):
                c = (rng.uniform(0.3, 0.7) * X, rng.uniform(0.3, 0.7) * Y, rng.uniform(0.25, 0.75) * Z)
                s = _sphere(shape, c, float(rng.uniform(2.0, 4.0)))
                if (s & body).sum() == s.sum():
                    pet[s] = float(rng.uniform(8, 14))
                    mask[s] = 1
                    break
    pet = gaussian_filter(pet, 0.8) + rng.normal(0, 0.05, shape).astype(np.float32)
    return pet.astype(np.float32), ct.astype(np.float32), mask


def make_phantom_dataset(root, n_patients=10, seed=0, shape=(48, 48, 40), spacing=(4.0, 4.0, 4.0),
                         p_second_study=0.3, p_lesion_free=0.4) -> Path:
    """Write <root>/PETCT_<id>/<study>/{SUV,CTres,SEG}.nii.gz for n_patients patients."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    affine = np.diag([*spacing, 1.0])
    for i in range(n_patients):
        pid = f"PETCT_{i:04d}phantom"
        n_studies = 2 if rng.random() < p_second_study else 1
        for k in range(n_studies):
            sdir = root / pid / f"2003-0{k + 1}-01-NA-PET-CT-phantom-{i:03d}{k}"
            sdir.mkdir(parents=True, exist_ok=True)
            pet, ct, mask = make_case(rng, shape, lesion_free=bool(rng.random() < p_lesion_free))
            nib.save(nib.Nifti1Image(pet, affine), str(sdir / "SUV.nii.gz"))
            nib.save(nib.Nifti1Image(ct, affine), str(sdir / "CTres.nii.gz"))
            nib.save(nib.Nifti1Image(mask, affine), str(sdir / "SEG.nii.gz"))
    return root


def make_phantom_nnunet(root, n_patients=8, seed=0, shape=(48, 48, 40), spacing=(4.0, 4.0, 4.0), n_psma=2) -> Path:
    """Write the real dataset's nnU-Net naming: imagesTr/fdg_<pid>_<study with spaces>_0000(CT)/_0001(PET),
    labelsTr/fdg_<pid>_<study>; patient 0 gets two studies; extra psma_ cases must be ignored."""
    root = Path(root)
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    affine = np.diag([-spacing[0], spacing[1], spacing[2], 1.0])
    names = [(f"{i:02d}ab{i:02d}cd{i:02d}", k) for i in range(n_patients) for k in range(2 if i == 0 else 1)]
    names += [(f"psma{i}", 0) for i in range(n_psma)]
    for pid, k in names:
        tracer = "psma" if pid.startswith("psma") else "fdg"
        stem = f"{tracer}_{pid}_03-2{k}-2003-NA-PET-CT Ganzkoerper  primaer mit KM-1{pid[:2]}{k}"
        pet, ct, mask = make_case(rng, shape, lesion_free=bool(rng.random() < 0.4))
        nib.save(nib.Nifti1Image(ct.astype(np.int16), affine), str(root / "imagesTr" / f"{stem}_0000.nii.gz"))
        nib.save(nib.Nifti1Image(pet.astype(np.float64), affine), str(root / "imagesTr" / f"{stem}_0001.nii.gz"))
        nib.save(nib.Nifti1Image(mask, affine), str(root / "labelsTr" / f"{stem}.nii.gz"))
    return root
