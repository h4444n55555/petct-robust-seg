"""Figure helpers for the Week 1 perturbation preview.

All figures use the clean 2-channel sample from data.load_clean_case() and the conditions from
perturb.specs_from_config(). Titles describe the synthetic change only; they make no claim
about any scanner. Lesion mask is drawn as a contour so the image underneath stays visible.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import perturb

CH_NAME = {perturb.PET: "PET", perturb.CT: "CT"}
CMAP = {perturb.PET: "magma", perturb.CT: "gray"}
FAMILY_ORDER = ["lowres", "noise_pet", "noise_ct", "noise_both", "pet_removal"]
FAMILY_LABEL = {"lowres": "low-res (factor)", "noise_pet": "noise on PET (sigma)",
                "noise_ct": "noise on CT (sigma)", "noise_both": "noise on both (sigma)",
                "pet_removal": "PET removed"}


def _np(x) -> np.ndarray:
    return (x.as_tensor() if hasattr(x, "as_tensor") else torch.as_tensor(x)).detach().cpu().numpy()


def best_axial_slice(mask) -> int:
    """Axial slice (last axis) with the largest lesion area; middle slice if there is no lesion."""
    m = _np(mask)[0] > 0.5
    areas = m.sum(axis=(0, 1))
    return int(areas.argmax()) if areas.max() > 0 else m.shape[2] // 2


def _show(ax, vol2d, mask2d, cmap, title=None, vmin=0.0, vmax=1.0):
    ax.imshow(vol2d.T, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
    if mask2d is not None and mask2d.any():
        ax.contour(mask2d.T.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def _by_family(specs):
    fam = {}
    for s in specs:
        fam.setdefault(s.family, []).append(s)
    return fam


def perturbation_grid(sample: dict, specs, channel: int, path, case_id: str | None = None, z: int | None = None):
    """Rows = perturbation family, columns = severity (first column = clean). One channel per figure."""
    case_id = case_id or sample.get("case_id", "case")
    z = best_axial_slice(sample["mask"]) if z is None else z
    mask2d = _np(sample["mask"])[0][:, :, z] > 0.5
    fam = _by_family(specs)
    rows = [f for f in FAMILY_ORDER if f in fam]
    ncol = 1 + max(len(fam[f]) for f in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.7 * ncol, 1.8 * len(rows) + 0.4), squeeze=False)
    clean = perturb.apply_spec(perturb.Spec("clean", "clean", "clean", 0.0), sample["image"], case_id)
    for r, f in enumerate(rows):
        _show(axes[r, 0], clean[channel][:, :, z].numpy(), mask2d, CMAP[channel], "clean" if r == 0 else None)
        axes[r, 0].set_ylabel(FAMILY_LABEL[f], fontsize=7)
        for c in range(1, ncol):
            ax = axes[r, c]
            if c - 1 < len(fam[f]):
                s = fam[f][c - 1]
                out = perturb.apply_spec(s, sample["image"], case_id)
                _show(ax, out[channel][:, :, z].numpy(), mask2d, CMAP[channel], f"{s.level:g}")
            else:
                ax.axis("off")
    fig.suptitle(f"{CH_NAME[channel]} channel, axial slice {z} (cyan = expert lesion mask, unchanged)\n{case_id}", fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def mip_figure(sample: dict, specs, path, case_id: str | None = None, names=("clean", "lowres_x3", "noise_pet_s0.1", "pet_removed")):
    """Coronal PET maximum-intensity projections, clean vs selected perturbations."""
    case_id = case_id or sample.get("case_id", "case")
    chosen = [s for n in names for s in specs if s.name == n]
    fig, axes = plt.subplots(1, len(chosen), figsize=(2.2 * len(chosen), 2.6), squeeze=False)
    mask_mip = (_np(sample["mask"])[0] > 0.5).max(axis=1)
    for ax, s in zip(axes[0], chosen):
        img = perturb.apply_spec(s, sample["image"], case_id)[perturb.PET].numpy()
        mip = img.max(axis=1)
        ax.imshow(mip.T, cmap="gray_r", origin="lower", vmin=0, vmax=1)
        if mask_mip.any():
            ax.contour(mask_mip.T.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        ax.set_title(s.name, fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"PET coronal MIP: {case_id}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def difference_figure(sample: dict, specs, path, case_id: str | None = None, z: int | None = None,
                      names=("lowres_x2", "lowres_x4", "noise_pet_s0.1", "noise_ct_s0.1", "pet_removed")):
    """Perturbed minus clean, per channel (diverging colour map centred on zero)."""
    case_id = case_id or sample.get("case_id", "case")
    z = best_axial_slice(sample["mask"]) if z is None else z
    chosen = [s for n in names for s in specs if s.name == n]
    clean = perturb.apply_spec(perturb.Spec("clean", "clean", "clean", 0.0), sample["image"], case_id)
    fig, axes = plt.subplots(2, len(chosen), figsize=(1.9 * len(chosen), 4.2), squeeze=False)
    for c, s in enumerate(chosen):
        out = perturb.apply_spec(s, sample["image"], case_id)
        for r, ch in enumerate((perturb.PET, perturb.CT)):
            d = (out[ch] - clean[ch])[:, :, z].numpy()
            lim = max(float(np.abs(d).max()), 1e-6)
            axes[r, c].imshow(d.T, cmap="coolwarm", origin="lower", vmin=-lim, vmax=lim, interpolation="nearest")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, c].set_title(f"{s.name}\n{CH_NAME[ch]}  max|d|={lim:.2f}", fontsize=6)
    fig.suptitle(f"perturbed - clean, axial slice {z}: {case_id}", fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def histogram_figure(sample: dict, specs, path, case_id: str | None = None,
                     names=("clean", "lowres_x4", "noise_both_s0.2")):
    """Intensity histograms (log count) of each channel, clean vs the most severe perturbations."""
    case_id = case_id or sample.get("case_id", "case")
    chosen = [s for n in names for s in specs if s.name == n]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    for ch, ax in zip((perturb.PET, perturb.CT), axes):
        for s in chosen:
            v = perturb.apply_spec(s, sample["image"], case_id)[ch].numpy().ravel()
            ax.hist(v, bins=80, range=(-0.4, 1.4), histtype="step", label=s.name)
        ax.set_yscale("log"); ax.set_title(f"{CH_NAME[ch]} intensities (normalised)", fontsize=8)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
