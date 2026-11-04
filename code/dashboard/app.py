"""PET/CT lesion segmentation dashboard: load a trained model, pick a study (test set, upload, or files on disk),
optionally degrade it with one of the 15 frozen perturbations, and see the model's detection next to the ground truth.

Run from this folder:   streamlit run app.py
"""
import os, sys, json, time, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import petct_core as core

CODE_DIR = Path(os.environ.get("PETCT_CODE_DIR", HERE.parent)).resolve()
st.set_page_config(page_title="PET/CT lesion segmentation", layout="wide")


# ---------------------------------------------------------------- helpers
def default_runs_dirs():
    env = os.environ.get("PETCT_RUNS")
    dirs = env.split(os.pathsep) if env else [str(CODE_DIR / "runs"), str(Path.home() / "PETCT_C_E_results")]
    return "\n".join(dirs)


def default_data_root():
    env = os.environ.get("PETCT_DATA_ROOT")
    if env:
        return env
    cand = CODE_DIR.parent / "data" / "extracted" / "PSMA-FDG-PET-CT-Lesions_v2"
    return str(cand) if cand.exists() else ""


def find_runs(text):
    runs = {}
    for line in text.splitlines():
        root = Path(line.strip().strip('"'))
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "best.pt").exists() and (d / "config.json").exists() and not d.name.endswith("_old"):
                runs[d.name] = d
    return runs


@st.cache_resource(show_spinner="Loading model ...")
def get_model(run_dir: str, mtime: float):
    return core.load_model(run_dir)


@st.cache_data
def get_tables(code_dir: str):
    index = pd.read_csv(Path(code_dir) / "case_index.csv").set_index("case_id")
    splits = json.load(open(Path(code_dir) / "splits_v2.json"))
    specs = core.load_specs(Path(code_dir) / "perturbations_frozen.json")
    return index, splits, specs


def make_record(index, case_id, data_root):
    r = index.loc[case_id]
    return {"case_id": case_id, "patient_id": r["patient_id"], "pet": str(Path(data_root) / r["pet"]),
            "ct": str(Path(data_root) / r["ct"]), "mask": str(Path(data_root) / r["mask"]),
            "voxel_ml": float(r["dx"] * r["dy"] * r["dz"] / 1000.0), "aspect": float(r["dz"] / r["dx"]), "has_mask": True}


def record_from_files(ct_path, pet_path, mask_path):
    import nibabel as nib
    img = nib.load(str(ct_path)); zooms = img.header.get_zooms()[:3]
    has_mask = bool(mask_path)
    if not has_mask:   # no ground truth: an empty mask on the CT grid
        mask_path = Path(tempfile.mkdtemp(prefix="petct_mask_")) / "empty_mask.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(img.shape[:3], np.uint8), img.affine), str(mask_path))
    return {"case_id": "upload", "patient_id": "upload", "pet": str(pet_path), "ct": str(ct_path), "mask": str(mask_path),
            "voxel_ml": float(np.prod(zooms) / 1000.0), "aspect": float(zooms[2] / zooms[0]), "has_mask": has_mask}


def save_upload(up, folder):
    p = Path(folder) / up.name
    p.write_bytes(up.getbuffer())
    return p


def contour(ax, m2d, color, lw=1.0):
    if m2d.any():
        ax.contour(m2d.T.astype(float), levels=[0.5], colors=color, linewidths=lw, origin="lower")


def panel_figure(res, pred, view, idx, overlay, aspect):
    take = (lambda a: a[:, :, idx].T) if view == "axial" else (lambda a: a[:, idx, :].T)
    asp = 1.0 if view == "axial" else aspect
    ct, pet, gt, prob = take(res["ct_model"]), take(res["pet_model"]), take(res["gt"]), take(res["prob"].astype(np.float32))
    pr = take(pred)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2 if view == "axial" else 5.6))
    axs[0].imshow(ct, cmap="gray", vmin=0, vmax=1, origin="lower", aspect=asp); axs[0].set_title("CT (as the model sees it)")
    axs[1].imshow(pet, cmap="gray_r", vmin=0, vmax=0.4, origin="lower", aspect=asp); axs[1].set_title("PET (as the model sees it)")
    axs[2].imshow(ct, cmap="gray", vmin=0, vmax=1, origin="lower", aspect=asp)
    axs[2].imshow(np.ma.masked_less(pet, 0.05), cmap="hot", vmin=0, vmax=0.4, alpha=0.5, origin="lower", aspect=asp)
    if overlay == "Probability heatmap":
        axs[2].imshow(np.ma.masked_less(prob, 0.1), cmap="cool", vmin=0, vmax=1, alpha=0.6, origin="lower", aspect=asp)
    else:
        contour(axs[2], pr, "red", 1.4)
    contour(axs[2], gt, "lime", 1.0)
    axs[2].set_title("fusion: green = ground truth, red = prediction" if overlay != "Probability heatmap" else "fusion: green = ground truth, colour = probability")
    for a in axs:
        a.axis("off")
    fig.tight_layout()
    return fig


def mip_figure(res, pred):
    pet, gt = res["pet_model"].max(axis=1).T, res["gt"].any(axis=1)
    fig, axs = plt.subplots(1, 2, figsize=(10, 6.4))
    for a, title, show in zip(axs, ["PET MIP + ground truth", "PET MIP + prediction"], ["gt", "pred"]):
        a.imshow(pet, cmap="gray_r", vmin=0, vmax=0.6, origin="lower", aspect=res["aspect"])
        contour(a, gt if show == "gt" else pred.any(axis=1), "lime" if show == "gt" else "red", 1.2)
        a.set_title(title); a.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- sidebar
st.title("PET/CT lesion segmentation dashboard")
index, splits, specs = get_tables(str(CODE_DIR))
with st.sidebar:
    st.header("1. Model")
    runs_text = st.text_area("Folders that contain trained runs (one per line)", default_runs_dirs(), height=90,
                             help="Each run folder needs best.pt and config.json, as written by the training notebook.")
    runs = find_runs(runs_text)
    if not runs:
        st.error("No trained run found in these folders."); st.stop()
    run_name = st.selectbox("Model", list(runs))
    run_dir = runs[run_name]

    st.header("2. Image")
    mode = st.radio("Source", ["Test-set study", "Upload NIfTI files", "Files on disk"])
    data_root, rec, label = "", None, ""
    if mode == "Test-set study":
        data_root = st.text_input("Dataset folder (contains imagesTr/ and labelsTr/)", default_data_root())
        split = st.selectbox("Split", ["test_lung", "test_neg", "val", "train"], help="test_lung = held-out lung studies; test_neg = lesion-free studies")
        cases = splits["cases"][split]
        names = {c: f"{index.loc[c, 'patient_id'][:8]} | {index.loc[c, 'diagnosis']} | {int(index.loc[c, 'lesion_voxels']):,} lesion voxels" for c in cases}
        case_id = st.selectbox("Study", cases, format_func=lambda c: names[c])
        if data_root and (Path(data_root) / "imagesTr").is_dir():
            rec = make_record(index, case_id, data_root); label = names[case_id]
        else:
            st.warning("Set the dataset folder.")
    elif mode == "Upload NIfTI files":
        up_ct = st.file_uploader("CT (HU), .nii / .nii.gz", type=["nii", "gz"])
        up_pet = st.file_uploader("PET (SUV), same grid as the CT", type=["nii", "gz"])
        up_mask = st.file_uploader("Lesion mask (optional)", type=["nii", "gz"])
        if up_ct and up_pet:
            tmp = st.session_state.setdefault("tmp_dir", tempfile.mkdtemp(prefix="petct_up_"))
            rec = record_from_files(save_upload(up_ct, tmp), save_upload(up_pet, tmp), save_upload(up_mask, tmp) if up_mask else None)
            label = f"{up_ct.name}"
    else:
        p_ct, p_pet = st.text_input("CT path (.nii.gz)"), st.text_input("PET path (.nii.gz)")
        p_mask = st.text_input("Mask path (optional)")
        if p_ct and p_pet and Path(p_ct).exists() and Path(p_pet).exists():
            rec = record_from_files(p_ct, p_pet, p_mask if p_mask and Path(p_mask).exists() else None); label = Path(p_ct).name

    st.header("3. Image quality")
    spec_name = st.selectbox("Condition", [s.name for s in specs],
                             help="clean = unmodified. Low-res, noise (sigma in normalised units) and PET removal are the 15 frozen test conditions.")
    spec = next(s for s in specs if s.name == spec_name)
    go = st.button("Run detection", type="primary", disabled=rec is None, width="stretch")

if core.DEVICE.type == "cpu":
    st.warning("No GPU found: inference will take several minutes per study.")

# ---------------------------------------------------------------- run
key = (run_name, mode, rec["case_id"] if rec else None, rec["ct"] if rec else None, spec_name)
if go and rec is not None:
    net, cfg, ckinfo = get_model(str(run_dir), (run_dir / "best.pt").stat().st_mtime)
    t0 = time.time()
    try:
        with st.spinner(f"Running {run_name} on {label or 'the study'} ({spec_name}) ..."):
            res = core.run_case(net, rec, spec)
    except Exception as e:
        st.error(f"Could not process this study: {e}"); st.stop()
    st.session_state["res"] = {"key": key, "data": res, "seconds": time.time() - t0, "cfg": cfg, "ck": ckinfo, "has_mask": rec["has_mask"],
                               "label": label, "run_dir": str(run_dir)}

R = st.session_state.get("res")
if not R:
    st.info("Pick a model, an image and a condition in the sidebar, then press **Run detection**.")
    st.stop()
res = R["data"]

# ---------------------------------------------------------------- results
st.subheader(f"{R['cfg']['run']} on {R['label']} | condition: {R['key'][4]}")
c1, c2 = st.columns([1, 3])
thr = c1.slider("Detection threshold", 0.05, 0.95, 0.5, 0.05, help="Voxels with lesion probability above this count as lesion (0.5 was used in all evaluations).")
overlay = c1.radio("Overlay", ["Prediction contour", "Probability heatmap"])
pred, row, lesions = core.score(res["prob"], res["gt"], res["voxel_ml"], thr)

lab, n_pred = ndimage.label(pred, structure=core.STRUCT26)
m = c2.columns(5)
if R["has_mask"] and row["gt_voxels"] > 0:
    m[0].metric("Dice", f"{row['dice']:.3f}")
    m[1].metric("Lesions found", f"{row['n_detected']} / {row['n_gt_lesions']}", help="a lesion counts as found if at least 10% of its voxels are predicted")
    m[2].metric("Voxel recall", f"{row['voxel_recall']:.2f}")
    m[3].metric("False-positive blobs", f"{row['n_fp_components']}", f"{row['fp_volume_ml']:.1f} ml", delta_color="off")
    m[4].metric("Predicted volume", f"{row['pred_volume_ml']:.1f} ml")
else:
    m[0].metric("Predicted blobs", f"{n_pred}")
    m[1].metric("Predicted volume", f"{row['pred_volume_ml']:.1f} ml")
    m[2].metric("Max probability", f"{float(res['prob'].max()):.2f}")
    m[3].metric("Max SUV in prediction", f"{float(res['pet_suv'][pred].max()):.1f}" if pred.any() else "-")
    m[4].metric("Ground truth", "none" if R["has_mask"] else "not provided")

tab_slices, tab_mip, tab_les, tab_info = st.tabs(["Slices", "PET MIP", "Lesions", "Run info"])
with tab_slices:
    gt, prob = res["gt"], res["prob"]
    ref = gt if gt.any() else pred
    z0 = int(ref.sum(axis=(0, 1)).argmax()) if ref.any() else gt.shape[2] // 2
    y0 = int(ref.sum(axis=(0, 2)).argmax()) if ref.any() else gt.shape[1] // 2
    s1, s2 = st.columns(2)
    z = s1.slider("Axial slice", 0, gt.shape[2] - 1, z0)
    y = s2.slider("Coronal slice", 0, gt.shape[1] - 1, y0)
    fig = panel_figure(res, pred, "axial", z, overlay, 1.0); st.pyplot(fig); plt.close(fig)
    fig = panel_figure(res, pred, "coronal", y, overlay, res["aspect"]); st.pyplot(fig); plt.close(fig)
with tab_mip:
    fig = mip_figure(res, pred); st.pyplot(fig); plt.close(fig)
with tab_les:
    if lesions:
        df = pd.DataFrame(lesions)
        df["volume_ml"] = (df.gt_voxels * res["voxel_ml"]).round(2)
        df["covered_%"] = (df.overlap_frac * 100).round(0).astype(int)
        df["found"] = np.where(df.detected == 1, "yes", "missed")
        st.dataframe(df[["lesion_id", "volume_ml", "covered_%", "found"]], hide_index=True, width="stretch")
    else:
        st.write("No ground-truth lesions in this study." if R["has_mask"] else "No mask was provided, so there is no ground truth to compare with.")
with tab_info:
    cfg = R["cfg"]
    st.write(f"**Model** {cfg['model']} ({cfg['params_million']} M parameters), augmentation: {cfg['augment']}, patch {tuple(cfg['patch'])}, "
             f"best checkpoint from epoch {R['ck']['epoch']} (validation Dice {R['ck']['val_dice']:.3f}). Inference took {R['seconds']:.0f} s.")
    summ = Path(R["run_dir"]) / "summary_by_condition.csv"
    if summ.exists():
        s = pd.read_csv(summ); s = s[s.split == "test_lung"].set_index("condition")
        if R["key"][4] in s.index:
            r_ = s.loc[R["key"][4]]
            st.write(f"**This run on the whole lung test set under '{R['key'][4]}':** Dice {r_.dice:.3f} (95% CI {r_.dice_lo:.3f} to {r_.dice_hi:.3f}), "
                     f"lesion recall {r_.lesion_recall:.2f}, false-positive volume {r_.fp_volume_ml:.0f} ml per study.")
    st.caption("Inputs are expected as in the training data: CT in Hounsfield units, PET in SUV, both on the same voxel grid. "
               "Perturbations are applied after normalisation, exactly as in the evaluation notebook.")
