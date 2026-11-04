# PET/CT lesion segmentation dashboard

Load a trained model, pick a study (held-out test set, uploaded NIfTI files or files on disk), optionally degrade it with one of the
15 frozen perturbations, and compare the model's detection (red) with the ground truth (green).

```
pip install streamlit          # on top of the training environment
cd code/dashboard
streamlit run app.py
```

* **Model**: any run folder containing `best.pt` and `config.json` (written by the training notebook), e.g. `code/runs/A_dynunet_noaug`
  or `PETCT_C_E_results/C_segresnet_noaug`. List the parent folders in the sidebar, one per line.
* **Image**: a test-set study (needs the dataset folder with `imagesTr/` and `labelsTr/`), or your own CT (HU) and PET (SUV) on the same voxel grid;
  the lesion mask is optional.
* **Condition**: clean, low resolution x1.5 to x4, Gaussian noise on PET / CT / both, or PET removed (applied exactly as in the evaluation).
* After a run, the threshold slider, slice sliders and overlay can be changed without recomputing. Inference takes roughly 20-60 s per study on a GPU.

`petct_core.py` is copied from `notebooks/petct_lung_robustness.ipynb` (preprocessing, perturbations, models, scoring), so results match the notebook.
