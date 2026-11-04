# PETCT-RobustSeg, lung study: code folder

Train a 2-channel PET/CT segmentation network on all cancer types, then test it on lung-cancer patients under 15 frozen image-quality perturbations (low resolution, noise, missing PET) without retraining.

Everything runs from one notebook. You only edit its first code cell. **To train a model, open the guide for that model in [`run_guides/`](run_guides/)**; it gives the exact first-cell values and the settings to change for a faster or slower machine.

## Files
| file | purpose |
|---|---|
| `notebooks/petct_lung_robustness.ipynb` | the whole pipeline: data, visualisation, perturbations, augmentation, model, training, evaluation, results, run comparison |
| `notebooks/petct_lung_robustness_B_fixed.ipynb` | the same notebook preset for run B (augmented DynUNet), with retries for out-of-memory errors during loading |
| `notebooks/week1_checks.ipynb` | Week 1 data and perturbation checks (run from the repository root layout) |
| `dashboard/` | Streamlit app to load a trained model and inspect its detections ([README](dashboard/README.md)) |
| `runs/A_dynunet_noaug/` | results of run A, including the trained weights `best.pt` (44 MB, epoch 42) |
| `run_guides/` | one guide per planned model (A to E) |
| `splits_v2.json` | the shared patient-disjoint split (train 790 / val 40 / test_lung 50 / test_neg 20 patients). Do not edit or regenerate. |
| `perturbations_frozen.json` | the 15 frozen test conditions (fingerprint `bca490f76304...`). Do not edit. |
| `case_index.csv` | every study with paths relative to `DATA_ROOT`, lesion voxel count, diagnosis |
| `make_split_v2.py` | how the split was built (for the record; not needed to run anything) |
| `requirements.txt` | packages (install the right PyTorch build first) |

## Environment used for the reported results
| item | version |
|---|---|
| OS / GPU | Windows, NVIDIA GeForce RTX 3050 6 GB Laptop GPU |
| Python | 3.14.6 |
| PyTorch | 2.11.0+cu128 |
| MONAI | 1.6.1 |
| NumPy | 2.5.2 |

Create it with:
```
python -m venv .venv
.venv\Scripts\activate                  # Linux / macOS: source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r code/requirements.txt
python -c "import torch, monai; print(torch.__version__, monai.__version__, torch.cuda.is_available())"   # must end in True
```
Older Python or other CUDA builds work too (the code needs MONAI 1.3 or newer), but exact numbers can differ slightly between library versions.

## Evaluating the trained model (run A) without retraining
The weights are in `runs/A_dynunet_noaug/` (`best.pt`, plus `config.json` with the architecture). Input: 2 channels in the order [PET, CT], PET SUV clipped to [0, 20] and CT clipped to [-100, 250] HU, each scaled to [0, 1], RAS orientation. Sliding window 96x96x96, overlap 0.25, threshold 0.5.
* **Interactive:** `pip install streamlit`, then `cd code/dashboard && streamlit run app.py` and pick `code/runs/A_dynunet_noaug` as the model folder (needs the dataset for test-set studies).
* **In code:**
```python
import torch, json
from monai.networks.nets import DynUNet
cfg = json.load(open("runs/A_dynunet_noaug/config.json")); F = cfg["filters"]
strides = [[1, 1, 1]] + [[2, 2, 2]] * (len(F) - 1)
net = DynUNet(spatial_dims=3, in_channels=2, out_channels=2, kernel_size=[[3, 3, 3]] * len(F), strides=strides,
              upsample_kernel_size=strides[1:], filters=F, norm_name=("INSTANCE", {"affine": True}),
              act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}), deep_supervision=True, deep_supr_num=2, res_block=False)
net.load_state_dict(torch.load("runs/A_dynunet_noaug/best.pt", map_location="cpu", weights_only=False)["model"]); net.eval()
```
Run A's metrics under the 15 perturbations are in `runs/A_dynunet_noaug/summary_by_condition.csv`, `per_case_metrics.csv` and `per_lesion.csv`.

## One-time setup
1. **Data.** You need the autoPET FDG dataset `PSMA-FDG-PET-CT-Lesions_v2` in nnU-Net layout (about 143 GB): a folder containing `imagesTr/` (`..._0000.nii.gz` = CT, `..._0001.nii.gz` = PET) and `labelsTr/`. File names must match `case_index.csv`, e.g. `imagesTr/fdg_0011f3deaf_03-23-2003-NA-PET-CT Ganzkoerper  primaer mit KM-10445_0001.nii.gz`. Put it on an SSD if you can: training mostly waits for volumes to load.
2. **PyTorch with CUDA.** Install the CUDA build from https://pytorch.org/get-started/locally/ (a plain `pip install torch` on Windows gives the CPU-only build). Check it:
   ```
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   It must print `True`.
3. **Everything else:** `pip install -r requirements.txt` (this includes `einops`, which SwinUNETR needs).
4. Start Jupyter and open a notebook from `code/notebooks/`. It finds `splits_v2.json` etc. in `code/` (the folder above it); runs are written to `code/runs/<RUN_NAME>/`.

## Running a model
1. Open the guide for the model in `run_guides/` and copy its values into the first code cell. Set `DATA_ROOT` to your data folder.
2. Run all cells top to bottom. The first cell stops with an error if `RUN_NAME` does not match `MODEL_NAME` / `AUGMENT`, so results from different models cannot get mixed up.
3. The model cell prints the peak GPU memory of one training step. Check it against your GPU memory before leaving the run alone (see the guide).
4. Outputs go to `runs/<RUN_NAME>/`: `best.pt`, `last.pt`, `train_log.csv`, `status.log`, `per_case_metrics.csv`, `per_lesion.csv`, `lesion_features.csv`, `summary_by_condition.csv`, `config.json`, `figures/`, `predictions/`, and `COMPLETE` when everything has finished.

**Interrupted?** Training and evaluation both resume: restart the kernel and run all cells again with the same first cell.

**Checking on a long run:** `runs/<RUN_NAME>/status.log` gets one line per epoch and per evaluated study, with a timestamp.

## Planned runs
| run | `MODEL_NAME` | `AUGMENT` | purpose | guide |
|---|---|---|---|---|
| `A_dynunet_noaug` | `dynunet` | `none` | clean baseline | [A](run_guides/A_dynunet_noaug.md) |
| `B_dynunet_aug` | `dynunet` | `perturb` | same network with augmentation: isolates the effect of augmentation (compare with A) | [B](run_guides/B_dynunet_aug.md) |
| `C_segresnet_noaug` | `segresnet` | `none` | architecture comparison against A | [C](run_guides/C_segresnet_noaug.md) |
| `D_swinunetr_noaug` | `swinunetr` | `none` | architecture comparison against A (transformer) | [D](run_guides/D_swinunetr_noaug.md) |
| `E_basic_unet_noaug` | `basic_unet` | `none` | architecture comparison against A (plain U-Net) | [E](run_guides/E_basic_unet_noaug.md) |

## Rules so that runs are comparable
* Same `splits_v2.json`, same `perturbations_frozen.json`, same metrics. The notebook checks the fingerprints, stores them in `config.json`, and the comparison cell flags runs that do not match.
* Change one thing between runs (architecture **or** augmentation). Keep `PATCH`, `BATCH`, `FILTERS`, `PATCHES_PER_VOLUME`, `MAX_EPOCHS`, `VOLUMES_PER_EPOCH`, `LR` and `SEED` at their defaults. Only lower `PATCH` / `BATCH` if the model does not fit in GPU memory, and then say so in the report (`config.json` records it).
* `LOAD_THREADS` only changes speed, never results: set it for your machine.
* Do not tune anything on `test_lung` / `test_neg`. Checkpoint selection uses `val` only.
* `EVAL_MAX_CASES` must be `None` for real results (a number is only for a quick look).

## Sending results back
Run the last cell. It writes `runs/<RUN_NAME>_results.zip` with the metrics, figures and config but **without** checkpoints. Send that zip; the receiver unzips it into their own `runs/` folder and runs the "Compare runs" section, which overlays every run on the same plots.

## Adding another architecture
Add a branch in `build_model` returning `(model, uses_deep_supervision)`. Input `(B, 2, X, Y, Z)` ([PET, CT]); output `(B, 2, X, Y, Z)` logits in eval mode. The patch size must suit the network (divisible by 16 for the U-Nets here, by 32 for SwinUNETR). Name the run `<letter>_<MODEL_NAME>_noaug` (or `_aug`).

## Metric definitions (identical for all runs)
* Dice: voxel overlap on lesion-positive studies (undefined without lesions).
* Lesion recall: ground-truth lesions are 26-connected components; detected if at least 10% of voxels are predicted (`LESION_MIN_OVERLAP`; the raw overlap fraction is saved in `per_lesion.csv`). Pooled over lesions.
* False positives: predicted components touching no ground-truth lesion (count and ml), mainly on `test_neg`.
* 95% intervals: patient-level bootstrap.

## Known limits
* `dynunet` is MONAI's nnU-Net-style `DynUNet` (about 11 M parameters at the default widths), not the official nnU-Net pipeline.
* The frozen PET noise (sigma 0.05 on the [0, 1] scale, i.e. 1 SUV) is strong relative to normal background uptake; this was noted in Week 1 and left unchanged. Even the mildest PET-noise level may therefore cost a lot of Dice.
* "Low-resolution" perturbations are synthetic and are not claimed to represent a particular scanner.
* Volumes are read from `.nii.gz` on the fly (about 4 s each), which is why each loaded volume yields several patches and why training speed depends mostly on the disk and `LOAD_THREADS`.
