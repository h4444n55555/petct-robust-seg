# Run D: `swinunetr`, no augmentation

An architecture comparison against run A: a transformer encoder instead of a convolutional one.

* **Network:** MONAI `SwinUNETR`, a Swin-transformer encoder with a convolutional decoder (Hatamizadeh et al., 2022), `feature_size = 24`, about 15.7 M parameters. It has no deep supervision and is trained from scratch (no pre-trained weights).
* **Gradient checkpointing** is switched on in `build_model`. It gives the same result with less memory (6.9 GB down to 4.6 GB at the default settings), at the cost of extra compute. Without it, this model does not fit a 6 GB GPU at the shared settings.
* **Training:** clean images only (left-right flip), as in A.
* **Compared with:** A.

## Extra requirement
`pip install einops` (it is in `requirements.txt`). Without it the model cell fails with `OptionalImportError ... einops`.

## First code cell
Change only these lines (set `DATA_ROOT` to your own data folder); leave everything else at its default.
```python
RUN_NAME   = "D_swinunetr_noaug"
MODEL_NAME = "swinunetr"
AUGMENT    = "none"
DATA_ROOT  = r"<your path>\PSMA-FDG-PET-CT-Lesions_v2"   # folder that contains imagesTr/ and labelsTr/
```
Then run all cells. `FILTERS` only applies to `dynunet` and is ignored here. `PATCH` must be divisible by 32 for this network; the default 96 is.

## What to expect
This is the heaviest model. Measured on an RTX 3050 Laptop GPU (6 GB), 16 CPU threads, 16 GB RAM:

| | value |
|---|---|
| peak GPU memory, default settings (patch 96, batch 2) | 4.4-4.6 GB |
| GPU time per training step | about 1.0 s (300 steps per epoch, about 5 min) |
| time per epoch | about 5-6 min |
| training, 60 epochs | about 5-6 h (estimate) |
| evaluation, 71 studies x 15 conditions | about 3-4 h (2-3.7 min per study, measured) |

## Settings for your machine
**Keep `PATCH = (96, 96, 96)` and `BATCH = 2`**, the same as run A, whenever it fits.

| your machine | what to change |
|---|---|
| GPU with 6 GB or more | nothing (needs 4.6 GB; on exactly 6 GB, close other programs that use the GPU) |
| GPU with less than 6 GB | `PATCH = (64, 64, 64)` (1.9 GB). Training then differs from A: mention it in the report. |
| 8+ CPU cores and 32 GB+ RAM | `LOAD_THREADS = 5` or `6` (faster, results do not change; less important here because the GPU is the bottleneck) |
| 16 GB RAM or less | keep `LOAD_THREADS = 3` |
| no NVIDIA GPU | not possible in reasonable time |

Do not raise `BATCH` to 4: at patch 96 it needs about 8.6 GB. Patch 128 / batch 2 did not fit in 6 GB.

**Check the memory line.** The model cell prints `peak GPU memory ... X GB of Y GB`. Leave at least 0.5 GB spare. On Windows, a model that does not fit often does **not** fail with an error. It spills into normal RAM and runs 5-20 times slower: without checkpointing this model took about 7 s per step instead of 1 s. If an epoch takes much longer than about 6 min, this is the likely cause.

## Before trusting the results
* `runs/D_swinunetr_noaug/COMPLETE` exists.
* `EVAL_MAX_CASES = None`.

Then run the last cell and send `runs/D_swinunetr_noaug_results.zip`.
