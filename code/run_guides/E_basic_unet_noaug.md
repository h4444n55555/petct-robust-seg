# Run E: `basic_unet`, no augmentation

An architecture comparison against run A: a plain 3D U-Net without nnU-Net's extras.

* **Network:** MONAI `BasicUNet`, a classic 3D U-Net with features `(16, 32, 64, 128, 256, 16)`, about 5.7 M parameters. It has no deep supervision. It is the lightest and fastest of the five models.
* **Training:** clean images only (left-right flip), as in A.
* **Compared with:** A.

## First code cell
Change only these lines (set `DATA_ROOT` to your own data folder); leave everything else at its default.
```python
RUN_NAME   = "E_basic_unet_noaug"
MODEL_NAME = "basic_unet"
AUGMENT    = "none"
DATA_ROOT  = r"<your path>\PSMA-FDG-PET-CT-Lesions_v2"   # folder that contains imagesTr/ and labelsTr/
```
Then run all cells. `FILTERS` only applies to `dynunet` and is ignored here.

## What to expect
Measured on an RTX 3050 Laptop GPU (6 GB), 16 CPU threads, 16 GB RAM:

| | value |
|---|---|
| peak GPU memory, default settings (patch 96, batch 2) | 1.3 GB |
| GPU time per training step | 0.21 s (300 steps per epoch, about 1 min) |
| time per epoch | about 4 min (almost all waiting for volumes to load) |
| training, 60 epochs | about 4 h (estimate) |
| evaluation, 71 studies x 15 conditions | about 1 h (estimate) |

## Settings for your machine
**Keep `PATCH = (96, 96, 96)` and `BATCH = 2`**, the same as run A, even on a bigger GPU.

| your machine | what to change |
|---|---|
| GPU with 2 GB or more | nothing (needs 1.3 GB) |
| GPU with less than 2 GB | `PATCH = (64, 64, 64)` (0.5 GB). Training then differs from A: mention it in the report. |
| 8+ CPU cores and 32 GB+ RAM | `LOAD_THREADS = 5` or `6`. This is the main speed-up for this model, since the GPU mostly waits for data. Results do not change. |
| 16 GB RAM or less | keep `LOAD_THREADS = 3` |
| data on a slow or external hard disk | copy it to an SSD if possible |
| no NVIDIA GPU | not practical (tens of hours on a CPU) |

For reference, other sizes measured on the 6 GB GPU: patch 96 / batch 4 needs 2.6 GB, and patch 128 / batch 2 needs 3.0 GB. Do not use them unless every run uses them.

**Check the memory line.** The model cell prints `peak GPU memory ... X GB of Y GB`. Leave at least 0.5 GB spare. On Windows, a model that does not fit often runs 5-20 times slower instead of failing with an error.

## Before trusting the results
* `runs/E_basic_unet_noaug/COMPLETE` exists.
* `EVAL_MAX_CASES = None`.

Then run the last cell and send `runs/E_basic_unet_noaug_results.zip`.
