# Run C: `segresnet`, no augmentation

An architecture comparison against run A: same data, same training settings, different network.

* **Network:** MONAI `SegResNet`, an encoder-decoder with residual blocks (Myronenko, BraTS 2018), about 4.7 M parameters. It has no deep supervision.
* **Training:** clean images only (left-right flip), as in A.
* **Compared with:** A.

## First code cell
Change only these lines (set `DATA_ROOT` to your own data folder); leave everything else at its default.
```python
RUN_NAME   = "C_segresnet_noaug"
MODEL_NAME = "segresnet"
AUGMENT    = "none"
DATA_ROOT  = r"<your path>\PSMA-FDG-PET-CT-Lesions_v2"   # folder that contains imagesTr/ and labelsTr/
```
Then run all cells. `FILTERS` only applies to `dynunet` and is ignored here.

## What to expect
Measured on an RTX 3050 Laptop GPU (6 GB), 16 CPU threads, 16 GB RAM:

| | value |
|---|---|
| peak GPU memory, default settings (patch 96, batch 2) | 2.4 GB |
| GPU time per training step | 0.26 s (300 steps per epoch, about 1.3 min) |
| time per epoch | about 4-5 min (mostly waiting for volumes to load) |
| training, 60 epochs | about 4-5 h (estimate) |
| evaluation, 71 studies x 15 conditions | about 1 h (estimate) |

## Settings for your machine
**Keep `PATCH = (96, 96, 96)` and `BATCH = 2`**, the same as run A, even on a bigger GPU.

| your machine | what to change |
|---|---|
| GPU with 4 GB or more | nothing (needs 2.4 GB) |
| GPU with less than 4 GB | `PATCH = (64, 64, 64)` (0.8 GB). Training then differs from A: mention it in the report. |
| 8+ CPU cores and 32 GB+ RAM | `LOAD_THREADS = 5` or `6` (faster, results do not change) |
| 16 GB RAM or less | keep `LOAD_THREADS = 3` |
| data on a slow or external hard disk | copy it to an SSD if possible |
| no NVIDIA GPU | not practical (about 55 h of training on a CPU) |

For reference, other sizes measured on the 6 GB GPU: patch 96 / batch 4 needs 4.6 GB, and patch 128 / batch 2 needs 5.5 GB. Do not use them unless every run uses them.

**Check the memory line.** The model cell prints `peak GPU memory ... X GB of Y GB`. Leave at least 0.5 GB spare. On Windows, a model that does not fit often runs 5-20 times slower instead of failing with an error.

## Before trusting the results
* `runs/C_segresnet_noaug/COMPLETE` exists.
* `EVAL_MAX_CASES = None`.

Then run the last cell and send `runs/C_segresnet_noaug_results.zip`.
