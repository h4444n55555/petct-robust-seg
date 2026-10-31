# Run A: `dynunet`, no augmentation (clean baseline)

The reference model. Every other run is compared against it.

* **Network:** MONAI `DynUNet`, nnU-Net-style 3D U-Net (instance norm, leaky ReLU, deep supervision), about 11 M parameters.
* **Training:** clean images only (a random left-right flip is the only augmentation).
* **Compared with:** B (effect of augmentation), C, D and E (effect of architecture).

## First code cell
Change only these lines (set `DATA_ROOT` to your own data folder); leave everything else at its default.
```python
RUN_NAME   = "A_dynunet_noaug"
MODEL_NAME = "dynunet"
AUGMENT    = "none"
DATA_ROOT  = r"<your path>\PSMA-FDG-PET-CT-Lesions_v2"   # folder that contains imagesTr/ and labelsTr/
```
Then run all cells.

## What to expect
Measured on an RTX 3050 Laptop GPU (6 GB), 16 CPU threads, 16 GB RAM:

| | value |
|---|---|
| peak GPU memory, default settings (patch 96, batch 2) | 1.7 GB |
| GPU time per training step | 0.30 s (300 steps per epoch, about 1.5 min) |
| time per epoch | about 4-5 min (mostly waiting for volumes to load) |
| training, 60 epochs | about 4-5 h (estimate) |
| evaluation, 71 studies x 15 conditions | about 1 h (0.9 min per study, measured) |

## Settings for your machine
**Keep `PATCH = (96, 96, 96)`, `BATCH = 2` and `FILTERS` at their defaults**, even on a bigger GPU. All runs must use the same values to be comparable.

| your machine | what to change |
|---|---|
| GPU with 4 GB or more | nothing (needs 1.7 GB) |
| GPU with less than 4 GB | `PATCH = (64, 64, 64)` (0.6 GB). Training then differs from the other runs: mention it in the report. |
| 8+ CPU cores and 32 GB+ RAM | `LOAD_THREADS = 5` or `6`. This is the main speed-up, since training waits for data. Results do not change. |
| 16 GB RAM or less | keep `LOAD_THREADS = 3` (each loading thread holds roughly 1 GB while it reads a volume) |
| data on a slow or external hard disk | copy it to an SSD if possible; loading is the bottleneck |
| no NVIDIA GPU | not practical: on a CPU one epoch takes about 55 min (about 55 h of training) |

**Check the memory line.** The model cell prints `peak GPU memory for a training step ... X GB of Y GB`. Leave at least 0.5 GB spare. On Windows, a model that does not fit often does **not** fail with an out-of-memory error. It spills into normal RAM and runs 5-20 times slower. If epochs take far longer than the times above, this is the likely cause.

## Before trusting the results
* `runs/A_dynunet_noaug/COMPLETE` exists.
* `EVAL_MAX_CASES = None` (the default).
* The evaluation cell printed 50 `test_lung` and 21 `test_neg` studies.

Then run the last cell and send `runs/A_dynunet_noaug_results.zip`.
