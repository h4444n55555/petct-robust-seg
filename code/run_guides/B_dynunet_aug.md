# Run B: `dynunet` with perturbation augmentation

The same network and settings as run A. The only difference is that training patches are randomly degraded. Comparing B with A shows how much augmentation improves robustness.

* **Network:** MONAI `DynUNet`, the same as A (about 11 M parameters).
* **Augmentation (`AUGMENT = "perturb"`):** for each training patch, a left-right flip plus:
  * random blur (low-resolution factor 1.2-4, probability 0.3);
  * random Gaussian noise (sigma 0.02-0.15 on PET, CT or both, probability 0.4);
  * PET dropout (PET set to 0, probability 0.15).

  The ranges bracket the frozen test levels without copying them.
* **Compared with:** A only. For a clean comparison, B must use exactly the same settings as A, apart from `AUGMENT`.

## First code cell
Change only these lines (set `DATA_ROOT` to your own data folder); leave everything else at its default.
```python
RUN_NAME   = "B_dynunet_aug"
MODEL_NAME = "dynunet"
AUGMENT    = "perturb"
DATA_ROOT  = r"<your path>\PSMA-FDG-PET-CT-Lesions_v2"   # folder that contains imagesTr/ and labelsTr/
```
Then run all cells. The augmentation preview figure (section 5) should show visibly blurred, noisy or PET-free patches in some of the six draws. If all six look clean, `AUGMENT` is not set.

## What to expect
Measured on an RTX 3050 Laptop GPU (6 GB), 16 CPU threads, 16 GB RAM:

| | value |
|---|---|
| peak GPU memory, default settings (patch 96, batch 2) | 1.7 GB |
| GPU time per training step | 0.30 s (300 steps per epoch, about 1.5 min) |
| time per epoch | about 4-5 min (augmentation adds a little CPU work) |
| training, 60 epochs | about 4-5 h (estimate) |
| evaluation, 71 studies x 15 conditions | about 1 h |

Clean-image Dice may come out slightly lower than A's. That is a normal cost of augmentation; the interesting part is the curves under perturbation.

## Settings for your machine
**Use the same `PATCH`, `BATCH` and `FILTERS` as run A** (defaults: `(96, 96, 96)`, `2`, `(24, 48, 96, 192, 320)`). If A was run with smaller values because of memory, B must use the same smaller values.

| your machine | what to change |
|---|---|
| GPU with 4 GB or more | nothing (needs 1.7 GB) |
| GPU with less than 4 GB | `PATCH = (64, 64, 64)`, and only if A also used it |
| 8+ CPU cores and 32 GB+ RAM | `LOAD_THREADS = 5` or `6` (faster, results do not change) |
| 16 GB RAM or less | keep `LOAD_THREADS = 3` |
| data on a slow or external hard disk | copy it to an SSD if possible |
| no NVIDIA GPU | not practical (about 55 h of training on a CPU) |

**Check the memory line.** The model cell prints `peak GPU memory ... X GB of Y GB`. Leave at least 0.5 GB spare. On Windows, a model that does not fit often runs 5-20 times slower instead of failing with an error.

## Before trusting the results
* `runs/B_dynunet_aug/COMPLETE` exists.
* `EVAL_MAX_CASES = None`.
* `config.json` shows `"augment": "perturb"`.

Then run the last cell and send `runs/B_dynunet_aug_results.zip`.
