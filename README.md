# PETCT-RobustSeg

Robustness of PET/CT lesion segmentation to image-quality changes

**Course:** AI512
**Institution:** Indian Institute of Technology Ropar

## Team

| Name | Entry number |
|---|---|
| Nongmaithem Hans | 2024AIB1011 |
| Ravikant Sharma | 2024AIB1013 |
| Rishav Kumar | 2024AIB1014 |

## About

PET and CT scans are often used together to find tumours. PET shows which tissue is metabolically active, and CT shows the anatomy. Automatic segmentation models can combine both, but real scans vary in quality, so it matters how well such a model holds up when its input gets worse.

This project measures exactly that. A small two-channel PET+CT U-Net is trained once on clean images and then tested, without retraining, on degraded versions of the same scans:

- Lower spatial resolution
- Added intensity noise
- A missing PET channel

Performance is reported with Dice and lesion recall as the degradation gets stronger, along with example overlays. The goal is to understand how much each channel contributes and where the model breaks, not to build a new architecture.

## Data

The project uses the FDG studies of the public autoPET dataset: 1,014 whole-body PET/CT studies from 900 patients, each with an expert lesion mask. The data is split by patient, so no patient appears in more than one of the training, validation and test sets. The dataset is not included in this repository.

## Repository layout
| path | content |
|---|---|
| `src/`, `tests/`, `configs/`, `results/week1/` | Week 1: data loading, preprocessing, the perturbations, tests, the frozen perturbation list and the data index and split |
| `code/` | Lung robustness study: shared split v2, frozen perturbations, `notebooks/` (training and evaluation), `run_guides/`, `dashboard/`, and `runs/` with the results and trained weights of run A |

The training and evaluation steps, the environment used for the reported results (Python 3.14.6, PyTorch 2.11.0+cu128, MONAI 1.6.1) and how to load the trained model are described in [`code/README.md`](code/README.md).

## Contents

The repository contains the data loading and preprocessing code, the code that applies the degradations, tests, and a fast local PET/CT viewer for browsing the studies, lesions and degraded versions of each scan.
