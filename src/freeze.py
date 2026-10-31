"""Write configs/perturbations_frozen.json from configs/week1.yaml.

Run once, BEFORE any evaluation:   python -m src.freeze
After that, tests/test_week1.py fails if the perturbation levels in the config ever change.
"""
import json
from pathlib import Path

from . import data, perturb

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = data.load_config(ROOT / "configs" / "week1.yaml")
    rec = perturb.frozen_record(perturb.specs_from_config(cfg))
    out = ROOT / "configs" / "perturbations_frozen.json"
    out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"wrote {out}  ({len(rec['conditions'])} conditions, fingerprint {rec['fingerprint'][:12]}...)")


if __name__ == "__main__":
    main()
