"""Build splits_v2.json: patient-disjoint split for the LUNG-focused study.

Design (v2, 2026-10-26)
  * Train on ALL cancer types (+ lesion-free studies), evaluate on LUNG cancer only.
  * test_lung : 50 lung-cancer patients (the 36 v1 lung test patients + 14 more from v1 train).
                Headline results, per-lesion analysis.
  * test_neg  : 20 lesion-free (NEGATIVE) patients, used ONLY for the false-positive-under-perturbation question.
  * val       : 40 patients (18 lung, 14 other cancers, 8 lesion-free) for checkpoint selection.
  * train     : everything else (non-lung patients that were in the v1 test set are now training data).
Selection inside each group is by SHA-256(seed|patient_id) rank, so it is reproducible and version-independent.
The v1 split (results/week1/splits.json) is left untouched.
Run:  python make_split_v2.py <index.csv> <v1 splits.json> <out splits_v2.json>
"""
import hashlib, json, sys
import pandas as pd

SEED = 42
N_TEST_LUNG_EXTRA, N_TEST_NEG = 14, 20            # 36 + 14 = 50 lung test patients
N_VAL_OTHER_POS, N_VAL_NEG = 14, 8                # val: 18 lung (v1 val) + 14 other cancers + 8 lesion-free


def rank(pid):
    return hashlib.sha256(f"{SEED}|{pid}".encode()).hexdigest()


def main(index_csv, v1_json, out_json):
    df = pd.read_csv(index_csv)
    v1 = json.load(open(v1_json))["patients"]
    v1_of = {p: s for s, v in v1.items() for p in v}
    pat = df.groupby("patient_id").agg(dx=("diagnosis", "first"), pos=("lesion_voxels", lambda s: bool((s > 0).any()))).reset_index()
    pat["v1"] = pat.patient_id.map(v1_of)
    pat["rank"] = pat.patient_id.map(rank)
    lung = pat[pat.dx == "LUNG_CANCER"]
    other_pos = pat[(pat.dx.isin(["LYMPHOMA", "MELANOMA"])) & pat.pos]
    neg = pat[(pat.dx == "NEGATIVE") & ~pat.pos]

    def pick(frame, v1_split, n):
        return list(frame[frame.v1 == v1_split].sort_values("rank").patient_id[:n])

    test_lung = list(lung[lung.v1 == "test"].patient_id) + pick(lung, "train", N_TEST_LUNG_EXTRA)
    val_lung = list(lung[lung.v1 == "val"].patient_id)
    test_neg = pick(neg, "test", N_TEST_NEG)
    val_neg = pick(neg, "val", N_VAL_NEG)
    vo = other_pos[other_pos.v1 == "val"]
    val_other = []                                    # balance lymphoma / melanoma
    for dx in ("LYMPHOMA", "MELANOMA"):
        val_other += list(vo[vo.dx == dx].sort_values("rank").patient_id[: N_VAL_OTHER_POS // 2])
    val = val_lung + val_other + val_neg
    taken = set(test_lung) | set(test_neg) | set(val)
    train = [p for p in pat.patient_id if p not in taken]
    patients = {"train": sorted(train), "val": sorted(val), "test_lung": sorted(test_lung), "test_neg": sorted(test_neg)}
    names = list(patients)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (set(patients[a]) & set(patients[b])), f"leak {a}/{b}"
    assert sum(len(v) for v in patients.values()) == len(pat)
    cases = {k: sorted(df[df.patient_id.isin(v)].case_id) for k, v in patients.items()}
    out = {"version": 2, "seed": SEED, "note": "train on all cancers; evaluate on lung (test_lung) + lesion-free (test_neg, false positives only)",
           "patients": patients, "cases": cases}
    json.dump(out, open(out_json, "w"), indent=1)
    rows = []
    d2 = df.assign(split=df.patient_id.map({p: s for s, v in patients.items() for p in v}), pos=df.lesion_voxels > 0)
    for s in names:
        x = d2[d2.split == s]
        rows.append({"split": s, "patients": x.patient_id.nunique(), "studies": len(x), "lesion_positive": int(x.pos.sum()),
                     "lesion_free": int((~x.pos).sum()), **{f"dx_{k}": int((x.diagnosis == k).sum()) for k in ["LUNG_CANCER", "LYMPHOMA", "MELANOMA", "NEGATIVE"]}})
    summ = pd.DataFrame(rows)
    print(summ.to_string(index=False))
    summ.to_csv(out_json.replace(".json", "_summary.csv"), index=False)


if __name__ == "__main__":
    main(*sys.argv[1:4])
