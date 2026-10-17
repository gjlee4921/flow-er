"""Template-copy and classical baselines (spec §3 rows 1-2) -> predictions in data/_results.

Reads the per-sample cache (no FCS), takes every gate's template from the evaluation's reference
sample (ALS: donor 2008's unstimulated well) and writes per-cell predictions for every sample of
the evaluation, in the predictions format of docs/cache-format.md. Prints a quick per-gate
summary; scripts/report.py makes the full tables.

    uv run python scripts/run_baselines.py --evaluation als
"""

import argparse
import sys
import time
from collections import defaultdict

import flowkit as fk
import numpy as np
import pandas as pd
from build_dataset import is_cytokine, name_and_path, reference_sample, workspace_path

from flow_er.ml import baseline, cache, splits


def f1(pred: np.ndarray, true: np.ndarray) -> float:
    if true.sum() < 20:
        return np.nan
    tp = (pred & true).sum()
    return 2 * tp / (pred.sum() + true.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evaluation", default="als")
    parser.add_argument("--fit-roles", nargs="+", default=["train", "finetune"],
                        help="roles whose unstimulated wells set the cytokine background rate")
    args = parser.parse_args()
    t0 = time.time()

    split, registry = splits.load_split(), splits.load_registry()
    stages = dict(splits._evaluations(split))[args.evaluation]
    by_key = {splits.key(s): s for s in registry["samples"]}
    role_of = {k: role for role, keys in stages.items() for k in keys}
    samples = [by_key[k] for k in sorted(role_of)]
    print(f"{len(samples)} samples")

    # Each gate's template comes from its reference sample (ALS: donor 2008 for every gate;
    # elsewhere the first training sample of the workspace that has the gate).
    refs: dict[str, tuple] = {}

    def template(s: dict, gk: str):
        ref = reference_sample(s, gk, stages["train"], by_key)
        if ref is None:
            return None
        rk = splits.key(ref)
        if rk not in refs:
            gs = fk.Workspace(str(workspace_path(ref)), load_missing_file_data=True)
            refs[rk] = (gs.get_gating_strategy(ref["sample"]), cache.load_sample(ref))
        gs, rc = refs[rk]
        return gs.get_gate(*name_and_path(gk)), gs.transformations, rc

    def unstim_of(s: dict) -> dict | None:
        return next((u for u in samples if u["individual"] == s["individual"]
                     and u["condition"] == "unstimulated"), None)

    # Cytokine background rate: median expert positive fraction in the training donors'
    # unstimulated wells.
    background = defaultdict(list)
    for s in samples:
        if role_of[splits.key(s)] in args.fit_roles and s["condition"] == "unstimulated":
            c = cache.load_sample(s)
            for gk in c.gate_keys:
                if is_cytokine(s, gk):
                    background[gk].append(c.labels[c.gate(gk)][c.in_parent(c.gate(gk))].mean())
    background = {gk: float(np.median(v)) for gk, v in background.items()}

    shifts = {}
    rows = []
    for s in samples:            # one sample in memory at a time
        k = splits.key(s)
        c = cache.load_sample(s)
        for gk in s["usable_gates"]:
            found = template(s, gk)
            if found is None or not cache.gate_path(s, gk).exists():
                continue                    # no training sample has this gate: not scored
            gate, xforms, ref_cache = found
            g = c.gate(gk)
            truth, meta = cache.load_gate(s, gk)
            # Predict exactly the model file's cells (all parent cells, or its 200k subsample).
            cells = truth["cell_index"].astype(np.int64)
            assert c.in_parent(g)[cells].all(), (k, gk)
            raw = c.gate_raw(g, cells)
            chans, pnrs = c.gate_channels(g)
            preds = {"template_copy": baseline.apply_template(gate, xforms, raw, chans, pnrs)}
            if gk in background and (u := unstim_of(s)) is not None:
                key = (s["individual"], gk)
                if key not in shifts:
                    uc = cache.load_sample(u)
                    ug = uc.gate(gk)
                    shifts[key] = (baseline.threshold_shift(
                        gate, xforms, uc.gate_raw(ug, uc.in_parent(ug)), chans, pnrs,
                        background[gk]), 0.0)
                shift = shifts[key]
            else:
                shift = baseline.peak_shift(c.images[g], ref_cache.images[ref_cache.gate(gk)],
                                            ref_cache.masks[ref_cache.gate(gk)], len(cells))
            preds["classical"] = baseline.apply_template(gate, xforms, raw, chans, pnrs, shift)
            true = truth["cell_label"]
            cnt = meta["counts"]
            noise = 100 * abs(cnt["flowjo"] / max(cnt["flowjo_parent"], 1)
                              - cnt["flowkit"] / max(cnt["flowkit_parent"], 1))
            for method, pred in preds.items():
                out = cache.results_path(args.evaluation, method, s, gk)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, pred=pred)
                rows.append({"role": role_of[k], "gate": gk, "method": method,
                             "f1": f1(pred, true),
                             "freq_err": 100 * abs(pred.mean() - true.mean()) if len(true) else 0,
                             "noise": noise})
    print(f"predictions written in {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    df["gate"] = df["gate"].str.split("/").str[-2:].str.join(" < ").str.slice(-48)
    for role in ("val", "test"):
        d = df[df["role"] == role]
        t = d.pivot_table(index="gate", columns="method", values=["f1", "freq_err"],
                          aggfunc={"f1": "median", "freq_err": "mean"}, sort=False)
        t["noise_pp"] = d.groupby("gate", sort=False)["noise"].mean()
        print(f"\n{role}: median F1 (wells with >= 20 positives) and mean |freq error| (pp)")
        print(t.round(3).to_string())
    print(f"\ncytokine background rates fitted on {args.fit_roles}: "
          f"{ {g.split('/')[-2][:3] + ' ' + g.split('/')[-1]: round(100 * r, 3) for g, r in background.items()} } %")
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
