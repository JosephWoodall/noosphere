#!/usr/bin/env python3
"""
Analyze LOSO results and regenerate key LaTeX table fragments + article summary.

Usage:
    .venv/bin/python v2_digital_self_replication/scripts/analyze_results.py \
        --v1 v2_digital_self_replication/logs/loso_results.json \
        --v3 v2_digital_self_replication/logs/loso_results_v3.json

Prints:
  - Comparison table (v1 vs v3)
  - Key numbers for article copy-paste
  - Wilcoxon p-values for primary comparisons
  - Decision: which checkpoint to use for the paper
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def subject_means(results: dict, cond: str) -> list[float]:
    return [s[cond]["mean"] for s in results["subject_results"] if cond in s]


def fmt(agg: dict) -> str:
    return f"{agg['mean']*100:.1f}% [{agg['ci_lo']*100:.1f}–{agg['ci_hi']*100:.1f}%]"


def wilcoxon_p(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    try:
        _, p = wilcoxon(a[:n], b[:n], alternative="two-sided")
        return p
    except Exception:
        return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v1", default="v2_digital_self_replication/logs/loso_results.json")
    p.add_argument("--v3", default="v2_digital_self_replication/logs/loso_results_v3.json")
    args = p.parse_args()

    v1 = load(args.v1) if Path(args.v1).exists() else None
    v3 = load(args.v3) if Path(args.v3).exists() else None

    if v3 is None:
        print("loso_results_v3.json not yet available.")
        if v1:
            print("v1 aggregate:")
            for k, v in v1["aggregate"].items():
                print(f"  {k}: {fmt(v)}")
        return

    print("=" * 70)
    print("LOSO v3 Results (all conditions, 20 subjects)")
    print("=" * 70)

    chance = v3["chance"]
    agg = v3["aggregate"]

    # Primary comparison table
    conditions_order = [
        ("encoder_ft_cls_planned",     "JEPA + world-model planner + cls [PRIMARY]"),
        ("encoder_ft_cls_cns_planned", "CNS encoder + world-model planner + cls [Phase 2+WM]"),
        ("encoder_ft_cls",             "JEPA encoder + cls head (e2e FT)"),
        ("encoder_ft_cls_cns",         "CNS cross-modal encoder + cls head [Phase 2]"),
        ("ablation_encoder_cls",       "Random encoder + cls head [ablation]"),
        ("zero_shot_probe",            "JEPA encoder + LogReg (zero-shot)"),
        ("jepa_ft",                    "JEPA + IntentDecoder FT"),
        ("csp_lda",                    "CSP + LDA"),
        ("mdm",                        "MDM (Riemannian)"),
    ]

    for cond, label in conditions_order:
        if cond not in agg:
            continue
        a = agg[cond]
        delta = (a["mean"] - chance) * 100
        print(f"  {label:<50} {a['mean']*100:5.1f}%  [{a['ci_lo']*100:.1f}–{a['ci_hi']*100:.1f}%]  "
              f"Δchance={delta:+.1f}pp")

    print()

    # Key statistical comparisons
    print("Statistical comparisons (Wilcoxon, two-sided):")
    enc_ft = subject_means(v3, "encoder_ft_cls")
    abl_enc = subject_means(v3, "ablation_encoder_cls")
    csp = subject_means(v3, "csp_lda")
    chance_arr = [chance] * len(enc_ft)

    p_vs_abl = wilcoxon_p(enc_ft, abl_enc)
    p_vs_chance = wilcoxon_p(enc_ft, chance_arr)
    p_vs_csp = wilcoxon_p(enc_ft, csp)
    print(f"  encoder_ft_cls vs ablation_encoder_cls: p={p_vs_abl:.4f}")
    print(f"  encoder_ft_cls vs chance:               p={p_vs_chance:.4f}")
    print(f"  encoder_ft_cls vs csp_lda:              p={p_vs_csp:.4f}")

    if "encoder_ft_cls_cns" in agg:
        enc_cns = subject_means(v3, "encoder_ft_cls_cns")
        p_cns_vs_ft = wilcoxon_p(enc_cns, enc_ft)
        print(f"  encoder_ft_cls_cns vs encoder_ft_cls:   p={p_cns_vs_ft:.4f}")

    print()

    # Phase 1 vs Phase 3 comparison
    if v1 and "encoder_ft_cls" in v3["aggregate"]:
        v1_zsp = v1["aggregate"].get("zero_shot_probe", {})
        v3_enc = v3["aggregate"]["encoder_ft_cls"]
        v3_cns = v3["aggregate"].get("encoder_ft_cls_cns")
        print("Phase comparison:")
        if v1_zsp:
            print(f"  Phase 1 zero_shot_probe:   {fmt(v1_zsp)}")
        print(f"  Phase 1 encoder_ft_cls:    {fmt(v3_enc)}")
        if v3_cns:
            delta_cns = (v3_cns["mean"] - v3_enc["mean"]) * 100
            print(f"  Phase 2 (CNS cross-modal): {fmt(v3_cns)}  Δ={delta_cns:+.1f}pp")
            if delta_cns > 1.0:
                print("  → Phase 2 IMPROVES results. Use CNS checkpoint.")
            elif delta_cns < -1.0:
                print("  → Phase 2 HURTS results. Use Phase 1 checkpoint.")
            else:
                print("  → Phase 2 shows no significant improvement (Δ<1pp). Use Phase 1 (synthetic teacher insufficient).")

    print()
    print("=" * 70)
    print("KEY NUMBERS FOR ARTICLE:")
    print("=" * 70)
    if "encoder_ft_cls" in agg:
        a = agg["encoder_ft_cls"]
        print(f"  Primary metric (encoder_ft_cls): {a['mean']*100:.1f}% [95% CI: {a['ci_lo']*100:.1f}–{a['ci_hi']*100:.1f}%]")
    if "ablation_encoder_cls" in agg:
        a = agg["ablation_encoder_cls"]
        print(f"  Ablation (random encoder):       {a['mean']*100:.1f}% [95% CI: {a['ci_lo']*100:.1f}–{a['ci_hi']*100:.1f}%]")
    if "csp_lda" in agg:
        a = agg["csp_lda"]
        print(f"  CSP+LDA baseline:                {a['mean']*100:.1f}% [95% CI: {a['ci_lo']*100:.1f}–{a['ci_hi']*100:.1f}%]")
    if "encoder_ft_cls_cns" in agg:
        a = agg["encoder_ft_cls_cns"]
        print(f"  Phase 2 CNS cross-modal:         {a['mean']*100:.1f}% [95% CI: {a['ci_lo']*100:.1f}–{a['ci_hi']*100:.1f}%]")

    # LaTeX table fragment
    print()
    print("LaTeX table rows (aggregate table):")
    latex_rows = [
        ("encoder_ft_cls",       r"\textbf{ZOH-SSM JEPA + cls (e2e FT)}"),
        ("ablation_encoder_cls", r"Random encoder + cls (e2e FT)"),
        ("zero_shot_probe",      r"JEPA + LogReg (zero-shot)"),
        ("csp_lda",              r"CSP+LDA"),
        ("mdm",                  r"MDM (Riemannian)"),
        ("encoder_ft_cls_cns",   r"ZOH-SSM Cross-Modal JEPA + cls"),
    ]
    for cond, label in latex_rows:
        if cond not in agg:
            continue
        a = agg[cond]
        bold_open = r"\textbf{" if cond == "encoder_ft_cls" else ""
        bold_close = "}" if cond == "encoder_ft_cls" else ""
        print(f"  {label} & {bold_open}{a['mean']*100:.1f}{bold_close} & "
              f"{a['ci_lo']*100:.1f} & {a['ci_hi']*100:.1f} \\\\")


if __name__ == "__main__":
    main()
