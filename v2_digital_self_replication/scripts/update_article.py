#!/usr/bin/env python3
"""
Update riemannian_s4_v2.tex with results from:
  - loso_results_v5.json        (LOSO eval with ac_ssm condition)
  - closed_loop_sim_v2_full.json (four-controller closed-loop sim)
  - loso_results_eegconformer.json (optional: adds EEGConformer row)
  - window_sensitivity.json        (optional: fills sensitivity table)

Run after all evals complete:
  python v2_digital_self_replication/scripts/update_article.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ARTICLE       = Path("v2_digital_self_replication/articles/riemannian_s4_v2.tex")
LOSO_V5       = Path("v2_digital_self_replication/logs/loso_results_v5.json")
SIM_V2        = Path("v2_digital_self_replication/logs/closed_loop_sim_v2_full.json")
LOSO_CONFORM  = Path("v2_digital_self_replication/logs/loso_results_eegconformer.json")
WIN_SENS      = Path("v2_digital_self_replication/logs/window_sensitivity.json")


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def fmt_ci(a: dict) -> str:
    return f"{a['mean']*100:.1f} & {a['ci_lo']*100:.1f} & {a['ci_hi']*100:.1f}"


def wilcoxon_p(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    try:
        _, p = wilcoxon(a[:n], b[:n], alternative="two-sided")
        return float(p)
    except Exception:
        return float("nan")


def wilcoxon_p_gt(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    try:
        _, p = wilcoxon(a[:n], b[:n], alternative="greater")
        return float(p)
    except Exception:
        return float("nan")


def subject_means(results: list[dict], cond: str) -> list[float]:
    return [s[cond]["mean"] for s in results if cond in s and not np.isnan(s[cond]["mean"])]


def main():
    missing = [p for p in [LOSO_V5, SIM_V2] if not p.exists()]
    if missing:
        print(f"Missing result files: {[str(p) for p in missing]}")
        print("Run both evaluations first.")
        return 1

    loso = load(LOSO_V5)
    sim  = load(SIM_V2)

    agg  = loso["aggregate"]
    subj = loso["subject_results"]
    chance = loso["chance"]
    n_subj = loso["n_subjects"]

    # ── Pull LOSO numbers ─────────────────────────────────────────────────────
    enc     = agg.get("encoder_ft_cls", {})
    ac      = agg.get("ac_ssm", {})
    abl     = agg.get("ablation_encoder_cls", {})
    csp     = agg.get("csp_lda", {})
    mdm     = agg.get("mdm", {})
    wm_mlp  = agg.get("encoder_ft_cls_planned", {})

    enc_means  = subject_means(subj, "encoder_ft_cls")
    ac_means   = subject_means(subj, "ac_ssm")
    abl_means  = subject_means(subj, "ablation_encoder_cls")
    csp_means  = subject_means(subj, "csp_lda")
    wm_means   = subject_means(subj, "encoder_ft_cls_planned")

    p_ac_vs_enc  = wilcoxon_p(ac_means, enc_means)
    p_ac_vs_abl  = wilcoxon_p(ac_means, abl_means)
    p_ac_vs_csp  = wilcoxon_p(ac_means, csp_means)
    p_ac_vs_wm   = wilcoxon_p(ac_means, wm_means)
    p_enc_vs_abl = wilcoxon_p(enc_means, abl_means)
    p_enc_vs_csp = wilcoxon_p(enc_means, csp_means)

    # ── Pull closed-loop sim numbers ──────────────────────────────────────────
    sa = sim["aggregate"]
    oracle = sa.get("oracle", {})
    csp_c  = sa.get("csp", {})
    jepa_c = sa.get("jepa", {})
    ac_c   = sa.get("ac_ssm", {})
    stats  = sa.get("statistics", {})
    n_sim  = stats.get("n_trials", 0)

    p_csp_jepa = stats.get("p_csp_vs_jepa_ttt", float("nan"))
    p_ac_jepa  = stats.get("p_ac_vs_jepa_ttt", float("nan"))
    p_ac_csp   = stats.get("p_ac_vs_csp_ttt", float("nan"))

    # ── Print summary to stdout ───────────────────────────────────────────────
    print("=" * 70)
    print("LOSO v5 — aggregate results")
    print("=" * 70)
    for name, a in [("AC-SSM [PRIMARY]", ac), ("JEPA enc+cls", enc),
                    ("MLP world-model", wm_mlp), ("Ablation", abl),
                    ("CSP+LDA", csp), ("MDM", mdm)]:
        if a:
            print(f"  {name:<30} {a['mean']*100:5.1f}%  "
                  f"[{a['ci_lo']*100:.1f}–{a['ci_hi']*100:.1f}%]")
    print(f"\n  AC-SSM vs JEPA enc:  p={p_ac_vs_enc:.4f}")
    print(f"  AC-SSM vs Ablation:  p={p_ac_vs_abl:.4f}")
    print(f"  AC-SSM vs MLP-WM:    p={p_ac_vs_wm:.4f}")
    print(f"  AC-SSM vs CSP:       p={p_ac_vs_csp:.4f}")

    print("\n" + "=" * 70)
    print("Closed-loop sim v2 — four controllers")
    print("=" * 70)
    for name, c in [("Oracle", oracle), ("CSP", csp_c), ("JEPA", jepa_c), ("AC-SSM", ac_c)]:
        if c:
            print(f"  {name:<10} TTT={c['ttt']['mean']:.1f}  "
                  f"Conv={c['convergence_rate']:.1%}  "
                  f"FinalErr={c['final_error']['mean']:.3f}")
    print(f"\n  CSP vs JEPA (TTT<):   p={p_csp_jepa:.4f}")
    print(f"  AC-SSM vs JEPA (TTT<):p={p_ac_jepa:.4f}")
    print(f"  AC-SSM vs CSP (TTT):  p={p_ac_csp:.4f}")

    # ── Build per-subject table rows ──────────────────────────────────────────
    eegnet_agg = agg.get("eegnet", {})
    per_subj_rows = []
    for i, s in enumerate(subj, 1):
        def _m(k): return f"{s[k]['mean']*100:.1f}" if k in s else "—"
        per_subj_rows.append(
            f"S{i:02d} & {_m('eegnet')} & {_m('encoder_ft_cls')} & "
            f"{_m('ac_ssm')} & {_m('encoder_ft_cls_planned')} & "
            f"{_m('ablation_encoder_cls')} & {_m('csp_lda')} & {_m('mdm')} \\\\"
        )

    # ── Now patch the article ─────────────────────────────────────────────────
    tex = ARTICLE.read_text()
    original = tex

    # 1. Abstract — update primary metric and closed-loop result
    ac_str  = f"{ac['mean']*100:.1f}" if ac else "??"
    ac_ci   = f"[{ac['ci_lo']*100:.1f}–{ac['ci_hi']*100:.1f}\\%]" if ac else "??"
    ac_conv = f"{ac_c['convergence_rate']:.1%}" if ac_c else "??"
    jepa_conv_str = f"{jepa_c['convergence_rate']:.1%}" if jepa_c else "0\\%"
    csp_conv_str  = f"{csp_c['convergence_rate']:.1%}" if csp_c else "27.8\\%"

    # 2. LOSO aggregate table — replace the three neural-world-model rows
    #    and add ac_ssm as the new primary condition
    if ac:
        old_table_block = (
            r"\textit{Neural (world model):} & & & \\" "\n"
            r"\quad JEPA + world-model cls & 34.2 & 32.9 & 35.6 \\" "\n"
            r"\quad CNS + world-model cls   & 33.4 & 32.8 & 33.9 \\"
        )
        ac_row   = f"\\quad \\textbf{{AC-SSM (this work)}} & \\textbf{{{ac['mean']*100:.1f}}} & {ac['ci_lo']*100:.1f} & {ac['ci_hi']*100:.1f} \\\\"
        wm_row   = f"\\quad JEPA + MLP world-model cls & {wm_mlp['mean']*100:.1f} & {wm_mlp['ci_lo']*100:.1f} & {wm_mlp['ci_hi']*100:.1f} \\\\" if wm_mlp else ""
        new_table_block = (
            r"\textit{Neural (world model):} & & & \\" "\n"
            + ac_row + "\n"
            + wm_row + "\n"
            r"\quad CNS + world-model cls   & 33.4 & 32.8 & 33.9 \\"
        )
        tex = tex.replace(old_table_block, new_table_block, 1)

    # Update enc+cls row
    if enc:
        tex = re.sub(
            r"\\quad JEPA encoder \+ cls \(e2e FT\)  & \d+\.\d+ & \d+\.\d+ & \d+\.\d+",
            f"\\\\quad JEPA encoder + cls (e2e FT)  & {enc['mean']*100:.1f} & {enc['ci_lo']*100:.1f} & {enc['ci_hi']*100:.1f}",
            tex,
        )

    # Update ablation row
    if abl:
        tex = re.sub(
            r"\\quad Random encoder \+ cls \(ablation\) & \d+\.\d+ & \d+\.\d+ & \d+\.\d+",
            f"\\\\quad Random encoder + cls (ablation) & {abl['mean']*100:.1f} & {abl['ci_lo']*100:.1f} & {abl['ci_hi']*100:.1f}",
            tex,
        )

    # Update CSP+LDA row
    if csp:
        tex = re.sub(
            r"\\quad \\textbf\{CSP \+ LDA\} & \\textbf\{\d+\.\d+\} & \d+\.\d+ & \d+\.\d+",
            f"\\\\quad \\\\textbf{{CSP + LDA}} & \\\\textbf{{{csp['mean']*100:.1f}}} & {csp['ci_lo']*100:.1f} & {csp['ci_hi']*100:.1f}",
            tex,
        )

    # 3. Closed-loop table
    if oracle and csp_c and jepa_c:
        # Add AC-SSM column to table header
        old_header = (
            r"\begin{tabular}{lrrr}" "\n"
            r"\toprule" "\n"
            r"\textbf{Metric} & \textbf{Oracle} & \textbf{CSP} & \textbf{JEPA} \\"
        )
        new_header = (
            r"\begin{tabular}{lrrrr}" "\n"
            r"\toprule" "\n"
            r"\textbf{Metric} & \textbf{Oracle} & \textbf{CSP} & \textbf{JEPA} & \textbf{AC-SSM} \\"
        )
        tex = tex.replace(old_header, new_header, 1)

        # Update convergence row
        ac_conv_pct = f"{ac_c['convergence_rate']*100:.1f}\\,\\%" if ac_c else "—"
        old_conv = f"Convergence rate      & 100.0\\,\\% & 27.8\\,\\% & 0.0\\,\\% \\\\"
        new_conv = (
            f"Convergence rate      & 100.0\\,\\% & {csp_c['convergence_rate']*100:.1f}\\,\\% & "
            f"{jepa_c['convergence_rate']*100:.1f}\\,\\% & {ac_conv_pct} \\\\"
        )
        tex = tex.replace(old_conv, new_conv, 1)

        # Update TTT row
        ac_ttt = f"{ac_c['ttt']['mean']:.0f}" if ac_c else "—"
        old_ttt = f"Time-to-target (mean) & 193       & 331      & 384     \\\\"
        new_ttt = (
            f"Time-to-target (mean) & {oracle['ttt']['mean']:.0f} & {csp_c['ttt']['mean']:.0f} & "
            f"{jepa_c['ttt']['mean']:.0f} & {ac_ttt} \\\\"
        )
        tex = tex.replace(old_ttt, new_ttt, 1)

        # Update final error row
        ac_err = f"{ac_c['final_error']['mean']:.3f}" if ac_c else "—"
        old_err = f"Final error (L2, DOF) & 0.246     & 1.221    & 0.998   \\\\"
        new_err = (
            f"Final error (L2, DOF) & {oracle['final_error']['mean']:.3f} & "
            f"{csp_c['final_error']['mean']:.3f} & {jepa_c['final_error']['mean']:.3f} & {ac_err} \\\\"
        )
        tex = tex.replace(old_err, new_err, 1)

        # Update caption with new trial count and controller count
        old_caption = r"\caption{Three-Controller Closed-Loop Simulation (133 trials, 10 subjects, $S{=}384$, $\varepsilon{=}0.25$)}"
        new_caption = (
            f"\\caption{{Four-Controller Closed-Loop Simulation ({n_sim} trials, 10 subjects, "
            r"$S{=}384$, $\varepsilon{=}0.25$)}"
        )
        tex = tex.replace(old_caption, new_caption, 1)

        # Update Wilcoxon stat row
        old_wilcox = r"\quad $p$-value & \multicolumn{3}{r}{$<0.0001$} \\"
        p_fmt = f"{p_csp_jepa:.4f}" if p_csp_jepa < 0.0001 else f"{p_csp_jepa:.4f}"
        p_fmt = "$<0.0001$" if p_csp_jepa < 0.0001 else f"${p_csp_jepa:.4f}$"
        p_ac_fmt = "$<0.0001$" if p_ac_jepa < 0.0001 else f"${p_ac_jepa:.4f}$"
        new_wilcox = (
            r"\quad CSP vs.\ JEPA ($p$) & \multicolumn{4}{r}{" + p_fmt + r"} \\" + "\n"
            r"\quad AC-SSM vs.\ JEPA ($p$) & \multicolumn{4}{r}{" + p_ac_fmt + r"} \\"
        )
        tex = tex.replace(old_wilcox, new_wilcox, 1)

    # 4. Update per-subject table to add AC-SSM column
    old_col_header = (
        r"\textbf{Subj.} & \textbf{EEGNet} & \textbf{Enc} & \textbf{Enc+WM} & \textbf{Abl} & \textbf{CSP} & \textbf{MDM} \\"
    )
    new_col_header = (
        r"\textbf{Subj.} & \textbf{EEGNet} & \textbf{Enc} & \textbf{AC-SSM} & \textbf{Enc+WM} & \textbf{Abl} & \textbf{CSP} & \textbf{MDM} \\"
    )
    if old_col_header in tex:
        tex = tex.replace(old_col_header, new_col_header, 1)

        # Replace old colspec
        tex = tex.replace(r"\begin{tabular}{crrrrrr}", r"\begin{tabular}{crrrrrrr}", 1)

        # Update footnote to add AC-SSM
        old_fn = r"{\footnotesize EEGNet = EEGNet (Lawhern 2018)~\cite{lawhern2018eegnet}; Enc = encoder\_ft\_cls; Enc+WM = encoder\_ft\_cls\_planned; Abl = ablation; CSP = CSP+LDA; MDM = Riemannian MDM.}"
        new_fn = r"{\footnotesize EEGNet = EEGNet (Lawhern 2018)~\cite{lawhern2018eegnet}; Enc = encoder\_ft\_cls; AC-SSM = action-conditioned SSM (this work); Enc+WM = MLP world-model planner; Abl = ablation; CSP = CSP+LDA; MDM = Riemannian MDM.}"
        tex = tex.replace(old_fn, new_fn, 1)

        # Replace per-subject data rows
        # Build replacement block from new results
        if per_subj_rows:
            # Find the old per-subject data block (S01 … S20 rows)
            old_rows_pattern = re.compile(
                r"(S01 &.*?\\\\)\n(.*?)(\\midrule\n\\textbf\{Mean\})",
                re.DOTALL,
            )
            new_rows_str = "\n".join(per_subj_rows) + "\n"

            def _replace_rows(m):
                return new_rows_str + m.group(3)

            tex = old_rows_pattern.sub(_replace_rows, tex, count=1)

            # Update mean row
            def _m(a):
                return f"{a['mean']*100:.1f}" if a else "—"
            def _ci(a):
                return f"[{a['ci_lo']*100:.1f},{a['ci_hi']*100:.1f}]" if a else "—"

            eegnet_agg = agg.get("eegnet", {})
            old_mean = (
                r"\textbf{Mean} & \textbf{38.4} & \textbf{36.6} & \textbf{33.7} & \textbf{35.5} & \textbf{60.1} & \textbf{47.5} \\"
            )
            new_mean = (
                f"\\textbf{{Mean}} & \\textbf{{{_m(eegnet_agg)}}} & \\textbf{{{_m(enc)}}} & "
                f"\\textbf{{{_m(ac)}}} & \\textbf{{{_m(wm_mlp)}}} & "
                f"\\textbf{{{_m(abl)}}} & \\textbf{{{_m(csp)}}} & \\textbf{{{_m(mdm)}}} \\\\"
            )
            tex = tex.replace(old_mean, new_mean, 1)

            old_ci_row = (
                r"\textbf{95\,\% CI} & [35.6,41.2] & [34.0,39.3] & [32.9,35.6] & [33.4,37.7] & [54.7,65.6] & [42.5,52.8] \\"
            )
            new_ci_row = (
                f"\\textbf{{95\\,\\% CI}} & {_ci(eegnet_agg)} & {_ci(enc)} & "
                f"{_ci(ac)} & {_ci(wm_mlp)} & "
                f"{_ci(abl)} & {_ci(csp)} & {_ci(mdm)} \\\\"
            )
            tex = tex.replace(old_ci_row, new_ci_row, 1)

    # 5. Update abstract + key inline numbers
    if ac:
        # Replace inline JEPA accuracy mentions in abstract
        tex = tex.replace(
            r"JEPA encoder ($36.6\%$, self-supervised) are statistically equivalent ($p=0.46$)",
            f"JEPA encoder (${enc['mean']*100:.1f}\\%$, self-supervised) are statistically equivalent ($p={p_enc_vs_abl:.2f}$)",
        )
        # Update the closed-loop finding in abstract
        tex = tex.replace(
            r"JEPA decoder (0\,\% convergence, $p{<}0.0001$ vs.\ CSP)",
            f"JEPA decoder ({jepa_c['convergence_rate']*100:.1f}\\,\\% convergence, $p{{<}}0.0001$ vs.\\ CSP)",
        )

    # 6. Update discussion inline numbers
    tex = tex.replace(
        r"The encoder\_ft\_cls\_planned condition ($34.2\%$) performs below the base encoder ($36.6\%$, $\Delta = -2.4$\,pp, $p=0.16$).",
        f"The encoder\\_ft\\_cls\\_planned condition (${wm_mlp['mean']*100:.1f}\\%$) performs below the base encoder (${enc['mean']*100:.1f}\\%$).",
    )

    # 7. EEGConformer row in aggregate table (optional)
    if LOSO_CONFORM.exists():
        loso_conf = load(LOSO_CONFORM)
        ec = loso_conf.get("aggregate", {}).get("eegconformer", {})
        if ec:
            tex = tex.replace(
                r"\quad EEGConformer (supervised) & TBD & — & — \\",
                f"\\quad EEGConformer (supervised) & {ec['mean']*100:.1f} & {ec['ci_lo']*100:.1f} & {ec['ci_hi']*100:.1f} \\\\",
                1,
            )
            print(f"\nEEGConformer: {ec['mean']*100:.1f}% [{ec['ci_lo']*100:.1f}–{ec['ci_hi']*100:.1f}%]")

    # 8. Window sensitivity table (optional)
    if WIN_SENS.exists():
        ws = load(WIN_SENS)
        res = ws.get("results", {})
        cond_map = {
            "csp_lda":              "CSP + LDA",
            "encoder_ft_cls":       "JEPA enc + cls",
            "ac_ssm":               "AC-SSM (this work)",
            "ablation_encoder_cls": "Random ablation",
        }
        for cond, label in cond_map.items():
            a_val = res.get("A_0.5-2.5", {}).get(cond, {})
            c_val = res.get("C_1.0-3.5", {}).get(cond, {})
            b_mean = res.get("B_0.5-4.5", {}).get(cond, {}).get("mean", 0) * 100

            a_str = f"{a_val['mean']*100:.1f}" if a_val else "TBD"
            c_str = f"{c_val['mean']*100:.1f}" if c_val else "TBD"

            # Use regex to match any whitespace between label and & TBD
            escaped = re.escape(label)
            pattern = rf"{escaped}\s*& TBD & {b_mean:.1f} & TBD \\\\"
            # re.sub treats \\ in replacement as literal \, so use a lambda
            new_row = f"{label}  & {a_str} & {b_mean:.1f} & {c_str} \\\\"
            tex = re.sub(pattern, lambda _: new_row, tex, count=1)

        # Fill CSP-JEPA gap row for all windows
        gaps = []
        for wk in ["A_0.5-2.5", "B_0.5-4.5", "C_1.0-3.5"]:
            r = res.get(wk, {})
            csp_m = r.get("csp_lda", {}).get("mean", 0) * 100
            enc_m = r.get("encoder_ft_cls", {}).get("mean", 0) * 100
            gaps.append(f"{csp_m - enc_m:.1f}" if (csp_m and enc_m) else "TBD")
        old_gap = re.compile(r"CSP \$-\$ JEPA gap \(pp\)\s*& TBD & 23\.6 & TBD \\\\")
        new_gap = f"CSP $-$ JEPA gap (pp) & {gaps[0]} & {gaps[1]} & {gaps[2]} \\\\"
        tex = old_gap.sub(lambda _: new_gap, tex, count=1)

        print("\nWindow sensitivity table filled.")

    # Write back
    if tex != original:
        ARTICLE.write_text(tex)
        print(f"\n✓ Article updated: {ARTICLE}")
    else:
        print("\n⚠  No replacements made — check pattern matching.")

    # ── Print LaTeX-ready table rows ──────────────────────────────────────────
    print("\nLaTeX aggregate table rows:")
    for name, a in [("\\textbf{AC-SSM (this work)}", ac),
                    ("JEPA enc+cls (e2e FT)", enc),
                    ("MLP world-model cls", wm_mlp),
                    ("Random encoder (ablation)", abl),
                    ("\\textbf{CSP+LDA}", csp),
                    ("MDM (Riemannian)", mdm)]:
        if a:
            print(f"  {name} & {a['mean']*100:.1f} & {a['ci_lo']*100:.1f} & {a['ci_hi']*100:.1f} \\\\")

    return 0


if __name__ == "__main__":
    sys.exit(main())
