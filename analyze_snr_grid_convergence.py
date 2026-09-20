"""R0 SNR-Grid Convergence Study & Resolution Sensitivity Analysis.

Investigates switching boundary threshold convergence as local SNR resolution
is systematically refined:
    1.0-dB local transition view (integer-spaced grid) ->
    0.5-dB local transition view (Deep 70k grid) ->
    0.25-dB local transition refinement view (transition-centered refinement)

Note on Grid Terminology:
    These resolution views are NOT uniform global grids across the entire SNR span;
    they describe the local resolution of SNR points available around mode switching boundaries.

Consumes:
    - Deep reference calibration table: results/r0_mc_deep_70k/calibration_1d_cuda_pooled.csv
    - Transition refinement calibration: results/r0_snr_grid_refine_025/calibration_1d_cuda_pooled.csv

Produces in results/r0_snr_grid_convergence/:
    1. snr_grid_convergence.csv (per-transition bracket endpoints, widths, midpoints, shifts, and DT metrics)
    2. snr_grid_views_labels.csv (per-SNR classification labels and eligibility across the views)
    3. snr_grid_convergence.md (data-driven publication report)

Strict Constraint:
    This script performs post-processing analysis only and MUST NOT invoke PHY simulation.
"""
import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ground_truth import (
    BER_TARGET,
    CONFIDENCE_K,
    GroundTruthConfig,
    GroundTruthRow,
    LookupTable1D,
    compute_ground_truth,
    load_calibration_csv,
)
from compare_mc_budgets import select_optimal_cart_depth


TRANSITION_SPECS = [
    ("BPSK->QPSK", "BPSK", "QPSK", 16.75),
    ("QPSK->16QAM", "QPSK", "16QAM", 22.75),
    ("16QAM->64QAM", "16QAM", "64QAM", 28.25),
]


def load_and_merge_datasets(
    deep_csv_path: Union[str, Path],
    refine_csv_path: Union[str, Path],
) -> Tuple[Dict[float, Dict[str, Dict[str, float]]], Dict[float, Dict[str, Dict[str, float]]], Dict[float, Dict[str, Dict[str, float]]]]:
    """Load Deep and Refinement calibration datasets and produce merged dataset.

    Returns:
        (deep_cal, refine_cal, merged_cal)
    """
    deep_p = Path(deep_csv_path)
    refine_p = Path(refine_csv_path)

    if not deep_p.exists():
        raise FileNotFoundError(
            f"Deep calibration dataset not found at: {deep_p}. "
            "Please run 'python calibration_l2_cuda.py --profile deep' first."
        )
    if not refine_p.exists():
        raise FileNotFoundError(
            f"Refinement calibration dataset not found at: {refine_p}. "
            "Please run 'python refine_snr_grid_cuda.py' first."
        )

    deep_cal = load_calibration_csv(deep_p)
    refine_cal = load_calibration_csv(refine_p)

    merged_cal: Dict[float, Dict[str, Dict[str, float]]] = {}
    for snr, mods in deep_cal.items():
        merged_cal[snr] = dict(mods)
    for snr, mods in refine_cal.items():
        merged_cal[snr] = dict(mods)

    return deep_cal, refine_cal, merged_cal


def construct_resolution_views(
    deep_cal: Dict[float, Dict[str, Dict[str, float]]],
    merged_cal: Dict[float, Dict[str, Dict[str, float]]],
) -> Tuple[Dict[float, Dict[str, Dict[str, float]]], Dict[float, Dict[str, Dict[str, float]]], Dict[float, Dict[str, Dict[str, float]]]]:
    """Construct the three local resolution calibration views: 1.0 dB, 0.5 dB, and 0.25 dB.

    These describe local transition bracket resolution, NOT uniform global grids:
    - 1.0-dB local transition view: integer-spaced SNR points from Deep 70k.
    - 0.5-dB local transition view: canonical Deep 70k grid points.
    - 0.25-dB local transition refinement view: merged Deep 70k plus 16.75, 22.75, 28.25 dB refinement points.
    """
    view_1_0_snrs = [s for s in sorted(deep_cal.keys()) if abs(s - round(s)) < 1e-4]
    view_1_0_cal = {s: deep_cal[s] for s in view_1_0_snrs}

    view_0_5_cal = dict(deep_cal)
    view_0_25_cal = dict(merged_cal)

    return view_1_0_cal, view_0_5_cal, view_0_25_cal


def find_transition_bracket(
    gt_rows: Sequence[GroundTruthRow],
    from_mode: str,
    to_mode: str,
) -> Optional[Dict[str, float]]:
    """Identify the switching bracket and midpoint threshold between two modulation modes."""
    sorted_rows = sorted(gt_rows, key=lambda r: r.snr_db)
    for i in range(len(sorted_rows) - 1):
        curr = sorted_rows[i]
        nxt = sorted_rows[i + 1]
        if curr.best_mode == from_mode and nxt.best_mode == to_mode:
            low = curr.snr_db
            high = nxt.snr_db
            return {
                "lower_endpoint": low,
                "upper_endpoint": high,
                "midpoint": 0.5 * (low + high),
                "bracket_width": high - low,
            }

    # Fallback if non-adjacent: find highest from_mode below lowest to_mode
    from_snrs = [r.snr_db for r in sorted_rows if r.best_mode == from_mode]
    to_snrs = [r.snr_db for r in sorted_rows if r.best_mode == to_mode]
    if from_snrs and to_snrs and max(from_snrs) < min(to_snrs):
        low = max(from_snrs)
        high = min(to_snrs)
        return {
            "lower_endpoint": low,
            "upper_endpoint": high,
            "midpoint": 0.5 * (low + high),
            "bracket_width": high - low,
        }
    return None


def match_cart_threshold(th_list: Sequence[float], nominal_snr: float, tol: float = 2.5) -> Optional[float]:
    """Match closest learned CART split threshold within tolerance of nominal transition."""
    if not th_list:
        return None
    closest = min(th_list, key=lambda t: abs(t - nominal_snr))
    return float(closest) if abs(closest - nominal_snr) <= tol else None


def run_snr_grid_convergence_study(
    deep_dir: Union[str, Path] = "results/r0_mc_deep_70k",
    refine_dir: Union[str, Path] = "results/r0_snr_grid_refine_025",
    output_dir: Union[str, Path] = "results/r0_snr_grid_convergence",
    ber_target: float = BER_TARGET,
    confidence_k: float = CONFIDENCE_K,
) -> Dict[str, Any]:
    """Execute complete post-processing SNR-grid convergence and resolution sensitivity study."""
    deep_p = Path(deep_dir)
    refine_p = Path(refine_dir)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    deep_csv = deep_p / "calibration_1d_cuda_pooled.csv"
    refine_csv = refine_p / "calibration_1d_cuda_pooled.csv"

    # 1. Load and merge datasets
    deep_cal, refine_cal, merged_cal = load_and_merge_datasets(deep_csv, refine_csv)

    # 2. Construct three resolution views
    cal_1_0, cal_0_5, cal_0_25 = construct_resolution_views(deep_cal, merged_cal)

    gt_cfg = GroundTruthConfig(
        ber_target=ber_target,
        fallback_policy="robustest_mode",
        confidence_k=confidence_k,
    )

    # 3. Ground Truth Synthesis per view
    gt_1_0 = compute_ground_truth(cal_1_0, gt_cfg)
    gt_0_5 = compute_ground_truth(cal_0_5, gt_cfg)
    gt_0_25 = compute_ground_truth(cal_0_25, gt_cfg)

    # 4. CART Depth Model Selection per view (sweeping 1..5)
    snrs_1_0 = [r.snr_db for r in gt_1_0]
    labels_1_0 = [r.best_mode for r in gt_1_0]
    clf_1_0, dt_1_0 = select_optimal_cart_depth(snrs_1_0, labels_1_0)

    snrs_0_5 = [r.snr_db for r in gt_0_5]
    labels_0_5 = [r.best_mode for r in gt_0_5]
    clf_0_5, dt_0_5 = select_optimal_cart_depth(snrs_0_5, labels_0_5)

    snrs_0_25 = [r.snr_db for r in gt_0_25]
    labels_0_25 = [r.best_mode for r in gt_0_25]
    clf_0_25, dt_0_25 = select_optimal_cart_depth(snrs_0_25, labels_0_25)

    dt_metrics = {
        "view_1_0": {
            "selected_depth": dt_1_0.max_depth_param,
            "actual_depth": dt_1_0.actual_depth,
            "fidelity": dt_1_0.accuracy,
            "thresholds": dt_1_0.learned_thresholds,
            "leaves": dt_1_0.leaf_count,
        },
        "view_0_5": {
            "selected_depth": dt_0_5.max_depth_param,
            "actual_depth": dt_0_5.actual_depth,
            "fidelity": dt_0_5.accuracy,
            "thresholds": dt_0_5.learned_thresholds,
            "leaves": dt_0_5.leaf_count,
        },
        "view_0_25": {
            "selected_depth": dt_0_25.max_depth_param,
            "actual_depth": dt_0_25.actual_depth,
            "fidelity": dt_0_25.accuracy,
            "thresholds": dt_0_25.learned_thresholds,
            "leaves": dt_0_25.leaf_count,
        },
    }

    # 5. Transition bracket analysis across views
    convergence_rows: List[Dict[str, Any]] = []
    for tr_name, from_m, to_m, nominal_snr in TRANSITION_SPECS:
        b_1_0 = find_transition_bracket(gt_1_0, from_m, to_m)
        b_0_5 = find_transition_bracket(gt_0_5, from_m, to_m)
        b_0_25 = find_transition_bracket(gt_0_25, from_m, to_m)

        th_1_0 = b_1_0["midpoint"] if b_1_0 else None
        th_0_5 = b_0_5["midpoint"] if b_0_5 else None
        th_0_25 = b_0_25["midpoint"] if b_0_25 else None

        lut_shift_10_05 = (th_0_5 - th_1_0) if (th_1_0 is not None and th_0_5 is not None) else None
        lut_shift_05_025 = (th_0_25 - th_0_5) if (th_0_5 is not None and th_0_25 is not None) else None

        cart_1_0 = match_cart_threshold(dt_1_0.learned_thresholds, nominal_snr)
        cart_0_5 = match_cart_threshold(dt_0_5.learned_thresholds, nominal_snr)
        cart_0_25 = match_cart_threshold(dt_0_25.learned_thresholds, nominal_snr)

        cart_shift_10_05 = (cart_0_5 - cart_1_0) if (cart_1_0 is not None and cart_0_5 is not None) else None
        cart_shift_05_025 = (cart_0_25 - cart_0_5) if (cart_0_5 is not None and cart_0_25 is not None) else None

        convergence_rows.append({
            "transition": tr_name,
            "from_mode": from_m,
            "to_mode": to_m,
            "nominal_target_db": nominal_snr,
            # 1.0-dB View
            "view_1_0_lower_db": b_1_0["lower_endpoint"] if b_1_0 else None,
            "view_1_0_upper_db": b_1_0["upper_endpoint"] if b_1_0 else None,
            "view_1_0_midpoint_db": th_1_0,
            "view_1_0_bracket_width_db": b_1_0["bracket_width"] if b_1_0 else None,
            "view_1_0_cart_threshold_db": cart_1_0,
            # 0.5-dB View
            "view_0_5_lower_db": b_0_5["lower_endpoint"] if b_0_5 else None,
            "view_0_5_upper_db": b_0_5["upper_endpoint"] if b_0_5 else None,
            "view_0_5_midpoint_db": th_0_5,
            "view_0_5_bracket_width_db": b_0_5["bracket_width"] if b_0_5 else None,
            "view_0_5_cart_threshold_db": cart_0_5,
            # 0.25-dB View
            "view_0_25_lower_db": b_0_25["lower_endpoint"] if b_0_25 else None,
            "view_0_25_upper_db": b_0_25["upper_endpoint"] if b_0_25 else None,
            "view_0_25_midpoint_db": th_0_25,
            "view_0_25_bracket_width_db": b_0_25["bracket_width"] if b_0_25 else None,
            "view_0_25_cart_threshold_db": cart_0_25,
            # Threshold shifts
            "lut_shift_10_to_05_db": lut_shift_10_05,
            "lut_shift_05_to_025_db": lut_shift_05_025,
            "cart_shift_10_to_05_db": cart_shift_10_05,
            "cart_shift_05_to_025_db": cart_shift_05_025,
        })

    # 6. Save snr_grid_convergence.csv
    csv_conv_path = out_p / "snr_grid_convergence.csv"
    with open(csv_conv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(convergence_rows[0].keys()))
        writer.writeheader()
        writer.writerows(convergence_rows)

    # 7. Generate view label table (all unique SNR points across the 3 views)
    all_unique_snrs = sorted(list(merged_cal.keys()))
    gt_map_1_0 = {r.snr_db: r for r in gt_1_0}
    gt_map_0_5 = {r.snr_db: r for r in gt_0_5}
    gt_map_0_25 = {r.snr_db: r for r in gt_0_25}

    view_label_rows: List[Dict[str, Any]] = []
    for snr in all_unique_snrs:
        r_025 = gt_map_0_25[snr]
        r_05 = gt_map_0_5.get(snr)
        r_10 = gt_map_1_0.get(snr)

        eligs = []
        if r_025.bpsk_eligible: eligs.append("BPSK")
        if r_025.qpsk_eligible: eligs.append("QPSK")
        if r_025.qam16_eligible: eligs.append("16QAM")
        if r_025.qam64_eligible: eligs.append("64QAM")

        view_label_rows.append({
            "snr_db": snr,
            "in_view_1_0": (r_10 is not None),
            "in_view_0_5": (r_05 is not None),
            "in_view_0_25": True,
            "view_1_0_label": r_10.best_mode if r_10 else "N/A",
            "view_0_5_label": r_05.best_mode if r_05 else "N/A",
            "view_0_25_label": r_025.best_mode,
            "view_0_25_eligible_modes": ",".join(eligs) if eligs else "None",
            "BPSK_BER": r_025.bpsk_ber,
            "BPSK_SE": r_025.bpsk_se,
            "BPSK_CI_high": r_025.bpsk_ci_high,
            "QPSK_BER": r_025.qpsk_ber,
            "QPSK_SE": r_025.qpsk_se,
            "QPSK_CI_high": r_025.qpsk_ci_high,
            "16QAM_BER": r_025.qam16_ber,
            "16QAM_SE": r_025.qam16_se,
            "16QAM_CI_high": r_025.qam16_ci_high,
            "64QAM_BER": r_025.qam64_ber,
            "64QAM_SE": r_025.qam64_se,
            "64QAM_CI_high": r_025.qam64_ci_high,
        })

    csv_labels_path = out_p / "snr_grid_views_labels.csv"
    with open(csv_labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(view_label_rows[0].keys()))
        writer.writeheader()
        writer.writerows(view_label_rows)

    # 8. Generate Markdown Report
    report_path = out_p / "snr_grid_convergence.md"
    generate_convergence_markdown_report(
        output_path=report_path,
        convergence_rows=convergence_rows,
        view_label_rows=view_label_rows,
        dt_metrics=dt_metrics,
        ber_target=ber_target,
        confidence_k=confidence_k,
    )

    print(f"[Convergence Study] Saved Convergence CSV:   {csv_conv_path}")
    print(f"[Convergence Study] Saved View Labels CSV:   {csv_labels_path}")
    print(f"[Convergence Study] Saved Markdown Report:   {report_path}")
    for v_key, v_name in [
        ("view_1_0", "1.0-dB Local Transition View"),
        ("view_0_5", "0.5-dB Local Transition View"),
        ("view_0_25", "0.25-dB Local Transition Refinement View"),
    ]:
        dm = dt_metrics[v_key]
        print(f"[Convergence Study] {v_name} DT Selection: Depth={dm['selected_depth']}, Actual={dm['actual_depth']}, Fidelity={dm['fidelity'] * 100:.1f}%, Thresholds={dm['thresholds']}")

    return {
        "convergence_rows": convergence_rows,
        "view_label_rows": view_label_rows,
        "dt_metrics": dt_metrics,
        "output_dir": str(out_p),
    }


def compute_data_driven_findings(
    convergence_rows: List[Dict[str, Any]],
    dt_metrics: Dict[str, Any],
) -> List[str]:
    """Compute empirical findings conditionally from actual results without pre-ordained conclusions."""
    findings = []

    # 1. Bracket Width Behavior
    bracket_summaries = []
    all_narrowed = True
    all_halved = True
    for r in convergence_rows:
        tr = r["transition"]
        w_10 = r["view_1_0_bracket_width_db"]
        w_05 = r["view_0_5_bracket_width_db"]
        w_025 = r["view_0_25_bracket_width_db"]

        if w_10 is not None and w_05 is not None and w_025 is not None:
            bracket_summaries.append(f"{tr}: {w_10:.2f} dB -> {w_05:.2f} dB -> {w_025:.2f} dB")
            if not (w_05 < w_10 and w_025 < w_05):
                all_narrowed = False
            # Check within 0.05 dB tolerance of exact geometric halving
            if not (abs(w_05 - 0.5 * w_10) <= 0.05 and abs(w_025 - 0.5 * w_05) <= 0.05):
                all_halved = False
        else:
            all_narrowed = False
            all_halved = False
            bracket_summaries.append(f"{tr}: incomplete bracket data")

    if all_halved and bracket_summaries:
        findings.append(
            f"- **Switching Bracket Halving:** Uncertainty brackets halved systematically across all transitions "
            f"as local resolution was refined: {'; '.join(bracket_summaries)}."
        )
    elif all_narrowed and bracket_summaries:
        findings.append(
            f"- **Switching Bracket Narrowing:** Uncertainty brackets narrowed monotonically across all transitions: "
            f"{'; '.join(bracket_summaries)}."
        )
    elif bracket_summaries:
        findings.append(
            f"- **Switching Bracket Resolution:** Empirical bracket widths observed across views: "
            f"{'; '.join(bracket_summaries)}."
        )
    else:
        findings.append("- **Switching Bracket Resolution:** No transition brackets detected.")

    # 2. Threshold Shift Dynamics
    shift_summaries = []
    diminishing_count = 0
    bounded_count = 0
    total_shifts = 0

    for r in convergence_rows:
        tr = r["transition"]
        s1 = r["lut_shift_10_to_05_db"]
        s2 = r["lut_shift_05_to_025_db"]
        if s1 is not None and s2 is not None:
            total_shifts += 1
            is_dim = abs(s2) <= abs(s1)
            is_bnd = (abs(s1) <= 0.25 + 1e-4) and (abs(s2) <= 0.125 + 1e-4)
            if is_dim:
                diminishing_count += 1
            if is_bnd:
                bounded_count += 1
            shift_summaries.append(f"{tr}: Δθ(1.0→0.5) = {s1:+.3f} dB, Δθ(0.5→0.25) = {s2:+.3f} dB")
        elif s1 is not None:
            shift_summaries.append(f"{tr}: Δθ(1.0→0.5) = {s1:+.3f} dB, Δθ(0.5→0.25) = N/A")
        elif s2 is not None:
            shift_summaries.append(f"{tr}: Δθ(1.0→0.5) = N/A, Δθ(0.5→0.25) = {s2:+.3f} dB")

    if total_shifts > 0 and diminishing_count == total_shifts:
        findings.append(
            f"- **Threshold Shift Diminution:** Empirical threshold shifts diminished in magnitude with finer local resolution "
            f"across all {total_shifts} transitions ({'; '.join(shift_summaries)})."
        )
    elif total_shifts > 0:
        findings.append(
            f"- **Empirical Threshold Shifts:** Shifts observed across local resolution transitions: "
            f"{'; '.join(shift_summaries)} ({diminishing_count}/{total_shifts} transitions showed diminishing magnitude)."
        )
    else:
        findings.append("- **Empirical Threshold Shifts:** Insufficient adjacent transition points to compute shift progression.")

    if total_shifts > 0 and bounded_count == total_shifts:
        findings.append(
            "- **Theoretical Bound Adherence:** All observed midpoint shifts satisfied the theoretical symmetric bisection bounds "
            "($|\\Delta \\theta_{1.0 \\to 0.5}| \\le 0.25\\text{ dB}$, $|\\Delta \\theta_{0.5 \\to 0.25}| \\le 0.125\\text{ dB}$)."
        )
    elif total_shifts > 0:
        findings.append(
            f"- **Theoretical Bound Evaluation:** {bounded_count}/{total_shifts} transitions satisfied theoretical bisection bounds "
            "($|\\Delta \\theta_{1.0 \\to 0.5}| \\le 0.25\\text{ dB}$, $|\\Delta \\theta_{0.5 \\to 0.25}| \\le 0.125\\text{ dB}$); "
            "see Section 2 for individual bracket endpoints."
        )

    # 3. Decision Tree Model Selection & Fidelity
    d_10 = dt_metrics["view_1_0"]["selected_depth"]
    d_05 = dt_metrics["view_0_5"]["selected_depth"]
    d_025 = dt_metrics["view_0_25"]["selected_depth"]

    f_10 = dt_metrics["view_1_0"]["fidelity"]
    f_05 = dt_metrics["view_0_5"]["fidelity"]
    f_025 = dt_metrics["view_0_25"]["fidelity"]

    all_same_depth = (d_10 == d_05 == d_025)
    all_100_fidelity = (abs(f_10 - 1.0) < 1e-4 and abs(f_05 - 1.0) < 1e-4 and abs(f_025 - 1.0) < 1e-4)

    if all_same_depth and all_100_fidelity:
        findings.append(
            f"- **Decision Tree Policy Stability:** The optimal CART depth selected via model selection remained constant at "
            f"$d={d_10}$ across all three local transition views, each achieving 100.0% fidelity to ground-truth labels."
        )
    elif all_100_fidelity:
        findings.append(
            f"- **Decision Tree Model Selection:** 100.0% training fidelity was achieved across all views, with selected depths "
            f"varying with local resolution: $d={d_10}$ (1.0-dB local view), $d={d_05}$ (0.5-dB local view), $d={d_025}$ (0.25-dB local refinement view)."
        )
    else:
        findings.append(
            f"- **Decision Tree Model Selection:** Selected depths and fidelities observed across views: "
            f"1.0-dB local view ($d={d_10}$, fidelity={f_10 * 100:.1f}%), "
            f"0.5-dB local view ($d={d_05}$, fidelity={f_05 * 100:.1f}%), "
            f"0.25-dB local refinement view ($d={d_025}$, fidelity={f_025 * 100:.1f}%)."
        )

    return findings


def generate_convergence_markdown_report(
    output_path: Path,
    convergence_rows: List[Dict[str, Any]],
    view_label_rows: List[Dict[str, Any]],
    dt_metrics: Dict[str, Any],
    ber_target: float,
    confidence_k: float,
) -> None:
    """Format publication-quality Markdown report for the SNR-grid convergence study."""
    data_findings = compute_data_driven_findings(convergence_rows, dt_metrics)

    lines = [
        "# PHY-ML R0 SNR-Grid Convergence & Resolution Sensitivity Study",
        "",
        "**Topic:** Empirical Switching-Boundary Convergence and Decision Tree Policy Stability under Local Grid Refinement  ",
        "**Resolution Progression:** `1.0-dB local transition view` (integer-spaced grid) $\\to$ `0.5-dB local transition view` (Deep reference grid) $\\to$ `0.25-dB local transition refinement view` (transition-centered refinement)  ",
        "**Methodology:** Conservative 95% Confidence Interval Upper Bound ($\\text{BER} + 1.96 \\cdot \\text{SE} \\le 0.0100$)  ",
        "**Monte Carlo Budget:** Deep 70,000 blocks/seed (140,000 pooled blocks/point) | FP64 CUDA  ",
        "**Study Characterization:** Formal Grid Convergence / Resolution Sensitivity (NOT an ablation)  ",
        "**Output Directory:** `results/r0_snr_grid_convergence/`  ",
        "**Date:** 2026-09-20  ",
        "",
        "---",
        "",
        "> [!NOTE]",
        "> **Local Transition Views vs Global Grids:** The \"1.0-dB local transition view\", \"0.5-dB local transition view\", and \"0.25-dB local transition refinement view\" are **NOT uniform global grids** across the full SNR span. They describe the local resolution of SNR evaluation points available around mode switching boundaries. This targeted refinement evaluates switching sensitivity without redundant dense simulation across stable single-mode regions.",
        "",
        "## 1. Executive Summary",
        "",
        "This study measures the sensitivity and convergence behavior of AMC switching boundaries as local SNR resolution is refined around mode transitions.",
        "By probing candidate midpoints of the 0.5-dB transition intervals (**16.75 dB**, **22.75 dB**, and **28.25 dB**), the local switching brackets can be resolved down to $0.25\\text{ dB}$ without re-running any existing calibration points.",
        "",
        "### Empirical Convergence Findings (Computed from Data):",
    ]
    lines.extend(data_findings)
    lines.extend([
        "",
        "---",
        "",
        "## 2. Transition Switching-Boundary Convergence",
        "",
        "| Transition | Resolution View | Lower Bracket (dB) | Upper Bracket (dB) | Bracket Width (dB) | Midpoint Threshold (dB) | Midpoint Shift (dB) | CART Threshold (dB) | CART Shift (dB) |",
        "|:----------:|:---------------:|:------------------:|:------------------:|:------------------:|:-----------------------:|:-------------------:|:-------------------:|:---------------:|",
    ])

    for r in convergence_rows:
        tr = r["transition"]
        # 1.0-dB local transition view row
        w_10 = f"{r['view_1_0_bracket_width_db']:.2f}" if r["view_1_0_bracket_width_db"] is not None else "N/A"
        th_10 = f"{r['view_1_0_midpoint_db']:.3f}" if r["view_1_0_midpoint_db"] is not None else "N/A"
        c_10 = f"{r['view_1_0_cart_threshold_db']:.3f}" if r["view_1_0_cart_threshold_db"] is not None else "N/A"
        lines.append(
            f"| {tr:12s} | 1.0-dB local transition view | {r['view_1_0_lower_db']:18.2f} | {r['view_1_0_upper_db']:18.2f} | {w_10:18s} | {th_10:23s} | {'---':19s} | {c_10:19s} | {'---':15s} |"
        )

        # 0.5-dB local transition view row
        w_05 = f"{r['view_0_5_bracket_width_db']:.2f}" if r["view_0_5_bracket_width_db"] is not None else "N/A"
        th_05 = f"{r['view_0_5_midpoint_db']:.3f}" if r["view_0_5_midpoint_db"] is not None else "N/A"
        sh_lut_10_05 = f"{r['lut_shift_10_to_05_db']:+.3f}" if r["lut_shift_10_to_05_db"] is not None else "N/A"
        c_05 = f"{r['view_0_5_cart_threshold_db']:.3f}" if r["view_0_5_cart_threshold_db"] is not None else "N/A"
        sh_c_10_05 = f"{r['cart_shift_10_to_05_db']:+.3f}" if r["cart_shift_10_to_05_db"] is not None else "N/A"
        lines.append(
            f"| {tr:12s} | 0.5-dB local transition view | {r['view_0_5_lower_db']:18.2f} | {r['view_0_5_upper_db']:18.2f} | {w_05:18s} | {th_05:23s} | {sh_lut_10_05:19s} | {c_05:19s} | {sh_c_10_05:15s} |"
        )

        # 0.25-dB local transition view row
        w_025 = f"{r['view_0_25_bracket_width_db']:.2f}" if r["view_0_25_bracket_width_db"] is not None else "N/A"
        th_025 = f"{r['view_0_25_midpoint_db']:.3f}" if r["view_0_25_midpoint_db"] is not None else "N/A"
        sh_lut_05_025 = f"{r['lut_shift_05_to_025_db']:+.3f}" if r["lut_shift_05_to_025_db"] is not None else "N/A"
        c_025 = f"{r['view_0_25_cart_threshold_db']:.3f}" if r["view_0_25_cart_threshold_db"] is not None else "N/A"
        sh_c_05_025 = f"{r['cart_shift_05_to_025_db']:+.3f}" if r["cart_shift_05_to_025_db"] is not None else "N/A"
        lines.append(
            f"| {tr:12s} | 0.25-dB local transition view | {r['view_0_25_lower_db']:18.2f} | {r['view_0_25_upper_db']:18.2f} | {w_025:18s} | {th_025:23s} | {sh_lut_05_025:19s} | {c_025:19s} | {sh_c_05_025:15s} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Decision Tree Model Selection & Policy Stability",
        "",
        "Under the canonical model-selection rule, maximum depth was swept over $d \\in [1, 5]$ independently for each resolution view to identify the smallest depth achieving 100% fidelity to that view's ground-truth labels:",
        "",
    ])

    for v_key, v_title in [
        ("view_1_0", "1.0-dB Local Transition View (Integer Points)"),
        ("view_0_5", "0.5-dB Local Transition View (Canonical Deep Grid)"),
        ("view_0_25", "0.25-dB Local Transition Refinement View (Refined Midpoints Grid)"),
    ]:
        dm = dt_metrics[v_key]
        lines.extend([
            f"### {v_title}",
            f"- **Selected Depth:** `{dm['selected_depth']}`",
            f"- **Actual Depth:** `{dm['actual_depth']}`",
            f"- **Leaf Nodes:** `{dm['leaves']}`",
            f"- **Empirical Fidelity:** `{dm['fidelity'] * 100:.1f}%` ({dm['fidelity']:.4f})",
            f"- **Learned Thresholds:** `{dm['thresholds']}`",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 4. Detailed Operating Point Classifications across Views",
        "",
        "| SNR (dB) | In 1.0-dB Local View? | In 0.5-dB Local View? | In 0.25-dB Local View? | 1.0-dB Local BestMode | 0.5-dB Local BestMode | 0.25-dB Local BestMode | Eligible Modes (0.25-dB Local View) |",
        "|:--------:|:---------------------:|:---------------------:|:----------------------:|:---------------------:|:---------------------:|:----------------------:|:-----------------------------------:|",
    ])

    for row in view_label_rows:
        in_10 = "YES" if row["in_view_1_0"] else "No"
        in_05 = "YES" if row["in_view_0_5"] else "No"
        in_025 = "YES" if row["in_view_0_25"] else "No"
        lines.append(
            f"| {row['snr_db']:8.2f} | {in_10:21s} | {in_05:21s} | {in_025:22s} | {row['view_1_0_label']:21s} | {row['view_0_5_label']:21s} | {row['view_0_25_label']:22s} | {row['view_0_25_eligible_modes']:35s} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Scientific Methodological Stance",
        "",
        "1. **Formal Convergence Analysis:**",
        "   - The progression from 1.0-dB to 0.5-dB to 0.25-dB local transition views provides an empirical framework to test spatial resolution sensitivity around switching boundaries.",
        "   - It assesses whether the discrete 1D look-up table and CART boundaries converge toward stable operating thresholds as spatial sampling is refined around switching boundaries.",
        "",
        "2. **Compute Efficiency via Targeted Refinement:**",
        "   - Evaluating only 3 targeted refinement points (16.75, 22.75, 28.25 dB) instead of a dense 0.25-dB global grid (which would require 120 SNR points) achieves 97.5% compute savings while providing identical switching-boundary resolution.",
        "",
        "3. **Artifact Manifest:**",
        f"   - `results/r0_snr_grid_convergence/snr_grid_convergence.csv`: Quantitative bracket endpoints, widths, midpoints, and shifts across transitions.",
        f"   - `results/r0_snr_grid_convergence/snr_grid_views_labels.csv`: SNR point membership, classifications, and candidate mode error metrics.",
        f"   - `results/r0_snr_grid_convergence/snr_grid_convergence.md`: This comprehensive publication-grade report.",
    ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """CLI entry point for SNR-grid convergence study."""
    parser = argparse.ArgumentParser(
        description="PHY-ML R0: SNR-Grid Convergence Study (1.0 dB -> 0.5 dB -> 0.25 dB).",
    )
    parser.add_argument(
        "--deep-dir",
        type=str,
        default="results/r0_mc_deep_70k",
        help="Directory containing Deep calibration artifacts (results/r0_mc_deep_70k).",
    )
    parser.add_argument(
        "--refine-dir",
        type=str,
        default="results/r0_snr_grid_refine_025",
        help="Directory containing 0.25-dB refinement artifacts (results/r0_snr_grid_refine_025).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/r0_snr_grid_convergence",
        help="Output directory for convergence artifacts (results/r0_snr_grid_convergence).",
    )
    parser.add_argument(
        "--ber-target",
        type=float,
        default=BER_TARGET,
        help=f"Target bit error rate constraint (default: {BER_TARGET}).",
    )
    parser.add_argument(
        "--confidence-k",
        type=float,
        default=CONFIDENCE_K,
        help=f"Confidence interval multiplier (default: {CONFIDENCE_K}).",
    )

    args = parser.parse_args()

    run_snr_grid_convergence_study(
        deep_dir=args.deep_dir,
        refine_dir=args.refine_dir,
        output_dir=args.output_dir,
        ber_target=args.ber_target,
        confidence_k=args.confidence_k,
    )


if __name__ == "__main__":
    main()
