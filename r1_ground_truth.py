"""R1: Configurable Ground Truth & Baseline Policy Engine for Coded Link Adaptation.

Consumes calibration data from LDPC-coded Monte Carlo trials and synthesizes
traceable ground-truth link adaptation decisions.

Design Principles:
- Generic and fully decoupled: State = [SNR_setup], Action = (modulation, code_rate),
  Utility = Spectral Efficiency (eta), Primary Reliability Constraint = BLER <= BLER_target.
- No hard-coded scientific decisions:
  - BLER target is an external parameter (e.g. 0.10 or 0.01).
  - Confidence interval method is selectable (Wald, Wilson, Clopper-Pearson).
  - Eligibility rule is selectable (conservative upper-CI vs nominal).
  - Equal-eta tie-break rule is selectable (lowest BLER, lower modulation, stronger coding, predefined).
  - Fallback rule is selectable (lowest-rate, fixed robust).
- Traceable audit trail: records eligible sets, tie-break activations, label uncertainty,
  and fallback usage for every SNR operating point.
- Does NOT execute or generate production Ground Truth from uncalibrated/pilot data.
"""
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.stats


CONFIDENCE_K_DEFAULT: float = 1.96
BLER_TARGET_DEFAULT: float = 0.10


@dataclass(frozen=True)
class CodedCandidateRecord:
    """Evaluation record of a single candidate action at an SNR point."""
    action_name: str
    modulation: str
    spectral_efficiency: float
    bler: float
    bler_se: float
    bler_ci_low: float
    bler_ci_high: float
    eligible: bool
    info_ber: float = 0.0


@dataclass(frozen=True)
class R1GroundTruthConfig:
    """Configurable policy parameters for R1 Ground Truth decision engine."""
    bler_target: float = BLER_TARGET_DEFAULT
    ci_method: str = "wald"                  # "wald", "wilson", "clopper_pearson"
    confidence_k: float = CONFIDENCE_K_DEFAULT
    eligibility_rule: str = "conservative_ci" # "conservative_ci" (upper CI <= target) or "nominal" (bler <= target)
    tie_break_rule: str = "lowest_bler"      # "lowest_bler", "lower_modulation", "stronger_coding", "predefined"
    fallback_rule: str = "lowest_rate"       # "lowest_rate" (lowest eta among all candidates) or "fixed_action"
    fixed_robust_action: str = "BPSK-1/2"
    fixed_high_tp_action: str = "64QAM-3/4"
    predefined_action_order: Sequence[str] = (
        "BPSK-1/2", "QPSK-1/2", "QPSK-2/3", "16QAM-1/2",
        "16QAM-3/4", "64QAM-1/2", "64QAM-2/3", "64QAM-3/4"
    )


@dataclass(frozen=True)
class R1GroundTruthRow:
    """Traceable ground truth decision record for one SNR operating point."""
    snr_db: float
    best_action: str
    best_spectral_efficiency: float
    selection_reason: str
    fallback_used: bool
    label_uncertain: bool
    equal_eta_tie_broken: bool
    candidates: Dict[str, CodedCandidateRecord] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "snr_db": self.snr_db,
            "best_action": self.best_action,
            "best_spectral_efficiency": self.best_spectral_efficiency,
            "selection_reason": self.selection_reason,
            "fallback_used": self.fallback_used,
            "label_uncertain": self.label_uncertain,
            "equal_eta_tie_broken": self.equal_eta_tie_broken,
        }
        for name, cand in self.candidates.items():
            prefix = name.replace("-", "_")
            d[f"{prefix}_bler"] = cand.bler
            d[f"{prefix}_ci_low"] = cand.bler_ci_low
            d[f"{prefix}_ci_high"] = cand.bler_ci_high
            d[f"{prefix}_eligible"] = cand.eligible
        return d


def compute_confidence_interval(
    errors: int,
    total_trials: int,
    method: str = "wald",
    k: float = CONFIDENCE_K_DEFAULT,
) -> Tuple[float, float]:
    """Compute confidence interval for Bernoulli trial under chosen method."""
    if total_trials <= 0:
        return 0.0, 1.0

    p = float(errors) / float(total_trials)
    m = method.lower()

    if m == "wald":
        se = math.sqrt(max(0.0, p * (1.0 - p) / float(total_trials)))
        return max(0.0, p - k * se), min(1.0, p + k * se)

    elif m == "wilson":
        n = float(total_trials)
        z = k
        denom = 1.0 + (z ** 2) / n
        center = (p + (z ** 2) / (2.0 * n)) / denom
        margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z ** 2) / (4.0 * (n ** 2))))
        return max(0.0, center - margin), min(1.0, center + margin)

    elif m in ("clopper_pearson", "exact"):
        alpha = 2.0 * (1.0 - scipy.stats.norm.cdf(k))
        n = total_trials
        x = errors
        low = 0.0 if x == 0 else float(scipy.stats.beta.ppf(alpha / 2.0, x, n - x + 1))
        high = 1.0 if x == n else float(scipy.stats.beta.ppf(1.0 - alpha / 2.0, x + 1, n - x))
        return low, high

    else:
        raise ValueError(f"Unknown CI method: {method}")


def resolve_equal_eta_tie(
    candidates: Sequence[CodedCandidateRecord],
    tie_break_rule: str,
    predefined_order: Sequence[str],
) -> Tuple[CodedCandidateRecord, str]:
    """Resolve equal-spectral-efficiency tie among eligible candidates."""
    if len(candidates) == 1:
        return candidates[0], "single_candidate"

    rule = tie_break_rule.lower()

    if rule == "lowest_bler":
        # Strategy A: choose action with lowest measured BLER
        sorted_cands = sorted(candidates, key=lambda c: c.bler)
        return sorted_cands[0], "tie_break:lowest_bler"

    elif rule == "lower_modulation":
        # Strategy C: choose lower modulation order (e.g. 16QAM over 64QAM)
        mod_order = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6}
        sorted_cands = sorted(candidates, key=lambda c: mod_order.get(c.modulation.upper(), 99))
        return sorted_cands[0], "tie_break:lower_modulation"

    elif rule == "stronger_coding":
        # Strategy D: choose stronger coding (lower code rate, denser constellation)
        # e.g. 64QAM-1/2 (Rc=1/2) over 16QAM-3/4 (Rc=3/4)
        mod_order = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6}
        # lower code rate means higher modulation order for the same eta
        sorted_cands = sorted(candidates, key=lambda c: -mod_order.get(c.modulation.upper(), 0))
        return sorted_cands[0], "tie_break:stronger_coding"

    elif rule == "predefined":
        # Strategy F: deterministic predefined preference ordering
        order_map = {name: idx for idx, name in enumerate(predefined_order)}
        sorted_cands = sorted(candidates, key=lambda c: order_map.get(c.action_name, 999))
        return sorted_cands[0], "tie_break:predefined_order"

    else:
        raise ValueError(f"Unknown tie_break_rule: {tie_break_rule}")


def compute_r1_ground_truth(
    calibration_data: Dict[float, Dict[str, Dict[str, Any]]],
    config: Optional[R1GroundTruthConfig] = None,
) -> List[R1GroundTruthRow]:
    """Compute R1 ground truth rows from a calibration dictionary.

    Parameters
    ----------
    calibration_data : Dict[snr_db, Dict[action_name, action_metrics]]
        Calibration data structure where action_metrics includes:
        - bler or (block_errors and num_blocks)
        - modulation
        - spectral_efficiency
    config : R1GroundTruthConfig, optional
        Configuration settings. If None, default settings are used.

    Returns
    -------
    List[R1GroundTruthRow]
        Sorted sequence of ground truth records.
    """
    if config is None:
        config = R1GroundTruthConfig()

    rows: List[R1GroundTruthRow] = []

    for snr_db in sorted(calibration_data.keys()):
        action_map = calibration_data[snr_db]
        candidate_records: Dict[str, CodedCandidateRecord] = {}

        for act_name, metrics in action_map.items():
            num_blocks = int(metrics.get("num_blocks", 1))
            bler = float(metrics.get("bler", 0.0))
            errors = int(metrics.get("block_errors", int(round(bler * num_blocks))))
            eta = float(metrics.get("spectral_efficiency", metrics.get("effective_spectral_efficiency", 1.0)))
            mod = str(metrics.get("modulation", "QPSK"))
            info_ber = float(metrics.get("info_ber", 0.0))

            ci_low, ci_high = compute_confidence_interval(
                errors=errors,
                total_trials=num_blocks,
                method=config.ci_method,
                k=config.confidence_k,
            )
            se = math.sqrt(max(0.0, bler * (1.0 - bler) / float(num_blocks))) if num_blocks > 1 else 0.0

            if config.eligibility_rule == "conservative_ci":
                eligible = (ci_high <= config.bler_target)
            else:
                eligible = (bler <= config.bler_target)

            candidate_records[act_name] = CodedCandidateRecord(
                action_name=act_name,
                modulation=mod,
                spectral_efficiency=eta,
                bler=bler,
                bler_se=se,
                bler_ci_low=ci_low,
                bler_ci_high=ci_high,
                eligible=eligible,
                info_ber=info_ber,
            )

        # 1. Determine eligible candidate set
        eligible_cands = [c for c in candidate_records.values() if c.eligible]

        fallback_used = False
        equal_eta_tie = False
        selection_reason = ""
        best_cand: Optional[CodedCandidateRecord] = None

        if eligible_cands:
            # Find maximum spectral efficiency among eligible candidates
            max_eta = max(c.spectral_efficiency for c in eligible_cands)
            top_cands = [c for c in eligible_cands if abs(c.spectral_efficiency - max_eta) < 1e-9]

            if len(top_cands) == 1:
                best_cand = top_cands[0]
                selection_reason = f"highest_eligible_eta ({max_eta:.2f})"
            else:
                equal_eta_tie = True
                best_cand, tie_reason = resolve_equal_eta_tie(
                    candidates=top_cands,
                    tie_break_rule=config.tie_break_rule,
                    predefined_order=config.predefined_action_order,
                )
                selection_reason = f"highest_eligible_eta ({max_eta:.2f}) + {tie_reason}"
        else:
            # Fallback policy
            fallback_used = True
            all_cands = list(candidate_records.values())
            if config.fallback_rule == "lowest_rate":
                # Most robust / lowest spectral efficiency candidate
                sorted_by_rate = sorted(all_cands, key=lambda c: (c.spectral_efficiency, c.bler))
                best_cand = sorted_by_rate[0]
                selection_reason = f"fallback:lowest_spectral_efficiency ({best_cand.spectral_efficiency:.2f})"
            elif config.fallback_rule == "fixed_action":
                best_cand = candidate_records.get(
                    config.fixed_robust_action,
                    sorted(all_cands, key=lambda c: c.spectral_efficiency)[0]
                )
                selection_reason = f"fallback:fixed_robust ({config.fixed_robust_action})"
            else:
                raise ValueError(f"Unknown fallback_rule: {config.fallback_rule}")

        # Check label uncertainty (pessimistic vs optimistic eligibility)
        opt_eligible = [c for c in candidate_records.values() if c.bler_ci_low <= config.bler_target]
        opt_max_eta = max((c.spectral_efficiency for c in opt_eligible), default=0.0)
        label_uncertain = (abs(opt_max_eta - best_cand.spectral_efficiency) > 1e-9 and not fallback_used)

        rows.append(
            R1GroundTruthRow(
                snr_db=snr_db,
                best_action=best_cand.action_name,
                best_spectral_efficiency=best_cand.spectral_efficiency,
                selection_reason=selection_reason,
                fallback_used=fallback_used,
                label_uncertain=label_uncertain,
                equal_eta_tie_broken=equal_eta_tie,
                candidates=candidate_records,
            )
        )

    return rows


def evaluate_r1_baselines(
    gt_rows: Sequence[R1GroundTruthRow],
    config: Optional[R1GroundTruthConfig] = None,
) -> Dict[str, Any]:
    """Evaluate candidate baseline adaptation policies against Ground Truth."""
    if config is None:
        config = R1GroundTruthConfig()

    fixed_robust_name = config.fixed_robust_action
    fixed_high_tp_name = config.fixed_high_tp_action

    lut_etas: List[float] = []
    lut_violations: int = 0

    robust_etas: List[float] = []
    robust_violations: int = 0

    high_tp_etas: List[float] = []
    high_tp_violations: int = 0

    for r in gt_rows:
        # 1. 1D Coded Ground Truth LUT
        lut_etas.append(r.best_spectral_efficiency)
        chosen_cand = r.candidates.get(r.best_action)
        if chosen_cand and chosen_cand.bler > config.bler_target:
            lut_violations += 1

        # 2. Fixed Robust Baseline
        rob_cand = r.candidates.get(fixed_robust_name)
        if rob_cand:
            robust_etas.append(rob_cand.spectral_efficiency)
            if rob_cand.bler > config.bler_target:
                robust_violations += 1

        # 3. Fixed High-Throughput Baseline
        high_cand = r.candidates.get(fixed_high_tp_name)
        if high_cand:
            high_tp_etas.append(high_cand.spectral_efficiency)
            if high_cand.bler > config.bler_target:
                high_tp_violations += 1

    n_points = len(gt_rows)
    return {
        "num_operating_points": n_points,
        "mean_spectral_efficiency": {
            "ground_truth_lut": float(np.mean(lut_etas)) if lut_etas else 0.0,
            "fixed_robust": float(np.mean(robust_etas)) if robust_etas else 0.0,
            "fixed_high_throughput": float(np.mean(high_tp_etas)) if high_tp_etas else 0.0,
        },
        "bler_violation_rate": {
            "ground_truth_lut": float(lut_violations) / float(n_points) if n_points > 0 else 0.0,
            "fixed_robust": float(robust_violations) / float(n_points) if n_points > 0 else 0.0,
            "fixed_high_throughput": float(high_tp_violations) / float(n_points) if n_points > 0 else 0.0,
        },
        "total_violations": {
            "ground_truth_lut": lut_violations,
            "fixed_robust": robust_violations,
            "fixed_high_throughput": high_tp_violations,
        },
    }
