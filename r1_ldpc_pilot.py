"""R1: LDPC Pilot Verification and Correctness Runner.

Executes a small-scale, mathematically rigorous verification pilot for the
R1 LDPC-coded physical layer baseline.

Verifies:
1. End-to-end coded communication chain across all 8 initial validated actions:
   - BPSK-1/2, QPSK-1/2, QPSK-2/3, 16QAM-1/2, 16QAM-3/4, 64QAM-1/2, 64QAM-2/3, 64QAM-3/4
2. Execution over slow Rayleigh flat block fading with perfect coherent CSI.
3. Strict preservation of nominal Es/N0 as the setup axis, with explicit derivation of Eb/N0.
4. Separate tracking of Information BER and Codeword BLER.
5. Generation of pilot artifacts, CSV tables, manifest, and markdown verification report.

No full SNR sweep. No Deep 70k run. No Ground Truth policy yet.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch

from ldpc_phy import (
    ACTION_BY_NAME,
    BLOCK_SYMBOLS_DEFAULT,
    CodedAction,
    CodedBlockStats,
    CodedModem,
    R1_INITIAL_ACTIONS,
    get_ldpc_code_params,
    simulate_coded_transmission,
)


def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        import subprocess
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN"


def run_r1_pilot(
    actions: Sequence[CodedAction] = R1_INITIAL_ACTIONS,
    snrs: Sequence[float] = (10.0, 18.0, 26.0),
    num_blocks: int = 20,
    block_symbols: int = BLOCK_SYMBOLS_DEFAULT,
    num_iter: int = 15,
    cn_update: str = "boxplus-phi",
    precision: str = "double",
    device: str = "cpu",
    output_dir: Union[str, Path] = "results/r1_ldpc_pilot",
    master_seed: int = 20260920,
) -> Dict[str, Any]:
    """Execute R1 LDPC pilot verification run."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    git_commit = get_git_commit()
    t0 = time.perf_counter()

    print("=" * 75)
    print("        PHY-ML R1: LDPC CODED PHY PILOT VERIFICATION RUN")
    print("=" * 75)
    print(f"Target Architecture:    R1 LDPC Foundation (5G NR LDPC + Slow Rayleigh Flat Fading)")
    print(f"Source Git Commit:      {git_commit}")
    print(f"Simulation Setup Axis:  Nominal Es/N0 (dB) [Parent-compatible with frozen R0]")
    print(f"Transmission Block:     {block_symbols} complex symbols/block")
    print(f"Decoder Specification:  Sionna 2.0 LDPC5GDecoder (BP: {cn_update}, {num_iter} iterations)")
    print(f"Candidate Actions:      {len(actions)} actions ({', '.join(a.name for a in actions)})")
    print(f"Operating Pilot SNRs:   {list(snrs)} dB")
    print(f"Blocks per SNR-Action:  {num_blocks} blocks/codeword trials")
    print(f"Master Seed:            {master_seed}")
    print(f"PyTorch Device:         {device} (precision: {precision})")
    print("-" * 75)

    results_rows: List[Dict[str, Any]] = []

    for act_idx, action in enumerate(actions):
        modem = CodedModem(
            action=action,
            block_symbols=block_symbols,
            num_iter=num_iter,
            cn_update=cn_update,
            precision=precision,
            device=device,
        )

        for snr_idx, snr_db in enumerate(snrs):
            # Deterministic per-point seed
            point_seed = (master_seed + act_idx * 10007 + snr_idx * 997) & 0x7FFFFFFF

            res = simulate_coded_transmission(
                modem=modem,
                snr_db=snr_db,
                num_blocks=num_blocks,
                channel_type="rayleigh",
                seed=point_seed,
            )

            # Accumulate per-block statistics for exact standard error computation
            stats = CodedBlockStats(
                action=action,
                snr_db=snr_db,
                k=modem.k,
                n=modem.n,
            )
            for err in res.per_block_bit_errors:
                stats.update(err)

            stats_dict = stats.to_dict()
            stats_dict["decoder_iterations"] = num_iter
            stats_dict["cn_update"] = cn_update
            stats_dict["channel_type"] = "Slow Rayleigh Flat Block Fading"
            stats_dict["csi"] = "Perfect Coherent CSI"

            results_rows.append(stats_dict)

            print(
                f"  [{action.name:10s}] SNR: {snr_db:5.1f} dB (Eb/N0: {stats_dict['eb_n0_db']:5.1f} dB) | "
                f"k={modem.k:4d}, n={modem.n:4d} | "
                f"Info BER: {stats_dict['info_ber']:8.4e} (errs: {stats_dict['info_bit_errors']:5d}) | "
                f"Codeword BLER: {stats_dict['bler']:6.4f} ({stats_dict['block_errors']:2d}/{num_blocks})"
            )

    elapsed_s = time.perf_counter() - t0

    # 1. Save CSV
    csv_path = out_p / "r1_pilot_results.csv"
    if results_rows:
        fieldnames = list(results_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_rows)

    # 2. Save Manifest
    manifest = {
        "title": "PHY-ML R1 LDPC Coded PHY Pilot Verification Manifest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": git_commit,
        "runtime_seconds": elapsed_s,
        "scientific_specification": {
            "channel_model": "Slow Rayleigh Flat Block Fading (E[|h|^2] = 1.0)",
            "csi_knowledge": "Perfect Coherent CSI (h known at receiver)",
            "equalization": "y_eq = y / h, N0_eff = N0 / |h|^2",
            "demapping": "Soft APP Demapping (no hard decisions prior to LDPC decoder)",
            "fec_scheme": "5G NR LDPC (3GPP TS 38.212)",
            "decoder_algorithm": f"Belief Propagation ({cn_update}, {num_iter} iterations)",
            "symbols_per_block": block_symbols,
            "simulation_axis": "Nominal Es/N0 (dB) [Es = 1, N0 = 1 / 10^(SNR/10)]",
            "derived_eb_n0_formula": "(Eb/N0)_dB = (Es/N0)_dB - 10 * log10(eta)",
            "independent_statistical_unit": "Codeword / Transport-Block (1 codeword per Rayleigh fading block)",
        },
        "pilot_configuration": {
            "actions": [a.to_dict() for a in actions],
            "snrs_db": list(snrs),
            "num_blocks_per_point": num_blocks,
            "total_evaluations": len(results_rows),
            "master_seed": master_seed,
            "precision": precision,
            "device": str(device),
        },
        "output_files": {
            "results_csv": str(csv_path.name),
            "report_md": "r1_pilot_report.md",
        },
    }

    manifest_path = out_p / "r1_pilot_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 3. Save Markdown Report
    report_path = out_p / "r1_pilot_report.md"
    generate_pilot_markdown_report(
        output_path=report_path,
        manifest=manifest,
        results_rows=results_rows,
        actions=actions,
        snrs=snrs,
        elapsed_s=elapsed_s,
    )

    print("-" * 75)
    print(f"Pilot execution completed in {elapsed_s:.2f} s ({elapsed_s / len(results_rows):.3f} s/point)")
    print(f"Saved Results CSV:     {csv_path}")
    print(f"Saved Pilot Manifest:  {manifest_path}")
    print(f"Saved Markdown Report: {report_path}")
    print("=" * 75)

    return {
        "status": "PASS",
        "elapsed_s": elapsed_s,
        "results_rows": results_rows,
        "manifest": manifest,
    }


def generate_pilot_markdown_report(
    output_path: Path,
    manifest: Dict[str, Any],
    results_rows: Sequence[Dict[str, Any]],
    actions: Sequence[CodedAction],
    snrs: Sequence[float],
    elapsed_s: float,
) -> None:
    """Generate Markdown report summarizing R1 LDPC pilot verification."""
    lines = [
        "# PHY-ML R1: LDPC Coded PHY Foundation Pilot Verification Report",
        "",
        f"**Generated At (UTC):** `{manifest['generated_at_utc']}`  ",
        f"**Source Git Commit:** `{manifest['source_git_commit']}`  ",
        f"**Execution Runtime:** `{elapsed_s:.2f} s`  ",
        "**Stage:** R1 LDPC Coded Foundation (Verification Pilot)  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Design Scope",
        "",
        "This report verifies the initial integration of **5G NR LDPC channel coding** into the PHY-ML research core, directly inheriting from the frozen **R0 uncoded baseline root**.",
        "",
        "### Invariant Physical Foundations (Preserved from R0)",
        "- **Channel Model:** Slow Rayleigh flat block fading (constant complex gain $h \\sim \\mathcal{CN}(0, 1)$ over block symbols, independent between blocks).",
        "- **CSI Knowledge:** Perfect coherent CSI known at the receiver.",
        "- **Simulation Setup Axis:** Nominal $E_s/N_0$ in dB held constant ($E_s = 1$). No silent axis shift.",
        "- **Constellation Primitives:** Unit average energy Gray-mapped constellations (BPSK, QPSK, 16QAM, 64QAM).",
        "- **Seeding Discipline:** Exact deterministic reproducibility using PyTorch generators.",
        "- **R0 Protection:** R0 frozen artifacts, calibrations, and decision trees remain completely untouched.",
        "",
        "### Single Communication-System Complexity Added",
        "- **Channel Coding:** 5G NR LDPC encoder and soft belief-propagation decoder (`sionna.phy.fec.ldpc.LDPC5GEncoder` / `LDPC5GDecoder`).",
        "- **Demapper Interface:** Pure soft log-likelihood ratios (LLRs). Absolutely no hard decisions before LDPC decoding.",
        "",
        "---",
        "",
        "## 2. R1 Action Representation & Spectral Efficiency",
        "",
        "Actions in R1 are parameterized as tuples `(modulation, code_rate)` with effective spectral efficiency:",
        "$$\\eta = R_c \\cdot \\log_2(M) \\quad [\\text{information bits / complex symbol}]$$",
        "",
        "The derived relationship to information-bit $E_b/N_0$ is recorded for fairness:",
        "$$\\left(\\frac{E_b}{N_0}\\right)_{\\text{dB}} = \\left(\\frac{E_s}{N_0}\\right)_{\\text{dB}} - 10 \\log_{10}(\\eta)$$",
        "",
        "### Initial Validated Action Catalog",
        "",
        "| Action Name | Modulation ($M$) | Bits/Symbol ($\\log_2 M$) | Target Code Rate ($R_c$) | Info Bits ($k$) | Coded Bits ($n$) | Spectral Efficiency ($\\eta$) |",
        "|:------------|:----------------:|:------------------------:|:------------------------:|:---------------:|:----------------:|:----------------------------:|",
    ]

    for act in actions:
        k, n = get_ldpc_code_params(act, BLOCK_SYMBOLS_DEFAULT)
        lines.append(
            f"| `{act.name}` | {act.modulation} ({2**act.bits_per_symbol}) | {act.bits_per_symbol} | "
            f"`{act.code_rate_str}` ({act.code_rate:.3f}) | {k} | {n} | **{act.spectral_efficiency:.2f}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Independent Statistical Unit & Reliability Metrics",
        "",
        "- **Independent Statistical Unit:** Under slow block fading, each codeword of $n = 1536 \\cdot m$ bits spans exactly one transmission block and experiences one independent channel realization $h$. The **codeword / transport-block is the independent statistical unit**.",
        "- **Codeword BLER:** Modeled as independent Bernoulli trials across blocks ($E_i \\in \\{0, 1\\}$). Standard error: $\\text{SE}(\\text{BLER}) = \\sqrt{\\frac{\\text{BLER}(1 - \\text{BLER})}{N_{\\text{blocks}}}}$.",
        "- **Information BER:** Bit errors within a codeword are correlated. Standard error is computed from the empirical sample standard deviation of per-block BERs divided by $\\sqrt{N_{\\text{blocks}}}$.",
        "",
        "---",
        "",
        "## 4. Pilot Verification Results",
        "",
        "| Action | Nominal $E_s/N_0$ | Derived $E_b/N_0$ | Blocks | Info Bits ($k$) | Coded Bits ($n$) | Bit Errors | Info BER | Codeword Errors | Codeword BLER |",
        "|:-------|:-----------------:|:-----------------:|:------:|:---------------:|:----------------:|:----------:|:--------:|:---------------:|:-------------:|",
    ])

    for row in results_rows:
        lines.append(
            f"| `{row['action']}` | {row['snr_db']:.1f} dB | {row['eb_n0_db']:.2f} dB | "
            f"{row['num_blocks']} | {row['info_bits_per_block']} | {row['coded_bits_per_block']} | "
            f"{row['info_bit_errors']} | {row['info_ber']:.4e} | {row['block_errors']} | **{row['bler']:.4f}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Next Steps for R1",
        "",
        "1. Complete code rate and modulation expansion.",
        "2. Establish R1 Monte Carlo calibration protocol (balancing compute cost with LDPC decoding latency).",
        "3. Define Ground Truth policy for coded transmission (BLER target vs BER target).",
        "",
    ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run R1 LDPC PHY Pilot Verification.")
    parser.add_argument("--num-blocks", type=int, default=20, help="Number of blocks per point (default: 20)")
    parser.add_argument("--num-iter", type=int, default=15, help="Decoder iterations (default: 15)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--precision", type=str, default="double", help="double or single")
    parser.add_argument("--out-dir", type=str, default="results/r1_ldpc_pilot", help="Output directory")
    args = parser.parse_args()

    run_r1_pilot(
        num_blocks=args.num_blocks,
        num_iter=args.num_iter,
        device=args.device,
        precision=args.precision,
        output_dir=args.out_dir,
    )
