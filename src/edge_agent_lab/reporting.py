"""Portable research artifacts: plots, Markdown tables, and an offline dashboard."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import read_json, write_json

PALETTE = ["#47c9ad", "#7b9ff2", "#e0b36a", "#bf92e8", "#e67d8e", "#75c7df"]


def build_reports(
    artifacts: Path, output: Path, data_root: Path | None = None, edge_root: Path | None = None
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    edge_root = edge_root or artifacts / "edge"
    data_root = data_root or Path("data/har")
    edge_path = edge_root / "summary.json"
    if not edge_path.exists():
        raise ValueError("Training summary is missing; finish the edge study first")
    edge = read_json(edge_path)
    tests = (
        read_json(edge_root / "test_evaluation.json")
        if (edge_root / "test_evaluation.json").exists()
        else {}
    )
    steward = (
        read_json(artifacts / "steward/evaluation.json")
        if (artifacts / "steward/evaluation.json").exists()
        else {}
    )
    pilot = (
        read_json(artifacts / "pilot/selection.json")
        if (artifacts / "pilot/selection.json").exists()
        else {}
    )
    apple = (
        read_json(artifacts / "apple/export.json")
        if (artifacts / "apple/export.json").exists()
        else {}
    )
    telemetry = (
        read_json(artifacts / "pilot/telemetry.json")
        if (artifacts / "pilot/telemetry.json").exists()
        else {}
    )
    local_llm = (
        read_json(artifacts / "steward/local_llm_evaluation.json")
        if (artifacts / "steward/local_llm_evaluation.json").exists()
        else {}
    )
    extras = {}
    for key, relative in {
        "baselines": "baselines/results.json",
        "external": "external/results.json",
        "native": "native/parity.json",
        "native_search": "native-search/parity.json",
        "search": "search/selection.json",
        "search_test": "search/test_evaluation.json",
        "infeasible_pilot": "pilot-infeasible/selection.json",
        "apple_search": "apple-search/export.json",
        "llm_schema_only": "steward/local_llm_unscaffolded.json",
        "llm_large": "steward-large/local_llm_evaluation.json",
        "llm_grounded": "steward-grounded/local_llm_evaluation.json",
        "llm_intent": "steward-intent/evaluation.json",
        "llm_policy_guarded": "steward-policy-guarded/local_llm_evaluation.json",
    }.items():
        path = artifacts / relative
        extras[key] = read_json(path) if path.exists() else {}
    grouped = defaultdict(list)
    for row in edge["results"]:
        grouped[row["name"]].append(row)
    test_by = {row["candidate"]: row for row in tests.get("results", [])}
    aggregates = []
    for name, rows in grouped.items():
        test_rows = [
            test_by[f"seed-{r['seed']}/{name}"]
            for r in rows
            if f"seed-{r['seed']}/{name}" in test_by
        ]
        values = [r["validation"]["macro_f1"] for r in rows]
        test_values = [r["clean"]["macro_f1"] for r in test_rows]
        aggregates.append(
            {
                "name": name,
                "seeds": len(rows),
                "val_f1_mean": float(np.mean(values)),
                "val_f1_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0,
                "test_f1_mean": float(np.mean(test_values)) if test_values else None,
                "test_f1_std": float(np.std(test_values, ddof=1)) if len(test_values) > 1 else 0,
                "p95_ms_median": float(np.median([r["benchmark"]["p95_ms"] for r in rows])),
                "bytes": int(np.median([r["benchmark"]["state_dict_bytes"] for r in rows])),
                "noise_f1_mean": float(
                    np.mean([r["stress"]["noise"]["macro_f1"] for r in test_rows])
                )
                if test_rows
                else None,
                "dropout_f1_mean": float(
                    np.mean([r["stress"]["gyro_dropout"]["macro_f1"] for r in test_rows])
                )
                if test_rows
                else None,
            }
        )
    bundle = {
        **extras,
        "edge": edge,
        "test": tests,
        "aggregates": aggregates,
        "steward": steward,
        "pilot": pilot,
        "apple": apple,
        "telemetry": telemetry,
        "local_llm": local_llm,
    }
    bundle["agent_examples"] = {}
    for label, report, folder, case_id in [
        ("policy-gyro", steward, "steward", "gyro"),
        ("intent-router-gyro", extras["llm_intent"], "steward-intent", "gyro"),
        ("small-planner-failure", local_llm, "steward", "provenance"),
        ("large-planner-gyro", extras["llm_large"], "steward-large", "gyro"),
        ("policy-fixed-gyro", extras["llm_policy_guarded"], "steward-policy-guarded", "gyro"),
    ]:
        row = next((r for r in report.get("results", []) if r["id"] == case_id), None)
        if row:
            path = artifacts / folder / "runs" / f"{row['run_id']}.json"
            if path.exists():
                trace = read_json(path)
                bundle["agent_examples"][label] = trace
                write_json(output / "traces" / f"{label}.json", trace)
    write_json(output / "results.json", bundle)
    # Keep machine-readable per-project summaries portable in Git.
    write_json(
        output / "data_manifest.json",
        read_json(data_root / "manifest.json") if (data_root / "manifest.json").exists() else {},
    )
    _plots(aggregates, test_by, telemetry, output)
    (output / "RESULTS.md").write_text(_markdown(bundle))
    _dashboard(bundle, output)
    return {
        "report": str(output / "RESULTS.md"),
        "dashboard": str(output / "index.html"),
        "candidates": len(aggregates),
        "training_runs": len(edge["results"]),
    }


def _plots(rows, tests, telemetry, output):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for i, row in enumerate(rows):
        color = PALETTE[i % len(PALETTE)]
        axes[0].scatter(
            row["bytes"] / 1024,
            row["val_f1_mean"],
            s=65,
            color=color,
            marker=["o", "s", "^", "v", "D", "P", "X", "<", ">", "*"][i % 10],
            label=row["name"].replace("_", " "),
        )
    axes[0].set(
        xlabel="Serialized state size (KiB)",
        ylabel="Validation macro F1",
        title="Accuracy–size tradeoff",
    )
    names = [r["name"] for r in rows]
    axes[1].barh(names, [r["val_f1_mean"] for r in rows], color="#7b9ff2", label="Validation")
    if all(r["test_f1_mean"] is not None for r in rows):
        axes[1].scatter(
            [r["test_f1_mean"] for r in rows],
            range(len(rows)),
            color="#253c56",
            marker="D",
            zorder=3,
            label="Official test",
        )
    axes[1].set(xlabel="Macro F1", xlim=(0, 1), title="Real-data evaluation")
    axes[1].legend(loc="lower right")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    fig.savefig(output / "model_tradeoffs.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    subset = [(name, row) for name, row in tests.items() if name.startswith("seed-7/")]
    if subset:
        values = [
            [r["clean"]["macro_f1"]]
            + [r["stress"][k]["macro_f1"] for k in ("noise", "gain", "gyro_dropout", "rotation")]
            for _, r in subset
        ]
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        im = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_xticks(range(5), ["clean", "noise", "gain", "gyro outage", "rotation"])
        ax.set_yticks(range(len(subset)), [n.split("/")[1] for n, _ in subset])
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.65 else "black",
                )
        ax.set_title("Held-out test stress evaluation · seed 7")
        fig.colorbar(im, ax=ax, label="Macro F1")
        fig.savefig(output / "robustness.png", dpi=160)
        plt.close(fig)
    if telemetry.get("events"):
        events = telemetry["events"]
        fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
        ax.plot(
            [e["window"] for e in events], [e["accuracy"] for e in events], "o-", color="#2c7a72"
        )
        ax.axvspan(5.5, 11.5, color="#e67d8e", alpha=0.2, label="Synthetic gyroscope outage")
        ax.set(
            xlabel="Validation replay batch",
            ylabel="Accuracy",
            ylim=(0, 1.05),
            title="Shadow monitoring replay",
        )
        ax.legend()
        fig.savefig(output / "telemetry.png", dpi=160)
        plt.close(fig)


def _dashboard(bundle, output):
    # Escape script delimiters even though all content originates in local artifacts.
    payload = json.dumps(bundle).replace("<", "\\u003c").replace(">", "\\u003e")
    page = (Path(__file__).parent / "templates/report.html").read_text()
    (output / "index.html").write_text(page.replace("__PAYLOAD__", payload))


def _markdown(bundle: dict) -> str:
    rows = bundle["aggregates"]
    by_name = {row["name"]: row for row in rows}
    lines = [
        "# Measured experiment results",
        "",
        "Generated from saved artifacts. Validation selects models; official held-out test",
        "results are reporting only. All figures and tables use observed measurements.",
        "",
        "## Edge model study",
        "",
        "| Candidate | Seeds | Val F1 | Test F1 | p95 ms | KiB | Outage F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        test = (
            f"{row['test_f1_mean']:.4f} ± {row['test_f1_std']:.4f}"
            if row["test_f1_mean"] is not None
            else "pending"
        )
        outage = (
            f"{row['dropout_f1_mean']:.4f}" if row["dropout_f1_mean"] is not None else "pending"
        )
        lines.append(
            f"| {row['name']} | {row['seeds']} | "
            f"{row['val_f1_mean']:.4f} ± {row['val_f1_std']:.4f} | {test} | "
            f"{row['p95_ms_median']:.3f} | {row['bytes'] / 1024:.1f} | {outage} |"
        )
    lines += [
        "",
        "± is sample standard deviation across training seeds. Per-model subject-bootstrap",
        "accuracy intervals are in `results.json`. Storage is serialized state, not peak RAM.",
        "CPU p95 is the median of per-run p95 values. Development timing had concurrent load.",
        "",
    ]
    if "teacher" in by_name and "student" in by_name:
        ratio = by_name["teacher"]["bytes"] / by_name["student"]["bytes"]
        lines.append(
            f"The supervised student state is **{ratio:.2f}× smaller** than the teacher state."
        )
    if all(
        name in by_name and by_name[name]["test_f1_mean"] is not None
        for name in ("sim_to_real", "fewshot_scratch")
    ):
        delta = 100 * (
            by_name["sim_to_real"]["test_f1_mean"] - by_name["fewshot_scratch"]["test_f1_mean"]
        )
        lines += [
            f"Simulation pretraining changes equal-label-budget test macro F1 by **{delta:+.2f}",
            "percentage points** relative to training the same 480-window budget from scratch.",
            "This is a descriptive seed average, not a significance claim.",
        ]
    lines += [
        "",
        "![Model tradeoffs](model_tradeoffs.png)",
        "",
        "![Held-out robustness](robustness.png)",
        "",
        "## Classical references",
        "",
        "Fixed model parameters; inference timing includes statistical feature extraction.",
        "",
        "| Model | Val F1 | Test F1 | p95 ms |",
        "|---|---:|---:|---:|",
    ]
    for row in bundle.get("baselines", {}).get("results", []):
        lines.append(
            f"| {row['name']} | {row['validation']['macro_f1']:.4f} | "
            f"{row['test']['macro_f1']:.4f} | {row['benchmark']['p95_ms']:.3f} |"
        )
    lines += [
        "",
        "## Data stewardship agents",
        "",
        "Strict pass requires completion, relevant retrieval, correct incident arguments, and",
        "no unnecessary simulation tools. A completed but incorrect workflow fails.",
        "",
        "| System | Cases | Strict success | Retrieval hit@3 | Mean task seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    comparisons = [
        ("Policy", bundle["steward"]),
        ("Qwen 1.5B schema only", bundle.get("llm_schema_only", {})),
        ("Qwen 1.5B workflow guards", bundle["local_llm"]),
        ("Qwen 7B workflow guards", bundle.get("llm_large", {})),
        ("Qwen 1.5B grounded per-tool schema", bundle.get("llm_grounded", {})),
        ("Qwen 1.5B intent + typed workflow", bundle.get("llm_intent", {})),
        ("Policy-fixed Qwen realization", bundle.get("llm_policy_guarded", {})),
    ]
    for name, report in comparisons:
        if report:
            duration = np.mean([row["elapsed_seconds"] for row in report["results"]])
            lines.append(
                f"| {name} | {report['cases']} | {report['task_success_rate']:.3f} | "
                f"{report['retrieval_hit_at_3']:.3f} | {duration:.2f} |"
            )
    lines += [
        "",
        "The corpus and golden suite are small curated regressions. Schema/workflow guards",
        "do not guarantee semantic correctness. Policy-fixed realization removes model decision",
        "authority: policy chooses the tool and arguments; the LLM only supplies a reason.",
        "Its success must not be attributed to autonomous planning. Inspect failures in JSON.",
        "Intent routing infers task intent before retrieval, then executes",
        "typed steps, separating model intent from execution authority.",
        "",
        "## Constraint-aware model selection",
        "",
    ]
    for title, key in [("Saved-candidate pilot", "pilot"), ("On-demand training search", "search")]:
        report = bundle.get(key, {})
        winner = report.get("winner") or {}
        lines += [
            f"### {title}",
            "",
            f"Status: **{report.get('status', 'pending')}**. "
            f"Winner: **{winner.get('candidate', 'none')}**. Trials: {report.get('trials', 0)}.",
            "",
        ]
        if winner:
            lines += [
                f"Validation F1: {winner['validation']['macro_f1']:.4f}; "
                f"outage F1: {winner['dropout_macro_f1']:.4f}; "
                f"p95: {winner['benchmark']['p95_ms']:.3f} ms; "
                f"state: {winner['benchmark']['state_dict_bytes']:,} bytes.",
                "",
            ]
        lines += [
            "All hard gates use validation measurements. Test metrics are excluded.",
            "Objectives, critic margins, trained proposals, and Pareto frontier are in JSON.",
            "",
        ]
    if bundle.get("search_test"):
        lines += [
            "### Search-model held-out evaluation",
            "",
            "Single seed; test reporting follows validation selection.",
            "",
            "| Candidate | Test macro F1 | Test gyro-outage F1 |",
            "|---|---:|---:|",
        ]
        for row in bundle["search_test"].get("results", []):
            lines.append(
                f"| {row['candidate']} | {row['clean']['macro_f1']:.4f} | "
                f"{row['stress']['gyro_dropout']['macro_f1']:.4f} |"
            )
        lines.append("")
    if bundle.get("infeasible_pilot"):
        lines += [
            "Actual 1,024-byte storage-gate probe: **"
            + bundle["infeasible_pilot"]["status"]
            + "**; no fallback was promoted.",
            "",
        ]
    lines += [
        "![Shadow telemetry](telemetry.png)",
        "",
        "## Core ML deployment",
        "",
        "CPU_ONLY timings include Python API overhead. No ANE placement or energy claim.",
        "",
        "| Variant | Package bytes | Runtime | p95 ms | PyTorch top-1 agreement |",
        "|---|---:|---|---:|---:|",
    ]
    for row in bundle["apple"].get("variants", []):
        p95 = f"{row['p95_ms']:.3f}" if "p95_ms" in row else "unavailable"
        agreement = (
            f"{row['pytorch_top1_agreement']:.4f}"
            if "pytorch_top1_agreement" in row
            else "unavailable"
        )
        lines.append(
            f"| {row['variant']} | {row['package_bytes']:,} | "
            f"{row['runtime']} | {p95} | {agreement} |"
        )
    native = bundle.get("native", {})
    if native:
        lines += [
            "",
            f"Native Swift parity passed: **{native['passed']}** on one saved input. "
            f"Maximum probability difference: {native['max_probability_difference']:.3g}.",
            "Direct swiftc compilation and Core ML inference were executed locally.",
        ]
    if bundle.get("apple_search"):
        lines += [
            "",
            "### Search-selected model deployment",
            "",
            "The newly trained candidate is exported independently of the original student.",
            "",
            "| Variant | Package bytes | p95 ms | PyTorch agreement |",
            "|---|---:|---:|---:|",
        ]
        for row in bundle["apple_search"].get("variants", []):
            p95 = f"{row['p95_ms']:.3f}" if "p95_ms" in row else "unavailable"
            agreement = (
                f"{row['pytorch_top1_agreement']:.4f}"
                if "pytorch_top1_agreement" in row
                else "unavailable"
            )
            lines.append(f"| {row['variant']} | {row['package_bytes']:,} | {p95} | {agreement} |")
        if bundle.get("native_search"):
            parity = bundle["native_search"]
            lines += [
                "",
                f"Native Swift parity passed: **{parity['passed']}**; maximum probability "
                f"difference: {parity['max_probability_difference']:.3g} on the saved example.",
            ]
    external = bundle.get("external", {})
    if external:
        lines += [
            "",
            "## Independent real-data transfer probe",
            "",
            f"HAR70+: {external['dataset']['rows']:,} retained windows from "
            f"{len(external['dataset']['subjects'])} independent subjects.",
            "Population, placement, preprocessing, and modalities differ. Gyro is absent.",
            "Both predefined orientations are reported; these labels do not select an orientation",
            "or model. This is a descriptive mismatch probe, not a comparable HAR70+ benchmark.",
            "",
            "| Candidate | Orientation | External macro F1 |",
            "|---|---|---:|",
        ]
        for row in external.get("results", []):
            lines.append(
                f"| {row['candidate']} | {row['orientation']} | {row['metrics']['macro_f1']:.4f} |"
            )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Four primary validation subjects and nine official test subjects limit confidence.",
        "- Validation is reused for epoch and candidate selection.",
        "- Synthetic simulation and corruption do not establish new-device generalization.",
        "- Few-shot uses 480 labeled windows from known training subjects.",
        "- Body-acceleration-only removes gravity information as well as the gyroscope.",
        "- Dynamic INT8 quantizes Linear layers only; Core ML compression is weight-only.",
        "- Storage, latency, RAM, energy, and hardware placement are different measurements.",
        "- The agent benchmark is curated and small; it does not prove enterprise readiness.",
        "- Workflow actors are software roles; no independent multi-agent reasoning claim.",
        "- No production deployment, clinical validation, or state-of-the-art claim.",
        "",
    ]
    return "\n".join(lines)
