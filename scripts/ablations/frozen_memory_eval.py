#!/usr/bin/env python3
"""Run a MemCo eval-only ablation from a staged frozen-memory snapshot.

This experiment wrapper changes no production MemCo code.  It replaces only
the final global rebuild used by eval_collab_multidomain_global.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
for path in (str(SCRIPTS_DIR), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

# mas.llm reads these variables at import time.  The launch scripts export the
# real values; defaults keep --help and static validation usable on their own.
os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
os.environ.setdefault("OPENAI_API_KEY", "dummy")

import eval_collab_multidomain_global as eval_runner  # noqa: E402
from mas.memory.mas_memory.memco_backend.build_memory_graph import _global_to_dict  # noqa: E402
from mas.memory.mas_memory.memco_backend.construction_graph import GlobalPromoter  # noqa: E402
from mas.memory.mas_memory.memco_backend.graph_types import GlobalGraphMemory  # noqa: E402
from mas.memory.mas_memory.memco_backend.promotion import wilson_lower_bound  # noqa: E402
from mas.memory.mas_memory.memco_backend.serialization import load_local_memory  # noqa: E402


class AblationPromoter(GlobalPromoter):
    """Keep Wilson aggregation/filtering and replace only its decision score."""

    def __init__(self, *, decision_policy: str, **kwargs: Any) -> None:
        super().__init__(policy="wilson", **kwargs)
        self.decision_policy = decision_policy

    def _wilson_decision(
        self,
        *,
        record_kind: str,
        record_id: str,
        evidence: Any,
        structurally_valid: bool,
        low_information: bool = False,
    ) -> tuple[bool, dict[str, Any]]:
        if self.decision_policy == "empirical_rate":
            decision_score = float(evidence.confidence)
            score_name = "empirical_support_rate"
        elif self.decision_policy == "positive_only":
            decision_score = wilson_lower_bound(
                int(evidence.supporting),
                0,
                alpha=self.wilson_alpha,
            )
            score_name = "positive_only_wilson_lower_bound"
        else:
            raise ValueError(f"unsupported ablation decision policy: {self.decision_policy}")

        reasons: list[str] = []
        if not structurally_valid:
            reasons.append("invalid_structure")
        if low_information:
            reasons.append("low_information")
        if decision_score < self.wilson_threshold:
            reasons.append(f"{score_name}_below_threshold")
        accepted = not reasons
        decision = {
            "record_kind": record_kind,
            "record_id": record_id,
            **evidence.to_dict(alpha=self.wilson_alpha),
            "decision_score_name": score_name,
            "decision_score": decision_score,
            "decision_threshold": self.wilson_threshold,
            "ignored_contradicting_trials": (
                int(evidence.contradicting) if self.decision_policy == "positive_only" else 0
            ),
            "structurally_valid": structurally_valid,
            "low_information": low_information,
            "accepted": accepted,
            "reasons": reasons,
        }
        return accepted, decision


def _reuse_staged_global(*, global_dir: str, memory_namespace: str = "memco", **_: Any) -> None:
    global_path = Path(global_dir) / memory_namespace / "global_memory.json"
    if not global_path.is_file() or global_path.stat().st_size == 0:
        raise FileNotFoundError(f"staged frozen global memory is missing: {global_path}")


def _make_ablation_rebuilder(decision_policy: str):
    def rebuild(
        *,
        local_dirs: list[str],
        global_dir: str,
        promotion_threshold: float,
        memory_namespace: str = "memco",
        promotion_policy: str = "wilson",
        wilson_alpha: float = 0.05,
        wilson_threshold: float = 0.35,
        wilson_min_coverage: int | None = None,
    ) -> None:
        del promotion_policy, wilson_min_coverage
        local_memories = []
        for local_root in local_dirs:
            memory_dir = Path(local_root) / memory_namespace
            for local_path in sorted(memory_dir.glob("local_*.json")):
                local_memories.append(load_local_memory(str(local_path)))
        if not local_memories:
            raise ValueError("no staged local-memory snapshots were found")

        promoter = AblationPromoter(
            decision_policy=decision_policy,
            score_threshold=float(promotion_threshold),
            wilson_alpha=float(wilson_alpha),
            wilson_threshold=float(wilson_threshold),
        )
        global_memory = promoter.promote(
            GlobalGraphMemory(),
            local_memories,
            batch_name=f"frozen_memory_ablation:{decision_policy}",
        )

        output_dir = Path(global_dir) / memory_namespace
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "global_memory.json").open("w", encoding="utf-8") as writer:
            json.dump(_global_to_dict(global_memory), writer, ensure_ascii=False, indent=2)

        report = dict(promoter.last_promotion_report)
        report["active_policy"] = decision_policy
        report["ablation"] = {
            "kind": "frozen_memory_promotion_decision",
            "decision_policy": decision_policy,
            "source_run_id": os.getenv("WILSON_ABLATION_SOURCE_RUN_ID", ""),
            "structural_checks_unchanged": True,
            "low_information_filter_unchanged": True,
            "source_coverage": "diagnostic_only",
        }
        with (output_dir / "promotion_wilson.json").open("w", encoding="utf-8") as writer:
            json.dump(report, writer, ensure_ascii=False, indent=2)

        summary = {
            "mode": "frozen_memory_ablation",
            "decision_policy": decision_policy,
            "source_run_id": os.getenv("WILSON_ABLATION_SOURCE_RUN_ID", ""),
            "source_local_count": len(local_memories),
            "global": {
                "candidate_count": len(global_memory.candidates),
                "rule_count": len(global_memory.rules_by_id),
                "artifact_count": len(global_memory.artifacts_by_id),
                "promoted_batches": list(global_memory.promoted_batches),
            },
        }
        with (output_dir / "summary.json").open("w", encoding="utf-8") as writer:
            json.dump(summary, writer, ensure_ascii=False, indent=2)

    return rebuild


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--ablation-promotion-policy",
        required=True,
        choices=("reuse", "wilson", "empirical_rate", "positive_only"),
    )
    known, remaining = parser.parse_known_args()

    if known.ablation_promotion_policy == "reuse":
        eval_runner.rebuild_memco_global_from_locals = _reuse_staged_global
    elif known.ablation_promotion_policy != "wilson":
        eval_runner.rebuild_memco_global_from_locals = _make_ablation_rebuilder(
            known.ablation_promotion_policy
        )

    sys.argv = [sys.argv[0], *remaining]
    eval_runner.main()


if __name__ == "__main__":
    main()
