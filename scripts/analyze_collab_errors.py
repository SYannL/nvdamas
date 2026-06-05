"""Analyze failed tasks from collab eval logs. Usage: python scripts/analyze_collab_errors.py --log_base logs/medmcqa_collab_eval/20260218_132751/autogen/memory/memgraph/unknown/eval --report_dir reports/collab/20260305_133814 --tasks_per 20"""
import os
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from reports.analyze_medmcqa_results import parse_log

SCENARIOS = [
    "inner_baseline_A_on_A",
    "inner_baseline_B_on_B",
    "inner_ours_A_on_A",
    "inner_ours_B_on_B",
    "cross_baseline_A_on_B",
    "cross_baseline_B_on_A",
]


def classify_fail(record):
    actions = record.actions
    last_obs = (actions[-1].get("observation") or "").strip() if actions else ""
    has_finish = any((a.get("action") or "").startswith("Finish[") for a in actions)
    if "Answer is INCORRECT" in last_obs:
        return "wrong_finish"
    if any("Invalid Action" in (a.get("observation") or "") for a in actions):
        return "invalid_action"
    if not has_finish:
        return "no_finish"
    return "other"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--log_base", default="logs/medmcqa_collab_eval/20260218_132751/autogen/memory/memgraph/unknown/eval")
    p.add_argument("--report_dir", default="reports/collab/20260305_133814")
    p.add_argument("--tasks_per", type=int, default=20)
    args = p.parse_args()

    log_base = Path(args.log_base)
    report_dir = Path(args.report_dir)
    tasks_per = args.tasks_per

    all_failed = []
    for scen in SCENARIOS:
        log_path = log_base / scen / "total_task.log"
        if not log_path.exists():
            continue
        tasks = parse_log(log_path)
        recent = tasks[-tasks_per:] if len(tasks) >= tasks_per else tasks
        for t in recent:
            if t.label is not True and (t.question or t.actions):
                t.scenario = scen
                t.fail_type = classify_fail(t)
                all_failed.append(t)

    by_scenario = Counter(t.scenario for t in all_failed)
    by_type = Counter(t.fail_type for t in all_failed)

    out = []
    out.append("# Collab 答错题目汇总与分类分析（本次运行）")
    out.append("")
    out.append("**报告**: reports/collab/20260305_133814  |  Accuracy 30%~45%")
    out.append("**Log**: 每 scenario 取最后 " + str(tasks_per) + " 题")
    out.append("")
    out.append("## 1. 按 scenario 失败数")
    out.append("")
    out.append("| Scenario | 失败数 |")
    out.append("|---|---:|")
    for scen in SCENARIOS:
        out.append(f"| {scen} | {by_scenario.get(scen, 0)} |")
    out.append("")
    out.append("## 2. 按失败类型")
    out.append("")
    out.append("| 类型 | 数量 | 说明 |")
    out.append("|---|---|---|")
    out.append("| wrong_finish | " + str(by_type.get("wrong_finish", 0)) + " | 有 Finish 但答错 |")
    out.append("| no_finish | " + str(by_type.get("no_finish", 0)) + " | 步数用尽未交卷 |")
    out.append("| invalid_action | " + str(by_type.get("invalid_action", 0)) + " | Invalid Action |")
    out.append("| other | " + str(by_type.get("other", 0)) + " | 其他 |")
    out.append("")
    out.append("## 3. 逐题摘要")
    out.append("")

    for t in all_failed:
        q_short = (t.question or "")[:85] + ("..." if len(t.question or "") > 85 else "")
        out.append(f"### [{t.scenario}] Task {t.task_id} | {t.fail_type}")
        out.append("")
        out.append(f"- **Q**: {q_short}")
        kas = t.key_actions()
        if not kas:
            out.append("- **KeyActions**: (无)")
        else:
            for step in kas[-4:]:
                act = step.get("action", "")
                obs = (step.get("observation") or "").replace("\n", " ")[:100]
                if len((step.get("observation") or "")) > 100:
                    obs += "..."
                out.append(f"  - {act}")
                if obs:
                    out.append(f"    -> {obs}")
        out.append("")

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "error_analysis.md"
    report_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Total failed: {len(all_failed)}")
    print(f"By type: {dict(by_type)}")
    print(f"By scenario: {dict(by_scenario)}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
