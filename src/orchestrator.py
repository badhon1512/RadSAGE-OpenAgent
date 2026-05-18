import argparse
import json
import time
from pathlib import Path

import workflow


VALID_ACTIONS = {
    "run_findings_judge",
    "run_anatomy_judge",
    "revise_report",
    "select_best_candidate",
    "finalize_report",
}


def build_orchestrator_system_prompt(mute_findings: bool = False, mute_anatomy: bool = False):
    """Static system message — sent once, stays fixed across all turns."""
    findings_block = "" if mute_findings else (
        "run_findings_judge\n"
        "  Checks whether the report faithfully captures all source findings (no missing, no hallucinated).\n"
        "  Produces findings feedback for the current report version.\n\n"
    )
    anatomy_block = "" if mute_anatomy else (
        "run_anatomy_judge\n"
        "  Checks section placement and duplicate findings.\n"
        "  Produces anatomy feedback for the current report version.\n\n"
    )
    if mute_findings and mute_anatomy:
        heuristic = (
            "No judges are active in this ablation run. "
            "Structure the report carefully and finalize promptly."
        )
    elif mute_findings:
        heuristic = (
            "Only the anatomy judge is active (findings judge is disabled for this ablation run).\n"
            "When the anatomy judge returns completely empty feedback, finalize."
        )
    elif mute_anatomy:
        heuristic = (
            "Only the findings judge is active (anatomy judge is disabled for this ablation run).\n"
            "When the findings judge returns completely empty feedback, finalize."
        )
    else:
        heuristic = (
            "The two judges are independent — findings judge checks clinical completeness, "
            "anatomy judge checks structural correctness.\n"
            "When BOTH judges return completely empty feedback on the same report version, this is strong evidence that the report is\n"
            "clinically faithful and structurally correct. That is the right moment to finalize.\n"
            "If either judge has not yet run on the current version, or returned non-empty feedback, more work is needed."
        )
    active_action_names = [
        *( ["run_findings_judge"] if not mute_findings else [] ),
        *( ["run_anatomy_judge"]  if not mute_anatomy  else [] ),
        "revise_report", "select_best_candidate", "finalize_report",
    ]
    return f"""You are an autonomous radiology report quality controller.
You have verification tools available. Use your judgement to decide which action produces the best final report.
At each turn you will receive the current state. Choose exactly one action and return only JSON.

=== STRUCTURING GUIDELINES ===
{workflow.base_agent.main_prompt}

=== AVAILABLE ACTIONS ===
{findings_block}{anatomy_block}revise_report
  Uses current judge feedback to produce an improved report version.
  Requires non-empty feedback. Consumes one revision round.

select_best_candidate
  Picks the best report from all candidates generated so far.
  Requires at least 2 candidates.

finalize_report
  Accepts the current report as the final output.

=== CONFIDENCE HEURISTIC ===
{heuristic}

=== OUTPUT FORMAT ===
{{"action": "{' | '.join(active_action_names)}", "reason": "one sentence", "certainty": "judge-verified | self-verified | uncertain"}}"""


def build_orchestrator_user_message(
    free_text,
    current_report,
    findings_feedback,
    anatomy_feedback,
    candidates,
    state,
    max_tool_calls,
    max_revision_rounds,
    mute_findings: bool = False,
    mute_anatomy: bool = False,
):
    """Dynamic user message — sent each turn with the latest state."""
    missing       = findings_feedback.get("missing_findings") or []
    unsupported   = findings_feedback.get("unsupported_findings") or []
    wrong_section = anatomy_feedback.get("wrong_section_findings") or []
    duplicates    = anatomy_feedback.get("duplicate_findings") or []

    findings_ran = state.get("findings_judge_ran_since_last_revision", False)
    anatomy_ran  = state.get("anatomy_judge_ran_since_last_revision", False)

    tool_calls_remaining      = max_tool_calls      - state["tool_call"]
    revision_rounds_remaining = max_revision_rounds - state["revision_rounds_used"]
    total_revisions           = state["revision_rounds_used"]
    current_version           = "initial" if total_revisions == 0 else f"revision {total_revisions}"
    candidate_stages          = [c["stage"] for c in candidates]

    if findings_ran:
        findings_summary = (
            f"  missing_findings ({len(missing)}): {missing}\n"
            f"  unsupported_findings ({len(unsupported)}): {unsupported}"
        )
    else:
        findings_summary = "  NOT YET RUN on the current report version."

    if anatomy_ran:
        anatomy_summary = (
            f"  wrong_section_findings ({len(wrong_section)}): {[f['finding'][:80] for f in wrong_section]}\n"
            f"  duplicate_findings ({len(duplicates)}): {[f['finding'][:80] for f in duplicates]}"
        )
    else:
        anatomy_summary = "  NOT YET RUN on the current report version."

    # Active-judge clean check: muted judges are excluded from the count
    findings_clean = findings_ran and not missing and not unsupported
    anatomy_clean  = anatomy_ran  and not wrong_section and not duplicates
    active_judges_clean = (
        (mute_findings or findings_clean) and (mute_anatomy or anatomy_clean)
    )

    # Build judge-status lines only for active judges
    judge_status_lines = ""
    if not mute_findings:
        judge_status_lines += f"Findings judge ran on THIS version: {findings_ran}\n"
    if not mute_anatomy:
        judge_status_lines += f"Anatomy judge ran on THIS version:  {anatomy_ran}\n"
    judge_status_lines += f"Active judges clean on THIS version: {active_judges_clean}"

    # Build session-history call counts only for active judges
    findings_calls_line = (
        f"Findings judge called:    {state.get('findings_judge_calls', 0)} time(s) total\n"
        if not mute_findings else ""
    )
    anatomy_calls_line = (
        f"Anatomy judge called:     {state.get('anatomy_judge_calls', 0)} time(s) total\n"
        if not mute_anatomy else ""
    )

    # Build feedback sections only for active judges
    feedback_sections = ""
    if not mute_findings:
        feedback_sections += f"Findings feedback:\n{findings_summary}\n\n"
    if not mute_anatomy:
        feedback_sections += f"Anatomy feedback:\n{anatomy_summary}\n\n"

    return f"""=== SESSION HISTORY ===
Current report version:   {current_version}
{findings_calls_line}{anatomy_calls_line}Revisions completed:      {total_revisions}
Candidates so far:        {candidate_stages}

=== CURRENT VERSION STATUS ===
Tool calls: {state["tool_call"]} / {max_tool_calls}  ({tool_calls_remaining} remaining)
Revisions:  {total_revisions} / {max_revision_rounds}  ({revision_rounds_remaining} remaining)
{judge_status_lines}

{feedback_sections}=== CURRENT REPORT ===
Source free-text:
{free_text}

Current structured report:
{current_report}

What is your next action?"""


# Keep backward-compatible alias used by langgraph_agent_workflow.py
def build_orchestrator_prompt(
    free_text, current_report, findings_feedback, anatomy_feedback,
    candidates, state, max_tool_calls, max_revision_rounds,
):
    return build_orchestrator_user_message(
        free_text, current_report, findings_feedback, anatomy_feedback,
        candidates, state, max_tool_calls, max_revision_rounds,
    )


def parse_orchestrator_action(response):
    parsed = workflow.base_agent.extract_json(response)
    if not isinstance(parsed, dict):
        return "finalize_report", "Invalid action returned; finalizing conservatively.", "uncertain"
    action    = parsed.get("action")
    reason    = parsed.get("reason", "")
    certainty = parsed.get("certainty", "uncertain")
    if action not in VALID_ACTIONS:
        return "finalize_report", "Invalid action returned; finalizing conservatively.", "uncertain"
    return action, reason, certainty


def feedback_has_issues(findings_feedback, anatomy_feedback):
    return workflow.has_actionable_feedback(findings_feedback, anatomy_feedback)


# ── Human-friendly logging helpers ────────────────────────────────────────────

_W = 60  # line width for separators

def _sep(title=""):
    if title:
        pad = max(0, _W - len(title) - 5)
        print(f"\n{'━' * 4} {title} {'━' * pad}")
    else:
        print("━" * _W)

def _print_report(report: str):
    print(report)
    print("─" * _W)

def _print_decision(tool_call, max_tool_calls, requested, final, certainty, reason, used_fallback):
    icon = "⚠" if used_fallback else "▶"
    override = f"  [overridden → {final}]" if used_fallback else ""
    print(f"\n{icon} [Call {tool_call}/{max_tool_calls}] {requested}{override}")
    print(f"  Certainty : {certainty}")
    print(f"  Reason    : {reason}")

def _print_findings_feedback(fb):
    missing     = fb.get("missing_findings") or []
    unsupported = fb.get("unsupported_findings") or []
    print("  Findings Judge →")
    if missing:
        print(f"    ✗ Missing ({len(missing)}):")
        for f in missing:
            print(f"      • {f}")
    else:
        print("    ✓ No missing findings")
    if unsupported:
        print(f"    ✗ Unsupported ({len(unsupported)}):")
        for f in unsupported:
            print(f"      • {f}")
    else:
        print("    ✓ No unsupported findings")

def _print_anatomy_feedback(fb):
    wrong  = fb.get("wrong_section_findings") or []
    dupes  = fb.get("duplicate_findings") or []
    print("  Anatomy Judge →")
    if wrong:
        print(f"    ✗ Wrong section ({len(wrong)}):")
        for f in wrong:
            print(f"      • {f.get('finding', f)[:100]}")
    else:
        print("    ✓ All findings in correct sections")
    if dupes:
        print(f"    ✗ Duplicates ({len(dupes)}):")
        for f in dupes:
            print(f"      • {f.get('finding', f)[:100]}")
    else:
        print("    ✓ No duplicate findings")

def _print_raw(label, text, max_chars=600):
    preview = text[:max_chars] + ("…" if len(text) > max_chars else "")
    print(f"  {label} raw response:")
    for line in preview.splitlines():
        print(f"    {line}")

def _print_summary(metadata, tool_calls_used, max_tool_calls, revision_rounds_used, max_revision_rounds):
    _sep("Summary")
    seq = " → ".join(metadata["action_sequence"].split("|"))
    print(f"  Stop reason    : {metadata['stop_reason']}")
    print(f"  Tool calls     : {tool_calls_used} / {max_tool_calls}")
    print(f"  Revisions      : {revision_rounds_used} / {max_revision_rounds}")
    print(f"  Findings judge : {metadata['findings_judge_calls']}x  |  Anatomy judge : {metadata['anatomy_judge_calls']}x")
    print(f"  Candidates     : {metadata['candidate_count']}  |  Fallbacks : {metadata['fallback_count']}")
    print(f"  Action sequence:")
    print(f"    {seq}")
    _sep()


def run_orchestrator_agent_pipeline(
    free_text, max_tool_calls, max_revision_rounds, select_final,
    mute_findings_judge: bool = False, mute_anatomy_judge: bool = False,
):
    prompt_log = []  # full record of every prompt + response for every agent

    ablation_label = (
        "no_findings_no_anatomy" if (mute_findings_judge and mute_anatomy_judge)
        else "no_findings" if mute_findings_judge
        else "no_anatomy"  if mute_anatomy_judge
        else "all_judges"
    )
    if ablation_label != "all_judges":
        print(f"[ablation] Muted judges: {ablation_label}")

    # ── Step 0: Initial structuring ───────────────────────────────────────────
    structuring_prompt = workflow.base_agent.build_structuring_prompt(free_text)
    current_report = workflow.base_agent.call_llm(structuring_prompt).strip()
    prompt_log.append({
        "step": 0, "agent": "structuring", "tool_call": 0,
        "prompt": structuring_prompt, "response": current_report,
    })

    candidates = [{"stage": "initial", "report": current_report}]
    findings_feedback = workflow.EMPTY_FINDINGS_FEEDBACK.copy()
    anatomy_feedback  = workflow.EMPTY_ANATOMY_FEEDBACK.copy()
    tool_calls_used   = 0
    revision_rounds_used = 0
    findings_judge_calls = 0
    anatomy_judge_calls  = 0
    selection_calls      = 0
    fallback_count       = 0
    stop_reason          = "max_tool_calls"
    trace_events         = []
    findings_ran_since_revision = False
    anatomy_ran_since_revision  = False

    # history lists for richer stats CSV
    findings_feedback_history = []
    anatomy_feedback_history  = []
    certainty_sequence        = []
    revision_reports          = []

    _sep("Initial Structured Report")
    _print_report(current_report)

    # Initialize orchestrator conversation with a fixed system message.
    messages = [{"role": "system", "content": build_orchestrator_system_prompt(
        mute_findings=mute_findings_judge, mute_anatomy=mute_anatomy_judge,
    )}]

    for tool_call in range(1, max_tool_calls + 1):
        tool_calls_used = tool_call
        state = {
            "tool_call": tool_call,
            "revision_rounds_used": revision_rounds_used,
            "findings_judge_calls": findings_judge_calls,
            "anatomy_judge_calls": anatomy_judge_calls,
            "selection_calls": selection_calls,
            "candidate_count": len(candidates),
            "has_actionable_feedback": feedback_has_issues(findings_feedback, anatomy_feedback),
            "findings_judge_ran_since_last_revision": findings_ran_since_revision,
            "anatomy_judge_ran_since_last_revision": anatomy_ran_since_revision,
        }
        user_msg = build_orchestrator_user_message(
            free_text=free_text,
            current_report=current_report,
            findings_feedback=findings_feedback,
            anatomy_feedback=anatomy_feedback,
            candidates=candidates,
            state=state,
            max_tool_calls=max_tool_calls,
            max_revision_rounds=max_revision_rounds,
            mute_findings=mute_findings_judge,
            mute_anatomy=mute_anatomy_judge,
        )
        messages.append({"role": "user", "content": user_msg})
        raw_response = workflow.base_agent.call_llm_chat(messages)
        messages.append({"role": "assistant", "content": raw_response})

        requested_action, requested_reason, certainty = parse_orchestrator_action(raw_response)
        action, reason = requested_action, requested_reason

        # Ablation guards — redirect muted-judge actions as a safety net
        if action == "run_findings_judge" and mute_findings_judge:
            action = "finalize_report"
            reason = "Findings judge is muted in this ablation run."
        elif action == "run_anatomy_judge" and mute_anatomy_judge:
            action = "finalize_report"
            reason = "Anatomy judge is muted in this ablation run."

        # Minimal budget guards only — no routing overrides.
        if action == "revise_report" and revision_rounds_used >= max_revision_rounds:
            action = "select_best_candidate" if select_final and len(candidates) > 1 else "finalize_report"
            reason = "Revision budget exhausted."
        elif action == "select_best_candidate" and (not select_final or len(candidates) <= 1):
            action = "finalize_report"
            reason = "Selection requested without multiple candidates or selection disabled."

        used_fallback = action != requested_action
        if used_fallback:
            fallback_count += 1

        certainty_sequence.append(certainty)

        event = {
            "tool_call": tool_call,
            "requested_action": requested_action,
            "action": action,
            "used_fallback": used_fallback,
            "reason": reason,
            "certainty": certainty,
            "state_before": state,
            "raw_response": raw_response,
        }
        trace_events.append(event)

        # Log orchestrator turn (messages_sent = all messages that were sent, i.e. before appending assistant)
        prompt_log.append({
            "step": tool_call, "agent": "orchestrator", "tool_call": tool_call,
            "messages_sent": messages[:-1],  # excludes the assistant reply we just got
            "response": raw_response,
            "parsed_action": action,
            "requested_action": requested_action,
            "reason": reason,
            "certainty": certainty,
            "used_fallback": used_fallback,
            "state_before": state,
            # explicit snapshots so they're easy to find without parsing the user message
            "source_free_text": free_text,
            "current_report_at_decision": current_report,
            "findings_feedback_at_decision": findings_feedback,
            "anatomy_feedback_at_decision": anatomy_feedback,
        })

        _print_decision(tool_call, max_tool_calls, requested_action, action, certainty, reason, used_fallback)
        _print_raw("Orchestrator", raw_response)

        # ── Execute the chosen action ──────────────────────────────────────────
        if action == "run_findings_judge":
            judge_prompt = workflow.base_agent.build_findings_judge_prompt(free_text, current_report)
            feedback, findings_raw = workflow.call_json_judge(judge_prompt, workflow.EMPTY_FINDINGS_FEEDBACK)
            findings_feedback = workflow.normalize_findings_feedback(feedback)
            findings_judge_calls += 1
            findings_ran_since_revision = True
            findings_feedback_history.append({
                "tool_call": tool_call,
                "revision_round": revision_rounds_used,
                "feedback": findings_feedback,
                "raw_response": findings_raw,
            })
            prompt_log.append({
                "step": tool_call, "agent": "findings_judge", "tool_call": tool_call,
                "prompt": judge_prompt,
                "response": findings_raw,
                "parsed": findings_feedback,
            })
            _print_findings_feedback(findings_feedback)
            _print_raw("Findings judge", findings_raw)

        elif action == "run_anatomy_judge":
            judge_prompt = workflow.base_agent.build_anatomy_duplication_judge_prompt(current_report)
            feedback, anatomy_raw = workflow.call_json_judge(judge_prompt, workflow.EMPTY_ANATOMY_FEEDBACK)
            anatomy_feedback = workflow.normalize_anatomy_feedback(feedback)
            anatomy_judge_calls += 1
            anatomy_ran_since_revision = True
            anatomy_feedback_history.append({
                "tool_call": tool_call,
                "revision_round": revision_rounds_used,
                "feedback": anatomy_feedback,
                "raw_response": anatomy_raw,
            })
            prompt_log.append({
                "step": tool_call, "agent": "anatomy_judge", "tool_call": tool_call,
                "prompt": judge_prompt,
                "response": anatomy_raw,
                "parsed": anatomy_feedback,
            })
            _print_anatomy_feedback(anatomy_feedback)
            _print_raw("Anatomy judge", anatomy_raw)

        elif action == "revise_report":
            revision_prompt = workflow.base_agent.build_revision_prompt(
                free_text, current_report, findings_feedback, anatomy_feedback,
            )
            revised_report = workflow.base_agent.call_llm(revision_prompt).strip()
            prompt_log.append({
                "step": tool_call, "agent": "revision", "tool_call": tool_call,
                "prompt": revision_prompt,
                "response": revised_report,
            })
            if not revised_report or revised_report == current_report:
                print("  ⚠ Revision produced no change — stopping.")
                stop_reason = "revision_no_change"
                break
            revision_rounds_used += 1
            current_report = revised_report
            candidates.append({"stage": f"orchestrator_revision_{revision_rounds_used}", "report": current_report})
            revision_reports.append({"round": revision_rounds_used, "report": current_report})
            findings_feedback = workflow.EMPTY_FINDINGS_FEEDBACK.copy()
            anatomy_feedback  = workflow.EMPTY_ANATOMY_FEEDBACK.copy()
            findings_ran_since_revision = False
            anatomy_ran_since_revision  = False
            _sep(f"Revised Report (round {revision_rounds_used})")
            _print_report(current_report)

        elif action == "select_best_candidate":
            selection_prompt = workflow.build_revision_selection_prompt(free_text, candidates)
            selected = workflow.base_agent.call_llm(selection_prompt).strip()
            prompt_log.append({
                "step": tool_call, "agent": "selection", "tool_call": tool_call,
                "prompt": selection_prompt,
                "response": selected,
                "candidates_count": len(candidates),
            })
            if selected:
                current_report = selected
                selection_calls += 1
                stop_reason = "selected_final"
                _sep("Selected Best Candidate")
                _print_report(current_report)
            break

        elif action == "finalize_report":
            stop_reason = "finalize_report"
            break

    if stop_reason == "max_tool_calls" and select_final and len(candidates) > 1:
        print("\n  ℹ Max tool calls reached — selecting best candidate.")
        selection_prompt = workflow.build_revision_selection_prompt(free_text, candidates)
        selected = workflow.base_agent.call_llm(selection_prompt).strip()
        prompt_log.append({
            "step": tool_calls_used, "agent": "selection_budget_exhausted", "tool_call": tool_calls_used,
            "prompt": selection_prompt,
            "response": selected,
            "candidates_count": len(candidates),
        })
        if selected:
            current_report = selected
            selection_calls += 1
            stop_reason = "max_tool_calls_selected_final"
            _sep("Selected Best Candidate (budget exhausted)")
            _print_report(current_report)

    metadata = {
        # ── core counters ──────────────────────────────────────────────────────
        "tool_calls_used":       tool_calls_used,
        "revision_rounds_used":  revision_rounds_used,
        "findings_judge_calls":  findings_judge_calls,
        "anatomy_judge_calls":   anatomy_judge_calls,
        "selection_calls":       selection_calls,
        "candidate_count":       len(candidates),
        "fallback_count":        fallback_count,
        "stop_reason":           stop_reason,
        "ablation_muted_judges": ablation_label,
        # ── action sequences ───────────────────────────────────────────────────
        "action_sequence":           "|".join(e["action"]            for e in trace_events),
        "requested_action_sequence": "|".join(e["requested_action"]  for e in trace_events),
        "certainty_sequence":        "|".join(certainty_sequence),
        # ── report snapshots ───────────────────────────────────────────────────
        "initial_report":    candidates[0]["report"],
        "final_report":      current_report,
        "candidates_json":   json.dumps(candidates, ensure_ascii=False),
        "revision_reports_json": json.dumps(revision_reports, ensure_ascii=False),
        # ── feedback histories ─────────────────────────────────────────────────
        "findings_feedback_history_json": json.dumps(findings_feedback_history, ensure_ascii=False),
        "anatomy_feedback_history_json":  json.dumps(anatomy_feedback_history,  ensure_ascii=False),
        # ── full trace ─────────────────────────────────────────────────────────
        "trace_json": json.dumps(trace_events, ensure_ascii=False),
    }
    for action_name in sorted(VALID_ACTIONS):
        metadata[f"action_count_{action_name}"] = sum(1 for e in trace_events if e["action"] == action_name)

    _sep("Final Report")
    _print_report(current_report)
    _print_summary(metadata, tool_calls_used, max_tool_calls, revision_rounds_used, max_revision_rounds)
    return current_report, metadata, prompt_log


def build_study_json(study_id, status, prompt_log):
    """Convert raw prompt_log into the clean per-study JSON structure."""
    steps = []
    for entry in prompt_log:
        agent = entry["agent"]
        step  = entry["step"]

        if agent == "orchestrator":
            steps.append({
                "step": step,
                "agent": agent,
                "action_requested": entry.get("requested_action"),
                "action_executed":  entry.get("parsed_action"),
                "certainty":        entry.get("certainty"),
                "reason":           entry.get("reason"),
                "used_fallback":    entry.get("used_fallback"),
                "source_free_text":              entry.get("source_free_text", ""),
                "current_report_at_decision":    entry.get("current_report_at_decision", ""),
                "findings_feedback_at_decision": entry.get("findings_feedback_at_decision", {}),
                "anatomy_feedback_at_decision":  entry.get("anatomy_feedback_at_decision", {}),
                "state_before":  entry.get("state_before", {}),
                "conversation":  [{"role": m["role"], "content": m["content"]}
                                  for m in entry.get("messages_sent", [])],
                "response": entry.get("response"),
            })
        elif agent in ("findings_judge", "anatomy_judge"):
            steps.append({
                "step": step,
                "agent": agent,
                "prompt":          entry.get("prompt"),
                "response":        entry.get("response"),
                "parsed_feedback": entry.get("parsed"),
            })
        elif agent == "revision":
            steps.append({
                "step":     step,
                "agent":    agent,
                "prompt":   entry.get("prompt"),
                "response": entry.get("response"),
            })
        elif agent in ("structuring",):
            steps.append({
                "step":     step,
                "agent":    agent,
                "prompt":   entry.get("prompt"),
                "response": entry.get("response"),
            })
        else:  # selection, selection_budget_exhausted
            steps.append({
                "step":             step,
                "agent":            agent,
                "candidates_count": entry.get("candidates_count"),
                "prompt":           entry.get("prompt"),
                "response":         entry.get("response"),
            })

    return {
        "study_id":    study_id,
        "status":      status,
        "total_steps": len(steps),
        "steps":       steps,
    }


def run_with_retries(free_text, args):
    last_error = ""
    for attempt in range(1, args.max_retries + 1):
        try:
            output, metadata, prompt_log = run_orchestrator_agent_pipeline(
                free_text=free_text,
                max_tool_calls=args.max_tool_calls,
                max_revision_rounds=args.max_revision_rounds,
                select_final=args.select_final,
                mute_findings_judge=getattr(args, "mute_findings_judge", False),
                mute_anatomy_judge=getattr(args, "mute_anatomy_judge", False),
            )
            # strip any preamble the model echoed before FINDINGS before storing
            return workflow.base_agent.extract_findings_section(output), metadata, prompt_log, "", attempt
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[warn] Attempt {attempt}/{args.max_retries} failed: {last_error}")
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            if attempt < args.max_retries and args.retry_sleep > 0:
                time.sleep(args.retry_sleep)
    return "", {}, [], last_error, args.max_retries


def process_csv(args):
    import pandas as pd

    provider, resolved_model_name = workflow.initialize_backend(args)
    input_csv = Path(args.input_csv)

    # Build ablation suffix for default output naming
    ablation_parts = []
    if getattr(args, "mute_findings_judge", False):
        ablation_parts.append("no_findings")
    if getattr(args, "mute_anatomy_judge", False):
        ablation_parts.append("no_anatomy")
    ablation_suffix = ("_ablation_" + "_".join(ablation_parts)) if ablation_parts else ""

    output_csv = Path(args.output_csv) if args.output_csv else Path(
        f"{workflow.safe_slug(args.model_name)}-orchestrator_agent{ablation_suffix}.csv"
    )
    stats_csv = Path(args.stats_csv) if args.stats_csv else output_csv.with_name(
        output_csv.stem + "_stat" + output_csv.suffix
    )
    prompts_jsonl = output_csv.with_name(output_csv.stem + "_prompts.jsonl")
    prompts_json  = output_csv.with_name(output_csv.stem + "_study_prompts.json")
    gen_column = args.output_column or f"{args.model_name}-orchestrator_agent{ablation_suffix}"

    df = pd.read_csv(input_csv)
    required_columns = {args.id_column, args.text_column}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    if args.start:
        df = df.iloc[args.start:]
    if args.limit:
        df = df.head(args.limit)

    if args.resume:
        existing_df, completed_ids = workflow.load_existing_output(output_csv, args.id_column)
        records       = existing_df.to_dict("records") if not existing_df.empty else []
        existing_stats_df, _ = workflow.load_existing_output(stats_csv, args.id_column)
        stats_records = existing_stats_df.to_dict("records") if not existing_stats_df.empty else []
        jsonl_mode    = "a"
        prompts_dict  = json.loads(prompts_json.read_text(encoding="utf-8")) if prompts_json.exists() else {}
    else:
        # Fresh start — clear all existing output files
        records, stats_records, completed_ids = [], [], set()
        prompts_dict = {}
        jsonl_mode = "w"
        for f in (output_csv, stats_csv, prompts_jsonl, prompts_json):
            if f.exists():
                f.unlink()
                print(f"[info] Cleared existing file: {f}")

    num_workers = args.num_workers if getattr(args, "use_vllm", False) else 1
    if num_workers > 1 and not getattr(args, "use_vllm", False):
        print("[warn] --num_workers > 1 only works with --use_vllm; falling back to 1 worker.")
        num_workers = 1

    total = len(df)
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Stats CSV: {stats_csv}")
    print(f"Prompts JSONL: {prompts_jsonl}")
    print(f"Study prompts JSON: {prompts_json}")
    print(f"Backend: {provider} ({resolved_model_name})")
    print(f"Workers: {num_workers} {'(parallel — vLLM)' if num_workers > 1 else '(sequential)'}")
    print("Workflow: orchestrator_agent")
    print(f"Max tool calls: {args.max_tool_calls}")
    print(f"Max revision rounds: {args.max_revision_rounds}")
    print(f"Select final: {args.select_final}")
    print(f"Rows selected: {total}")
    print(f"Resume: {'on — appending to existing files' if args.resume else 'off — starting fresh'}")
    print(f"Ablation — muted judges: {ablation_suffix[len('_ablation_'):] if ablation_suffix else 'none (all active)'}")

    # Rows to process (skip already completed when resuming)
    pending = [
        (i + 1, row)
        for i, (_, row) in enumerate(df.iterrows())
        if not (args.resume and str(row[args.id_column]) in completed_ids)
    ]
    for pos, row in [(p, r) for p, r in enumerate(df.iterrows(), 1) if args.resume and str(r[1][args.id_column]) in completed_ids]:
        print(f"[{pos}/{total}] Skipping completed: {row[1][args.id_column]}")

    import threading
    save_lock = threading.Lock()
    processed_since_save = 0

    def process_one(position, row):
        nonlocal processed_since_save, jsonl_mode
        study_id = str(row[args.id_column])
        free_text = "" if pd.isna(row[args.text_column]) else str(row[args.text_column])
        print(f"[{position}/{total}] Processing: {study_id}")
        print(f"  Free text preview: {free_text}")
        start_time = time.time()
        output, metadata, prompt_log, error, attempts = run_with_retries(free_text, args)
        elapsed = round(time.time() - start_time, 3)
        status = "ok" if not error else "error"

        record = {
            args.id_column: study_id,
            "ref": free_text,
            gen_column: output,
            "status": status,
            "error": error,
            "attempts": attempts,
            "elapsed_sec": elapsed,
        }
        stats_record = {args.id_column: study_id, "status": status, "error": error,
                        "attempts": attempts, "elapsed_sec": elapsed}
        stats_record.update(metadata)

        study_json = build_study_json(study_id, status, prompt_log)

        with save_lock:
            records.append(record)
            stats_records.append(stats_record)
            prompts_dict[study_id] = study_json
            tmp = prompts_json.with_suffix(".tmp")
            tmp.write_text(json.dumps(prompts_dict, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(prompts_json)
            with prompts_jsonl.open(jsonl_mode, encoding="utf-8") as pf:
                jsonl_mode = "a"
                pf.write(json.dumps({args.id_column: study_id, "status": status,
                                     "prompt_log": prompt_log}, ensure_ascii=False) + "\n")
            print(f"[row {position}/{total}] {study_id} — status={status} "
                  f"calls={metadata.get('tool_calls_used',0)} "
                  f"revisions={metadata.get('revision_rounds_used',0)} "
                  f"stop={metadata.get('stop_reason','')} elapsed={elapsed}s")
            nonlocal processed_since_save
            processed_since_save += 1
            if processed_since_save >= args.save_every:
                workflow.atomic_write_csv(pd.DataFrame(records), output_csv)
                workflow.atomic_write_csv(pd.DataFrame(stats_records), stats_csv)
                processed_since_save = 0
                print(f"[info] Saved progress → {output_csv}")

    pending_rows = [
        (i + 1, row)
        for i, (_, row) in enumerate(df.iterrows())
        if not (args.resume and str(row[args.id_column]) in completed_ids)
    ]

    if num_workers == 1:
        for position, row in pending_rows:
            process_one(position, row)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_one, pos, row): pos for pos, row in pending_rows}
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    print(f"[error] Worker failed: {exc}")

    workflow.atomic_write_csv(pd.DataFrame(records), output_csv)
    workflow.atomic_write_csv(pd.DataFrame(stats_records), stats_csv)
    print(f"[done] Wrote {len(records)} rows to {output_csv}")
    print(f"[done] Wrote {len(stats_records)} stats rows to {stats_csv}")
    print(f"[done] Prompt log → {prompts_jsonl}")
    print(f"[done] Study prompts JSON → {prompts_json}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrator-agent workflow for structured radiology report refinement."
    )
    parser.add_argument("--input_csv", default="/home/hpc/iwi5/iwi5284h/RRG/srr_eval_all.csv")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--stats_csv", default=None)
    parser.add_argument("--id_column", default="StudyInstanceUid")
    parser.add_argument("--text_column", default="findings")
    parser.add_argument("--output_column", default=None)
    parser.add_argument("--model_name", default="Qwen3-14B")
    parser.add_argument("--provider", choices=["qwen", "gemma", "medgemma", "gpt"], default=None)
    parser.add_argument("--max_tool_calls", type=int, default=10)
    parser.add_argument("--max_orchestrator_steps", dest="max_tool_calls", type=int)
    parser.add_argument("--max_revision_rounds", type=int, default=3)
    parser.add_argument("--select_final", dest="select_final", action="store_true", default=True)
    parser.add_argument("--no-select_final", dest="select_final", action="store_false")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--use_vllm", dest="use_vllm", action="store_true", default=False,
                        help="Route all LLM calls to a running vLLM server instead of loading locally.")
    parser.add_argument("--openai_base_url", default="http://127.0.0.1:8050/v1")
    parser.add_argument("--openai_api_key", default="EMPTY")
    parser.add_argument("--openai_model_name", default=None)
    parser.add_argument("--openai_timeout", type=float, default=600)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Parallel study workers (only effective with --use_vllm; ignored otherwise).")
    # Ablation study: mute individual judges
    parser.add_argument("--mute_findings_judge", action="store_true", default=False,
                        help="Disable the findings judge (ablation: no clinical-faithfulness feedback).")
    parser.add_argument("--mute_anatomy_judge", action="store_true", default=False,
                        help="Disable the anatomy judge (ablation: no section-placement/duplicate feedback).")
    return parser.parse_args()


if __name__ == "__main__":
    process_csv(parse_args())
