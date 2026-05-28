import argparse
import json
import os
import re
import time
from pathlib import Path

base_agent = None


EMPTY_FINDINGS_FEEDBACK = {
    "missing_findings": [],
    "unsupported_findings": [],
}

EMPTY_ANATOMY_FEEDBACK = {
    "wrong_section_findings": [],
    "duplicate_findings": [],
}


def safe_slug(value: str) -> str:
    value = value.strip().replace("/", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def workflow_label(args) -> str:
    if not args.agent_mode:
        return "one_pass_model"
    if args.workflow == "original_agent":
        return "original_agent"
    return "enhanced_agent"


def normalize_model_name(model_name: str, provider: str) -> str:
    if "/" in model_name:
        return model_name
    if provider == "qwen":
        return f"Qwen/{model_name}"
    if provider in {"gemma", "medgemma"}:
        return f"google/{model_name}"
    return model_name


def ensure_base_agent():
    global base_agent
    if base_agent is None:
        import agent

        base_agent = agent
    return base_agent


def normalize_findings_feedback(feedback):
    if not isinstance(feedback, dict):
        return EMPTY_FINDINGS_FEEDBACK.copy()
    missing     = feedback.get("missing_findings") or []
    unsupported = feedback.get("unsupported_findings") or []
    # A finding simultaneously marked missing AND unsupported is a judge contradiction;
    # satisfying both constraints is impossible, so drop from both lists.
    missing_texts     = {_finding_text(f) for f in missing}
    unsupported_texts = {_finding_text(f) for f in unsupported}
    contradicted = missing_texts & unsupported_texts
    if contradicted:
        missing     = [f for f in missing     if _finding_text(f) not in contradicted]
        unsupported = [f for f in unsupported if _finding_text(f) not in contradicted]
    return {"missing_findings": missing, "unsupported_findings": unsupported}


def _finding_text(entry) -> str:
    if isinstance(entry, dict):
        return (entry.get("finding") or "").strip().rstrip(".")
    return str(entry).strip().rstrip(".")


def normalize_anatomy_feedback(feedback):
    if not isinstance(feedback, dict):
        return EMPTY_ANATOMY_FEEDBACK.copy()
    wrong = feedback.get("wrong_section_findings") or []
    # Drop entries where current_section == correct_section — the judge made no real finding.
    wrong = [
        f for f in wrong
        if not (isinstance(f, dict) and f.get("current_section") == f.get("correct_section"))
    ]
    return {
        "wrong_section_findings": wrong,
        "duplicate_findings": feedback.get("duplicate_findings") or [],
    }


def has_actionable_feedback(findings_feedback, anatomy_feedback):
    return bool(
        findings_feedback.get("missing_findings")
        or findings_feedback.get("unsupported_findings")
        or anatomy_feedback.get("wrong_section_findings")
        or anatomy_feedback.get("duplicate_findings")
    )


def call_json_judge(prompt: str, expected: dict):
    response = base_agent.call_llm(prompt)
    parsed = base_agent.extract_json(response)
    if parsed:
        return parsed, response
    return expected.copy(), response


def build_revision_selection_prompt(free_text: str, candidates) -> str:
    candidates_text = json.dumps(candidates, indent=2, ensure_ascii=False)
    return f"""
You are an expert radiology report quality controller.

Your task:
Select the best final structured report from the candidate reports.

Original structuring instructions:
{base_agent.main_prompt}

Use these original instructions only to judge which candidate best follows the required structured-report format. Do not create new findings from these instructions.

Selection criteria, in order:
1. The report must be clinically faithful to the source free-text report.
2. It must preserve all clinically meaningful source findings.
3. It must not include unsupported or hallucinated findings.
4. Findings should be placed under appropriate anatomical section headers.
5. The report should be concise and not duplicate findings.

Important rules:
- Use only the source report as the clinical truth.
- Do not add new findings while selecting.
- Prefer a later candidate only if it improves clinical faithfulness: it preserves all supported findings from the earlier candidate while reducing missing source findings, unsupported findings, wrong-section findings, or duplicate findings.

Output format:
- Return ONLY the text of the selected structured report, exactly as it appears in the candidates.
- Do not include any analysis, reasoning, commentary, or explanations.
- Do not modify the selected report in any way.
- Do not add any text outside of the clinical findings in structured format.

Source free-text report:
{free_text}

Candidate structured reports:
{candidates_text}
""".strip()


def infer_provider(model_name: str) -> str:
    name = model_name.lower()
    if "medgemma" in name:
        return "medgemma"
    if "gemma" in name:
        return "gemma"
    if "qwen" in name:
        return "qwen"
    if "gpt" in name or "openai" in name:
        return "gpt"
    raise ValueError(
        f"Could not infer provider from model name '{model_name}'. "
        "Use --provider qwen, gemma, medgemma, or gpt."
    )


def initialize_backend(args):
    import httpx
    import torch
    from openai import OpenAI
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

    ensure_base_agent()
    # vLLM is accessed through an OpenAI-compatible API, so do not infer provider
    # from model family names such as Llama. For local loading, keep inference.
    provider = args.provider or ("gpt" if getattr(args, "use_vllm", False) else infer_provider(args.model_name))
    model_name = normalize_model_name(args.model_name, provider)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_agent.model_name = model_name
    base_agent.device = device

    # ── vLLM mode: skip local model loading, route to vLLM server ─────────────
    if getattr(args, "use_vllm", False):
        print(f"[vllm] Backend: {model_name} @ {args.openai_base_url}")
        base_agent.model_name    = args.openai_model_name or model_name
        base_agent.call_llm      = base_agent.call_gpt
        base_agent.call_llm_chat = base_agent.call_gpt_chat
        os.environ.pop("http_proxy",  None)
        os.environ.pop("https_proxy", None)
        os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
        base_agent.client = OpenAI(
            base_url=args.openai_base_url,
            api_key=args.openai_api_key,
            http_client=httpx.Client(trust_env=False, timeout=args.openai_timeout),
        )
        return "vllm", model_name

    if provider == "medgemma":
        print(f"Using MedGemma backend: {model_name}")
        base_agent.call_llm      = base_agent.call_medgemma
        base_agent.call_llm_chat = base_agent.call_medgemma_chat
        base_agent.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            quantization_config=base_agent.bnb_config,
            torch_dtype="auto",
            device_map=device,
            attn_implementation=base_agent.get_attn_impl(),
            token=hf_token,
        )
        base_agent.processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
    elif provider == "gemma":
        print(f"Using Gemma backend: {model_name}")
        base_agent.call_llm      = base_agent.call_gemma
        base_agent.call_llm_chat = base_agent.call_gemma_chat
        base_agent.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=base_agent.bnb_config,
            torch_dtype="auto",
            device_map=device,
            attn_implementation=base_agent.get_attn_impl(),
            token=hf_token,
        )
        base_agent.processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
    elif provider == "qwen":
        print(f"Using Qwen backend: {model_name}")
        base_agent.call_llm      = base_agent.call_qwen3
        base_agent.call_llm_chat = base_agent.call_qwen3_chat
        base_agent.processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
        base_agent.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=base_agent.bnb_config,
            torch_dtype="auto",
            attn_implementation=base_agent.get_attn_impl(),
            device_map="auto",
            token=hf_token,
        )
    elif provider == "gpt":
        print(f"Using OpenAI-compatible backend: {model_name}")
        base_agent.call_llm      = base_agent.call_gpt
        base_agent.call_llm_chat = base_agent.call_gpt_chat
        base_agent.model_name = args.openai_model_name or model_name
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"
        base_agent.client = OpenAI(
            base_url=args.openai_base_url,
            api_key=args.openai_api_key,
            http_client=httpx.Client(trust_env=False, timeout=args.openai_timeout),
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return provider, model_name


def atomic_write_csv(df, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_csv.with_suffix(output_csv.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(output_csv)


def run_with_retries(free_text: str, args, max_retries: int, retry_sleep: float):
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            return run_selected_pipeline(free_text, args).strip(), "", attempt
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[warn] Attempt {attempt}/{max_retries} failed: {last_error}")
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            if attempt < max_retries and retry_sleep > 0:
                time.sleep(retry_sleep)
    return "", last_error, max_retries


def run_enhanced_agent_pipeline(free_text: str, revision_rounds: int, select_final: bool):
    candidates = []

    initial_report = base_agent.call_llm(base_agent.build_structuring_prompt(free_text)).strip()
    current_report = initial_report
    candidates.append({"stage": "initial", "report": current_report})
    print("Initial structured report response:", current_report)

    for round_idx in range(1, revision_rounds + 1):
        print(f"[agent] Critique round {round_idx}/{revision_rounds}")

        findings_feedback, findings_raw = call_json_judge(
            base_agent.build_findings_judge_prompt(free_text, current_report),
            EMPTY_FINDINGS_FEEDBACK,
        )
        findings_feedback = normalize_findings_feedback(findings_feedback)
        print("findings_feedback", findings_feedback)
        if findings_feedback == EMPTY_FINDINGS_FEEDBACK and findings_raw:
            print("[agent] Findings judge returned no parsed actionable feedback.")

        anatomy_feedback, anatomy_raw = call_json_judge(
            base_agent.build_anatomy_duplication_judge_prompt(current_report),
            EMPTY_ANATOMY_FEEDBACK,
        )
        anatomy_feedback = normalize_anatomy_feedback(anatomy_feedback)
        print("anatomy_feedback", anatomy_feedback)
        if anatomy_feedback == EMPTY_ANATOMY_FEEDBACK and anatomy_raw:
            print("[agent] Anatomy judge returned no parsed actionable feedback.")

        if not has_actionable_feedback(findings_feedback, anatomy_feedback):
            print("[agent] No actionable feedback found; stopping critique loop.")
            break

        # Apply findings feedback first, then anatomy feedback — each in its own call
        # so the model only handles one concern at a time.
        intermediate = current_report
        has_findings = bool(
            findings_feedback.get("missing_findings") or findings_feedback.get("unsupported_findings")
        )
        has_anatomy = bool(
            anatomy_feedback.get("wrong_section_findings") or anatomy_feedback.get("duplicate_findings")
        )
        if has_findings:
            intermediate = base_agent.call_llm(
                base_agent.build_findings_revision_prompt(free_text, current_report, findings_feedback)
            ).strip()
            print(f"[agent] After findings revision:\n{intermediate}")
        if has_anatomy:
            intermediate = base_agent.call_llm(
                base_agent.build_anatomy_revision_prompt(intermediate, anatomy_feedback)
            ).strip()
            print(f"[agent] After anatomy revision:\n{intermediate}")
        revised_report = intermediate
        print(f"Revised structured report response:\n{revised_report}")

        if revised_report == current_report:
            print("[agent] Revision did not change the report; stopping critique loop.")
            break

        current_report = revised_report
        candidates.append({"stage": f"revision_round_{round_idx}", "report": current_report})

    if select_final and len(candidates) > 1:
        print("[agent] Selecting best report from candidate revisions.")
        selected = base_agent.call_llm(build_revision_selection_prompt(free_text, candidates)).strip()
        if selected:
            current_report = selected
            print(f"Selected final structured report:\n{current_report}")

    return current_report


def run_selected_pipeline(free_text: str, args):
    if not args.agent_mode:
        return base_agent.run_pipeline(free_text, is_agent=False).strip()
    if args.workflow == "original_agent":
        return base_agent.run_pipeline(free_text, is_agent=True).strip()
    return run_enhanced_agent_pipeline(
        free_text=free_text,
        revision_rounds=args.revision_rounds,
        select_final=args.select_final,
    ).strip()


def load_existing_output(output_csv: Path, id_column: str):
    import pandas as pd

    if not output_csv.exists():
        return pd.DataFrame(), set()
    df = pd.read_csv(output_csv)
    completed = set()
    if id_column in df.columns and "status" in df.columns:
        completed = set(df.loc[df["status"] == "ok", id_column].astype(str))
    return df, completed


def process_csv(args):
    import pandas as pd

    provider, resolved_model_name = initialize_backend(args)
    input_csv = Path(args.input_csv)
    run_label = workflow_label(args)
    output_csv = Path(args.output_csv) if args.output_csv else Path(
        f"{safe_slug(args.model_name)}-{run_label}.csv"
    )
    gen_column = args.output_column or f"{args.model_name}-{run_label}"

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
        existing_df, completed_ids = load_existing_output(output_csv, args.id_column)
        if not existing_df.empty and "status" in existing_df.columns:
            ok_mask = existing_df["status"].fillna("").astype(str).str.lower().eq("ok")
            retry_ids = set(existing_df.loc[~ok_mask, args.id_column].astype(str))
            records = existing_df.loc[ok_mask].to_dict("records")
        else:
            retry_ids = set()
            records = existing_df.to_dict("records") if not existing_df.empty else []
        if retry_ids:
            print(f"[resume] Preserved {len(completed_ids)} ok rows; retrying {len(retry_ids)} error/incomplete rows.")
    else:
        records, completed_ids = [], set()
        if output_csv.exists():
            output_csv.unlink()
            print(f"[info] Cleared existing file: {output_csv}")

    total = len(df)
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Backend: {provider} ({resolved_model_name})")
    print(f"Workflow: {run_label}")
    print(f"Rows selected: {total}")
    print(f"Resume: {'on' if args.resume else 'off'}")

    processed_since_save = 0
    for position, (_, row) in enumerate(df.iterrows(), start=1):
        study_id = str(row[args.id_column])
        if args.resume and study_id in completed_ids:
            print(f"[{position}/{total}] Skipping completed row: {study_id}")
            continue

        free_text = "" if pd.isna(row[args.text_column]) else str(row[args.text_column])
        print(f"[{position}/{total}] Processing row: {study_id}")

        start_time = time.time()
        output, error, attempts = run_with_retries(
            free_text=free_text,
            args=args,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )
        elapsed = round(time.time() - start_time, 3)
        status = "ok" if not error else "error"

        records.append(
            {
                args.id_column: study_id,
                "ref": free_text,
                gen_column: output,
                "status": status,
                "error": error,
                "attempts": attempts,
                "elapsed_sec": elapsed,
            }
        )
        processed_since_save += 1

        if processed_since_save >= args.save_every:
            atomic_write_csv(pd.DataFrame(records), output_csv)
            processed_since_save = 0
            print(f"[info] Saved progress to {output_csv}")

    atomic_write_csv(pd.DataFrame(records), output_csv)
    print(f"[done] Wrote {len(records)} rows to {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Safer batch workflow for the radiology report structuring agent."
    )
    parser.add_argument("--input_csv", default="/home/hpc/iwi5/iwi5284h/RRG/srr_eval_all.csv")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--id_column", default="StudyInstanceUid")
    parser.add_argument("--text_column", default="findings")
    parser.add_argument("--output_column", default=None)
    parser.add_argument("--model_name", default="Qwen3-14B")
    parser.add_argument("--provider", choices=["qwen", "gemma", "medgemma", "gpt"], default=None)
    parser.add_argument("--agent_mode", dest="agent_mode", action="store_true", default=True)
    parser.add_argument("--no-agent_mode", dest="agent_mode", action="store_false")
    parser.add_argument("--workflow", choices=["enhanced_agent", "original_agent"], default="enhanced_agent")
    parser.add_argument("--revision_rounds", type=int, default=2)
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
    return parser.parse_args()


if __name__ == "__main__":
    process_csv(parse_args())
