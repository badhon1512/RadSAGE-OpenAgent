import json
# pip install accelerate
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch
import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import httpx, os, argparse
from openai import OpenAI
import re as _re

from prompts import *
from backends import bnb_config, get_attn_impl, _strip_thinking


hf_toekn = os.environ.get("HF_TOKEN", "")
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen3-14B"
print(f"Loading model {model_name} on {device} with 4-bit quantization...")

processor = None
model = None
call_llm = None
call_llm_chat = None
model_name = None
clint = None


def call_qwen3_chat(messages: list) -> str:
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = processor([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    return processor.decode(output_ids[index:], skip_special_tokens=True).strip("\n")


def call_qwen3(prompt: str) -> str:
    messages = [
        {
            "role": "system", "content": "You are a radiologiest to strcture and improve free-text report into strctured format "
        },
        {
            "role": "user", "content":  prompt
        }
    ]

    text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = processor([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=200,
        do_sample=False
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    #thinking_content = processor.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = processor.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    #print("thinking content:", thinking_content)

    return content


def call_medgemma_chat(messages: list) -> str:
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generation = generation[0][input_len:]
    return processor.decode(generation, skip_special_tokens=True).strip()


def call_medgemma(prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content":[
             {
                "type":"text", "text":"You are a radiologiest to strcture and improve free-text report into strctured format "
             }
            ]
        },
        {
            "role": "user",

            "content":[
             {
                "type":"text", "text":prompt
             }
            ]
        }
    ]

    inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True).strip()

    return decoded


def call_gemma_chat(messages: list) -> str:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    res = processor.parse_response(response)
    return res.get("content", "").strip()


def call_gemma(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    res = processor.parse_response(response)
    return res.get("content", "").strip()


_FINDINGS_RE = _re.compile(r'(\*{0,2}FINDINGS\b.*)', _re.IGNORECASE | _re.DOTALL)

def extract_findings_section(text: str) -> str:
    """Strip any preamble before FINDINGS — keeps output clean when small models echo the prompt template."""
    m = _FINDINGS_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def call_gpt_chat(messages: list) -> str:
    result = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return _strip_thinking(result.choices[0].message.content or "")


def call_gpt(prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a radiologist to structure and improve free-text report into structured format"},
        {"role": "user", "content": prompt},
    ]
    result = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return _strip_thinking(result.choices[0].message.content or "")


def extract_json(text: str):
    # Collect all top-level {...} blocks — handles reasoning text before the JSON.
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : i + 1])
                start = None
    # Return the last valid JSON object (reasoning text may precede the answer).
    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {}


def render_report(report: dict) -> str:
    ordered_sections = [
        "Lungs and Airways",
        "Pleura",
        "Cardiovascular",
        "Hila and Mediastinum",
        "Tubes, Catheters, and Support Devices",
        "Musculoskeletal and Chest Wall",
        "Abdominal",
        "Other",
    ]

    lines = []
    for section in ordered_sections:
        findings = report.get(section, [])
        if findings:
            lines.append(f"{section}:")
            for item in findings:
                lines.append(f"- {item}")
            lines.append("")
    return "\n".join(lines).strip()


def run_pipeline(free_text: str, is_agent:bool = True) -> str:
    structuring_prompt = build_structuring_prompt(free_text)
    initial_report_response = call_llm(structuring_prompt)
    if not is_agent:
        print("Initial structured report response:", initial_report_response)
        return initial_report_response

    initial_report = initial_report_response
    print("Initial structured report response:", initial_report)

    findings_judge_prompt = build_findings_judge_prompt(free_text, initial_report)
    findings_judge_response = call_llm(findings_judge_prompt)

    findings_feedback = extract_json(findings_judge_response) or findings_judge_response
    print("findings_feedback", findings_feedback)

    anatomy_judge_prompt = build_anatomy_duplication_judge_prompt(initial_report)
    anatomy_judge_response = call_llm(anatomy_judge_prompt)

    anatomy_feedback = extract_json(anatomy_judge_response) or anatomy_judge_response

    print("anatomy_feedback", anatomy_feedback)

    revision_prompt = build_revision_prompt(
        free_text,
        initial_report,
        findings_feedback,
        anatomy_feedback,
    )
    final_report_response = call_llm(revision_prompt)
    print(f"Final structured report response:\n{final_report_response}")

    return final_report_response


import time
import pandas as pd
from argparse import ArgumentParser
if __name__ == "__main__":

    args = ArgumentParser()
    args.add_argument("--model_name", default="Qwen/Qwen3-14B", help="model name to use")
    args.add_argument("--is_agent_mode", action="store_true", help="whether to run in agent mode with feedback loop or just one-pass structuring")

    args = args.parse_args()
    model_name = args.model_name
    is_agent_mode = args.is_agent_mode

    if "medgemma" in model_name.lower():
        model_name = f"google/{model_name}"
        print("Using MedGemma model and processor")
        call_llm = call_medgemma
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype="auto",
            device_map=device,
            attn_implementation=get_attn_impl(),
            token=hf_toekn
        )
        processor = AutoProcessor.from_pretrained(model_name, token=hf_toekn)
    elif "gemma" in model_name.lower():
        model_name = f"google/{model_name}"
        print("Using Gemma model and processor")
        call_llm = call_gemma
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype="auto",
            device_map=device,
            #attn_implementation=get_attn_impl(),
            token=hf_toekn
        )
        processor = AutoProcessor.from_pretrained(model_name, token=hf_toekn)
    elif "qwen" in model_name.lower():
        print("Using Qwen model and processor")
        model_name = f"Qwen/{model_name}"
        call_llm = call_qwen3
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype="auto",
            attn_implementation=get_attn_impl(),
            device_map="auto"
        )
    elif "gpt" in model_name.lower():
        print("Using GPT model and processor")
        call_llm = call_gpt
        model_name = f"openai/{model_name}"
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
        os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1"

        client = OpenAI(
            base_url="http://127.0.0.1:8050/v1",
            api_key="EMPTY",
            http_client=httpx.Client(trust_env=False, timeout=600),
        )
    else:
        raise ValueError(f"Model {model_name} not supported. Only MedGemma and Qwen models are supported.")


    gen_column = f"{args.model_name}-agent" if is_agent_mode else f"{args.model_name}-model"
    new_df = pd.DataFrame(columns=['StudyInstanceUid','ref',gen_column])
    df = pd.read_csv("/home/hpc/iwi5/iwi5284h/RRG/srr_eval_all.csv")
    for idx,row in df.iterrows():
        start =  time.time()
        free_text = row['findings']
        print(row['StudyInstanceUid'])

        new_df.loc[len(new_df)] = [row['StudyInstanceUid'],row['findings'],run_pipeline(free_text,is_agent=is_agent_mode).strip()]
        end = time.time()
        print(f"Time taken for row {idx}: {end - start} seconds")

    new_df.to_csv(f"/home/hpc/iwi5/iwi5284h/RRG/{gen_column}.csv",index=False)
