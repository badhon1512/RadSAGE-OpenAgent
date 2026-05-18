import torch
import re as _re
from transformers import BitsAndBytesConfig


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


def get_attn_impl():
    # default fallback
    attn = "eager"
    try:
        import flash_attn
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            # FlashAttention 2 usually requires Compute Capability >= 8.0 (Ampere or newer)
            if cc[0] >= 8:
                attn = "flash_attention_2"
                print(f"[info] Using FlashAttention-2 (GPU CC {cc[0]}.{cc[1]})")
            else:
                print(f"[warn] FlashAttention found but GPU CC {cc[0]}.{cc[1]} < 8.0 → fallback to eager")
        else:
            print("[warn] No CUDA device detected → fallback to eager")
    except ImportError:
        print("[warn] flash_attn not installed -> fallback to eager")
    return attn


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen3 emits in thinking mode."""
    return _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()
