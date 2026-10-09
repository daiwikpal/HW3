"""Model loading and LoRA training configuration helpers."""

import inspect
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments


def load_base_model(model_id, dtype):
    """Load a fresh causal LM directly onto the current CUDA device."""
    version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
    keyword = "dtype" if version >= (4, 56) else "torch_dtype"
    return AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", **{keyword: dtype})


def base_model_stats(model, model_id):
    """Measure base parameter count, runtime footprint, and cached weight bytes."""
    snapshot = Path(snapshot_download(model_id, local_files_only=True))
    return {"total_params": sum(parameter.numel() for parameter in model.parameters()),
            "memory_footprint_bytes": model.get_memory_footprint(),
            "disk_bytes": sum(path.stat().st_size for path in snapshot.glob("*.safetensors"))}


def expected_lora_params(config, rank):
    """Compute LoRA parameters for every adapted Qwen attention/MLP projection."""
    hidden, intermediate, layers = config.hidden_size, config.intermediate_size, config.num_hidden_layers
    kv = config.num_key_value_heads * (hidden // config.num_attention_heads)
    per_layer = rank * (2 * (hidden + hidden) + 2 * (hidden + kv)
                        + 2 * (hidden + intermediate) + (intermediate + hidden))
    return per_layer * layers


def attach_lora(model, rank, alpha_per_rank, dropout, target_modules):
    """Attach a causal-LM LoRA adapter to a freshly loaded base model."""
    config = LoraConfig(r=rank, lora_alpha=alpha_per_rank * rank, lora_dropout=dropout,
                        target_modules=target_modules, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, config)


def make_training_args(output_dir, epochs, learning_rate, batch_size, grad_accum,
                       max_steps, dtype, seed):
    """Build version-compatible Trainer arguments for one LoRA run."""
    parameters = inspect.signature(TrainingArguments).parameters
    values = dict(output_dir=output_dir, num_train_epochs=epochs, learning_rate=learning_rate,
                  per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                  gradient_accumulation_steps=grad_accum, lr_scheduler_type="cosine",
                  logging_steps=5, save_strategy="no", max_steps=max_steps, report_to="none",
                  seed=seed, bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16)
    values["warmup_ratio" if "warmup_ratio" in parameters else "warmup_steps"] = 0.03
    evaluation_key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    values[evaluation_key] = "epoch"
    return TrainingArguments(**values)


def adapter_file_bytes(adapter_dir):
    """Return the measured size of the saved adapter weights."""
    files = list(Path(adapter_dir).glob("adapter_model.*"))
    return sum(path.stat().st_size for path in files)
