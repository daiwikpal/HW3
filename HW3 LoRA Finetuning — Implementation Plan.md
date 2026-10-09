# HW3 LoRA Finetuning — Implementation Plan

Oct 8, 2026 · @Daiwik Pal

## Goal and context

Build one Jupyter notebook that compares three models on a 100-question TruthfulQA test set: base Qwen2.5-3B-Instruct, a LoRA adapter with rank 8, and a LoRA adapter with rank 64. This is CS6220 HW3, design option 3.2 (two LoRA ranks). It is due Oct 9, 2026 at midnight.

- **Accuracy metric:** TruthfulQA MC1 by log-likelihood. Score each MC1 answer choice by the summed log-probability of its tokens given the question. The prediction is the highest-scoring choice; the query is correct if that choice is the true one.
- **Training data:** the 717 non-test questions, turned into question → correct-answer pairs.
- **Hardware:** Georgia Tech PACE ICE, one GPU in a 4-hour interactive Jupyter session. The notebook does the split, training and evaluation top to bottom inside that session.
- **Purpose of the notebook:** produce every number, table, example and screenshot the report needs. The last section maps each deliverable to the cell that answers it. Daiwik writes the report text himself.

## Rules for the coding agent

Readability beats cleverness: a grader must be able to read the notebook top to bottom and follow it.

- **One notebook as the entry point,** `hw3_truthfulqa_lora.ipynb`. It must run top to bottom with Restart & Run All inside one 4-hour GPU session.
- **Helper `.py` modules are allowed** for reusable functions, in the same folder as the notebook. Suggested split: `data_utils.py` (loading, split, SFT pairs), `scoring.py` (MC1 scoring and evaluation), `train_utils.py` (LoRA setup and training). Keep to a few flat files, no package structure.
- **Module functions take their inputs as arguments** (tokenizer, model, config values) instead of reading notebook globals, so each file reads on its own.
- **The notebook stays the readable story.** It keeps the CONFIG cell, the calls in order, and every printed table and plot. A reader should be able to follow the experiment from the notebook alone, opening a module only for details.
- **For interactive work,** start the notebook with `%load_ext autoreload` and `%autoreload 2`, so module edits take effect without restarting the kernel.
- **Every output appears in the notebook and on disk.** Display each table in the notebook and also write it to `results/` as CSV; write headline numbers to JSON; display each plot and save it as PNG. Save right after computing, so a late crash loses nothing. No Excel files.
- **Report cells read from disk.** Sections 12–15 and 17 load the saved CSV and JSON files, not in-memory variables, so they can be re-run in seconds to redraw any output without retraining.
- **Markdown before every section** naming the deliverable it serves (see the last section of this plan).
- **All settings in one CONFIG cell** at the top; modules hold no settings of their own. No argparse or YAML config.
- **Small plain functions,** ideally 25 lines or fewer, each with a one-line docstring. No classes, decorators, multiprocessing, or custom Trainer callbacks.
- **Libraries allowed:** torch, transformers (`Trainer`), datasets, peft, accelerate, pandas, numpy, matplotlib, tqdm.
- **Do not use:** TRL, Unsloth, DeepSpeed, FSDP, bitsandbytes/QLoRA, vLLM, flash-attn, Weights & Biases, openpyxl or any Excel output. Plain `Trainer` with manual label masking keeps training and scoring on identical tokenization, and avoids TRL's frequent API changes.
- **Target about 300 lines of code in total,** notebook and modules combined.
- **Seed 42 everywhere:** the split, training, and example sampling.
- **Print key numbers as readable tables while running.** These printed outputs become the screenshots for deliverable 3.3.
- **Comments explain why, not what.**

## Fixed design decisions

These are settled; implement them as given, and put every value in the CONFIG cell.

| Setting | Value | Reason |
| --- | --- | --- |
| Base model | `Qwen/Qwen2.5-3B-Instruct` | Fits one GPU; the chat template keeps prompting simple |
| Fallback model | `Qwen/Qwen2.5-3B` (non-instruct) | Only if baseline MC1 accuracy is above 70% |
| Dataset | `truthfulqa/truthful_qa`, configs `generation` + `multiple_choice` | 817 questions, 38 categories |
| Test set | 100 questions, uniform random, seed 42 | Assignment step 1 |
| Train pool | Remaining 717; 10% of those questions held out for validation loss | Split by question, so no question appears in both |
| Metric | MC1 accuracy: summed answer-token log-prob, no length normalization | Standard lm-evaluation-harness definition |
| LoRA ranks | r1 = 8, r2 = 64 | Option 3.2 |
| LoRA alpha | 2 × r (16 and 128) | Fixed alpha/r, so rank is the only variable |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | All linear layers |
| LoRA dropout | 0.05 |  |
| Epochs | 3 |  |
| Learning rate | 2e-4, cosine schedule, 3% warmup |  |
| Batch | 8 per device × gradient accumulation 2 = 16 effective |  |
| Max sequence length | 256 tokens | Questions and answers are short |
| Precision | bf16 if GPU compute capability ≥ 8 (A100, A40, L40S, H100), else fp16 | RTX 6000 and V100 lack bf16 |
| Termination | Fixed 3 epochs; final train and validation loss are reported as the error at termination | Assignment 3.3 |

**Prompt format.** The user message is the question alone, formatted with `tokenizer.apply_chat_template(..., add_generation_prompt=True, tokenize=False)`. Qwen's default system prompt is inserted automatically. One shared helper builds this prompt for both training and scoring.

**Query latency** is the wall time to score all MC1 choices for one question.

## PACE environment setup

Everything lives on scratch, because the 15 GB home quota cannot hold model weights. Do setup and downloads on the login node; run compute only inside Slurm jobs.

1. Connect the GlobalProtect VPN, then `ssh` to `login-ice.pace.gatech.edu`.
2. Create the working folders:

```bash
mkdir -p ~/scratch/hw3/{results,outputs,logs} && cd ~/scratch/hw3
```

3. Create the Python environment on scratch. Check the module name with `module avail anaconda` first.

```bash
module load anaconda3
conda create -p ~/scratch/hw3/env python=3.11 -y
conda activate ~/scratch/hw3/env
pip install torch transformers datasets peft accelerate pandas numpy matplotlib tqdm jupyter ipykernel
python -m ipykernel install --user --name hw3-env --display-name "Python (hw3)"   # makes the env selectable in Jupyter
```

4. Prefetch the model and dataset, so the batch job can run offline:

```bash
export HF_HOME=~/scratch/hf_cache
hf download Qwen/Qwen2.5-3B-Instruct   # older huggingface_hub: huggingface-cli download
python -c "from datasets import load_dataset; [load_dataset('truthfulqa/truthful_qa', c) for c in ('generation', 'multiple_choice')]"
```

5. Put the notebook and helper modules in `~/scratch/hw3`.
6. Run everything else in the GPU session described in the next section.

## Notebook specification

The notebook has 18 sections (0 to 17), in this order. Function names below are the required names; bodies are up to the agent unless a reference snippet is given. Each function may live in the notebook or in a helper module. In a module, it takes the tokenizer and config values as arguments instead of using notebook globals.

### 0. Title (markdown)

Title, author (Daiwik), and a short table mapping notebook sections to assignment deliverables (copy from the last section of this plan).

### 1. CONFIG cell

```python
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATASET_ID = "truthfulqa/truthful_qa"
SEED = 42
N_TEST = 100
VAL_FRACTION = 0.10
LORA_RANKS = [8, 64]
LORA_ALPHA_PER_RANK = 2          # alpha = 2 * r
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
GRAD_ACCUM = 2
MAX_LEN = 256
GRADIENT_CHECKPOINTING = False   # set True only if out of memory
RUN_MERGE_CHECK = True           # optional section 16
SMOKE_TEST = False

MAX_TRAIN_EXAMPLES = None        # None = use all
MAX_STEPS = -1                   # -1 = use EPOCHS
RESULTS_DIR, OUTPUT_DIR = "results", "outputs"
if SMOKE_TEST:                   # tiny end-to-end run to catch bugs fast
    N_TEST, MAX_TRAIN_EXAMPLES, MAX_STEPS = 10, 64, 5
    RESULTS_DIR, OUTPUT_DIR = "results_smoke", "outputs_smoke"
```

### 2. Imports and environment

First set os.environ\["HF\_HOME"\] to \~/scratch/hf\_cache, before importing transformers or datasets, so the cache stays off the home quota. Then import everything here. Print the versions of torch, transformers, peft and datasets, plus the GPU name and memory. Choose the dtype from the GPU's compute capability, not from `torch.cuda.is_bf16_supported()`, which reports emulated bf16 as supported:

```python
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
```

Call `transformers.set_seed(SEED)` and create the results and output folders.

### 3. Packages used (markdown, deliverable 1)

One entry each, with the URL and a one-to-two-sentence description written by the agent:

- Pretrained LLM: Qwen2.5-3B-Instruct, [huggingface.co/Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- Benchmark dataset: TruthfulQA, [github.com/sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA) (Hugging Face copy: [truthfulqa/truthful\_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa))
- Finetuning algorithm: supervised finetuning with the Hugging Face Transformers `Trainer`, [github.com/huggingface/transformers](https://github.com/huggingface/transformers); LoRA reference implementation, [github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)
- LoRA package: Hugging Face PEFT, [github.com/huggingface/peft](https://github.com/huggingface/peft)

### 4. Load the dataset

`load_truthfulqa()` returns one DataFrame with these columns: `qid, category, question, best_answer, correct_answers, incorrect_answers, mc1_choices, mc1_labels`.

- Load the `generation` and `multiple_choice` configs (each has a single `validation` split of 817 rows).
- Only `multiple_choice` has `mc1_targets`; only `generation` has `category`. Join the two on the stripped question text.
- `qid` is the row index from 0 to 816.
- Assert 817 rows and no missing MC1 targets. Print the number of categories (expect 38) and one example row.

### 5. Train/test split (assignment step 1)

- Draw the test set with `df.sample(n=N_TEST, random_state=SEED)`; the rest is the train pool.
- Assert the train and test qids are disjoint.
- Save `test_questions.csv` (qid, category, question, best\_answer) and `train_qids.csv`.
- Print how many categories the test set covers, with counts per category.
- Split the train pool by question into train (90%) and validation (10%) with the same seed.

### 6. Prompt and encoding helpers

These two functions are shared by training and scoring, so both see exactly the same token layout.

```python
def build_prompt(question):
    """Chat-formatted prompt that ends where the assistant's answer begins."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def encode_pair(question, answer):
    """Token ids of the prompt and of the answer, tokenized separately."""
    prompt_ids = tokenizer(build_prompt(question), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    return prompt_ids, answer_ids
```

### 7. Load the base model and record its size

- `load_base_model()` loads the model with `AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map="cuda")`. Transformers versions before 4.56 take `torch_dtype=` instead.
- Load the tokenizer once, globally.
- Record and print three numbers: total parameters (`sum(p.numel() ...)`), bytes in GPU memory (`model.get_memory_footprint()`), and bytes on disk. For bytes on disk, sum the `*.safetensors` files in `snapshot_download(MODEL_ID, local_files_only=True)`.

### 8. MC1 accuracy function

Reference implementation of the scorer. Watch the off-by-one: the logit at position t predicts token t+1.

```python
def answer_logprob(model, question, answer):
    """Sum of log P(answer tokens | prompt)."""
    prompt_ids, answer_ids = encode_pair(question, answer)
    input_ids = torch.tensor([prompt_ids + answer_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, :-1].float()
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(1, input_ids[0, 1:].unsqueeze(1)).squeeze(1)
    return token_logprobs[len(prompt_ids) - 1:].sum().item()

def mc1_predict(model, row):
    """Pick the highest-scoring MC1 choice; return index, correctness, scores."""
    scores = [answer_logprob(model, row.question, choice) for choice in row.mc1_choices]
    pred = int(np.argmax(scores))
    return pred, int(row.mc1_labels[pred] == 1), scores
```

`evaluate_model(model, test_df, name)`:

- Call `model.eval()`, then score two test questions untimed as a GPU warm-up.
- For each test question, call `torch.cuda.synchronize()` and `time.perf_counter()` before and after `mc1_predict`. Show progress with `tqdm`.
- One row per question: `qid, category, question, best_answer, predicted_answer, correct, latency_s, margin`. Here `margin` is the true choice's score minus the best wrong choice's score.
- Display the per-question table in the notebook and save it as `eval_{name}.csv`. For the baseline, this CSV is the step 2 table of test queries and results.
- Print the accuracy and the mean latency.

Two notes: in the dataset the true MC1 choice is always at index 0, but log-likelihood scores do not depend on order, so no shuffling is needed. Ties between float scores are practically impossible.

### 9. Baseline evaluation (assignment step 2)

Run `evaluate_model(base_model, test_df, "base")`, then print a check:

- If accuracy is 100%, say to re-draw the test set with a new seed.
- If accuracy is above 70%, say to switch `MODEL_ID` to `Qwen/Qwen2.5-3B` and keep this CSV as documentation of the first choice.
- Do not automate the switch.

Free the base model afterward.

### 10. Build the SFT dataset

- `make_sft_pairs(questions_df)` returns (question, answer) pairs. Use every correct answer, make sure `best_answer` is included, and de-duplicate. Drop "I have no comment", since it teaches evasion and never appears as an MC1 choice. Truncate to `MAX_TRAIN_EXAMPLES` if it is set.
- `tokenize_pair(question, answer)` returns `input_ids`, `attention_mask` and `labels`:
  - `input_ids` = prompt + answer + end token.
  - `labels` = `[-100] * len(prompt)` + answer + end token, so loss is computed on the answer only.
  - The end token is `tokenizer.convert_tokens_to_ids("<|im_end|>")`, the token that closes a turn in Qwen's chat template.
  - Truncate everything to `MAX_LEN`.
- Build `datasets.Dataset.from_list(...)` for train and validation.
- Print the example counts and the mean token length. Decode one example, showing that the prompt is masked.

### 11. LoRA training loop (one iteration per rank)

For each `rank` in `LORA_RANKS`:

1. **Load a fresh base model** with `load_base_model()`. Never stack adapters. If `GRADIENT_CHECKPOINTING` is on, call `model.gradient_checkpointing_enable()` and `model.enable_input_require_grads()`.
2. **Attach LoRA:** `LoraConfig(r=rank, lora_alpha=LORA_ALPHA_PER_RANK * rank, lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM")`, then `get_peft_model`, then `print_trainable_parameters()`.
3. **Assert the trainable parameter count** equals `expected_lora_params(model.config, rank)`, the helper shown below this list.
4. **Configure training:** `TrainingArguments` with these settings:
   - the epochs, learning rate, batch size and gradient accumulation from CONFIG
   - `lr_scheduler_type="cosine"`, `warmup_ratio=0.03`
   - `logging_steps=5`, `eval_strategy="epoch"` (older versions: `evaluation_strategy`), `save_strategy="no"`
   - `bf16` or `fp16` to match `DTYPE`
   - `max_steps=MAX_STEPS`, `report_to="none"`, `seed=SEED`
5. **Create the trainer:** `Trainer(...)` with the collator `DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100)`.
6. **Print a hyperparameter table** for this run (one DataFrame).
7. **Train and time it:** reset peak GPU memory, time `trainer.train()` with `perf_counter`, then run `trainer.evaluate()`.
8. **Save the adapter** to `OUTPUT_DIR/lora_r{rank}`. Record the size of `adapter_model.safetensors` on disk. PEFT stores adapters in fp32, so measure it rather than assume.
9. **Save the training log:** `trainer.state.log_history` to `train_log_r{rank}.csv`.
10. **Record a run summary row** and append it to `training_runs.csv`. Fields: rank, alpha, trainable params, % trainable, total params (base + LoRA), adapter bytes, train time (s), steps, final logged train loss, final validation loss, peak GPU memory (GB).
11. **Evaluate now,** while the unmerged adapter is loaded: `evaluate_model(model, test_df, f"lora_r{rank}")`.
12. **Free memory:** `del` the model and trainer, then `gc.collect()` and `torch.cuda.empty_cache()`.

```python
def expected_lora_params(cfg, r):
    """LoRA adds r*(d_in + d_out) parameters per adapted linear layer."""
    h, i, layers = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
    kv = cfg.num_key_value_heads * (h // cfg.num_attention_heads)
    per_layer = r * (2 * (h + h) + 2 * (h + kv) + 2 * (h + i) + (i + h))
    return per_layer * layers
```

For Qwen2.5-3B this gives 14,966,784 parameters at r = 8 and 119,734,272 at r = 64.

### 12. Model size and training time (deliverable 2)

One table with rows `base`, `lora_r8` and `lora_r64`. Columns:

- total params, trainable params, % trainable
- size in bytes: for the base, params × bytes per param; for each fine-tuned model, base bytes + adapter bytes. Add a note that a merged model is the same size as the base.
- adapter file size (MB)
- train time (s), peak GPU memory (GB)
- final train loss, final validation loss, test accuracy

Save it as `size_time_table.csv`.

### 13. Test performance comparison (deliverable 3.1)

Per model, from the three eval CSVs: n, mean accuracy, standard deviation of accuracy, mean latency (ms), standard deviation of latency (ms). Accuracy SD is the pandas `.std()` of the per-query 0/1 correctness. Save it as `comparison_table.csv`, and add one markdown sentence explaining the accuracy SD.

### 14. Example queries (deliverable 3.2)

Pivot the eval CSVs into a wide table by qid, with each model's correctness and chosen answer. Define four buckets:

- **all\_correct:** all three models right
- **fixed\_by\_finetuning:** base wrong, both LoRA models right
- **two\_failed:** exactly two of the three wrong
- **all\_failed:** all three wrong

Print each bucket's size. Then pick 2 per bucket with `sample(random_state=SEED)`, printing a note if a bucket has fewer than 2. For each example, print the question, the best answer, and each model's chosen answer with right/wrong. Save `examples.csv`.

### 15. Training curves and hyperparameters (deliverable 3.3)

- Plot training loss against step for both ranks, with validation loss per epoch marked, on one figure. Save it as `loss_curves.png`.
- Print the two runs' hyperparameters side by side, and the final train and validation loss as the error at termination.

### 16. Optional: merged vs unmerged latency

Run only if `RUN_MERGE_CHECK` is set. Load the base model with the r = 64 adapter, then `merge_and_unload()`. Time 20 test queries and compare against the unmerged latency from section 11. This checks the lecture's claim that a merged model has no extra inference cost. Save `merge_check.csv`.

### 17. Summary for the report

Write `summary.json` with every headline number, and print a compact summary. Add a per-category accuracy table for the three models, covering only categories in the test set. End with a markdown cell of open questions for Daiwik's Final Remarks:

- Did r = 64 beat r = 8?
- What did the extra rank cost in time and size?
- Which categories improved?

## Running it in a 4-hour GPU session

Request one GPU for 4 hours and run the whole notebook inside that session; no batch script is needed. The full run should take about an hour on an A100, which leaves time for the smoke test and a rerun.

1. Connect the VPN and open [ICE Open OnDemand](https://ondemand-ice.pace.gatech.edu).
2. Launch the Jupyter interactive app with 1 node, 1 GPU, 4 hours and about 48 GB of memory. If the form offers a choice, pick a bf16-capable GPU: A100, A40, L40S or H100.
3. Once the session starts, open `~/scratch/hw3/hw3_truthfulqa_lora.ipynb` and select the "Python (hw3)" kernel.
4. Run once with `SMOKE_TEST = True` (about 10 minutes) and check the outputs.
5. Set `SMOKE_TEST = False`, then choose Restart & Run All.

**Stepping away.** In a standard Jupyter setup, cell output reaches the notebook only while a browser tab is connected. If the tab closes mid-run, the code keeps running and the CSV, JSON and PNG files are still written, but printed output from that stretch is missing from the notebook. There are two ways around this:

- Keep the tab open and the laptop awake until the run finishes.
- Run the notebook headless from a Terminal tab inside the same Jupyter session. It keeps going with the browser closed and writes every output into an executed copy of the notebook:

```bash
cd ~/scratch/hw3 && module load anaconda3 && conda activate ~/scratch/hw3/env
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output hw3_truthfulqa_lora_executed.ipynb hw3_truthfulqa_lora.ipynb
```

If some displayed output goes missing anyway, re-run sections 12–15 and 17. They read from `results/` and redraw everything in seconds.

Without OnDemand, `salloc -N1 --gres=gpu:1 --mem=48G -t 4:00:00` gives the same 4-hour GPU. Jupyter is then started on the compute node and reached through an SSH tunnel.

**Troubleshooting**

- **Out of memory:** set `GRADIENT_CHECKPOINTING = True`, or use `BATCH_SIZE = 4` with `GRAD_ACCUM = 4`.
- **NaN loss under fp16:** start a new session on an A100, A40, L40S or H100.
- **Unknown-argument errors:** these come from library version drift. Use the names the installed version accepts, such as `dtype` vs `torch_dtype` or `eval_strategy` vs `evaluation_strategy`.

## Acceptance checklist

The handoff is done when every box below is true. The checks marked (assert) must be `assert` statements in the notebook.

- [ ] The smoke test (`SMOKE_TEST = True`) runs end to end in the GPU session in about 10 minutes, before the full run.
- [ ] The joined dataset has 817 questions and 38 categories (assert).
- [ ] The test set has 100 questions, and train and test qids are disjoint (assert).
- [ ] Baseline accuracy is printed together with the 70% check.
- [ ] One decoded training example shows the prompt masked, with labels covering only the answer and the end token.
- [ ] The trainable LoRA parameter count equals `expected_lora_params` for both ranks (assert).
- [ ] Each LoRA run starts from a freshly loaded base model.
- [ ] Training loss goes down, and the final validation loss is reported for both ranks.
- [ ] `eval_base.csv`, `eval_lora_r8.csv` and `eval_lora_r64.csv` each have 100 rows.
- [ ] Every table and plot appears in the notebook outputs and is also saved in `results/` as CSV, JSON or PNG. There are no Excel files.
- [ ] The notebook and helper modules together have about 300 lines of code or fewer, and use no TRL, DeepSpeed or quantization.

## Deliverable to notebook output map

Every deliverable is answered by a specific notebook section and results file. The Final Remarks are the only part the notebook does not produce.

| Deliverable | Notebook section | Output file in `results/` |
| --- | --- | --- |
| Step 2 baseline test queries and results | 9 | `eval_base.csv` |
| (1) URLs and descriptions of model, dataset, finetuning code, LoRA package | 3 | none (markdown) |
| (2) Params and bytes for the base and both fine-tuned models | 7, 12 | `size_time_table.csv` |
| (2) How rank affects training time, accuracy and size | 11, 12, 16 | `training_runs.csv`, `size_time_table.csv`, `merge_check.csv` |
| (3.1) Mean ± SD of accuracy and per-query latency, three models | 13 | `comparison_table.csv` |
| (3.2) Four sets of 2 example queries | 14 | `examples.csv` |
| (3.3) Screenshots: training iterations, hyperparameters, error at termination | 11, 15 | `train_log_r8.csv`, `train_log_r64.csv`, `loss_curves.png` |
| (4) Final Remarks, three learnings | 17 (numbers only) | `summary.json` |
