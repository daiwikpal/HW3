"""TruthfulQA MC1 log-likelihood scoring helpers."""

import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


def answer_logprob(model, tokenizer, build_prompt, question, answer):
    """Return summed log P(answer tokens | chat-formatted question)."""
    prompt_ids = tokenizer(build_prompt(question), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([prompt_ids + answer_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids).logits[0, :-1].float()
    targets = input_ids[0, 1:].unsqueeze(1)
    token_scores = torch.log_softmax(logits, dim=-1).gather(1, targets).squeeze(1)
    return token_scores[len(prompt_ids) - 1:].sum().item()


def mc1_predict(model, tokenizer, build_prompt, row):
    """Pick the highest-likelihood MC1 answer and return its scores."""
    scores = [answer_logprob(model, tokenizer, build_prompt, row.question, choice)
              for choice in row.mc1_choices]
    prediction = int(np.argmax(scores))
    return prediction, int(row.mc1_labels[prediction] == 1), scores


def evaluate_model(model, tokenizer, build_prompt, test_frame, name, results_dir):
    """Evaluate MC1 accuracy and per-question wall-clock latency."""
    model.eval()
    for row in list(test_frame.itertuples())[:2]:
        mc1_predict(model, tokenizer, build_prompt, row)
    records = []
    for row in tqdm(test_frame.itertuples(), total=len(test_frame), desc=name):
        torch.cuda.synchronize()
        start = time.perf_counter()
        prediction, correct, scores = mc1_predict(model, tokenizer, build_prompt, row)
        torch.cuda.synchronize()
        true_index = row.mc1_labels.index(1)
        wrong_scores = [score for index, score in enumerate(scores) if index != true_index]
        records.append({"qid": row.qid, "category": row.category, "question": row.question,
                        "best_answer": row.best_answer,
                        "predicted_answer": row.mc1_choices[prediction], "correct": correct,
                        "latency_s": time.perf_counter() - start,
                        "margin": scores[true_index] - max(wrong_scores)})
    results = pd.DataFrame(records)
    results.to_csv(f"{results_dir}/eval_{name}.csv", index=False)
    print(f"{name}: accuracy={results.correct.mean():.3f}, "
          f"mean latency={results.latency_s.mean():.3f}s")
    return results
