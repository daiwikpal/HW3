"""TruthfulQA loading, splitting, and supervised-finetuning data helpers."""

import pandas as pd
from datasets import Dataset, load_dataset


def load_truthfulqa(dataset_id):
    """Join TruthfulQA generation metadata to its multiple-choice targets."""
    generation = load_dataset(dataset_id, "generation", split="validation").to_pandas()
    multiple_choice = load_dataset(dataset_id, "multiple_choice", split="validation").to_pandas()
    generation["join_question"] = generation.question.str.strip()
    multiple_choice["join_question"] = multiple_choice.question.str.strip()
    columns = ["join_question", "category", "best_answer", "correct_answers", "incorrect_answers"]
    frame = multiple_choice.merge(generation[columns], on="join_question", validate="one_to_one")
    frame = frame.rename(columns={"join_question": "question"}).reset_index(drop=True)
    frame["qid"] = frame.index
    frame["mc1_choices"] = frame.mc1_targets.map(lambda target: list(target["choices"]))
    frame["mc1_labels"] = frame.mc1_targets.map(lambda target: list(target["labels"]))
    result = frame[["qid", "category", "question", "best_answer", "correct_answers",
                    "incorrect_answers", "mc1_choices", "mc1_labels"]]
    assert len(result) == 817 and result.mc1_choices.map(len).gt(0).all()
    assert result.category.nunique() == 38
    return result


def split_questions(frame, n_test, val_fraction, seed):
    """Create disjoint test, training-question, and validation-question splits."""
    test = frame.sample(n=n_test, random_state=seed).sort_values("qid").reset_index(drop=True)
    pool = frame.loc[~frame.qid.isin(test.qid)]
    validation = pool.sample(frac=val_fraction, random_state=seed)
    train = pool.loc[~pool.qid.isin(validation.qid)]
    assert set(test.qid).isdisjoint(pool.qid)
    assert set(train.qid).isdisjoint(validation.qid)
    return train.reset_index(drop=True), validation.reset_index(drop=True), test


def make_sft_pairs(questions, max_examples=None):
    """Expand each question into de-duplicated question/correct-answer pairs."""
    pairs = []
    for row in questions.itertuples():
        answers = list(row.correct_answers)
        if row.best_answer not in answers:
            answers.insert(0, row.best_answer)
        for answer in dict.fromkeys(answers):
            if answer.strip() != "I have no comment":
                pairs.append((row.question, answer))
    return pairs if max_examples is None else pairs[:max_examples]


def build_sft_dataset(pairs, tokenizer, build_prompt, max_len):
    """Tokenize SFT pairs while masking every prompt token from the loss."""
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert end_id != tokenizer.unk_token_id

    def tokenize_pair(pair):
        prompt_ids = tokenizer(build_prompt(pair[0]), add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(pair[1], add_special_tokens=False)["input_ids"] + [end_id]
        input_ids = (prompt_ids + answer_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}

    return Dataset.from_list([tokenize_pair(pair) for pair in pairs])
