"""Fine-tune a transformer model for Finnish grammatical case detection using HuggingFace Trainer."""

from argparse import ArgumentParser

import numpy as np
import sklearn.metrics
from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

from training.cases import CASE_TO_CLASS_ID, CLASS_ID_TO_CASE

TOKEN_IGNORED_IN_EVAL = {0, -100}


def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    """Compute macro precision, recall, F1, and accuracy, excluding padding and undefined tokens."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    y_true = [
        y_token_true
        for y_sentence_pred, y_sentence_true in zip(predictions, labels, strict=True)
        for (y_token_pred, y_token_true) in zip(y_sentence_pred, y_sentence_true, strict=True)
        if y_token_true not in TOKEN_IGNORED_IN_EVAL
    ]

    y_pred = [
        y_token_pred
        for y_sentence_pred, y_sentence_true in zip(predictions, labels, strict=True)
        for (y_token_pred, y_token_true) in zip(y_sentence_pred, y_sentence_true, strict=True)
        if y_token_true not in TOKEN_IGNORED_IN_EVAL
    ]

    return {
        "precision": sklearn.metrics.precision_score(y_true, y_pred, average="macro"),
        "recall": sklearn.metrics.recall_score(y_true, y_pred, average="macro"),
        "f1": sklearn.metrics.f1_score(y_true, y_pred, average="macro"),
        "accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="TurkuNLP/bert-base-finnish-uncased-v1")
    parser.add_argument("--dataset_path", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_split_size", type=float, default=0.002)
    parser.add_argument("--output_dir", type=str, default="./training_output")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(CLASS_ID_TO_CASE),
        id2label=CLASS_ID_TO_CASE,
        label2id=CASE_TO_CLASS_ID,
    )
    model.to(args.device)

    dataset = load_from_disk(args.dataset_path)
    ds = dataset["train"].train_test_split(test_size=args.eval_split_size)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        logging_dir=args.logging_dir,
        save_strategy="steps",
        eval_strategy="steps",
        eval_on_start=True,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        report_to=args.report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
