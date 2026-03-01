"""Converts pre-processed text files into tokenized HuggingFace datasets for training.

In the input text, each line contains tab-separated columns: `words | base_forms | cases`, where
each column holds space-separated values. Target words are either fully masked ([MASK]) in the
slot-only scenario or partially (using base form only) in the lemma-conditioned scenario.
The dataset will be used to teach the model to predict case from context in both scenarios.
"""

import random
from argparse import ArgumentParser
from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from training.cases import CASE_TO_CLASS_ID, UNDEFINED_CASE

MAX_TOKEN_LENGTH = 128
MASK_TYPE_SLOTONLY = 1
MASK_TYPE_LEMMA = 2
PUNCTUATION_WITHOUT_SPACE_SEP = [".", ",", ":", ";", "!", "?"]
PAD_CASE_ID = -100
DEFAULT_CASE_ID = 0  # Used for punctuation and undefined tokens
UNICODE_OFFSET = 100_000
PC_SLOTONLY_MASK = 0.5


def words_to_sentence(words: list[str]) -> str:
    """Join words into a sentence, removing spaces before punctuation."""
    sentence = " ".join(words)
    for char in PUNCTUATION_WITHOUT_SPACE_SEP:
        sentence = sentence.replace(" " + char, char)
    return sentence


def get_class_id_per_token(
    char_offset_mapping: list[tuple[int, int]],
    words: list[str],
    class_ids: list[int],
) -> list[int | None]:
    """Map a class ID to each token based on which word it belongs to.

    Uses a Unicode-encoded sentinel string to track word boundaries: each word
    is replaced by a run of a unique character, so any character position can be
    mapped back to its word index via character arithmetic.
    """
    # Build a character-level word-index string, e.g. "Test one!" → "████ ███!"
    # where each block character encodes the word's index via its Unicode codepoint.
    case_id_sentence = words_to_sentence(
        [chr(UNICODE_OFFSET + i) * len(w) if w not in PUNCTUATION_WITHOUT_SPACE_SEP else w for i, w in enumerate(words)]
    )

    token_class_ids = []
    for char_start, char_end in char_offset_mapping:
        if char_start == char_end:
            # Special tokens (CLS, SEP, PAD) have zero-length offsets.
            token_class_ids.append(PAD_CASE_ID)
            continue

        word_id = ord(case_id_sentence[char_end - 1]) - UNICODE_OFFSET

        if not (0 <= word_id < len(words)):
            # Punctuation characters are not encoded as sentinels, so fall back.
            token_class_ids.append(DEFAULT_CASE_ID)
        else:
            token_class_ids.append(class_ids[word_id])

    return token_class_ids


def select_random_samples(values: list[Any], keep_probability: float = 0.2) -> list[Any]:
    """Randomly subsample a list, always returning at least one element."""
    result = [v for v in values if random.random() < keep_probability]
    if not result and len(values) > 0:
        result = [random.choice(values)]
    return result


def datarow_to_tokenized_input(line: str) -> dict:
    """Tokenize one VRT data row into a model-ready input dict.

    Randomly selects a subset of cased words and masks them either fully
    ([MASK] token) or partially (base form), then returns the tokenized
    sentence with per-token case labels and masking-type labels.
    """
    parts = line.strip().split("\t")
    words = parts[0].split(" ")
    word_bases = parts[1].split(" ")
    cases = parts[2].split(" ")

    assert len(words) == len(word_bases) == len(cases), "Size mismatch between words and labels"

    masked_case_indexes = select_random_samples(np.where(np.array(cases) != UNDEFINED_CASE)[0])

    word_mask_types = [0] * len(words)
    for idx in masked_case_indexes:
        # Randomly decide whether the word will be fully masked (slot-only scenario)
        if random.random() >= PC_SLOTONLY_MASK:
            words[idx] = tokenizer.mask_token
            word_mask_types[idx] = MASK_TYPE_SLOTONLY
        # Or if we just replace the word by its baseform (lemma-conditioned scenario)
        else:
            words[idx] = word_bases[idx]
            word_mask_types[idx] = MASK_TYPE_LEMMA

    masked_sentence = words_to_sentence(words)
    tokenized = tokenizer(
        masked_sentence,
        padding="max_length",
        max_length=MAX_TOKEN_LENGTH,
        truncation=True,
        return_offsets_mapping=True,
    )

    case_class_ids = [CASE_TO_CLASS_ID[case] for case in cases]
    token_case_ids = get_class_id_per_token(tokenized["offset_mapping"], words, case_class_ids)
    mask_type_per_token = get_class_id_per_token(tokenized["offset_mapping"], words, word_mask_types)

    assert len(tokenized["input_ids"]) == len(token_case_ids)
    return {
        **tokenized,
        "labels": token_case_ids,
        "masking_type": mask_type_per_token,
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="TurkuNLP/bert-base-finnish-uncased-v1")
    parser.add_argument("--input_processed_text_file", type=str, required=True)
    parser.add_argument("--output_dataset_folder", type=str)
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset = load_dataset("text", data_files=args.input_processed_text_file)
    dataset = dataset.map(lambda line: datarow_to_tokenized_input(line["text"]))

    if args.split != "train":
        from datasets import DatasetDict
        dataset = DatasetDict({args.split: dataset["train"]})

    short_model_name = args.model_name.split("/")[-1]
    dataset.remove_columns(["text", "offset_mapping"]).save_to_disk(args.output_dataset_folder)
