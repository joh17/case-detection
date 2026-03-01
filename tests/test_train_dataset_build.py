from __future__ import annotations

from typing import TYPE_CHECKING

from training.build_train_dataset import get_class_id_per_token, words_to_sentence
from training.cases import CASE_TO_CLASS_ID

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


def test_get_class_id_per_token(tokenizer: PreTrainedTokenizer):
    test_words = ["Kissa", "kävelee", "!isolla", "lavalla", "joulupukin", "kanssa", "."]
    test_cases = ["Nom", "-", "-", "Ade", "Ade", "Gen", "Nom", "-"]
    test_case_ids = [CASE_TO_CLASS_ID[case_name] for case_name in test_cases]

    tokenized_sentence = tokenizer(words_to_sentence(test_words), return_offsets_mapping=True)

    token_cases = get_class_id_per_token(tokenized_sentence["offset_mapping"], test_words, test_case_ids)
    assert len(token_cases) == len(tokenized_sentence["input_ids"]), (
        "Size mismatch: there should be 1 case ID for each token"
    )


def test_get_class_id_per_token_with_multiwords():
    test_words = ["Machine", "Learning"]
    test_cases = ["Nom", "Nom"]
    test_case_ids = [CASE_TO_CLASS_ID[case_name] for case_name in test_cases]

    tokenized_offset_mapping = [(0, 0), (0, 16), (0, 0)]
    tokenized_input_ids = [0, 123, 0]

    token_cases = get_class_id_per_token(tokenized_offset_mapping, test_words, test_case_ids)
    assert len(token_cases) == len(tokenized_input_ids), "Size mismatch: there should be 1 case ID for each token"
