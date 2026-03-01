from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig

if TYPE_CHECKING:
    from case_detection.inflect import Inflector

DEFAULT_MODEL = "johl/fi-case-detection-finbert"
PLACEHOLDER = "{}"
MIN_PROBA = 0.05


class CaseDetector:
    """Predicts the grammatical case of words using a fine-tuned transformer model.

    Mark the target position in the sentence with {}.

    Usage:
        # Case prediction only
        detector = CaseDetector()
        detector.predict("Matkasi {} alkaa.", "Tampere")
        # {'predictions': [{'label': 'All', 'score': 0.87}]}

        # With inflection (requires pyhfst)
        detector = CaseDetector(inflect=True)
        detector.predict("Matkasi {} alkaa.", "Tampere")
        # {'predictions': [{'label': 'All', 'score': 0.87, 'inflected': 'Tampereelle'}],
        #  'sentence': 'Matkasi Tampereelle alkaa.'}
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        inflect: bool = False,
        inflector: Inflector | None = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, use_safetensors=True)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model.eval()

        if inflector is not None:
            self.inflector = inflector
        elif inflect:
            from case_detection.inflect import Inflector as _Inflector
            self.inflector = _Inflector()
        else:
            self.inflector = None

    def predict(self, sentence: str, lemma: str | None = None, *, min_proba: float = MIN_PROBA) -> dict:
        """Predict the grammatical case of the target word in a sentence.

        The sentence must contain a {} placeholder marking the target position.

        Args:
            sentence: A sentence with {} where the target word is.
            lemma: The base form (nominative) of the target word. If None,
                   the placeholder is replaced with the model's [MASK] token
                   and the case is predicted from context alone.
            min_proba: Minimum probability threshold for returned predictions.

        Returns:
            A dict with:
            - 'predictions': list of dicts with 'label', 'score', and optionally 'inflected'
            - 'sentence': the filled sentence (only when inflection is enabled and a lemma is provided)
        """
        predictions = self._predict_cases(sentence, lemma, min_proba)
        result = {"predictions": predictions}

        if self.inflector is not None and lemma is not None:
            _add_inflections(result, sentence, lemma, self.inflector)

        return result

    def predict_all_words(self, sentence: str, min_proba: float = MIN_PROBA) -> list[dict]:
        """Predict the grammatical case for every token in a sentence.

        Args:
            sentence: A sentence (no placeholder needed).
            min_proba: Minimum probability threshold for returned predictions.

        Returns:
            A list of dicts, one per token, with 'token', 'label' (top predicted case),
            and 'score'.
        """
        tokenized = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        with torch.no_grad():
            logits = self.model(**tokenized).logits
        probas = torch.nn.functional.softmax(logits, dim=2)

        token_ids = tokenized["input_ids"][0]
        pred_ids = probas.argmax(dim=2)[0]

        results = []
        for tok_id, pred_id, token_probas in zip(token_ids, pred_ids, probas[0]):
            label = self.config.id2label[pred_id.item()]
            score = token_probas[pred_id].item()
            token_str = self.tokenizer.decode(tok_id)
            if label != "_" and score > min_proba:
                results.append({"token": token_str, "label": label, "score": round(score, 4)})

        return results

    # -- Private --

    def _predict_cases(self, sentence: str, lemma: str | None, min_proba: float) -> list[dict]:
        if PLACEHOLDER not in sentence:
            msg = f"Sentence must contain a '{PLACEHOLDER}' placeholder marking the target word"
            raise ValueError(msg)

        fill_word = lemma if lemma is not None else self.tokenizer.mask_token
        filled_sentence = sentence.replace(PLACEHOLDER, fill_word, 1)

        start_pos = sentence.index(PLACEHOLDER)
        end_pos = start_pos + len(fill_word)

        tokenized = self.tokenizer(
            filled_sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            return_offsets_mapping=True,
        )
        offset_mapping = tokenized.pop("offset_mapping")[0]

        if lemma is None:
            mask_token_id = self.tokenizer.mask_token_id
            target_tokens = [
                i
                for i, token_id in enumerate(tokenized["input_ids"][0])
                if token_id == mask_token_id
            ]
        else:
            target_tokens = [
                i
                for i, (tok_start, tok_end) in enumerate(offset_mapping)
                if _overlaps(tok_start.item(), tok_end.item(), start_pos, end_pos)
            ]

        if not target_tokens:
            return []

        with torch.no_grad():
            logits = self.model(**tokenized).logits
        probas = torch.nn.functional.softmax(logits, dim=2)

        aggregated_scores = probas[0, target_tokens, :].mean(dim=0)

        predictions = [
            {"label": self.config.id2label[i], "score": round(score.item(), 4)}
            for i, score in enumerate(aggregated_scores)
            if score > min_proba
        ]
        predictions.sort(key=lambda d: d["score"], reverse=True)
        return predictions


def _add_inflections(result: dict, sentence: str, lemma: str, inflector: Inflector) -> None:
    for pred in result["predictions"]:
        inflected = inflector.inflect(lemma, pred["label"])
        if inflected is not None:
            pred["inflected"] = inflected

    top = result["predictions"][0] if result["predictions"] else None
    if top and "inflected" in top:
        result["sentence"] = sentence.replace(PLACEHOLDER, top["inflected"], 1)


def _overlaps(start_1: int, end_1: int, start_2: int, end_2: int) -> bool:
    return max(start_1, start_2) < min(end_1, end_2)
