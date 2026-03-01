from transformers import AutoTokenizer
import pytest


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-uncased-v1")