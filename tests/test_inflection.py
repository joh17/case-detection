import pytest

from case_detection.inflect import Inflector


@pytest.fixture(scope="module")
def inflector():
    return Inflector()


testdata = [
    ("Helsinki", "Gen", "Helsingin"),
    ("Tampere", "All", "Tampereelle"),
    ("koira", "Par", "koiraa"),
    ("talo", "Ine", "talossa"),
    ("auto", "Ela", "autosta"),
]


@pytest.mark.parametrize("word,case,expected", testdata)
def test_inflection(inflector, word: str, case: str, expected: str):
    assert inflector.inflect(word, case) == expected
