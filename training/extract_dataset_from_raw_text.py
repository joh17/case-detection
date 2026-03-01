"""Generate training data from raw Finnish text using HFST morphological analysis.

Reads a plain text file, splits it into sentences and words, analyzes each word
with an HFST analyzer to extract its base form and case, and writes the result
as a tab-separated file.

Output format (one line per sentence, tab-separated columns):
    words (space-separated) <TAB> baseforms (space-separated) <TAB> cases (space-separated)

Example output:
    Koira juoksi puistossa	koira juosta puisto	Nom _ Ine
    Annoin kirjan pojalle	antaa kirja poika	_ Gen All

Usage:
    python -m training.extract_dataset_from_raw_text --input_file input.txt --output_file output.vrt
"""

import re
from pathlib import Path

import pyhfst
from fire import Fire

from training.cases import GRAMMATICAL_CASE_TO_3_LETTER_CODE

CASES_3_LETTER_CODES = set(GRAMMATICAL_CASE_TO_3_LETTER_CODE.values())


def load_analyzer(model_path: str) -> pyhfst.Hfst:
    """Load an HFST morphological analyzer from disk."""
    analyzer_stream = pyhfst.HfstInputStream(model_path)
    return analyzer_stream.read()


def split_into_sentences(text: str) -> list[str]:
    """Split a text into sentences, using '.' as separator."""
    return re.split(r"[.!?]", text)


def split_into_words(text: str) -> list[str]:
    """Split a sentence into a list of words."""
    return re.findall(r"\b\w+\b", text.strip())


def get_word_baseform_case(
    word: str, analyzer: pyhfst.Hfst, *, accept_adjectives: bool = True,
) -> tuple[str, str]:
    """Return (baseform, 3-letter case code) for a word.

    Uses the HFST analyzer which returns forms like "katu+N+Sg+Gen".
    The case is the last segment after the final "+".
    If the case is not clearly defined, falls back to the word itself and '_'.
    """
    word_forms = analyzer.lookup(word)
    if not word_forms:
        return word, "_"

    candidates = []
    for form, _ in word_forms:
        is_form_in_scope = "+N+" in form
        if accept_adjectives:
            is_form_in_scope = is_form_in_scope or "+A+" in form

        if not is_form_in_scope:
            continue

        case_code = form.rsplit("+", 1)[-1]
        if case_code not in CASES_3_LETTER_CODES:
            continue

        parts = form.split("#")
        baseform = "".join(part.split("+")[0] for part in parts)
        candidates.append((form, baseform, case_code))

    # If the word's case cannot clearly be determined, leave it empty
    if len({case for _, _, case in candidates}) != 1:
        return word, "_"

    # Pick the simplest (shortest) analysis
    _, baseform, case_code = min(candidates, key=lambda c: len(c[0]))

    return baseform, case_code


def extract_dataset(
    input_file: str,
    output_file: str,
    analyzer_model: str = "models/analyser-gt-norm.hfstol",
) -> None:
    """Read raw text from input_file, analyze it with HFST, and write VRT to output_file."""
    analyzer = load_analyzer(analyzer_model)

    with Path(input_file).open("r") as f_in:
        text = f_in.read()

    with Path(output_file).open("w") as f_out:
        sentences = split_into_sentences(text)
        for sentence in sentences:
            words = split_into_words(sentence)
            baseforms = []
            word_cases = []
            for word in words:
                baseform, case = get_word_baseform_case(word, analyzer)
                baseforms.append(baseform)
                word_cases.append(case)

            # Skip sentences with only undefined cases
            if set(word_cases) - {"_"}:
                original_sentence = " ".join(words)
                baseform_sentence = " ".join(baseforms)
                cases_str = " ".join(word_cases)
                f_out.write(f"{original_sentence}\t{baseform_sentence}\t{cases_str}\n")


if __name__ == "__main__":
    Fire(extract_dataset)
