"""Extract training data from Yle news VRT files using HFST morphological analysis.

Processing script written for the Ylenews dataset from https://www.kielipankki.fi/aineistot/
Reads VRT files from an input folder, parses sentence boundaries and word-level
annotations, extracts case information, and writes processed sentences as
tab-separated output files.

Output format (one line per sentence, tab-separated columns):
    words (space-separated) <TAB> baseforms (space-separated) <TAB> cases (space-separated)

Usage:
    python -m training.extract_dataset_from_yle_vrt --input_folder data/vrt/ --output_folder data/processed/
"""

import logging
from pathlib import Path

import pyhfst
from fire import Fire

from training.cases import GRAMMATICAL_CASE_TO_3_LETTER_CODE

CASES_3_LETTER_CODES = set(GRAMMATICAL_CASE_TO_3_LETTER_CODE.values())
IGNORED_CASES = {"Lat", "Prl", "Dis"}

logger = logging.getLogger(__name__)


def load_analyzer(model_path: str) -> pyhfst.Hfst:
    """Load an HFST morphological analyzer from disk."""
    analyzer_stream = pyhfst.HfstInputStream(model_path)
    return analyzer_stream.read()


def is_noun(word: str, analyzer: pyhfst.Hfst, *, accept_adjectives: bool = True) -> bool | None:
    """Check whether a word is a noun according to the HFST analyzer.

    Returns None if the word is not recognized.
    """
    word_forms = analyzer.lookup(word)
    if not word_forms:
        return None

    matches_condition = any("+N+" in form for form, _ in word_forms)

    if accept_adjectives and not matches_condition:
        matches_condition = any("+A+" in form for form, _ in word_forms)

    return matches_condition


def extract_case_from_word_forms(word: str, word_forms: str) -> str | None:
    """Extract the 3-letter case code from a VRT word-forms field.

    Returns the case code, '_' if no case is found, or None if an
    unrecognized case is encountered (indicating the word should be skipped).
    """
    if "CASE_" not in word_forms:
        return "_"

    case_start = word_forms.index("CASE_") + len("CASE_")
    case_code = word_forms[case_start:][:3]

    if case_code in CASES_3_LETTER_CODES:
        return case_code

    if case_code not in IGNORED_CASES:
        logger.warning("Unrecognized case %r in word %r", case_code, word)
        return None

    return "_"


def process_vrt_file(
    input_path: Path,
    output_path: Path,
    analyzer: pyhfst.Hfst,
) -> None:
    """Process a single VRT file and write the extracted dataset."""
    with input_path.open("r") as f, output_path.open("w") as f_out:
        current_sentence_words = []
        current_sentence_base_forms = []
        current_sentence_cases = []
        currently_in_sentence = False

        for line in f:
            stripped = line.strip()

            if stripped.startswith("<sentence"):
                currently_in_sentence = True
            elif stripped.startswith("</sentence>"):
                currently_in_sentence = False
                if set(current_sentence_cases) != {"_"}:
                    words = " ".join(current_sentence_words)
                    baseforms = " ".join(current_sentence_base_forms)
                    word_cases = " ".join(current_sentence_cases)
                    f_out.write(f"{words}\t{baseforms}\t{word_cases}\n")
                current_sentence_words = []
                current_sentence_base_forms = []
                current_sentence_cases = []
            elif currently_in_sentence:
                columns = stripped.split("\t")
                word = columns[0]
                base_form = columns[2]
                word_forms = columns[5]

                case = extract_case_from_word_forms(word, word_forms)
                if case is None:
                    break

                if is_noun(word, analyzer) is False:
                    case = "_"

                current_sentence_words.append(word)
                current_sentence_base_forms.append(base_form)
                current_sentence_cases.append(case)


def extract_dataset(
    input_folder: str,
    output_folder: str,
    analyzer_model: str = "models/analyser-gt-norm.hfstol",
) -> None:
    """Extract case-annotated dataset from Yle VRT files."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    analyzer = load_analyzer(analyzer_model)

    input_path = Path(input_folder)
    vrt_files = input_path.glob("*.vrt")

    for vrt_file in vrt_files:
        logger.info("Processing %s", vrt_file.name)
        process_vrt_file(vrt_file, output_path / vrt_file.name, analyzer)

    logger.info("Done")


if __name__ == "__main__":
    Fire(extract_dataset)
