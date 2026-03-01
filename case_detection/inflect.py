"""Morphological inflection using HFST transducers.

Inflects a lemma (base form) to a target grammatical case using the
analyzer and generator models from the UralicNLP project.
"""

from pathlib import Path

import pyhfst

DEFAULT_ANALYZER = Path(__file__).parent.parent / "models" / "analyser-gt-norm.hfstol"
DEFAULT_GENERATOR = Path(__file__).parent.parent / "models" / "generator-gt-norm.hfstol"


class Inflector:
    """Inflects words to a target grammatical case using HFST transducers.

    Usage:
        inflector = Inflector()
        inflector.inflect("Tampere", "All")
        # "Tampereelle"
    """

    def __init__(
        self,
        analyzer_path: str | Path = DEFAULT_ANALYZER,
        generator_path: str | Path = DEFAULT_GENERATOR,
    ):
        self.analyzer = pyhfst.HfstInputStream(str(analyzer_path)).read()
        self.generator = pyhfst.HfstInputStream(str(generator_path)).read()

    def inflect(self, lemma: str, case_label: str) -> str | None:
        """Inflect a lemma to the given grammatical case.

        Args:
            lemma: The base form (nominative) of the word.
            case_label: A 3-letter case code (e.g. "All", "Ill", "Gen").

        Returns:
            The inflected word form, or None if inflection failed.
        """
        analyses = self.analyzer.lookup(lemma)
        if not analyses:
            return None

        analysis = _pick_best_analysis(analyses)
        if analysis is None:
            return None

        parts = analysis.split("+")
        parts[-1] = case_label
        generated = self.generator.lookup("+".join(parts))
        if not generated:
            return None

        return generated[0][0]


def _pick_best_analysis(analyses: list[list]) -> str | None:
    """Pick the most likely morphological analysis.

    Prefers singular forms, then shorter analyses (fewer morphological tags).
    """
    candidate_analyses = [a[0] for a in analyses]

    # Prefer analyses containing nominative
    nom_analyses = [a for a in candidate_analyses if "+Nom" in a]
    candidate_analyses = nom_analyses or candidate_analyses

    # Prefer analyses containing singular nouns
    sg_analyses = [a for a in candidate_analyses if "+Sg+" in a]
    candidate_analyses = sg_analyses or candidate_analyses

    # Among those, prefer the shortest (fewest tags = least complex)
    return min(candidate_analyses, key=lambda a: len(a.split("+")))
