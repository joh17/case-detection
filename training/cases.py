from enum import Enum


class GrammaticalCase(Enum):
    NOMINATIVE = "nominative"
    GENITIVE = "genitive"
    PARTITIVE = "partitive"
    ACCUSATIVE = "accusative"
    INESSIVE = "inessive"
    ELATIVE = "elative"
    ILLATIVE = "illative"
    ADESSIVE = "adessive"
    ABLATIVE = "ablative"
    ALLATIVE = "allative"
    ESSIVE = "essive"
    TRANSLATIVE = "translative"
    ABESSIVE = "abessive"
    INSTRUCTIVE = "instructive"
    COMITATIVE = "comitative"


# Grammatical case name in Finnish, as returned by the Pyvoikko library.
FI_CLASS_TO_GRAMMATICAL_CASE = {
    "nimento": GrammaticalCase.NOMINATIVE,
    "omanto": GrammaticalCase.GENITIVE,
    "osanto": GrammaticalCase.PARTITIVE,
    "sisaolento": GrammaticalCase.INESSIVE,
    "sisaeronto": GrammaticalCase.ELATIVE,
    "sisatulento": GrammaticalCase.ILLATIVE,
    "ulkoolento": GrammaticalCase.ADESSIVE,
    "ulkoeronto": GrammaticalCase.ABLATIVE,
    "ulkotulento": GrammaticalCase.ALLATIVE,
    "olento": GrammaticalCase.ESSIVE,
    "tulento": GrammaticalCase.TRANSLATIVE,
    "vajanto": GrammaticalCase.ABESSIVE,
    "keinonto": GrammaticalCase.INSTRUCTIVE,
}

GRAMMATICAL_CASE_TO_3_LETTER_CODE = {
    GrammaticalCase.NOMINATIVE: "Nom",
    GrammaticalCase.GENITIVE: "Gen",
    GrammaticalCase.PARTITIVE: "Par",
    GrammaticalCase.ACCUSATIVE: "Acc",
    GrammaticalCase.INESSIVE: "Ine",
    GrammaticalCase.ELATIVE: "Ela",
    GrammaticalCase.ILLATIVE: "Ill",
    GrammaticalCase.ADESSIVE: "Ade",
    GrammaticalCase.ABLATIVE: "Abl",
    GrammaticalCase.ALLATIVE: "All",
    GrammaticalCase.ESSIVE: "Ess",
    GrammaticalCase.TRANSLATIVE: "Tra",
    GrammaticalCase.ABESSIVE: "Abe",
    GrammaticalCase.INSTRUCTIVE: "Ins",
    GrammaticalCase.COMITATIVE: "Com",
}

UNDEFINED_CASE = "_"

GRAMMATICAL_CASE_CLASSES = [UNDEFINED_CASE] + list(GRAMMATICAL_CASE_TO_3_LETTER_CODE.values())
CLASS_ID_TO_CASE = dict([(case_id, case_name) for case_id, case_name in enumerate(GRAMMATICAL_CASE_CLASSES)])
CASE_TO_CLASS_ID = dict([(case_name, case_id) for case_id, case_name in CLASS_ID_TO_CASE.items()])
CASE_TO_CLASS_ID["-"] = 0
