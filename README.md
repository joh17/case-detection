# Grammatical Case Prediction for Template Filling

In case-marking languages like Finnish, filling a template placeholder isn't as simple as dropping in a word. The word ending changes depending on its grammatical role in the sentence. For example, the English template

> Your trip to **[CITY]** is starting

works with any city name. But the Finnish equivalent requires the city name to be inflected to the right case:

> Matkasi **Helsinkiin** alkaa

where *Helsinki* became *Helsinkiin* (illative case, indicating movement towards).

This repository provides:
- **Inference code** (`case_detection/`) — predict which grammatical case a placeholder needs, given a sentence template. Works with pre-trained models from Hugging Face, no training required.
- **Training pipeline** (`training/`) — everything to train your own model: dataset generation from raw text, preprocessing, fine-tuning, and evaluation.

Currently trained and evaluated on Finnish. The pipeline generalises to other case-marking languages — it only requires raw text and a morphological analyzer.

Pre-trained models on Hugging Face:
- [johl/fi-case-detection-finbert](https://huggingface.co/johl/fi-case-detection-finbert) (FinBERT-based)
- [johl/fi-case-detection-xlm-roberta](https://huggingface.co/johl/fi-case-detection-xlm-roberta) (XLM-RoBERTa-based)

Research publication: https://aclanthology.org/2026.sigtyp-main.1/

## Quick start

```bash
git clone https://github.com/johl/grammatical-case-detection.git
cd grammatical-case-detection
pip install torch transformers
```

Mark the target word position with `{}` in your sentence:

```python
from case_detection import CaseDetector

detector = CaseDetector()

# Predict case from context alone (placeholder becomes [MASK])
detector.predict("Matkasi {} alkaa.")
# {'predictions': [{'label': 'Ill', 'score': 0.48}, {'label': 'All', 'score': 0.32}]}

# Predict case knowing the base form of the word
detector.predict("Matkasi {} alkaa.", "Tampere")
# {'predictions': [{'label': 'All', 'score': 0.87}]}
```

### With inflection

If you also want to inflect the word to the predicted case (requires `pyhfst`):

```bash
pip install pyhfst
```

```python
detector = CaseDetector(inflect=True)

detector.predict("Matkasi {} alkaa.", "Tampere")
# {'predictions': [{'label': 'All', 'score': 0.87, 'inflected': 'Tampereelle'}],
#  'sentence': 'Matkasi Tampereelle alkaa.'}
```


### Predicted case labels

The model outputs 3-letter case labels:

| Label | Case name | Example | Rough meaning |
| :--- | :--- | :--- | :--- |
| Nom | Nominative | talo | house |
| Gen | Genitive | talon | of the house |
| Part | Partitive | taloa | (some) house |
| Acc | Accusative | talo(n) | the house (object) |
| Ine | Inessive | talossa | in the house |
| Ela | Elative | talosta | out of the house |
| Ill | Illative | taloon | into the house |
| Ade | Adessive | talolla | at the house |
| Abl | Ablative | talolta | from the house |
| All | Allative | talolle | to the house |
| Ess | Essive | talona | as a house |
| Tra | Translative | taloksi | into a house |
| Abe | Abessive | talotta | without a house |
| Ins | Instructive | taloin | by means of houses |
| Com | Comitative | taloineen | with houses |

## Training

### 1. Preprocess your corpus

The first step is to annotate your text corpus with the base form and grammatical case of each word. There are two scripts depending on your starting data:

**From raw text** — use `extract_dataset_from_raw_text.py` if you have a plain text file. It splits the text into sentences and words, runs each word through the HFST morphological analyzer to extract its base form and case, and writes the result to a tab-separated file.

```bash
python -m training.extract_dataset_from_raw_text \
    --input_file corpus.txt \
    --output_file data/processed.txt
```

**From Yle VRT** — use `extract_dataset_from_yle_vrt.py` if you have corpus data in VRT format (e.g. Ylenews data from [Kielipankki](https://www.kielipankki.fi/aineistot/)). It parses the existing word-level annotations for nouns from the VRT files.

```bash
python -m training.extract_dataset_from_yle_vrt \
    --input_folder data/vrt/ \
    --output_folder data/processed/
```

The output is a text file where each line represents one sentence as three tab-separated columns: surface words, base forms, and case labels — all space-separated.

```
Koira juoksi puistossa	koira juosta puisto	Nom _ Ine
```

### 2. Build a tokenized HuggingFace dataset

Convert the preprocessed text file into a tokenized HuggingFace dataset ready for training:

```bash
python -m training.build_train_dataset \
    --input_processed_text_file data/processed.txt \
    --output_dataset_folder data/dataset/ \
    --model_name TurkuNLP/bert-base-finnish-uncased-v1
```

This tokenizes each sentence and applies the training masking strategy: for each sentence, a random subset of cased words is selected, and each is either replaced with `[MASK]` (slot-only) or with its base form (lemma-conditioned). This teaches the model to predict case from context in both inference modes.

### 3. Fine-tune a model

Fine-tune a token classification model on the prepared dataset using HuggingFace Trainer:

```bash
python -m training.train \
    --dataset_path data/dataset/ \
    --model_name TurkuNLP/bert-base-finnish-uncased-v1 \
    --output_dir ./output/
```

Any HuggingFace encoder model can be used as a base.

Once trained, load your model checkpoint with `CaseDetector` as shown in the Quick start section above.
