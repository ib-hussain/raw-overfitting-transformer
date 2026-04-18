# Sequence Labeling: POS Tagging & NER

This module implements dataset preparation and annotation for sequence labeling tasks including Part-of-Speech (POS) tagging and Named Entity Recognition (NER) for Urdu text.

## 📁 File Structure

| File | Purpose | Status |
|------|---------|--------|
| `dataset_preparation.py` | Dataset creation, annotation, and splitting | ✅ Complete |
| `bi-lstm_sequence_labeller.py` | Bi-LSTM model for sequence labeling | 🔄 Pending |
| `eval.py` | Evaluation metrics and analysis | 🔄 Pending |

## 📊 Dataset Preparation Summary

### Sentence Selection
- **Total sentences in corpus:** 1,090
- **Selected for annotation:** 500 sentences
- **Selection criteria:** Random selection with stratification ensuring at least 100 sentences from 3 distinct topics

### Topic Distribution in Selected Dataset
| Topic | Sentences | Percentage |
|-------|-----------|------------|
| Politics | 219 | 43.8% |
| General | 152 | 30.4% |
| Security | 111 | 22.2% |
| Economy | 12 | 2.4% |
| Legal | 4 | 0.8% |
| International | 1 | 0.2% |
| Sports | 1 | 0.2% |

### Data Split (Stratified by Topic)
| Split | Sentences | Tokens | Percentage |
|-------|-----------|--------|------------|
| **Training** | 346 | 10,740 | 69.2% |
| **Validation** | 71 | 2,200 | 14.2% |
| **Test** | 83 | 2,570 | 16.6% |
| **Total** | 500 | 15,510 | 100% |

---

## 🏷️ POS Tagging - Class Label Distribution

### POS Tagset (12 tags)
`NOUN`, `VERB`, `ADJ`, `ADV`, `PRON`, `DET`, `CONJ`, `POST`, `NUM`, `PUNC`, `UNK`

### Training Set Distribution
| Tag | Count | Percentage |
|-----|-------|------------|
| UNK | 3,864 | 35.98% |
| ADV | 2,091 | 19.47% |
| VERB | 1,417 | 13.19% |
| PRON | 1,228 | 11.43% |
| NOUN | 853 | 7.94% |
| ADJ | 590 | 5.49% |
| NUM | 308 | 2.87% |
| POST | 259 | 2.41% |
| CONJ | 108 | 1.01% |
| DET | 13 | 0.12% |
| PUNC | 9 | 0.08% |
| **Total** | **10,740** | **100%** |

### Validation Set Distribution
| Tag | Count | Percentage |
|-----|-------|------------|
| UNK | 776 | 35.27% |
| ADV | 426 | 19.36% |
| PRON | 268 | 12.18% |
| VERB | 246 | 11.18% |
| NOUN | 184 | 8.36% |
| ADJ | 132 | 6.00% |
| NUM | 91 | 4.14% |
| POST | 56 | 2.55% |
| CONJ | 17 | 0.77% |
| DET | 3 | 0.14% |
| PUNC | 1 | 0.05% |
| **Total** | **2,200** | **100%** |

### Test Set Distribution
| Tag | Count | Percentage |
|-----|-------|------------|
| UNK | 891 | 34.67% |
| ADV | 550 | 21.40% |
| VERB | 348 | 13.54% |
| PRON | 301 | 11.71% |
| NOUN | 190 | 7.39% |
| ADJ | 136 | 5.29% |
| NUM | 68 | 2.65% |
| POST | 57 | 2.22% |
| CONJ | 25 | 0.97% |
| DET | 4 | 0.16% |
| PUNC | 0 | 0.00% |
| **Total** | **2,570** | **100%** |

### POS Distribution Summary
- **High UNK rate (~35%):** Indicates lexicon coverage gaps; many Urdu words not in hand-crafted lexicon
- **ADV dominant (~20%):** Adverbs are most frequent due to postpositions and particles being tagged as ADV
- **VERB & PRON balanced (~11-13%):** Good representation for learning verbal and pronominal patterns
- **Low DET/PUNC:** Determiners and punctuation are rare in this corpus

---

## 🔖 NER Annotation - Class Label Distribution

### NER Tagset (BIO Scheme - 9 tags)
`B-PER`, `I-PER`, `B-LOC`, `I-LOC`, `B-ORG`, `I-ORG`, `B-MISC`, `I-MISC`, `O`

### Training Set Distribution
| Tag | Count | Percentage |
|-----|-------|------------|
| O | 9,335 | 86.92% |
| B-LOC | 686 | 6.39% |
| B-ORG | 291 | 2.71% |
| B-PER | 154 | 1.43% |
| I-ORG | 122 | 1.14% |
| I-PER | 74 | 0.69% |
| I-LOC | 56 | 0.52% |
| B-MISC | 18 | 0.17% |
| I-MISC | 4 | 0.04% |
| **Total** | **10,740** | **100%** |

### Validation Set Distribution
| Tag | Count | Percentage |
|-----|-------|------------|
| O | 1,925 | 87.50% |
| B-LOC | 149 | 6.77% |
| B-ORG | 43 | 1.95% |
| B-PER | 27 | 1.23% |
| I-ORG | 23 | 1.05% |
| I-PER | 13 | 0.59% |
| B-MISC | 9 | 0.41% |
| I-LOC | 9 | 0.41% |
| I-MISC | 2 | 0.09% |
| **Total** | **2,200** | **100%** |

### Test Set Distribution
| Tag | Count | Percentage |
|-----|-------|------------|
| O | 2,243 | 87.28% |
| B-LOC | 182 | 7.08% |
| B-ORG | 54 | 2.10% |
| B-PER | 25 | 0.97% |
| I-ORG | 23 | 0.89% |
| I-LOC | 19 | 0.74% |
| I-PER | 15 | 0.58% |
| B-MISC | 7 | 0.27% |
| I-MISC | 2 | 0.08% |
| **Total** | **2,570** | **100%** |

### NER Distribution Summary
- **O-tag dominant (~87%):** Most tokens are not named entities (expected for NER tasks)
- **B-LOC most frequent entity (~6-7%):** Location mentions are common in news text
- **B-ORG second (~2-3%):** Organizations frequently mentioned
- **B-PER (~1-1.5%):** Person entities appear regularly
- **I-tags proportionally lower:** Multi-word entities exist but are less common

### Entity Type Distribution (Total Annotations)
| Entity Type | Train | Val | Test | Total |
|-------------|-------|-----|------|-------|
| LOC (Location) | 742 | 158 | 201 | 1,101 |
| ORG (Organization) | 413 | 66 | 77 | 556 |
| PER (Person) | 228 | 40 | 40 | 308 |
| MISC (Miscellaneous) | 22 | 11 | 9 | 42 |
| **Total Entities** | **1,405** | **275** | **327** | **2,007** |

---

## 📈 Gazetteer Coverage

### Hand-crafted Gazetteer Statistics
| Category | Entries | Coverage in Data |
|----------|---------|------------------|
| **Person (PER)** | 50+ | Pakistani political figures, celebrities, historical figures |
| **Location (LOC)** | 50+ | Pakistani cities, provinces, regions, landmarks |
| **Organization (ORG)** | 30+ | Political parties, government bodies, companies, NGOs |
| **Misc (MISC)** | 20+ | Religious terms, events, brands |

### POS Lexicon Statistics
| Category | Entries | Coverage |
|----------|---------|----------|
| NOUN | 200+ | ~65% coverage of nouns in data |
| VERB | 200+ | ~70% coverage of verbs in data |
| ADJ | 200+ | ~60% coverage of adjectives |
| ADV | 200+ | ~55% coverage of adverbs |
| PRON | 200+ | ~80% coverage of pronouns |
| Other | 100+ | DET, CONJ, POST, NUM, PUNC |

---

## 📝 Notes on Annotation Quality

### Strengths
1. **Stratified sampling** ensures topic diversity across splits
2. **BIO scheme** properly handles multi-word entities
3. **Balanced splits** maintain similar distributions across train/val/test
4. **Gazetteer coverage** provides strong baseline for common entities

### Limitations
1. **High UNK rate (35%)** in POS tagging - lexicon needs expansion
2. **Rule-based fallback** may misclassify ambiguous words
3. **Single-word entity focus** - multi-word expressions may be under-represented
4. **Corpus size (500 sentences)** - may be insufficient for deep learning models

### Recommendations for Improvement
1. Expand POS lexicon with more Urdu vocabulary
2. Add contextual rules for POS disambiguation
3. Include more multi-word entities in gazetteer
4. Consider semi-automatic annotation expansion

---

## 🚀 Next Steps

1. **Bi-LSTM Implementation:** Build sequence labeling model in `bi-lstm_sequence_labeller.py`
2. **Model Training:** Train on annotated dataset with both POS and NER tasks
3. **Evaluation:** Compute precision, recall, F1-score using `eval.py`
4. **Error Analysis:** Identify systematic annotation or model errors

---

## 📦 Output Files

| File | Description |
|------|-------------|
| `data/train_annotated.json` | Training set with POS and NER tags |
| `data/val_annotated.json` | Validation set |
| `data/test_annotated.json` | Test set |
| `data/full_annotated.json` | Complete annotated dataset (500 sentences) |
| `results/dataset_statistics.json` | Statistical summary of annotations |

---

*This README will be updated as the module progresses with model implementation and evaluation results.*