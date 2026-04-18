# Sequence Labeling: POS Tagging & NER

This module implements a complete sequence labeling pipeline for Urdu text, including dataset preparation, annotation, BiLSTM-based modeling, and comprehensive evaluation for both Part-of-Speech (POS) tagging and Named Entity Recognition (NER) tasks.

## 📁 File Structure

| File | Purpose | Status |
|------|---------|--------|
| `dataset_preparation.py` | Dataset creation, annotation, and splitting | ✅ Complete |
| `bi-lstm_sequence_labeller.py` | Bi-LSTM model for sequence labeling | ✅ Complete |
| `eval.py` | Evaluation metrics and analysis | ✅ Complete |
| `train_ablation.py` | Ablation study training and evaluation | ✅ Complete |

## 🎯 Module Overview

This module implements a 2-layer bidirectional LSTM sequence labeler with:
- **Word2Vec initialization** from Part 1 (Condition C3)
- **Frozen and fine-tuned** embedding modes
- **Dropout regularization** (p=0.5) between LSTM layers
- **CRF decoding** for NER with Viterbi algorithm
- **Linear classifier** for POS with cross-entropy loss
- **Checkpointing** and early stopping
- **Comprehensive evaluation** with confusion matrices and error analysis

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 100 |
| Hidden Dimension | 128 |
| LSTM Layers | 2 |
| Dropout | 0.5 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Weight Decay | 1e-4 |
| Early Stopping Patience | 5 epochs |
| Optimizer | Adam |

---

## 📊 Dataset Preparation Summary

### Sentence Selection
- **Total sentences in corpus:** 1,090
- **Selected for annotation:** 500 sentences
- **Selection criteria:** Random selection with stratification ensuring at least 100 sentences from 3 distinct topics (Politics, General, Security)

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

### Distribution Across Splits

| Tag | Train | Train% | Val | Val% | Test | Test% |
|-----|-------|--------|-----|------|------|-------|
| UNK | 3,864 | 35.98% | 776 | 35.27% | 891 | 34.67% |
| ADV | 2,091 | 19.47% | 426 | 19.36% | 550 | 21.40% |
| VERB | 1,417 | 13.19% | 246 | 11.18% | 348 | 13.54% |
| PRON | 1,228 | 11.43% | 268 | 12.18% | 301 | 11.71% |
| NOUN | 853 | 7.94% | 184 | 8.36% | 190 | 7.39% |
| ADJ | 590 | 5.49% | 132 | 6.00% | 136 | 5.29% |
| NUM | 308 | 2.87% | 91 | 4.14% | 68 | 2.65% |
| POST | 259 | 2.41% | 56 | 2.55% | 57 | 2.22% |
| CONJ | 108 | 1.01% | 17 | 0.77% | 25 | 0.97% |
| DET | 13 | 0.12% | 3 | 0.14% | 4 | 0.16% |
| PUNC | 9 | 0.08% | 1 | 0.05% | 0 | 0.00% |
| **Total** | **10,740** | **100%** | **2,200** | **100%** | **2,570** | **100%** |

### POS Distribution Summary
- **High UNK rate (~35%):** Indicates lexicon coverage gaps; many Urdu words not in hand-crafted lexicon
- **ADV dominant (~20%):** Adverbs are most frequent due to postpositions and particles being tagged as ADV
- **VERB & PRON balanced (~11-13%):** Good representation for learning verbal and pronominal patterns

---

## 🔖 NER Annotation - Class Label Distribution

### NER Tagset (BIO Scheme - 9 tags)
`B-PER`, `I-PER`, `B-LOC`, `I-LOC`, `B-ORG`, `I-ORG`, `B-MISC`, `I-MISC`, `O`

### Distribution Across Splits

| Tag | Train | Train% | Val | Val% | Test | Test% |
|-----|-------|--------|-----|------|------|-------|
| O | 9,335 | 86.92% | 1,925 | 87.50% | 2,243 | 87.28% |
| B-LOC | 686 | 6.39% | 149 | 6.77% | 182 | 7.08% |
| B-ORG | 291 | 2.71% | 43 | 1.95% | 54 | 2.10% |
| B-PER | 154 | 1.43% | 27 | 1.23% | 25 | 0.97% |
| I-ORG | 122 | 1.14% | 23 | 1.05% | 23 | 0.89% |
| I-PER | 74 | 0.69% | 13 | 0.59% | 15 | 0.58% |
| I-LOC | 56 | 0.52% | 9 | 0.41% | 19 | 0.74% |
| B-MISC | 18 | 0.17% | 9 | 0.41% | 7 | 0.27% |
| I-MISC | 4 | 0.04% | 2 | 0.09% | 2 | 0.08% |

### Entity Type Distribution (Total Annotations)
| Entity Type | Train | Val | Test | Total |
|-------------|-------|-----|------|-------|
| LOC (Location) | 742 | 158 | 201 | 1,101 |
| ORG (Organization) | 413 | 66 | 77 | 556 |
| PER (Person) | 228 | 40 | 40 | 308 |
| MISC (Miscellaneous) | 22 | 11 | 9 | 42 |
| **Total Entities** | **1,405** | **275** | **327** | **2,007** |

### NER Distribution Summary
- **O-tag dominant (~87%):** Most tokens are not named entities (expected for NER tasks)
- **B-LOC most frequent entity (~6-7%):** Location mentions are common in news text
- **B-ORG second (~2-3%):** Organizations frequently mentioned
- **B-PER (~1-1.5%):** Person entities appear regularly

---

## 📈 5.1 POS Tagging Results

### Test Set Performance

| Mode | Token Accuracy | Macro F1 | Weighted F1 |
|------|---------------|----------|-------------|
| **Frozen Embeddings** | 72.96% | 0.6766 | 0.7321 |
| **Fine-tuned Embeddings** | **83.81%** | **0.9244** | **0.8442** |

### Frozen vs Fine-tuned Comparison

| Metric | Frozen | Fine-tuned | Improvement |
|--------|--------|------------|-------------|
| Token Accuracy | 72.96% | 83.81% | **+10.85%** |
| Macro F1 | 0.6766 | 0.9244 | **+0.2478** |
| Weighted F1 | 0.7321 | 0.8442 | **+0.1121** |
| Epochs Trained | 5 | 8 | - |

**Finding:** Fine-tuning the Word2Vec embeddings provides substantial improvement across all metrics, with a 10.85% absolute gain in accuracy and 24.78% gain in macro F1.

### Per-Class F1 Scores (Fine-tuned)

| Tag | F1 Score |
|-----|----------|
| PRON | 0.9950 |
| POST | 0.9912 |
| CONJ | 0.9804 |
| NUM | 1.0000 |
| DET | 1.0000 |
| ADJ | 0.9297 |
| NOUN | 0.9244 |
| VERB | 0.9108 |
| UNK | 0.7809 |
| ADV | 0.7314 |
| PUNC | 0.0000 |

### Top 3 Most Confused Tag Pairs (Fine-tuned)

| Rank | True → Predicted | Count | Explanation |
|------|------------------|-------|-------------|
| 1 | **UNK → ADV** | 309 | Unknown words (often borrowed/foreign) incorrectly predicted as adverbs |
| 2 | **VERB → ADV** | 50 | Verbs with adverbial usage or participle forms confused |
| 3 | **NOUN → ADV** | 22 | Nouns used in adverbial phrases misclassified |

### Example Sentences for Confused Pairs

#### UNK → ADV (309 errors)
**Example 1:**
- **Sentence:** `<NUM> جنوری کو پاکستا وزیر اعظم شہباز شریف نے ڈیووس میں ٹرمپ کے بورڈ اف پیس کی سرکاری تقریب میں شرکت کی`
- **Error at:** `ڈیووس` (Davos - foreign location name)
- **True:** UNK | **Predicted:** ADV
- **Explanation:** Foreign word not in vocabulary, model defaults to ADV due to positional context.

**Example 2:**
- **Sentence:** `...فیلڈ مارشل سید عاصم منیر بھی موجود تھے`
- **Error at:** `عاصم` (personal name)
- **True:** UNK | **Predicted:** ADV
- **Explanation:** Proper noun not in training vocabulary, misclassified based on surrounding function words.

#### VERB → ADV (50 errors)
**Example 1:**
- **Sentence:** `ان میں سے کئی زخم اتنے گہرے تھے کہ پھیپھڑ اور دیگر اعضا کو شدید نقصان پہنچا`
- **Error at:** `گہرے` (deep)
- **True:** VERB | **Predicted:** ADV
- **Explanation:** Adjective/participle form used descriptively, model confused by adjectival usage.

**Example 2:**
- **Sentence:** `وکلا اور ان کے کلائینٹس سیکیورٹی رکاوٹ کے پاس سے تیزی سے گزر ہیں... اور کیفیٹریا کے ویٹرز گاہک کے انتظار میں کھڑے ہیں`
- **Error at:** `کھڑے` (standing)
- **True:** VERB | **Predicted:** ADV
- **Explanation:** Stative verb in continuous aspect confused with adverbial phrase.

#### NOUN → ADV (22 errors)
**Example 1:**
- **Sentence:** `سندھ اسمبلی کی روایت کے مطابق اجلاس کی کارروا سے قبل مرحومین کی مغفرت کے لیے دعا کی جا ہے`
- **Error at:** `روایت` (tradition)
- **True:** NOUN | **Predicted:** ADV
- **Explanation:** Noun in idiomatic phrase (`کے مطابق`) misclassified due to fixed expression pattern.

**Example 2:**
- **Sentence:** `لڑکا کا تعلق ایک زمیندار گھرانے سے تھا`
- **Error at:** `لڑکا` (boy)
- **True:** NOUN | **Predicted:** ADV
- **Explanation:** Sentence-initial noun with genitive marker confused by model.

### Confusion Matrix

Confusion matrices are saved in the `figures/` directory:
- `pos_confusion_matrix_frozen.png`
- `pos_confusion_matrix_fine-tuned.png`

---

## 🔖 5.2 NER Results

### Entity-Level Metrics (Fine-tuned with CRF)

| Entity Type | Precision | Recall | F1 | TP | FP | FN |
|-------------|-----------|--------|-----|----|----|-----|
| **PER** | 1.16% | 16.67% | 2.17% | 3 | 256 | 15 |
| **LOC** | 2.83% | 65.52% | 5.42% | 19 | 653 | 10 |
| **ORG** | 0.00% | 0.00% | 0.00% | 0 | 0 | 32 |
| **MISC** | 0.00% | 0.00% | 0.00% | 0 | 0 | 6 |
| **Overall** | 5.97% | 58.82% | 10.83% | 50 | 788 | 35 |

### Frozen vs Fine-tuned (with CRF)

| Mode | PER F1 | LOC F1 | ORG F1 | Overall F1 |
|------|--------|--------|--------|------------|
| Frozen | 2.17% | 0.00% | 3.97% | 11.07% |
| Fine-tuned | 2.17% | **5.42%** | 0.00% | 10.83% |

**Finding:** Fine-tuning improves location recognition but overall NER performance remains poor due to limited training data and class imbalance.

### CRF vs Softmax Comparison (Fine-tuned)

| Method | PER F1 | LOC F1 | ORG F1 | Overall F1 |
|--------|--------|--------|--------|------------|
| **With CRF** | **2.17%** | **5.42%** | 0.00% | **10.83%** |
| Without CRF | 0.00% | 0.00% | 0.00% | 0.00% |

**Finding:** The CRF layer is **essential** for NER; without it, the model fails to predict any entities (all predictions are 'O'). The structured decoding provided by CRF enables the model to learn valid BIO tag sequences.

### Error Analysis

#### False Positives (Top 5)

| # | Entity | Sentence Context | Explanation |
|---|--------|------------------|-------------|
| 1 | `وزیر` | `پاکستا وزیر اعظم شہباز شریف` | Common noun "minister" incorrectly tagged as entity; gazetteer missing multi-word context |
| 2 | `بھی` | `فیلڈ مارشل سید عاصم منیر بھی موجود` | Function word "also" adjacent to entity incorrectly included |
| 3 | `منیر` | `سید عاصم منیر بھی موجود تھے` | Partial name tagged; model fails to group "سید عاصم منیر" as single PER entity |
| 4 | `پیس` | `بورڈ اف پیس کی سرکاری تقریب` | Part of organization name "Board of Peace" incorrectly tagged as separate entity |
| 5 | `مارشل` | `فیلڈ مارشل سید عاصم` | Title "Marshal" tagged separately instead of as part of full name |

**Pattern:** Model struggles with multi-word entities and frequently predicts single tokens as entities without proper BIO consistency.

#### False Negatives (Top 5)

| # | Entity | Sentence Context | Explanation |
|---|--------|------------------|-------------|
| 1 | `اعظم شہباز شریف` | `پاکستا وزیر اعظم شہباز شریف` | Full person name missed; "وزیر اعظم" (Prime Minister) treated as title, not entity |
| 2 | `پاکستان کے` | `پاکستان کے فیلڈ مارشل` | Genitive phrase missed; model doesn't recognize possessive constructions |
| 3 | `سندھ` | `سندھ اسمبلی کی روایت` | Province name missed; gazetteer coverage issue |
| 4 | `ڈیووس` | `ڈیووس میں ٹرمپ کے بورڈ` | Foreign location (Davos) not in gazetteer |
| 5 | `ٹرمپ` | `ٹرمپ کے بورڈ اف پیس` | Foreign person name not in Pakistani-focused gazetteer |

**Pattern:** Model misses:
- Multi-word person names with titles
- Possessive/genitive constructions
- Out-of-gazetteer locations and foreign entities
- Entities requiring broader world knowledge

### NER Performance Limitations

The poor NER performance (overall F1 ~11%) can be attributed to:

1. **Limited training data:** Only 346 training sentences with ~1,405 entity annotations
2. **Severe class imbalance:** 87% of tokens are 'O' (non-entity)
3. **Gazetteer coverage gaps:** Foreign entities and less common Pakistani entities missing
4. **CRF loss instability:** Training loss values (~263,000) indicate numerical issues requiring further debugging
5. **Multi-word entity challenge:** BIO consistency difficult to learn with limited data

---

## 🔬 5.3 Ablation Study

The ablation study evaluates the contribution of each model component by training variants with specific features removed.

### Results Summary

| ID | Configuration | Accuracy | Macro F1 | Weighted F1 | Δ from Baseline |
|----|---------------|----------|----------|-------------|-----------------|
| **Baseline** | BiLSTM + Dropout + Pretrained | 83.81% | 0.9244 | 0.8442 | - |
| **A1** | Unidirectional LSTM | 27.12% | 0.0678 | 0.1959 | **-64.83%** |
| **A2** | No Dropout | 83.70% | 0.9216 | 0.8431 | -0.11% |
| **A3** | Random Embeddings | **91.01%** | 0.8641 | **0.9090** | **+6.48%** |
| **A4** | Softmax instead of CRF (NER) | - | - | 0.00% | -10.83% |

### Detailed Analysis

#### A1: Unidirectional LSTM (Value of Backward Context)

| Metric | Bidirectional | Unidirectional | Difference |
|--------|---------------|----------------|------------|
| Accuracy | 83.81% | 27.12% | **-56.69%** |
| Weighted F1 | 0.8442 | 0.1959 | **-0.6483** |

**Finding:** Bidirectional context is **critical** for sequence labeling. The unidirectional model fails to learn meaningful patterns, achieving only 27% accuracy (near random for 12-class task). This demonstrates that both left and right context are essential for POS disambiguation in Urdu.

#### A2: No Dropout (Effect of Regularization)

| Metric | With Dropout (0.5) | Without Dropout | Difference |
|--------|-------------------|-----------------|------------|
| Accuracy | 83.81% | 83.70% | -0.11% |
| Weighted F1 | 0.8442 | 0.8431 | -0.0011 |

**Finding:** Dropout provides a **marginal benefit** (+0.11% accuracy). The model without dropout converges faster (reaches best F1 at epoch 26 vs 48) but shows slight overfitting. Given the small dataset, dropout regularization is beneficial but not critical.

**Convergence Comparison:**
- **With Dropout:** Gradual improvement over 48 epochs
- **Without Dropout:** Rapid convergence, best F1 at epoch 26

#### A3: Random Embeddings (Contribution of Pretrained Word2Vec)

| Metric | Pretrained (C3) | Random | Difference |
|--------|-----------------|--------|------------|
| Accuracy | 83.81% | **91.01%** | **+7.20%** |
| Weighted F1 | 0.8442 | **0.9090** | **+0.0648** |

**Finding (Surprising):** Random embeddings **outperform** pretrained Word2Vec embeddings by 7.20% accuracy! This counter-intuitive result suggests:

1. **Domain mismatch:** The Word2Vec embeddings were trained on the same corpus (cleaned.txt) using skip-gram, but the representations may not capture POS-relevant features effectively
2. **Task-specific learning:** Random embeddings allow the model to learn POS-optimized representations from scratch
3. **Limited pretraining data:** The Word2Vec model was trained on only ~335K pairs from 210 documents, which may be insufficient for quality embeddings
4. **Fine-tuning limitations:** Even with fine-tuning, the pretrained embeddings may constrain the model to suboptimal representations

**Implication:** For small-domain POS tagging, training embeddings from scratch may be preferable to using weakly pretrained embeddings.

#### A4: Softmax vs CRF (NER Structured Decoding)

| Metric | With CRF | Without CRF |
|--------|----------|-------------|
| Overall F1 | 10.83% | **0.00%** |
| PER F1 | 2.17% | 0.00% |
| LOC F1 | 5.42% | 0.00% |
| ORG F1 | 0.00% | 0.00% |

**Finding:** The CRF layer is **absolutely essential** for NER. Without CRF:
- The model predicts 'O' for all tokens
- Zero entities detected
- F1 score of 0%

The CRF's structured decoding enforces valid BIO tag transitions (e.g., I-PER must follow B-PER), which is critical for learning entity boundaries with limited data.

### Ablation Study Conclusions

| Component | Importance | Key Insight |
|-----------|------------|-------------|
| **Bidirectional LSTM** | ⭐⭐⭐⭐⭐ Critical | Without it, performance collapses (27% accuracy) |
| **CRF Layer (NER)** | ⭐⭐⭐⭐⭐ Critical | Zero entities detected without structured decoding |
| **Pretrained Embeddings** | ⭐⭐ Situational | Random embeddings outperformed pretrained for this task |
| **Dropout** | ⭐ Marginal | Small benefit; model is not severely overfitting |

---

## 📊 Overall Results Summary

### POS Tagging Summary

| Mode | Accuracy | Macro F1 | Weighted F1 | Epochs |
|------|----------|----------|-------------|--------|
| Frozen | 72.96% | 0.6766 | 0.7321 | 5 |
| **Fine-tuned** | **83.81%** | **0.9244** | **0.8442** | 8 |

### NER Summary

| Mode | CRF | Overall F1 | PER F1 | LOC F1 | ORG F1 |
|------|-----|------------|--------|--------|--------|
| Frozen | ✓ | 11.07% | 2.17% | 0.00% | 3.97% |
| Fine-tuned | ✓ | **10.83%** | 2.17% | **5.42%** | 0.00% |
| Fine-tuned | ✗ | 0.00% | 0.00% | 0.00% | 0.00% |

### Ablation Summary (POS)

| Configuration | Accuracy | Weighted F1 |
|---------------|----------|-------------|
| Baseline (BiLSTM + Dropout + Pretrained) | 83.81% | 0.8442 |
| A1: Unidirectional LSTM | 27.12% | 0.1959 |
| A2: No Dropout | 83.70% | 0.8431 |
| A3: Random Embeddings | **91.01%** | **0.9090** |

---

## 📦 Output Files

### Data Files (`data/`)
| File | Description |
|------|-------------|
| `train_annotated.json` | Training set (346 sentences) |
| `val_annotated.json` | Validation set (71 sentences) |
| `test_annotated.json` | Test set (83 sentences) |
| `full_annotated.json` | Complete annotated dataset (500 sentences) |

### Results Files (`results/`)
| File | Description |
|------|-------------|
| `dataset_statistics.json` | Class distribution statistics |
| `bilstm_pos_frozen_results.json` | POS frozen embeddings results |
| `bilstm_pos_fine-tuned_results.json` | POS fine-tuned results |
| `bilstm_ner_frozen_results.json` | NER frozen embeddings results |
| `bilstm_ner_fine-tuned_results.json` | NER fine-tuned results |
| `bilstm_summary.json` | Combined summary of all models |
| `ablation_study_results.json` | Ablation study results |
| `sequence_labeling_evaluation.json` | Comprehensive evaluation results |

### Figure Files (`figures/`)
| File | Description |
|------|-------------|
| `bilstm_pos_*_loss.png` | POS training/validation loss curves |
| `bilstm_ner_*_loss.png` | NER training/validation loss curves |
| `pos_confusion_matrix_*.png` | POS confusion matrices |
| `ablation_*_loss.png` | Ablation study loss curves |
| `ablation_study_comparison.png` | Ablation results bar chart |

### Model Files (`models/`)
| Directory | Description |
|-----------|-------------|
| `bilstm_pos_frozen/` | POS frozen embeddings checkpoints |
| `bilstm_pos_fine-tuned/` | POS fine-tuned checkpoints |
| `bilstm_ner_frozen/` | NER frozen embeddings checkpoints |
| `bilstm_ner_fine-tuned/` | NER fine-tuned checkpoints |
| `ablation_*/` | Ablation study model checkpoints |

---

## 🚀 Usage

### 1. Dataset Preparation
```bash
python SequenceLabeling/dataset_preparation.py > outputs/dataset_preparation.txt
```

### 2. Train BiLSTM Models
```bash
# Train all 4 configurations (POS frozen/fine-tuned, NER frozen/fine-tuned)
python SequenceLabeling/bi-lstm_sequence_labeller.py > outputs/bi_lstm_sequence_labeller.txt
```

### 3. Run Evaluation
```bash
python SequenceLabeling/eval.py > outputs/SequenceLabelling_eval.txt
```

### 4. Run Ablation Study
```bash
python SequenceLabeling/train_ablation.py > outputs/ablation_study.txt
```

---

## 📝 Key Findings

1. **Fine-tuning embeddings improves POS tagging by 10.85% accuracy** over frozen embeddings
2. **Bidirectional context is critical** - unidirectional LSTM achieves only 27% accuracy
3. **CRF layer is essential for NER** - without it, zero entities are detected
4. **Random embeddings surprisingly outperform pretrained Word2Vec** for this POS task
5. **NER performance is limited** (~11% F1) due to small dataset and class imbalance
6. **Dropout provides marginal regularization benefit** for this dataset size

---

## 🔧 Recommendations for Improvement

1. **Expand training data:** Annotate more sentences for NER (currently only 346)
2. **Improve gazetteer coverage:** Add foreign entities, multi-word expressions
3. **Fix CRF numerical stability:** Address high loss values in NER training
4. **Address class imbalance:** Use weighted loss or sampling strategies for NER
5. **Subword modeling:** Use character-level features for OOV handling
6. **Data augmentation:** Generate synthetic training examples for rare entities
