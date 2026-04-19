<h1 align="center">raw-overfitting-transformer</h1>

A complete neural pipeline for Urdu text processing implementing Word Embeddings, Sequence Labeling (POS/NER), and Transformer-based Text Classification from scratch.


## Table of Contents

- [Project Overview](#-project-overview)
- [Module 1: Word Embeddings](#-module-1-word-embeddings)
- [Module 2: Sequence Labeling](#-module-2-sequence-labeling)
- [Module 3: Transformer Encoder](#-module-3-transformer-encoder)
- [Final Evaluation & Model Comparison](#-final-evaluation--model-comparison)
- [Installation & Setup](#-installation--setup)
- [Results Summary](#-results-summary)
- [Key Findings](#-key-findings)

---

## Project Overview

This project implements a multi-part NLP pipeline for Urdu text processing with models built entirely from scratch (no pretrained models, no HuggingFace, no PyTorch built-in Transformer classes).

### Modules

| Module | Task | Key Technologies |
|--------|------|------------------|
| **WordEmbeddings** | Word representation learning | TF-IDF, PPMI, Skip-gram Word2Vec |
| **SequenceLabeling** | POS tagging & NER | BiLSTM, CRF, Ablation studies |
| **TransformerEncoder** | Text classification | Multi-Head Attention, Pre-LN Transformer | 
| **Final Evaluation** | Cross-module comparison | Unified evaluation framework |

---

# Module 1: Word Embeddings

## Overview

Implements TF-IDF, PPMI, and Skip-gram Word2Vec for Urdu word representation learning. Evaluates four conditions (C1-C4) to find optimal configuration.

## Results Summary

### Four-Condition Comparison

| Condition | MRR | Analogy Acc | Embedding Dim | Vocab Size | Description |
|-----------|-----|-------------|---------------|------------|-------------|
| **C1** | 0.1869 | 10.00% | 4386 | 200 | PPMI Baseline |
| **C2** | 0.2699 | 20.00% | 100 | 1083 | Skip-gram on raw.txt |
| **C3** | **0.3145** | **20.00%** | 100 | 1033 | Skip-gram on cleaned.txt |
| **C4** | 0.2693 | 20.00% | 200 | 1033 | Skip-gram d=200 |

### Best Model: C3

- **MRR:** 0.3145
- **Vocabulary:** 1,033 tokens
- **Embedding Dimension:** 100
- **Training Pairs:** ~335K

### Nearest Neighbors (C3 - Top 5)

| Query Word | Nearest Neighbors |
|------------|-------------------|
| **پاکستان** | ائین, متحدہ, انڈیا, اپوزیشن, انٹرنیشنل |
| **حکومت** | حمایت, موجودہ, صوبا, طالبان, وفاقی |
| **عدالت** | ضمانت, منظور, بامشقت, ہائیکورٹ, ریپ |
| **فوج** | پاکستا, عامہ, ترجمان, شعبہ, تعلق |
| **تعلیم** | اعلی, حاصل, باوجود, سکول, خیبر |

### Analogy Tests (C3 - 2/10 Correct)

| Analogy | Expected | Top-3 Predictions | ✓/✗ |
|---------|----------|-------------------|-----|
| عمران : خان :: نواز : ? | شریف | مریم, شریف, اصف | ✓ |
| شہباز : شریف :: مریم : ? | نواز | نواز, پنجاب, اعلی | ✓ |

**Assessment:** The embeddings capture basic co-occurrence patterns and person-name associations but struggle with complex relational analogies (geographical capitals, occupational relationships) due to limited corpus size.

### Analysis: Which Condition Yields Best Embeddings?

**C3 (Skip-gram on cleaned.txt with d=100) is the best condition** (MRR=0.3145). Key observations:

1. **Word2Vec > PPMI:** All Word2Vec variants outperform PPMI by 44-68% in MRR, showing prediction-based embeddings better capture semantic relationships.

2. **Cleaned > Raw:** C3 (cleaned) achieves better MRR than C2 (raw: 0.2699), demonstrating that preprocessing improves embedding quality.

3. **Increasing d does NOT help:** C4 (d=200) degrades to MRR=0.2693 because larger dimension requires more training data to avoid overfitting.

---

# Module 2: Sequence Labeling

## Overview

Implements BiLSTM-based sequence labeling for POS tagging (12 tags) and NER (BIO scheme, 9 tags). Includes comprehensive ablation studies.

## POS Tagging Results

### Test Set Performance

| Mode | Accuracy | Macro F1 | Weighted F1 |
|------|----------|----------|-------------|
| Frozen Embeddings | 72.96% | 0.6766 | 0.7321 |
| **Fine-tuned Embeddings** | **83.81%** | **0.9244** | **0.8442** |

### Per-Class F1 (Fine-tuned)

| Tag | F1 | Tag | F1 |
|-----|-----|-----|-----|
| PRON | 0.9950 | POST | 0.9912 |
| NUM | 1.0000 | DET | 1.0000 |
| NOUN | 0.9244 | VERB | 0.9108 |
| ADJ | 0.9297 | ADV | 0.7314 |
| UNK | 0.7809 | CONJ | 0.9804 |

### Top 3 Most Confused Tag Pairs (Fine-tuned)

| Rank | True → Predicted | Count | Example |
|------|------------------|-------|---------|
| 1 | **UNK → ADV** | 309 | `ڈیووس` (Davos) misclassified as adverb |
| 2 | **VERB → ADV** | 50 | `گہرے` (deep) confused with adverbial usage |
| 3 | **NOUN → ADV** | 22 | `روایت` (tradition) in idiomatic phrase |

**Example 1 (UNK → ADV):**
- Sentence: `<NUM> جنوری کو پاکستا وزیر اعظم شہباز شریف نے ڈیووس میں ٹرمپ کے بورڈ اف پیس کی سرکاری تقریب میں شرکت کی`
- Error at: `ڈیووس` (Davos - foreign location)
- Explanation: Foreign word not in vocabulary, model defaults to ADV based on context.

**Example 2 (VERB → ADV):**
- Sentence: `ان میں سے کئی زخم اتنے گہرے تھے کہ پھیپھڑ اور دیگر اعضا کو شدید نقصان پہنچا`
- Error at: `گہرے` (deep)
- Explanation: Adjective/participle form confused by model.

## NER Results

### Entity-Level Metrics (Fine-tuned with CRF)

| Entity | Precision | Recall | F1 | TP | FP | FN |
|--------|-----------|--------|-----|----|----|-----|
| PER | 1.16% | 16.67% | 2.17% | 3 | 256 | 15 |
| LOC | 2.83% | 65.52% | 5.42% | 19 | 653 | 10 |
| ORG | 0.00% | 0.00% | 0.00% | 0 | 0 | 32 |
| MISC | 0.00% | 0.00% | 0.00% | 0 | 0 | 6 |
| **Overall** | 5.97% | 58.82% | **10.83%** | 50 | 788 | 35 |

### CRF vs Softmax Comparison

| Method | PER F1 | LOC F1 | ORG F1 | Overall F1 |
|--------|--------|--------|--------|------------|
| **With CRF** | **2.17%** | **5.42%** | 0.00% | **10.83%** |
| Without CRF | 0.00% | 0.00% | 0.00% | 0.00% |

**Finding:** CRF is **essential** - without it, zero entities detected.

### Error Analysis: False Positives (Top 5)

| # | Entity | Sentence Context | Explanation |
|---|--------|------------------|-------------|
| 1 | `وزیر` | `پاکستا وزیر اعظم شہباز شریف` | Common noun "minister" tagged as entity |
| 2 | `بھی` | `فیلڈ مارشل سید عاصم منیر بھی موجود` | Function word adjacent to entity incorrectly included |
| 3 | `منیر` | `سید عاصم منیر بھی موجود تھے` | Partial name tagged; fails to group full name |
| 4 | `پیس` | `بورڈ اف پیس کی سرکاری تقریب` | Part of org name tagged separately |
| 5 | `مارشل` | `فیلڈ مارشل سید عاصم` | Title tagged separately from full name |

### Error Analysis: False Negatives (Top 5)

| # | Entity | Sentence Context | Explanation |
|---|--------|------------------|-------------|
| 1 | `اعظم شہباز شریف` | `پاکستا وزیر اعظم شہباز شریف` | Full name missed; title treated separately |
| 2 | `پاکستان کے` | `پاکستان کے فیلڈ مارشل` | Genitive phrase missed |
| 3 | `سندھ` | `سندھ اسمبلی کی روایت` | Province name missed (gazetteer gap) |
| 4 | `ڈیووس` | `ڈیووس میں ٹرمپ کے بورڈ` | Foreign location not in gazetteer |
| 5 | `ٹرمپ` | `ٹرمپ کے بورڈ اف پیس` | Foreign person name not recognized |

## Ablation Study Results

| ID | Configuration | Accuracy | Macro F1 | Weighted F1 | Δ from Baseline |
|----|---------------|----------|----------|-------------|-----------------|
| **Baseline** | BiLSTM + Dropout + Pretrained | 83.81% | 0.9244 | 0.8442 | - |
| **A1** | Unidirectional LSTM | 27.12% | 0.0678 | 0.1959 | **-64.83%** |
| **A2** | No Dropout | 83.70% | 0.9216 | 0.8431 | -0.11% |
| **A3** | Random Embeddings | **91.01%** | 0.8641 | **0.9090** | **+6.48%** |
| **A4** | Softmax instead of CRF | - | - | 0.00% | -10.83% |

### Ablation Findings

| Component | Importance | Key Insight |
|-----------|------------|-------------|
| **Bidirectional LSTM** | ⭐⭐⭐⭐⭐ Critical | Without it, performance collapses (27% accuracy) |
| **CRF Layer (NER)** | ⭐⭐⭐⭐⭐ Critical | Zero entities detected without structured decoding |
| **Pretrained Embeddings** | ⭐⭐ Situational | Random embeddings outperformed pretrained for this task |
| **Dropout** | ⭐ Marginal | Small benefit; model not severely overfitting |

**Surprising Finding (A3):** Random embeddings (91.01%) outperform pretrained Word2Vec (83.81%) by 7.20%. This suggests the Word2Vec embeddings may not capture POS-relevant features effectively for this domain.

---

# Module 3: Transformer Encoder

## Overview

Implements a complete Transformer Encoder from scratch for 5-class text classification on Urdu news articles.

## Architecture

```
Input Tokens → Token Embedding → Positional Encoding → [CLS] Token
    → 4× Transformer Encoder Blocks (Pre-LN) → LayerNorm → MLP (128→64→5)
```

### Specifications

| Component | Configuration | Parameters |
|-----------|---------------|------------|
| Vocabulary Size | 1,033 tokens | - |
| Max Sequence Length | 256 | - |
| Token Embedding | d_model = 128 | 132,224 |
| Multi-Head Attention | h=4, d_k=d_v=32 | 65,536 |
| Feed-Forward | d_ff=512, ReLU | 131,712 |
| Encoder Layers | 4 blocks | 791,040 |
| MLP Classifier | 128 → 64 → 5 | 8,581 |
| **Total** | - | **932,229** |

## Dataset Distribution

| Category | Train | Val | Test | Total | % |
|----------|-------|-----|------|-------|-----|
| Politics | 22 | 4 | 6 | 32 | 15.2% |
| Sports | 44 | 9 | 11 | 64 | 30.5% |
| Economy | 39 | 8 | 9 | 56 | 26.7% |
| International | 19 | 4 | 5 | 28 | 13.3% |
| Health & Society | 21 | 4 | 5 | 30 | 14.3% |
| **Total** | **145** | **29** | **36** | **210** | **100%** |

## Training Results

### Best Performance

| Metric | Value | Epoch |
|--------|-------|-------|
| Best Validation Accuracy | **34.48%** | 7 |
| Test Accuracy | 27.78% | - |
| Test Macro F1 | 8.55% | - |
| Test Weighted F1 | 15.67% | - |

### Per-Class Test Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Politics | 0.00% | 0.00% | 0.00% | 6 |
| **Sports** | **35.71%** | **90.91%** | **51.28%** | 11 |
| Economy | 0.00% | 0.00% | 0.00% | 9 |
| International | 0.00% | 0.00% | 0.00% | 5 |
| Health & Society | 0.00% | 0.00% | 0.00% | 5 |

### Training Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1 | 1.8120 | 22.07% | 1.6415 | 17.24% |
| 2 | 1.6723 | 22.76% | 1.5629 | 31.03% |
| 7 | 1.5885 | 28.28% | 1.5786 | **34.48%** |
| 50 | 1.3357 | 42.07% | 2.1808 | 27.59% |

## Analysis

### Critical Issues

1. **Severe Overfitting:** Training improves to 42% while validation degrades (loss increases 33%)
2. **Class Collapse:** Model predicts only Sports class (90.91% recall, but 0% for 4/5 classes)
3. **Parameter-to-Sample Ratio:** ~6,429 params/sample (64× higher than recommended)

### Root Causes

| Issue | Impact | Evidence |
|-------|--------|----------|
| Small Dataset | Severe | 145 training samples for 932K parameters |
| High UNK Rate | Significant | 16.39% tokens unknown |
| Class Imbalance | Moderate | Sports (44) and Economy (39) dominate |

---

# Final Evaluation & Model Comparison

## 8.1 Transformer Results

### Test Metrics
- **Accuracy:** 27.78%
- **Macro F1:** 8.55%
- **Weighted F1:** 15.67%

### Confusion Matrix
The model predicts almost exclusively the Sports class:
- Sports: 10/11 correctly classified
- All other classes: 0 correctly classified

### Attention Heatmaps Analysis
From the 3 correctly classified Sports articles, attention patterns reveal:
- **Head 1-2:** Focus on local context (adjacent tokens)
- **Head 3-4:** Broader attention across sequence
- **[CLS] token:** Attends broadly across all positions
- Overall patterns are noisy due to overfitting; clear semantic attention not learned

## 8.2 BiLSTM vs. Transformer Comparison

### 1. Which model achieves higher accuracy, and by how much?

**BiLSTM POS achieves dramatically higher accuracy (83.81%) compared to Transformer (27.78%)** - a difference of 56.03 percentage points. For sequence labeling, BiLSTM generalizes well while Transformer severely overfits. Even considering different tasks (POS vs classification), the BiLSTM's ability to learn from limited data is clearly superior.

### 2. Which model converged in fewer epochs?

**BiLSTM converged in ~8 epochs** (with early stopping), while Transformer's best validation occurred at epoch 7 but training continued for 50 epochs with degrading validation performance. BiLSTM NER converged in ~11 epochs.

### 3. Which model was faster to train per epoch, and why?

**BiLSTM was faster (~4-6 seconds/epoch) vs Transformer (~8-11 seconds/epoch)** because:
- BiLSTM has O(n) complexity; Transformer has O(n²) self-attention
- BiLSTM has ~400K parameters vs Transformer's 932K
- Self-attention requires computing pairwise interactions across all positions

### 4. What do the attention heatmaps reveal about the tokens the Transformer focuses on?

The attention heatmaps from correctly classified Sports articles show:
- **[CLS] token** attends broadly across the sequence for classification
- **Some heads** focus on local context (adjacent tokens)
- **Other heads** show more distributed attention patterns
- Due to severe overfitting, attention patterns are **noisy and inconsistent** - the model hasn't learned clear semantic attention with only 145 training samples

### 5. Given a dataset of only 200-300 articles, which architecture is more appropriate and why?

**BiLSTM is far more appropriate** for datasets of 200-300 articles because:
1. **Parameter efficiency:** ~400K vs 932K parameters reduces overfitting
2. **Inductive bias:** Sequential processing suits small data; Transformers need massive data (100K+ samples) to learn effective attention
3. **Proven performance:** BiLSTM achieves 84% accuracy vs Transformer's 28%
4. **Training stability:** BiLSTM converges smoothly; Transformer overfits severely
5. **Data requirements:** Self-attention needs substantial data to learn meaningful token relationships - with 145 samples it memorizes rather than generalizes

---

# Results Summary

## Overall Model Comparison

| Model | Task | Accuracy | F1 (Weighted) | Parameters | Train Samples |
|-------|------|----------|---------------|------------|---------------|
| Word2Vec (C3) | Embedding Quality | MRR=0.3145 | - | 103,300 | 335K pairs |
| **BiLSTM** | POS Tagging | **83.81%** | **0.8442** | ~400K | 346 sent |
| BiLSTM-CRF | NER | 87.28%* | 0.1083 | ~400K | 346 sent |
| Transformer | Text Classification | 27.78% | 0.1567 | 932,229 | 145 articles |

*Token-level accuracy (most tokens are 'O')

## Best Performing Models

| Task | Best Model | Performance |
|------|------------|-------------|
| Word Embeddings | Word2Vec C3 (d=100) | MRR = 0.3145 |
| POS Tagging | BiLSTM (fine-tuned) | F1 = 0.8442 |
| Text Classification | BiLSTM POS* | Acc = 83.81% |

*Transformer underperforms due to data scarcity

---

# Key Findings

## Word Embeddings
1. **C3 (cleaned.txt, d=100) is optimal** - MRR=0.3145, outperforming PPMI by 68%
2. **Increasing dimension hurts** - d=200 degrades to MRR=0.2693 due to overfitting
3. **Cleaning matters** - C3 (cleaned) > C2 (raw) by 0.0446 MRR
4. **Limited semantic capture** - Only 2/10 analogies correct; needs more data

## Sequence Labeling
1. **Fine-tuning improves POS by 10.85%** - 83.81% vs 72.96% accuracy
2. **Bidirectional context is critical** - Unidirectional drops to 27% accuracy
3. **CRF essential for NER** - 0% F1 without structured decoding
4. **Random embeddings surprisingly outperform pretrained** - 91.01% vs 83.81% (ablation A3)
5. **NER limited by data** - Only 11% F1 due to 346 training sentences and 87% 'O' tags

## Transformer
1. **Severe overfitting** - 932K parameters for only 145 samples
2. **Class collapse** - Model predicts only Sports class (91% recall, 0% for others)
3. **UNK rate impact** - 16.39% unknown tokens limit semantic understanding
4. **Parameter-to-sample ratio** - ~6,429:1 (64× higher than recommended)

## Cross-Module Comparison
1. **BiLSTM dominates Transformer on small data** - 84% vs 28% accuracy
2. **Inductive bias matters** - Sequential processing (BiLSTM) > self-attention (Transformer) for small datasets
3. **Parameter efficiency is crucial** - Fewer parameters (BiLSTM) generalize better
4. **Transformers need data** - Self-attention requires 100K+ samples to be effective

---

## Setup


### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install torch numpy scipy scikit-learn matplotlib seaborn tqdm python-dotenv

# Configure .env file, use the same env as provided but change the paths to match your system like this: 
DEBUG_MODE = 1
PROCESSOR = "cpu"
repoDir = "/path/to/raw-overfitting-transformer"
```

---

## References

1. Mikolov et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
2. Pennington et al. (2014). "GloVe: Global Vectors for Word Representation"
3. Vaswani et al. (2017). "Attention Is All You Need"
4. Lample et al. (2016). "Neural Architectures for Named Entity Recognition"
5. Xiong et al. (2020). "On Layer Normalization in the Transformer Architecture"

---