# Transformer Encoder: Text Classification

This module implements a Transformer Encoder for multi-class text classification on Urdu news articles. The dataset is prepared from `cleaned.txt` and `Metadata.json`, with articles categorized into 5 distinct categories.

## 📁 File Structure

| File | Purpose | Status |
|------|---------|--------|
| `dataset_preparation.py` | Dataset creation, categorization, tokenization, and splitting | ✅ Complete |
| `scaled_dotProductAttention.py` | Scaled dot-product attention mechanism | 🔄 Pending |
| `MultiHead_selfAttention.py` | Multi-head self-attention implementation | 🔄 Pending |
| `positional_encoding.py` | Positional encoding for sequence order | 🔄 Pending |
| `PositionWise_feedForward_Network.py` | Position-wise feed-forward network | 🔄 Pending |
| `transformer_encoder.py` | Complete Transformer Encoder implementation | 🔄 Pending |

## 📊 Dataset Preparation Summary

### Article Categorization

Articles from `Metadata.json` are assigned to one of 5 categories based on keyword matching in titles.

| Category ID | Category Name | Indicative Keywords |
|-------------|---------------|---------------------|
| 1 | **Politics** | election, government, minister, parliament, party, vote, democracy, president, senate, assembly, political, الیکشن, انتخابات, حکومت, وزیر, پارلیمان, جماعت, ووٹ, جمہوریت, صدر, سینیٹ, اسمبلی, سیاسی |
| 2 | **Sports** | cricket, match, team, player, score, tournament, football, hockey, sports, stadium, champion, league, کرکٹ, میچ, ٹیم, کھلاڑی, سکور, ٹورنامنٹ, فٹبال, ہاکی, کھیل, سٹیڈیم, چیمپئن, لیگ |
| 3 | **Economy** | inflation, trade, bank, GDP, budget, economy, stock, market, investment, export, import, finance, مہنگائی, تجارت, بینک, جی ڈی پی, بجٹ, معیشت, سٹاک, مارکیٹ, سرمایہ کاری, برآمد, درآمد, مالیات |
| 4 | **International** | UN, treaty, foreign, bilateral, conflict, diplomatic, ambassador, sanction, war, peace, alliance, global, اقوام متحدہ, معاہدہ, خارجہ, دو طرفہ, تنازع, سفارتی, سفیر, پابندی, جنگ, امن, اتحاد, عالمی |
| 5 | **Health & Society** | hospital, disease, vaccine, flood, education, health, doctor, patient, school, college, university, student, teacher, earthquake, disaster, relief, welfare, ہسپتال, بیماری, ویکسین, سیلاب, تعلیم, صحت, ڈاکٹر, مریض, سکول, کالج, یونیورسٹی, طلبہ, استاد |

### Tokenization Details

- **Vocabulary Source:** Word2Vec embeddings from Part 1 (Condition C3)
- **Vocabulary Size:** 1,033 tokens
- **Sequence Length:** 256 tokens (padded/truncated)
- **UNK Token Rate:** 16.39% (5,593 out of 34,120 tokens)

### Dataset Splits (70/15/15 Stratified by Category)

| Split | Articles | Percentage |
|-------|----------|------------|
| **Training** | 145 | 69.0% |
| **Validation** | 29 | 13.8% |
| **Test** | 36 | 17.1% |
| **Total** | 210 | 100% |

---

## 📈 Class Distribution

### Overall Dataset Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| **Politics** | 32 | 15.2% |
| **Sports** | 64 | 30.5% |
| **Economy** | 56 | 26.7% |
| **International** | 28 | 13.3% |
| **Health & Society** | 30 | 14.3% |
| **Total** | **210** | **100%** |

### Distribution Across Splits

| Category | Train | Validation | Test | Total |
|----------|-------|------------|------|-------|
| **Politics** | 22 | 4 | 6 | 32 |
| **Sports** | 44 | 9 | 11 | 64 |
| **Economy** | 39 | 8 | 9 | 56 |
| **International** | 19 | 4 | 5 | 28 |
| **Health & Society** | 21 | 4 | 5 | 30 |
| **Total** | **145** | **29** | **36** | **210** |

### Distribution Visualization

```
Overall Category Distribution (210 articles)
============================================
Politics          15.2% (32)
Sports            30.5% (64)
Economy           26.7% (56)
International     13.3% (28)
Health & Society  14.3% (30)

Split Distribution
==================
                    Train    Val     Test
Politics            22       4       6
Sports              44       9       11
Economy             39       8       9
International       19       4       5
Health & Society    21       4       5
─────────────────────────────────────────
Total               145      29      36
Percentage          69.0%    13.8%   17.1%
```

---

## 📁 Output Files

### Data Files (`data/`)
| File | Description | Shape |
|------|-------------|-------|
| `transformer_train.json` | Training set (145 articles) | JSON array |
| `transformer_val.json` | Validation set (29 articles) | JSON array |
| `transformer_test.json` | Test set (36 articles) | JSON array |
| `transformer_full.json` | Complete dataset (210 articles) | JSON array |
| `transformer_train.npz` | Training set as numpy arrays | tokens: (145, 256), labels: (145,) |
| `transformer_val.npz` | Validation set as numpy arrays | tokens: (29, 256), labels: (29,) |
| `transformer_test.npz` | Test set as numpy arrays | tokens: (36, 256), labels: (36,) |
| `transformer_categories.json` | Category ID to name mapping | JSON object |

### Results Files (`results/`)
| File | Description |
|------|-------------|
| `transformer_dataset_stats.json` | Statistical summary of the dataset |

---

## 📊 Dataset Statistics Summary

```json
{
  "total_articles": 210,
  "train_articles": 145,
  "val_articles": 29,
  "test_articles": 36,
  "max_sequence_length": 256,
  "vocab_size": 1033,
  "distribution": {
    "overall": {
      "Politics": 32,
      "Sports": 64,
      "Economy": 56,
      "International": 28,
      "Health_Society": 30
    },
    "train": {
      "Politics": 22,
      "Sports": 44,
      "Economy": 39,
      "International": 19,
      "Health_Society": 21
    },
    "val": {
      "Politics": 4,
      "Sports": 9,
      "Economy": 8,
      "International": 4,
      "Health_Society": 4
    },
    "test": {
      "Politics": 6,
      "Sports": 11,
      "Economy": 9,
      "International": 5,
      "Health_Society": 5
    }
  }
}
```

---

## 🔧 Dataset Preparation Pipeline

### Step 1: Article Categorization
Each article from `Metadata.json` is assigned a category based on keyword matching in the title. Both English and Urdu keywords are used for robust categorization.

### Step 2: Tokenization
Articles from `cleaned.txt` are tokenized using the Word2Vec vocabulary from Part 1 (Condition C3). Tokens not in the vocabulary are mapped to `<UNK>` (index 0).

### Step 3: Sequence Padding/Truncation
Each article's token sequence is padded or truncated to exactly 256 tokens to ensure uniform input size for the Transformer model.

### Step 4: Stratified Split
The dataset is split into training (70%), validation (15%), and test (15%) sets while maintaining the original category distribution across all splits.

---

## 📝 Data Format

### JSON Format (for inspection)
Each article in the JSON files has the following structure:
```json
{
  "doc_id": 1,
  "category_id": 1,
  "category_name": "Politics",
  "title": "پنجاب کے ہسپتالوں میں ایمرجنسی میں کام کرنے والے طبّی عملے کے موبائل فون کے استعمال پر پابندی",
  "publish_date": "2026-01-26",
  "tokens": [15, 1, 477, 70, 2, 53, 241, 595, ...]
}
```

### NPZ Format (for efficient training)
- `tokens`: NumPy array of shape `(n_samples, 256)` containing token indices
- `labels`: NumPy array of shape `(n_samples,)` containing category IDs (1-5)

---

## 🚀 Usage

### Run Dataset Preparation
```bash
python TransformerEncoder/dataset_preparation.py > outputs/transformer_dataset_prep.txt
```

### Load Dataset in Python
```python
import numpy as np
import json

# Load as numpy arrays (recommended for training)
data = np.load('data/transformer_train.npz')
tokens = data['tokens']  # shape: (145, 256)
labels = data['labels']  # shape: (145,)

# Load as JSON (for inspection)
with open('data/transformer_train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)
```

---

## 📋 Category Mapping

| ID | Name | Description |
|----|------|-------------|
| 1 | Politics | Government, elections, political parties, leaders |
| 2 | Sports | Cricket, football, matches, tournaments, players |
| 3 | Economy | Trade, banking, stock market, inflation, budget |
| 4 | International | Foreign relations, diplomacy, conflicts, global affairs |
| 5 | Health & Society | Healthcare, education, disasters, social welfare |

---

## 🎯 Next Steps

1. **Implement Transformer Components:**
   - Scaled Dot-Product Attention
   - Multi-Head Self-Attention
   - Positional Encoding
   - Position-Wise Feed-Forward Network

2. **Build Transformer Encoder:**
   - Stack N encoder layers
   - Add classification head
   - Train on prepared dataset

3. **Evaluation:**
   - Accuracy, Precision, Recall, F1 per category
   - Confusion matrix
   - Comparison with baseline models

---

## 📊 Key Observations

1. **Class Imbalance:** Sports (30.5%) and Economy (26.7%) dominate the dataset, while Politics (15.2%), Health & Society (14.3%), and International (13.3%) are relatively balanced but smaller.

2. **UNK Rate:** 16.39% of tokens are unknown, indicating vocabulary coverage gaps that may affect model performance.

3. **Sequence Length:** 256 tokens capture most article content; average article length is ~162 tokens.

4. **Stratification:** The 70/15/15 split maintains category proportions across all splits, ensuring representative evaluation.

---

*This README will be updated as the Transformer Encoder implementation progresses.*

This README provides:

1. **Complete file structure documentation** with status indicators
2. **Dataset preparation summary** including categorization criteria
3. **Class distribution** across all 5 categories with visual representation
4. **Detailed split statistics** (70/15/15 stratified)
5. **Tokenization details** including vocabulary source and UNK rate
6. **Output file descriptions** with shapes and formats
7. **Usage instructions** for running the preparation script
8. **Category mapping** with descriptions
9. **Next steps** for the Transformer implementation
10. **Key observations** about the dataset




# UPDATED NEW INFORMATION ADDED TO THE README AS THE SECOND VERSION OF THE README:

## 1. Comprehensive README for TransformerEncoder Directory

# Transformer Encoder: Text Classification

This module implements a complete Transformer Encoder from scratch for 5-class topic classification on Urdu news articles. All components are built without using PyTorch's built-in Transformer classes.

## 📁 File Structure

| File | Purpose | Status |
|------|---------|--------|
| `dataset_preparation.py` | Dataset creation, categorization, tokenization, and splitting | ✅ Complete |
| `scaled_dotProductAttention.py` | Scaled dot-product attention mechanism | ✅ Complete |
| `MultiHead_selfAttention.py` | Multi-head self-attention (h=4, d_model=128) | ✅ Complete |
| `positional_encoding.py` | Sinusoidal positional encoding | ✅ Complete |
| `PositionWise_feedForward_Network.py` | Position-wise FFN (d_ff=512) | ✅ Complete |
| `transformer_encoder.py` | Single Transformer encoder block (Pre-LN) | ✅ Complete |

## 🎯 Module Overview

This module implements a Transformer Encoder with the following architecture:

```
Input Tokens → Token Embedding → Positional Encoding → [CLS] Token Prepended
    → 4× Transformer Encoder Blocks (Pre-LN) → LayerNorm → MLP Classifier (128→64→5)
```

### Architecture Specifications

| Component | Configuration |
|-----------|--------------|
| **Vocabulary Size** | 1,033 tokens |
| **Max Sequence Length** | 256 |
| **Embedding Dimension (d_model)** | 128 |
| **Attention Heads (h)** | 4 |
| **Key/Value Dimension (d_k, d_v)** | 32 per head |
| **Feed-Forward Dimension (d_ff)** | 512 |
| **Encoder Layers** | 4 |
| **Dropout** | 0.1 |
| **Output Classes** | 5 |

### Pre-Layer Normalization Architecture

Each encoder block uses Pre-LN (modern standard):

```
x ← x + Dropout(MultiHead(LN(x)))
x ← x + Dropout(FFN(LN(x)))
```

---

## 📊 Dataset Summary

### Category Distribution

| Category | Train | Val | Test | Total |
|----------|-------|-----|------|-------|
| Politics | 22 | 4 | 6 | 32 |
| Sports | 44 | 9 | 11 | 64 |
| Economy | 39 | 8 | 9 | 56 |
| International | 19 | 4 | 5 | 28 |
| Health & Society | 21 | 4 | 5 | 30 |
| **Total** | **145** | **29** | **36** | **210** |

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Articles | 210 |
| Training Samples | 145 (69.0%) |
| Validation Samples | 29 (13.8%) |
| Test Samples | 36 (17.1%) |
| Max Sequence Length | 256 |
| Vocabulary Size | 1,033 |
| UNK Token Rate | 16.39% |

---

## 🔧 Component Details

### 1. Scaled Dot-Product Attention (`scaled_dotProductAttention.py`)

Implements the core attention mechanism:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Features:**
- Optional padding mask support
- Returns both output and attention weights
- Dropout on attention weights
- Causal mask support for decoder

**Parameters:** None (stateless)

---

### 2. Multi-Head Self-Attention (`MultiHead_selfAttention.py`)

Splits input into multiple heads for parallel attention.

**Configuration:**
- h = 4 heads
- d_model = 128
- d_k = d_v = 32 per head

**Parameters:**
| Component | Parameters |
|-----------|------------|
| W_q | 16,384 |
| W_k | 16,384 |
| W_v | 16,384 |
| W_o | 16,384 |
| **Total** | **65,536** |

**Features:**
- Separate projection matrices per head (implemented via reshaping)
- Shared output projection
- Xavier uniform initialization
- Returns attention weights for visualization

---

### 3. Positional Encoding (`positional_encoding.py`)

Sinusoidal positional encoding as fixed (non-learned) buffer:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Features:**
- Fixed encoding (registered as buffer, not parameter)
- Max length: 512 positions
- Value range: [-1, 1]
- Nearby positions have higher similarity
- Consistent norm across all positions (8.0)

**Visualizations Generated:**
- `positional_encoding_visualization.png`: Heatmap and frequency analysis
- `positional_encoding_similarity.png`: Similarity vs position distance

---

### 4. Position-Wise Feed-Forward Network (`PositionWise_feedForward_Network.py`)

Two-layer MLP applied independently to each position:

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

**Configuration:**
- d_model = 128
- d_ff = 512
- Activation: ReLU

**Parameters:**
| Component | Parameters |
|-----------|------------|
| linear1.weight | 65,536 |
| linear1.bias | 512 |
| linear2.weight | 65,536 |
| linear2.bias | 128 |
| **Total** | **131,712** |

**Note:** FFN has ~2× more parameters than Multi-Head Attention.

---

### 5. Transformer Encoder Block (`transformer_encoder.py`)

Single encoder block with Pre-Layer Normalization.

**Parameters per Block:**
| Component | Parameters |
|-----------|------------|
| Multi-Head Attention | 65,536 |
| Feed-Forward Network | 131,712 |
| Layer Normalization (×2) | 512 |
| **Total per Block** | **197,760** |

**Stacked Parameters:**
| Layers | Parameters |
|--------|------------|
| 1 | 197,760 |
| 2 | 395,520 |
| **4** | **791,040** |
| 6 | 1,186,560 |
| 8 | 1,582,080 |

---

## 🚀 Training Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Learning Rate | 5 × 10⁻⁴ |
| Weight Decay | 0.01 |
| Warmup Steps | 50 |
| Total Epochs | 50 |
| Gradient Clip | 1.0 |
| Optimizer | AdamW |
| LR Schedule | Cosine with linear warmup |
| Loss Function | Cross-Entropy |

### Model Size

| Component | Parameters |
|-----------|------------|
| Token Embedding | 132,224 |
| Positional Encoding | 0 (fixed) |
| [CLS] Token | 128 |
| 4× Encoder Blocks | 791,040 |
| Layer Norm | 256 |
| MLP Classifier (128→64→5) | 8,581 |
| **Total** | **932,229** |

---

## 📈 Training Results

### Best Performance

| Metric | Value |
|--------|-------|
| Best Validation Accuracy | 34.48% |
| Test Accuracy | 22.22% |
| Test Precision (weighted) | 9.64% |
| Test Recall (weighted) | 22.22% |
| Test F1 (weighted) | 11.87% |

### Per-Class Test Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Politics | 25.00% | 16.67% | 20.00% |
| Sports | 0.00% | 0.00% | 0.00% |
| Economy | 21.88% | 77.78% | 34.15% |
| International | 0.00% | 0.00% | 0.00% |
| Health & Society | 0.00% | 0.00% | 0.00% |

### Training Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1 | 1.8120 | 22.07% | 1.6415 | 17.24% |
| 2 | 1.6723 | 22.76% | 1.5629 | 31.03% |
| 50 | 1.3357 | 42.07% | 2.1808 | 27.59% |

**Improvement (Epoch 1 → 50):**
- Val Loss: -0.5393 (-32.9%)
- Val Acc: +10.34% (+60.0% relative)

---

## 📊 Analysis & Observations

### Performance Issues

The model shows signs of **overfitting**:
1. Training accuracy improves to ~42% while validation accuracy peaks at 34% then degrades
2. Validation loss increases from 1.56 (epoch 2) to 2.18 (epoch 50)
3. Model only learns to predict Economy class effectively (77.78% recall)

### Limitations

1. **Small Dataset:** Only 145 training samples across 5 classes
   - Sports class has 44 samples but 0% recall
   - Model collapses to predicting Economy for most inputs

2. **Class Imbalance:** Sports (30.5%) and Economy (26.7%) dominate

3. **UNK Token Rate:** 16.39% of tokens are unknown, limiting semantic understanding

4. **Sequence Length:** 256 tokens may be insufficient for some articles

### Recommendations for Improvement

1. **Data Augmentation:** Back-translation, synonym replacement
2. **Class Weights:** Use weighted loss to handle imbalance
3. **Subword Tokenization:** Reduce UNK rate with BPE or character-level features
4. **Pretrained Embeddings:** Use larger pretrained Urdu embeddings
5. **Regularization:** Increase dropout, add weight decay, early stopping
6. **More Data:** Annotate additional articles for underrepresented classes

---

## 📦 Output Files

### Data Files (`data/`)
| File | Description |
|------|-------------|
| `transformer_train.npz` | Training set (145 samples) |
| `transformer_val.npz` | Validation set (29 samples) |
| `transformer_test.npz` | Test set (36 samples) |
| `transformer_categories.json` | Category mapping |

### Results Files (`results/`)
| File | Description |
|------|-------------|
| `transformer_dataset_stats.json` | Dataset statistics |
| `transformer_results.json` | Training and evaluation results |

### Figure Files (`figures/`)
| File | Description |
|------|-------------|
| `transformer_training_curves.png` | Loss and accuracy curves |
| `transformer_confusion_matrix.png` | Test set confusion matrix |
| `positional_encoding_visualization.png` | PE heatmap and analysis |
| `positional_encoding_similarity.png` | PE similarity vs distance |
| `attention_weights_visualization.png` | Sample attention weights |

### Model Files (`models/transformer_classifier/`)
| File | Description |
|------|-------------|
| `latest_checkpoint.pth` | Most recent training checkpoint |
| `best_model.pth` | Model with best validation accuracy |

---

## 🚀 Usage

### 1. Dataset Preparation
```bash
python TransformerEncoder/dataset_preparation.py > outputs/transformer_dataset_prep.txt
```

### 2. Test Individual Components
```bash
python TransformerEncoder/scaled_dotProductAttention.py > outputs/scaled_dotProductAttention.txt
python TransformerEncoder/MultiHead_selfAttention.py > outputs/MultiHead_selfAttention.txt
python TransformerEncoder/positional_encoding.py > outputs/positional_encoding.txt
python TransformerEncoder/PositionWise_feedForward_Network.py > outputs/PositionWise_feedForward_Network.txt
python TransformerEncoder/transformer_encoder.py > outputs/transformer_encoder.txt
```

---


All implementations are from scratch following the assignment constraints (no PyTorch built-in Transformer classes).

---

*This README documents the complete TransformerEncoder module as implemented.*

---
