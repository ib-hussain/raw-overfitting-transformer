I'll create a comprehensive, well-structured README.md for the TransformerEncoder module that properly documents all components, results, and analysis. This will be based on all the information from the output logs and results files.

```markdown
# Transformer Encoder: Text Classification

This module implements a complete Transformer Encoder **from scratch** for 5-class topic classification on Urdu news articles. All components are built without using PyTorch's built-in Transformer classes, following assignment constraints.

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

| Component | Configuration | Parameters |
|-----------|---------------|------------|
| **Vocabulary Size** | 1,033 tokens | - |
| **Max Sequence Length** | 256 | - |
| **Token Embedding** | d_model = 128 | 132,224 |
| **Positional Encoding** | Sinusoidal (fixed) | 0 |
| **[CLS] Token** | Learned embedding | 128 |
| **Multi-Head Attention** | h=4, d_k=d_v=32 | 65,536 |
| **Feed-Forward Network** | d_ff=512, ReLU | 131,712 |
| **Encoder Layers** | 4 blocks | 791,040 |
| **Layer Normalization** | 2 per block + final | 768 |
| **MLP Classifier** | 128 → 64 → 5 | 8,581 |
| **Dropout** | 0.1 | - |
| **Total Parameters** | - | **932,229** |

### Pre-Layer Normalization Architecture

Each encoder block uses Pre-LN (modern standard for better gradient flow):

```
x ← x + Dropout(MultiHead(LN(x)))
x ← x + Dropout(FFN(LN(x)))
```

---

## 📊 Dataset Summary

### Article Categorization

Articles from `cleaned.txt` and `Metadata.json` are assigned to one of 5 categories based on keyword matching:

| Category ID | Category Name | Indicative Keywords |
|-------------|---------------|---------------------|
| 1 | **Politics** | حکومت, وزیر, پارلیمان, جماعت, الیکشن, سیاست, government, minister, parliament |
| 2 | **Sports** | کرکٹ, میچ, ٹیم, کھلاڑی, ٹورنامنٹ, cricket, match, team, player, sports |
| 3 | **Economy** | معیشت, بینک, بجٹ, مہنگائی, تجارت, economy, bank, budget, trade, stock |
| 4 | **International** | خارجہ, سفارتی, عالمی, اقوام متحدہ, foreign, diplomatic, global, treaty |
| 5 | **Health & Society** | صحت, ہسپتال, تعلیم, سکول, ڈاکٹر, health, hospital, education, society |

### Category Distribution

| Category | Train | Val | Test | Total | % |
|----------|-------|-----|------|-------|-----|
| Politics | 22 | 4 | 6 | 32 | 15.2% |
| Sports | 44 | 9 | 11 | 64 | 30.5% |
| Economy | 39 | 8 | 9 | 56 | 26.7% |
| International | 19 | 4 | 5 | 28 | 13.3% |
| Health & Society | 21 | 4 | 5 | 30 | 14.3% |
| **Total** | **145** | **29** | **36** | **210** | **100%** |

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Articles | 210 |
| Training Samples | 145 (69.0%) |
| Validation Samples | 29 (13.8%) |
| Test Samples | 36 (17.1%) |
| Max Sequence Length | 256 |
| Vocabulary Size | 1,033 |
| **UNK Token Rate** | **16.39%** (5,593 / 34,120 tokens) |

**Note:** The high UNK rate significantly impacts model performance by limiting semantic understanding.

---

## 🔧 Component Details

### 1. Scaled Dot-Product Attention (`scaled_dotProductAttention.py`)

Implements the core attention mechanism:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Features:**
- Optional padding mask support (masks padded positions to -inf)
- Causal mask support (for decoder, upper triangular masking)
- Dropout on attention weights (p=0.1)
- Returns both output and attention weights for visualization

**Parameters:** None (stateless operation)

**Testing:** All tests passed including gradient flow, mask application, and causal masking.

---

### 2. Multi-Head Self-Attention (`MultiHead_selfAttention.py`)

Splits input into multiple heads for parallel attention computation.

**Configuration:**
- h = 4 heads
- d_model = 128
- d_k = d_v = 32 per head (d_model / h)
- Dropout = 0.1

**Parameters:**
| Component | Shape | Parameters |
|-----------|-------|------------|
| W_q | (128, 128) | 16,384 |
| W_k | (128, 128) | 16,384 |
| W_v | (128, 128) | 16,384 |
| W_o | (128, 128) | 16,384 |
| **Total** | - | **65,536** |

**Features:**
- Separate projection matrices per head implemented via reshaping
- Xavier uniform initialization for all weights
- Returns attention weights (shape: [batch, heads, seq_len, seq_len])
- Also includes `MultiHeadCrossAttention` for encoder-decoder scenarios

**Testing:** Dimension preservation, mask propagation, and gradient flow verified.

---

### 3. Positional Encoding (`positional_encoding.py`)

Sinusoidal positional encoding as fixed (non-learned) buffer:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Features:**
- Fixed encoding (registered as buffer, not trainable parameter)
- Max length: 512 positions
- Value range: [-1, 1]
- **Nearby positions have higher similarity:** pos₀ to pos₁ similarity = 0.9702, pos₀ to pos₁₀₀ = 0.4772
- **Consistent norm across all positions:** Mean = 8.0000, Std = 0.0000

**Visualizations Generated:**
- `positional_encoding_visualization.png`: Heatmap of first 50 positions × 64 dimensions, frequency analysis
- `positional_encoding_similarity.png`: Cosine similarity between positions

**Testing:** Mathematical properties (periodicity, orthogonality, norm consistency) verified.

---

### 4. Position-Wise Feed-Forward Network (`PositionWise_feedForward_Network.py`)

Two-layer MLP applied independently to each position:

```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

**Configuration:**
- d_model = 128
- d_ff = 512 (4× expansion)
- Activation: ReLU
- Dropout = 0.1

**Parameters:**
| Component | Shape | Parameters |
|-----------|-------|------------|
| linear1.weight | (128, 512) | 65,536 |
| linear1.bias | (512,) | 512 |
| linear2.weight | (512, 128) | 65,536 |
| linear2.bias | (128,) | 128 |
| **Total** | - | **131,712** |

**Note:** FFN has ~2× more parameters than Multi-Head Attention, following the standard Transformer design.

**Testing:** Position-wise independence verified (tokens processed independently across sequence dimension).

---

### 5. Transformer Encoder Block (`transformer_encoder.py`)

Single encoder block with Pre-Layer Normalization architecture.

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

**Features:**
- Pre-Layer Normalization for better gradient flow
- Residual connections around both sub-layers
- Dropout after each sub-layer
- Option to return attention weights from all layers

**Testing:**
- Forward pass with/without mask ✓
- Attention weights return ✓
- Residual connection verification ✓
- Pre-LN vs Post-LN comparison (Pre-LN has better gradient flow) ✓
- Stacked blocks (4 layers) ✓

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
| Checkpointing | Every 5 epochs |

### Model Size Summary

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

| Metric | Value | Epoch |
|--------|-------|-------|
| Best Validation Accuracy | **34.48%** | 7 |
| Test Accuracy | 22.22% | - |
| Test Precision (weighted) | 9.64% | - |
| Test Recall (weighted) | 22.22% | - |
| Test F1 (weighted) | 11.87% | - |
| Test Loss | 1.5426 | - |

### Per-Class Test Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Politics | 25.00% | 16.67% | 20.00% | 6 |
| Sports | 0.00% | 0.00% | 0.00% | 11 |
| **Economy** | **21.88%** | **77.78%** | **34.15%** | 9 |
| International | 0.00% | 0.00% | 0.00% | 5 |
| Health & Society | 0.00% | 0.00% | 0.00% | 5 |

**Observation:** The model collapses to predicting Economy for most inputs, achieving 77.78% recall but only 21.88% precision for this class. Three out of five classes have 0% recall.

### Training Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | LR |
|-------|------------|-----------|----------|---------|-----|
| 1 | 1.8120 | 22.07% | 1.6415 | 17.24% | 5.00e-4 |
| 2 | 1.6723 | 22.76% | 1.5629 | 31.03% | 5.00e-5 |
| 3 | 1.5882 | 26.90% | 1.5556 | 31.03% | 1.00e-4 |
| 7 | 1.5885 | 28.28% | **1.5786** | **34.48%** | 3.00e-4 |
| 25 | 1.4541 | 40.00% | 1.7401 | 27.59% | 3.63e-4 |
| 50 | 1.3357 | 42.07% | 2.1808 | 27.59% | 1.00e-6 |

**Training vs Validation Trends:**
- Training loss decreases from 1.81 → 1.34 (improves)
- Validation loss increases from 1.56 → 2.18 (degrades)
- **Clear overfitting signature:** Training improves while validation worsens

### Improvement Summary (Epoch 1 → 50)

| Metric | Epoch 1 | Epoch 50 | Absolute Change | Relative Change |
|--------|---------|----------|-----------------|-----------------|
| Train Loss | 1.8120 | 1.3357 | -0.4763 | -26.3% |
| Train Acc | 22.07% | 42.07% | +20.00% | +90.6% |
| Val Loss | 1.6415 | 2.1808 | **+0.5393** | **+32.9%** ⚠️ |
| Val Acc | 17.24% | 27.59% | +10.35% | +60.0% |

---

## 📊 Analysis & Observations

### Critical Performance Issues

The model exhibits **severe overfitting** with the following characteristics:

1. **Training-Validation Divergence:**
   - Training accuracy improves by 20% (22% → 42%)
   - Validation accuracy peaks early at 34% (epoch 7) then degrades
   - Validation loss increases 33% while training loss decreases 26%

2. **Class Collapse:**
   - Model predicts Economy for 77.78% of Economy samples
   - Zero recall for Sports, International, and Health & Society
   - Precision is poor even for predicted classes (max 25%)

3. **Overfitting Indicators:**
   - Early peak in validation performance (epoch 7)
   - Steady increase in validation loss after epoch 2
   - Gap between train and validation accuracy widens throughout training

### Root Causes

| Issue | Impact | Evidence |
|-------|--------|----------|
| **Small Dataset** | Severe | Only 145 training samples for 932K parameters |
| **Class Imbalance** | Moderate | Sports (44) and Economy (39) dominate; Politics (22), International (19), Health (21) underrepresented |
| **High UNK Rate** | Significant | 16.39% tokens unknown, limiting semantic understanding |
| **Parameter Count** | Critical | 932K parameters for 145 samples (~6,400 params/sample) |
| **Sequence Length** | Minor | 256 tokens sufficient (avg article ~162 tokens) |

### Parameter-to-Sample Ratio Analysis

| Metric | Value |
|--------|-------|
| Total Parameters | 932,229 |
| Training Samples | 145 |
| **Params per Sample** | **~6,429** |
| Recommended Ratio | < 100 |
| **Over-parameterization** | **64× recommended** |

The model has approximately 6,429 trainable parameters per training sample, which is ~64× higher than recommended for avoiding overfitting. This explains the severe memorization behavior.

### Recommendations for Improvement

#### Immediate Fixes (with current data):
1. **Reduce model size:**
   - Decrease layers: 4 → 2 (saves ~395K params)
   - Decrease d_model: 128 → 64 (saves ~66K params)
   - Decrease d_ff: 512 → 256 (saves ~66K params)

2. **Increase regularization:**
   - Increase dropout: 0.1 → 0.3 or 0.5
   - Add label smoothing (0.1)
   - Use weight decay = 0.1

3. **Class weights:** Apply weighted loss to penalize Economy/Sports over-prediction

4. **Early stopping:** Stop at epoch 7 (best validation accuracy)

#### Medium-Term Improvements:
1. **Data augmentation:**
   - Back-translation (Urdu → English → Urdu)
   - Synonym replacement using Urdu thesaurus
   - Random token masking/dropping

2. **Subword tokenization:**
   - Implement BPE or WordPiece to reduce UNK rate
   - Character-level CNN for OOV handling

3. **Pretrained embeddings:**
   - Use larger pretrained Urdu embeddings (fastText)
   - Consider multilingual BERT embeddings (mBERT)

#### Long-Term Solutions:
1. **Collect more data:** Annotate additional articles for underrepresented classes
2. **Transfer learning:** Fine-tune from a larger Urdu corpus
3. **Ensemble methods:** Combine with BiLSTM or simpler models

---

## 📦 Output Files

### Data Files (`data/`)
| File | Description | Shape |
|------|-------------|-------|
| `transformer_train.npz` | Training set | tokens: (145, 256), labels: (145,) |
| `transformer_val.npz` | Validation set | tokens: (29, 256), labels: (29,) |
| `transformer_test.npz` | Test set | tokens: (36, 256), labels: (36,) |
| `transformer_train.json` | Training set (human-readable) | JSON array |
| `transformer_val.json` | Validation set (human-readable) | JSON array |
| `transformer_test.json` | Test set (human-readable) | JSON array |
| `transformer_full.json` | Complete dataset | JSON array |
| `transformer_categories.json` | Category ID → name mapping | JSON object |

### Results Files (`results/`)
| File | Description |
|------|-------------|
| `transformer_dataset_stats.json` | Dataset statistics and distribution |
| `transformer_results.json` | Complete training metrics and test results |

### Figure Files (`figures/`)
| File | Description |
|------|-------------|
| `transformer_training_curves.png` | Loss and accuracy curves (train vs val) |
| `transformer_confusion_matrix.png` | Test set confusion matrix |
| `positional_encoding_visualization.png` | PE heatmap, frequency, and value distribution |
| `positional_encoding_similarity.png` | Cosine similarity vs position distance |
| `attention_weights_visualization.png` | Sample attention weight matrices |

### Model Files (`models/transformer_classifier/`)
| File | Description |
|------|-------------|
| `latest_checkpoint.pth` | Most recent training checkpoint (epoch 50) |
| `best_model.pth` | Model with best validation accuracy (epoch 7) |
| `checkpoint_epoch_*.pth` | Periodic checkpoints every 5 epochs |

---

## 🚀 Usage

### 1. Dataset Preparation
```bash
python TransformerEncoder/dataset_preparation.py > outputs/TransformerEncoder_dataset_preparation.txt
```

### 2. Test Individual Components
```bash
# Test each component independently
python TransformerEncoder/scaled_dotProductAttention.py > outputs/scaled_dotProductAttention.txt
python TransformerEncoder/MultiHead_selfAttention.py > outputs/MultiHead_selfAttention.txt
python TransformerEncoder/positional_encoding.py > outputs/positional_encoding.txt
python TransformerEncoder/PositionWise_feedForward_Network.py > outputs/PositionWise_feedForward_Network.txt
python TransformerEncoder/transformer_encoder.py > outputs/transformer_encoder.txt
```

### 3. Train Full Model
```bash
# Train the complete Transformer classifier
python main.py > outputs/transformer_training.txt
```

### 4. Load Trained Model
```python
import torch
from TransformerEncoder.transformer_encoder import TransformerEncoderBlock

# Load checkpoint
checkpoint = torch.load('models/transformer_classifier/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 📝 Key Findings

1. **Transformer implementation works correctly:** All components pass unit tests including gradient flow, masking, and residual connections.

2. **Severe overfitting due to data scarcity:** With only 145 training samples and 932K parameters, the model memorizes rather than generalizes.

3. **Class collapse:** Model learns to predict the majority class (Economy) for most inputs, achieving 0% recall for 3 of 5 classes.

4. **Pre-LN architecture verified:** Better gradient flow compared to Post-LN, consistent with modern Transformer implementations.

5. **Positional encoding properties confirmed:** Sinusoidal encoding produces consistent norms and higher similarity for nearby positions.

6. **Component parameter distribution:** FFN (131,712) has ~2× parameters of Multi-Head Attention (65,536), following standard design.

---

## 🔗 References

- Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
- Xiong et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.

---

*All implementations are from scratch following assignment constraints: no PyTorch built-in Transformer classes, no HuggingFace, no pretrained models.*
```

This README provides:

1. **Complete file structure** with status indicators
2. **Architecture overview** with parameter counts per component
3. **Dataset statistics** including category distribution and UNK rate
4. **Detailed component documentation** with test results from the output logs
5. **Training configuration** and hyperparameters
6. **Comprehensive results** including per-class metrics and training progress
7. **Critical analysis** of overfitting with root cause analysis
8. **Parameter-to-sample ratio analysis** (6,429 params/sample vs recommended <100)
9. **Actionable recommendations** for improvement at three levels (immediate, medium-term, long-term)
10. **Output file documentation** with shapes and descriptions
11. **Usage instructions** for all scripts

The README is now complete and properly documents the entire TransformerEncoder module with all results from the training logs you provided.