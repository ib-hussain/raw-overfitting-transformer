# Word Embeddings Module

This module implements various word embedding techniques from scratch using PyTorch for Urdu text processing. All implementations follow the assignment constraints: no pretrained models, no Gensim, no HuggingFace, and no built-in Transformer modules.

## 📁 File Structure

| File | Purpose | Key Features |
|------|---------|-------------|
| `tf-idf.py` | TF-IDF weighted document representations | Term-document matrix, top discriminative words per topic |
| `pmi.py` | PPMI word embeddings | Co-occurrence matrix, t-SNE visualization, nearest neighbors |
| `skip-gram_Word2Vec.py` | Skip-gram Word2Vec (configurable) | Negative sampling, checkpointing, configurable hyperparameters |
| `skip-gram_Word2Vec_old.py` | Skip-gram Word2Vec (baseline C3) | 50 epochs, batch=1024, d=100 on cleaned.txt |
| `train_w2v_raw.py` | Skip-gram on raw.txt (C2) | 50 epochs, batch=1024, d=100 on raw corpus |
| `train_w2v_d200.py` | Skip-gram with d=200 (C4) | 50 epochs, batch=1024, d=200 on cleaned.txt |
| `eval.py` | Comprehensive evaluation framework | Nearest neighbors, analogy tests, MRR, four-condition comparison |

## 🎯 Module Overview

### 1. TF-IDF Weighting (`tf-idf.py`)
- Builds term-document matrix from `cleaned.txt`
- Vocabulary restricted to 10,000 most frequent tokens (others → `<UNK>`)
- Implements standard TF-IDF formula: `TF-IDF(w,d) = TF(w,d) × log(N/(1+df(w)))`
- Saves weighted matrix as `tfidf_matrix.npy`
- Reports top-10 discriminative words per topic category

### 2. PPMI Word Embeddings (`pmi.py`)
- Builds word-word co-occurrence matrix with symmetric window size k=5
- Applies Positive PMI weighting: `PPMI(w₁,w₂) = max(0, log₂(P(w₁,w₂)/(P(w₁)P(w₂))))`
- Saves PPMI matrix as `ppmi_matrix.npy`
- Generates t-SNE visualization of top 200 words colored by semantic category
- Finds nearest neighbors using cosine similarity

### 3. Skip-gram Word2Vec (`skip-gram_Word2Vec.py`)
- Implements Skip-gram with negative sampling from scratch
- Separate center (V) and context (U) embedding matrices
- Noise distribution: Pn(w) ∝ f(w)^(3/4)
- Binary cross-entropy loss optimization
- Configurable hyperparameters with checkpointing support

### 4. Evaluation Framework (`eval.py`)
- Loads embeddings from all four conditions (C1-C4)
- Computes nearest neighbors for query words
- Performs analogy tests using vector arithmetic: v(b) - v(a) + v(c)
- Calculates Mean Reciprocal Rank (MRR) on word pairs
- Generates comprehensive comparison reports

## 🔧 Hyperparameters

| Parameter | C1 (PPMI) | C2 (Raw) | C3 (Cleaned) | C4 (d=200) |
|-----------|-----------|----------|--------------|------------|
| Method | PPMI | Skip-gram | Skip-gram | Skip-gram |
| Embedding Dim | 4386 | 100 | 100 | 200 |
| Window Size | 5 | 5 | 5 | 5 |
| Negative Samples | - | 10 | 10 | 10 |
| Batch Size | - | 1024 | 1024 | 1024 |
| Epochs | - | 50 | 50 | 50 |
| Learning Rate | - | 0.001 | 0.001 | 0.001 |
| Min Frequency | - | 5 | 5 | 5 |
| Corpus | cleaned.txt | raw.txt | cleaned.txt | cleaned.txt |

## 📊 Results Summary

### Four-Condition Comparison

| Condition | MRR | Analogy Accuracy | Embedding Dim | Vocab Size | Description |
|-----------|-----|-----------------|---------------|------------|-------------|
| **C1** | 0.1869 | 10.00% | 4386 | 200 | PPMI Baseline |
| **C2** | 0.2699 | 20.00% | 100 | 1083 | Skip-gram on raw.txt |
| **C3** | **0.3145** | **20.00%** | 100 | 1033 | Skip-gram on cleaned.txt |
| **C4** | 0.2693 | 20.00% | 200 | 1033 | Skip-gram d=200 |

### Key Findings

**Best Overall Performance:** Condition C3 (Skip-gram on cleaned.txt with d=100) achieved the highest MRR (0.3145), outperforming all other conditions including the d=200 variant.

**Analogy Accuracy:** C2, C3, and C4 all achieved 20% accuracy (2/10 correct analogies), while C1 achieved only 10% (1/10 correct).

---

## 📈 Detailed Analysis

### 2.2 Evaluation - Nearest Neighbors and Analogy

#### Top-5 Nearest Neighbors (from best model C3)

| Query Word | Nearest Neighbors |
|------------|-------------------|
| **پاکستان** (Pakistan) | ائین, متحدہ, انڈیا, اپوزیشن, انٹرنیشنل |
| **حکومت** (Hukumat) | حمایت, موجودہ, صوبا, طالبان, وفاقی |
| **عدالت** (Adalat) | ضمانت, منظور, بامشقت, ہائیکورٹ, ریپ |
| **معیشت** (Maeeshat) | *Not in vocabulary* |
| **فوج** (Fauj) | پاکستا, عامہ, ترجمان, شعبہ, تعلق |
| **صحت** (Sehat) | *Not in vocabulary* |
| **تعلیم** (Taleem) | اعلی, حاصل, باوجود, سکول, خیبر |
| **آبادی** (Aabadi) | *Not in vocabulary* |

**Note:** Several query words (معیشت, صحت, آبادی) were not present in the vocabulary due to the minimum frequency threshold (min_freq=5). This indicates these terms appear infrequently in the corpus.

#### Analogy Tests (C3 Results)

| # | Analogy | Expected | Top-3 Predictions | ✓/✗ |
|---|---------|----------|-------------------|-----|
| 1 | عمران : خان :: نواز : ? | شریف | مریم, شریف, اصف | ✓ |
| 2 | پاکستان : اسلام :: انڈیا : ? | ہندو | اباد, ہائیکورٹ, گردی | ✗ |
| 3 | وزیر : اعظم :: صدر : ? | مملکت | پوتن, زرداری, مشرف | ✗ |
| 4 | شہباز : شریف :: مریم : ? | نواز | نواز, پنجاب, اعلی | ✓ |
| 5 | پنجاب : لاہور :: سندھ : ? | کراچی | اسلام, تھانہ, رحمان | ✗ |
| 6 | پاکستان : اسلام :: انڈیا : ? | دہلی | اباد, ہائیکورٹ, گردی | ✗ |
| 7 | فوج : پولیس :: عدالت : ? | جج | مرکزی, متاثرہ, دائر | ✗ |
| 8 | عدالت : جج :: ہسپتال : ? | ڈاکٹر | میرا, تاکہ, لایا | ✗ |
| 9 | حکومت : وزیر :: جماعت : ? | رہنما | اعظم, سابق, خان | ✗ |
| 10 | جنگ : فوج :: امن : ? | سیاست | عامہ, غزہ, برقرار | ✗ |

**Correct Analogies:** 2/10 (20%)

**Semantic Relationship Assessment:**
The embeddings capture some meaningful semantic relationships, particularly person-name associations (عمران-خان and شہباز-شریف pairs). However, the model struggles with more complex relational analogies like geographical capitals (پنجاب-لاہور :: سندھ-کراچی) and occupational relationships. This suggests that while the embeddings learn basic co-occurrence patterns, the limited corpus size (210 documents, ~34K tokens) restricts the model's ability to capture deeper semantic relationships that typically require larger training data.

### Four-Condition Comparison Analysis

#### Performance Comparison

| Metric | C1 (PPMI) | C2 (Raw W2V) | C3 (Cleaned W2V) | C4 (d=200 W2V) |
|--------|-----------|--------------|------------------|----------------|
| MRR | 0.1869 | 0.2699 | **0.3145** | 0.2693 |
| Analogy Acc. | 10% | **20%** | **20%** | **20%** |
| Vocab Size | 200 | 1083 | 1033 | 1033 |
| Embedding Dim | 4386 | 100 | 100 | 200 |

#### Discussion

**1. Which condition yields the best embeddings?**

**C3 (Skip-gram on cleaned.txt with d=100) yields the best embeddings** based on MRR (0.3145), significantly outperforming the PPMI baseline (C1: 0.1869) and both alternative Word2Vec configurations (C2: 0.2699, C4: 0.2693).

Key observations:
- **Word2Vec > PPMI:** All Word2Vec variants (C2-C4) outperform PPMI (C1) by substantial margins (44-68% improvement in MRR), demonstrating that prediction-based embeddings capture semantic relationships more effectively than count-based methods.
- **Cleaned > Raw:** C3 (cleaned) achieves better MRR (0.3145) than C2 (raw: 0.2699), suggesting that preprocessing (removing special tokens, normalizing text) improves embedding quality by reducing noise.
- **Vocabulary size matters:** C1's small vocabulary (200 tokens) severely limits its coverage, missing many query words entirely. C2-C4 have 5x larger vocabularies (1000+ tokens), enabling better semantic coverage.

**2. Does increasing embedding dimension (d) help?**

**No, increasing d from 100 to 200 did not improve performance** and actually degraded results:
- C3 (d=100): MRR = **0.3145**
- C4 (d=200): MRR = 0.2693 (↓14.4%)

This counter-intuitive result can be explained by:
- **Limited training data:** With only ~335K training pairs and 1033 vocabulary items, the larger 200-dimensional model has more parameters (206,600 vs 103,300) but insufficient data to train them effectively, leading to overfitting.
- **Diminishing returns:** For small corpora, moderate dimensions (50-100) often suffice; larger dimensions require proportionally more data to learn meaningful representations.
- **Training convergence:** C4's final loss (2.383) was slightly lower than C3's (2.463), but this didn't translate to better semantic representation, suggesting the model may be memorizing noise rather than learning generalizable patterns.

### Recommendations

1. **For this corpus size**, d=100 provides the optimal balance between representational capacity and generalization.
2. **Text preprocessing (cleaning)** demonstrably improves embedding quality and should be applied.
3. **Future improvements** could include:
   - Lowering min_freq to include more vocabulary items
   - Increasing corpus size for better coverage
   - Using subword information for OOV handling
   - Implementing phrase detection for multi-word expressions

## 📦 Output Files

The module generates the following outputs:

| Directory | Files | Description |
|-----------|-------|-------------|
| `embeddings/` | `tfidf_matrix.npy`, `ppmi_matrix.npy`, `embeddings_w2v_*.npy` | Embedding matrices |
| `results/` | `*_evaluation.json`, `*_training_stats.json`, `*_vocab.json` | Evaluation results and vocabulary |
| `figures/` | `tsne_visualization.png`, `w2v_*_loss_curve.png` | Visualizations |
| `models/` | `skipgram_word2vec_*.pth`, `w2v_checkpoint_*.pth` | Trained model checkpoints |

## 🚀 Usage

### Train individual models:
```bash
# C1: PPMI (already included in pmi.py)
python WordEmbeddings/pmi.py > outputs/pmi.txt

# C2: Skip-gram on raw.txt
python WordEmbeddings/train_w2v_raw.py > outputs/WordEmbeddings_eval_c2.txt

# C3: Skip-gram on cleaned.txt (baseline)
python WordEmbeddings/skip-gram_Word2Vec_old.py > outputs/skip-gram_Word2Vec_old.txt

# C4: Skip-gram with d=200
python WordEmbeddings/train_w2v_d200.py > outputs/WordEmbeddings_eval_c4.txt
```

### Run comprehensive evaluation:
```bash
python WordEmbeddings/eval.py > outputs/WordEmbeddings_eval.txt
```

## 📝 Requirements
- Python 3.12.7+
- PyTorch
- NumPy, SciPy
- scikit-learn (for t-SNE)
- Matplotlib
- tqdm
- python-dotenv

This README.md provides:
1. **File documentation** with purpose and features of each file
2. **Module overview** explaining each component
3. **Hyperparameters table** comparing all four conditions
4. **Detailed evaluation results** including nearest neighbors and analogy tests
5. **Semantic relationship assessment** in 2-3 sentences
6. **Recommendations** based on the findings
7. **Usage instructions** for running the code