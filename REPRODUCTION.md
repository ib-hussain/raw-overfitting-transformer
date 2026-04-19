
# REPRODUCTION GUIDE

This document provides step-by-step instructions to reproduce all results in this project.

## Prerequisites

### Environment Setup
```bash
# 1. Clone repository
git clone https://github.com/ib-hussain/raw-overfitting-transformer.git
cd raw-overfitting-transformer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install torch numpy scipy scikit-learn matplotlib seaborn tqdm python-dotenv

# 4. Configure .env file, edit paths in .env to match your system absolute paths
```

### Required Data Files
Ensure these files exist in `data/` directory:
- `cleaned.txt` - Preprocessed Urdu corpus
- `raw.txt` - Raw Urdu corpus  
- `Metadata.json` - Article metadata (210 articles)

---

## Module 1: Word Embeddings

### C1: PPMI Baseline
```bash
python WordEmbeddings/pmi.py > outputs/pmi.txt
```
**Outputs:**
- `embeddings/ppmi_matrix.npy`
- `figures/tsne_visualization.png`
- `results/ppmi_nearest_neighbors.json`

### C2: Skip-gram on raw.txt
```bash
python WordEmbeddings/train_w2v_raw.py > outputs/WordEmbeddings_eval_c2.txt
```
**Outputs:**
- `embeddings/embeddings_w2v_raw.npy`
- `results/w2v_raw_vocab.json`
- `models/skipgram_word2vec_raw.pth`

### C3: Skip-gram on cleaned.txt (d=100)
```bash
python WordEmbeddings/skip-gram_Word2Vec_old.py > outputs/skip-gram_Word2Vec_old.txt
```
**Outputs:**
- `embeddings/embeddings_w2v_old.npy`
- `results/w2v_vocab_old.json`
- `models/skipgram_word2vec_final.pth`

### C4: Skip-gram on cleaned.txt (d=200)
```bash
python WordEmbeddings/train_w2v_d200.py > outputs/WordEmbeddings_eval_c4.txt
```
**Outputs:**
- `embeddings/embeddings_w2v_d200.npy`
- `results/w2v_d200_vocab.json`
- `models/skipgram_word2vec_d200.pth`

### TF-IDF
```bash
python WordEmbeddings/tf-idf.py > outputs/tf-idf.txt
```
**Outputs:**
- `embeddings/tfidf_matrix.npy`
- `results/tf-idf_topic_words.json`

### Word Embeddings Evaluation (All Conditions)
```bash
python WordEmbeddings/eval.py > outputs/WordEmbeddings_eval.txt
```
**Outputs:**
- `results/c1_ppmi_evaluation.json`
- `results/c2_w2v_raw_evaluation.json`
- `results/c3_w2v_cleaned_evaluation.json`
- `results/c4_w2v_d200_evaluation.json`
- `results/embedding_evaluation_summary_all.json`

---

## Module 2: Sequence Labeling

### Dataset Preparation
```bash
python SequenceLabeling/dataset_preparation.py > outputs/SequenceLabeling_dataset_preparation.txt
```
**Outputs:**
- `data/train_annotated.json` (346 sentences)
- `data/val_annotated.json` (71 sentences)
- `data/test_annotated.json` (83 sentences)
- `data/full_annotated.json` (500 sentences)
- `results/dataset_statistics.json`

### Train BiLSTM Models
```bash
# Trains all 4 configurations (POS frozen/fine-tuned, NER frozen/fine-tuned)
python SequenceLabeling/bi_lstm_sequence_labeller.py > outputs/bi_lstm_sequence_labeller.txt
```
**Outputs:**
- `models/bilstm_pos_frozen/best_model.pth`
- `models/bilstm_pos_fine-tuned/best_model.pth`
- `models/bilstm_ner_frozen/best_model.pth`
- `models/bilstm_ner_fine-tuned/best_model.pth`
- `results/bilstm_*_results.json`
- `results/bilstm_summary.json`
- `figures/bilstm_*_loss.png`

### Sequence Labeling Evaluation
```bash
python SequenceLabeling/eval.py > outputs/SequenceLabelling_eval.txt
```
**Outputs:**
- `figures/pos_confusion_matrix_*.png`
- `results/sequence_labeling_evaluation.json`

### Ablation Study
```bash
python SequenceLabeling/train_ablation.py > outputs/ablation_study.txt
```
**Outputs:**
- `models/ablation_a1_unidirectional/`
- `models/ablation_a2_nodropout/`
- `models/ablation_a3_randomembeddings/`
- `results/ablation_study_results.json`
- `figures/ablation_study_comparison.png`

---

## Module 3: Transformer Encoder

### Dataset Preparation
```bash
python TransformerEncoder/dataset_preparation.py > outputs/TransformerEncoder_dataset_preparation.txt
```
**Outputs:**
- `data/transformer_train.json` (145 articles)
- `data/transformer_val.json` (29 articles)
- `data/transformer_test.json` (36 articles)
- `embeddings/transformer_*.npz`
- `results/transformer_categories.json`
- `results/transformer_dataset_stats.json`

### Test Individual Components
```bash
python TransformerEncoder/scaled_dotProductAttention.py > outputs/scaled_dotProductAttention.txt
python TransformerEncoder/MultiHead_selfAttention.py > outputs/MultiHead_selfAttention.txt
python TransformerEncoder/positional_encoding.py > outputs/positional_encoding.txt
python TransformerEncoder/PositionWise_feedForward_Network.py > outputs/PositionWise_feedForward_Network.txt
python TransformerEncoder/transformer_encoder.py > outputs/transformer_encoder.txt
```
**Outputs:**
- `figures/attention_weights_visualization.png`
- `figures/positional_encoding_visualization.png`
- `figures/positional_encoding_similarity.png`

### Train Transformer Classifier
```bash
python main.py > outputs/transformer_training.txt
```
**Outputs:**
- `models/transformer_classifier/best_model.pth`
- `models/transformer_classifier/latest_checkpoint.pth`
- `results/transformer_results.json`
- `figures/transformer_training_curves.png`
- `figures/transformer_confusion_matrix.png`

---

## Final Evaluation

### Run Complete Evaluation
```bash
python final_eval.py > outputs/final_eval.txt
```
**Outputs:**
- `results/final_evaluation_report.json`
- `figures/final_transformer_confusion_matrix.png`
- `figures/final_transformer_attention_heatmaps.png`
- `figures/final_model_comparison.png`

---

## Expected Results Summary

### Word Embeddings (C3 - Best)
| Metric | Value |
|--------|-------|
| MRR | 0.3145 |
| Analogy Accuracy | 20% |
| Vocab Size | 1,033 |
| Embedding Dim | 100 |

### BiLSTM POS (Fine-tuned)
| Metric | Value |
|--------|-------|
| Accuracy | 83.81% |
| Macro F1 | 0.9244 |
| Weighted F1 | 0.8442 |

### BiLSTM NER (Fine-tuned + CRF)
| Metric | Value |
|--------|-------|
| Overall F1 | 10.83% |
| PER F1 | 2.17% |
| LOC F1 | 5.42% |

### Transformer Classifier
| Metric | Value |
|--------|-------|
| Best Val Acc | 34.48% |
| Test Acc | 27.78% |
| Test F1 | 15.67% |

---

## Troubleshooting

### Checkpoint Not Found
Ensure models are trained before evaluation:
```bash
# Train all models in order
python WordEmbeddings/skip-gram_Word2Vec_old.py
python SequenceLabeling/bi_lstm_sequence_labeller.py
python main.py
```

### CUDA Out of Memory
Reduce batch size in respective files or set `PROCESSOR = "cpu"` in `.env`.

### Import Errors
Run from repository root and ensure all module paths are correct.

---

## File Verification

After running all scripts, verify these key files exist:

```bash
# Embeddings
ls embeddings/embeddings_w2v_old.npy

# Sequence Labeling Models
ls models/bilstm_pos_fine-tuned/best_model.pth
ls models/bilstm_ner_fine-tuned/best_model.pth

# Transformer Model
ls models/transformer_classifier/best_model.pth

# Results
ls results/bilstm_summary.json
ls results/transformer_results.json
ls results/final_evaluation_report.json

# Figures
ls figures/final_model_comparison.png
ls figures/final_transformer_attention_heatmaps.png
```

---

## Complete Reproduction (All Steps)

```bash
# 1. Word Embeddings (C1-C4)
python WordEmbeddings/pmi.py > outputs/pmi.txt
python WordEmbeddings/train_w2v_raw.py > outputs/w2v_raw.txt
python WordEmbeddings/skip-gram_Word2Vec_old.py > outputs/w2v_cleaned.txt
python WordEmbeddings/train_w2v_d200.py > outputs/w2v_d200.txt
python WordEmbeddings/tf-idf.py > outputs/tfidf.txt
python WordEmbeddings/eval.py > outputs/word_embeddings_eval.txt

# 2. Sequence Labeling
python SequenceLabeling/dataset_preparation.py > outputs/seq_dataset_prep.txt
python SequenceLabeling/bi_lstm_sequence_labeller.py > outputs/bilstm_train.txt
python SequenceLabeling/eval.py > outputs/seq_eval.txt
python SequenceLabeling/train_ablation.py > outputs/ablation.txt

# 3. Transformer
python TransformerEncoder/dataset_preparation.py > outputs/trans_dataset_prep.txt
python main.py > outputs/transformer_train.txt

# 4. Final Evaluation
python final_eval.py > outputs/final_eval.txt
```
