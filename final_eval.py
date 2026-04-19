#!/usr/bin/env python3
"""
FINAL EVALUATION SCRIPT
======================
Comprehensive evaluation comparing all three modules:
1. WordEmbeddings (Word2Vec C3)
2. SequenceLabeling (BiLSTM POS, BiLSTM-CRF NER)
3. TransformerEncoder (Text Classification)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import sys
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix
)
import time

# Load environment variables
load_dotenv()

DEBUG_MODE = int(os.getenv('DEBUG_MODE', 0))
PROCESSOR = os.getenv('PROCESSOR', 'cpu')
dataDir = os.getenv('dataDir', './data')
modelsDir = os.getenv('modelsDir', './models')
resultsDir = os.getenv('resultsDir', './results')
embeddingsDir = os.getenv('embeddingsDir', './embeddings')
figuresDir = os.getenv('figuresDir', './figures')

device = torch.device(PROCESSOR if torch.cuda.is_available() and PROCESSOR == "cuda" else "cpu")

print("=" * 80)
print("FINAL EVALUATION - COMPARING ALL MODULES")
print("=" * 80)
print(f"Device: {device}")

# ============================================================================
# PART 1: LOAD WORD EMBEDDINGS MODEL (C3)
# ============================================================================

print("\n" + "=" * 80)
print("LOADING WORD EMBEDDINGS (C3 - BEST CONDITION)")
print("=" * 80)

sys.path.append(os.path.join(os.path.dirname(__file__), 'WordEmbeddings'))
from eval import WordEmbeddingEvaluator, load_w2v_embeddings

def load_word2vec_c3():
    embeddings_file = os.path.join(embeddingsDir, 'embeddings_w2v_old.npy')
    vocab_file = os.path.join(resultsDir, 'w2v_vocab_old.json')
    
    if os.path.exists(embeddings_file) and os.path.exists(vocab_file):
        embeddings, vocab, word_to_idx = load_w2v_embeddings(embeddings_file, vocab_file)
        evaluator = WordEmbeddingEvaluator(embeddings, vocab, word_to_idx, "Word2Vec")
        print(f"  Loaded Word2Vec C3: {len(vocab)} vocabulary, d={embeddings.shape[1]}")
        
        eval_file = os.path.join(resultsDir, 'c3_w2v_cleaned_evaluation.json')
        if os.path.exists(eval_file):
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_results = json.load(f)
            print(f"  - MRR: {eval_results.get('mrr', 'N/A')}")
            print(f"  - Analogy Accuracy: {eval_results.get('analogy_accuracy', 'N/A')}")
        
        return evaluator, embeddings, vocab, word_to_idx
    else:
        print("  Word2Vec C3 files not found")
        return None, None, None, None

w2v_evaluator, w2v_embeddings, w2v_vocab, w2v_word_to_idx = load_word2vec_c3()

# ============================================================================
# PART 2: LOAD SEQUENCE LABELING MODELS
# ============================================================================

print("\n" + "=" * 80)
print("LOADING SEQUENCE LABELING MODELS")
print("=" * 80)

sys.path.append(os.path.join(os.path.dirname(__file__), 'SequenceLabeling'))
from bi_lstm_sequence_labeller import (
    BiLSTMSequenceLabeler, BiLSTMCRF, SequenceLabelingDataset,
    collate_fn, load_embeddings, load_dataset,
    POS_TAGS, NER_TAGS, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, BATCH_SIZE
)

def load_bilstm_model(task, freeze_embeddings, use_crf=False):
    vocab_size = 1033
    num_pos_tags = len(POS_TAGS)
    num_ner_tags = len(NER_TAGS)
    
    pretrained_embeddings, vocab, word_to_idx = load_embeddings()
    
    model = BiLSTMSequenceLabeler(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=freeze_embeddings,
        num_pos_tags=num_pos_tags,
        num_ner_tags=num_ner_tags,
        task=task
    ).to(device)
    
    mode = 'frozen' if freeze_embeddings else 'fine-tuned'
    checkpoint_path = os.path.join(modelsDir, f'bilstm_{task}_{mode}', 'best_model.pth')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        crf_model = None
        if task == 'ner' and use_crf:
            crf_model = BiLSTMCRF(model, num_ner_tags).to(device)
        
        return model, crf_model, word_to_idx
    else:
        print(f"  Checkpoint not found: {checkpoint_path}")
        return None, None, None

pos_model, _, pos_word_to_idx = load_bilstm_model('pos', freeze_embeddings=False)
if pos_model:
    print(f"  Loaded BiLSTM POS (fine-tuned)")

ner_model, ner_crf, ner_word_to_idx = load_bilstm_model('ner', freeze_embeddings=False, use_crf=True)
if ner_model:
    print(f"  Loaded BiLSTM NER (fine-tuned with CRF)")

# ============================================================================
# PART 3: LOAD TRANSFORMER MODEL
# ============================================================================

print("\n" + "=" * 80)
print("LOADING TRANSFORMER MODEL")
print("=" * 80)

sys.path.append(os.path.join(os.path.dirname(__file__), 'TransformerEncoder'))
from transformer_encoder import TransformerEncoderBlock
from positional_encoding import PositionalEncoding

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size=1033, d_model=128, num_heads=4, num_layers=4,
                 d_ff=512, max_len=256, num_classes=5, dropout=0.1, pad_idx=0):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len + 1, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Fixed: TransformerEncoderBlock takes (d_model, h, d_k, d_v, d_ff, dropout)
        d_k = d_model // num_heads
        d_v = d_model // num_heads
        
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads, d_k, d_v, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tokens, mask=None, return_attention=False):
        batch_size, seq_len = tokens.shape
        
        if mask is None:
            mask = (tokens != self.pad_idx)
        
        x = self.token_embedding(tokens) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        cls_mask = torch.ones(batch_size, 1).to(mask.device)
        extended_mask = torch.cat([cls_mask, mask], dim=1)
        
        attentions = []
        for block in self.encoder_blocks:
            if return_attention:
                x, attn = block(x, extended_mask, return_attention=True)
                attentions.append(attn)
            else:
                x = block(x, extended_mask)
        
        x = self.final_norm(x)
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)
        
        if return_attention:
            return logits, attentions
        return logits

def load_transformer_model():
    vocab_size = 1033
    d_model = 128
    num_heads = 4
    num_layers = 4
    d_ff = 512
    num_classes = 5
    
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        num_classes=num_classes
    ).to(device)
    
    checkpoint_path = os.path.join(modelsDir, 'transformer_classifier', 'best_model.pth')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"  Loaded Transformer classifier")
        print(f"  - Best val accuracy: {checkpoint.get('val_acc', 'N/A')}")
        return model, checkpoint
    else:
        checkpoint_path = os.path.join(modelsDir, 'transformer_classifier', 'latest_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                              if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"  Loaded Transformer classifier (latest checkpoint)")
            return model, checkpoint
        else:
            print("  Transformer checkpoint not found")
            return None, None

transformer_model, transformer_checkpoint = load_transformer_model()

# ============================================================================
# PART 4: EVALUATE TRANSFORMER ON TEST SET
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATING TRANSFORMER ON TEST SET")
print("=" * 80)

test_data = np.load(os.path.join(embeddingsDir, 'transformer_test.npz'))
test_tokens = test_data['tokens']
test_labels = test_data['labels']

with open(os.path.join(resultsDir, 'transformer_categories.json'), 'r', encoding='utf-8') as f:
    categories_data = json.load(f)
    categories = categories_data['categories']
    category_names = [categories[str(i)] for i in range(1, 6)]

with open(os.path.join(resultsDir, 'transformer_test.json'), 'r', encoding='utf-8') as f:
    test_json = json.load(f)

transformer_predictions = []
transformer_attentions = []
transformer_correct_indices = []

if transformer_model:
    transformer_model.eval()
    batch_size = 32
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(test_tokens), batch_size):
            batch_tokens = torch.tensor(test_tokens[i:i+batch_size], dtype=torch.long).to(device)
            logits, attentions = transformer_model(batch_tokens, return_attention=True)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            
            for j, (pred, true) in enumerate(zip(preds, test_labels[i:i+batch_size])):
                if pred == true:
                    transformer_correct_indices.append(i + j)
                    final_attn = attentions[-1][j].cpu().numpy()
                    transformer_attentions.append({
                        'idx': i + j,
                        'attention': final_attn,
                        'true_label': true,
                        'pred_label': pred
                    })
    
    transformer_predictions = np.array(all_preds)
    transformer_acc = accuracy_score(test_labels, transformer_predictions)
    transformer_f1_macro = f1_score(test_labels, transformer_predictions, average='macro', zero_division=0)
    transformer_f1_weighted = f1_score(test_labels, transformer_predictions, average='weighted', zero_division=0)
    
    print(f"\nTransformer Test Results:")
    print(f"  Accuracy: {transformer_acc:.4f}")
    print(f"  Macro F1: {transformer_f1_macro:.4f}")
    print(f"  Weighted F1: {transformer_f1_weighted:.4f}")
    
    transformer_cm = confusion_matrix(test_labels, transformer_predictions, labels=range(1, 6))
    
    precision, recall, f1, support = precision_recall_fscore_support(
        test_labels, transformer_predictions, labels=range(1, 6), zero_division=0
    )
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<8}")
    print("-" * 58)
    for i, name in enumerate(category_names):
        print(f"{name:<20} {precision[i]:.4f}     {recall[i]:.4f}     {f1[i]:.4f}     {support[i]}")

# ============================================================================
# PART 5: EVALUATE BiLSTM POS ON TEST SET
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATING BiLSTM POS ON TEST SET")
print("=" * 80)

pos_acc = None
pos_f1_macro = None
pos_f1_weighted = None

test_annotated_file = os.path.join(dataDir, 'test_annotated.json')
if os.path.exists(test_annotated_file):
    with open(test_annotated_file, 'r', encoding='utf-8') as f:
        test_annotated = json.load(f)
    
    pos_tag_to_idx = {tag: i for i, tag in enumerate(POS_TAGS)}
    pos_test_dataset = SequenceLabelingDataset(
        test_annotated, pos_word_to_idx, pos_tag_to_idx=pos_tag_to_idx, task='pos'
    )
    pos_test_loader = DataLoader(
        pos_test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    
    if pos_model:
        pos_model.eval()
        all_pos_true = []
        all_pos_pred = []
        
        with torch.no_grad():
            for batch in pos_test_loader:
                tokens = batch['tokens'].to(device)
                lengths = batch['lengths']
                pos_tags = batch['pos_tags']
                
                outputs = pos_model(tokens, lengths)
                logits = outputs['pos']
                predictions = logits.argmax(dim=-1)
                
                for b in range(tokens.shape[0]):
                    seq_len = lengths[b].item()
                    true_tags = pos_tags[b][:seq_len].cpu().tolist()
                    pred_tags = predictions[b][:seq_len].cpu().tolist()
                    
                    valid_pairs = [(t, p) for t, p in zip(true_tags, pred_tags) if t != -1]
                    all_pos_true.extend([t for t, _ in valid_pairs])
                    all_pos_pred.extend([p for _, p in valid_pairs])
        
        pos_acc = accuracy_score(all_pos_true, all_pos_pred)
        pos_f1_macro = f1_score(all_pos_true, all_pos_pred, average='macro', zero_division=0)
        pos_f1_weighted = f1_score(all_pos_true, all_pos_pred, average='weighted', zero_division=0)
        
        print(f"\nBiLSTM POS Test Results:")
        print(f"  Accuracy: {pos_acc:.4f}")
        print(f"  Macro F1: {pos_f1_macro:.4f}")
        print(f"  Weighted F1: {pos_f1_weighted:.4f}")

# ============================================================================
# PART 6: EVALUATE BiLSTM NER ON TEST SET
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATING BiLSTM NER ON TEST SET")
print("=" * 80)

ner_acc = None
ner_f1_macro = None
ner_f1_weighted = None

if os.path.exists(test_annotated_file):
    ner_tag_to_idx = {tag: i for i, tag in enumerate(NER_TAGS)}
    ner_test_dataset = SequenceLabelingDataset(
        test_annotated, ner_word_to_idx, ner_tag_to_idx=ner_tag_to_idx, task='ner'
    )
    ner_test_loader = DataLoader(
        ner_test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    
    if ner_model and ner_crf:
        ner_model.eval()
        all_ner_true = []
        all_ner_pred = []
        
        with torch.no_grad():
            for batch in ner_test_loader:
                tokens = batch['tokens'].to(device)
                lengths = batch['lengths']
                mask = batch['mask'].to(device)
                ner_tags = batch['ner_tags']
                
                predictions = ner_crf(tokens, lengths, mask=mask)
                
                for b in range(tokens.shape[0]):
                    seq_len = lengths[b].item()
                    true_tags = ner_tags[b][:seq_len].cpu().tolist()
                    pred_tags = predictions[b][:seq_len]
                    
                    valid_pairs = [(t, p) for t, p in zip(true_tags, pred_tags) if t != -1]
                    all_ner_true.extend([t for t, _ in valid_pairs])
                    all_ner_pred.extend([p for _, p in valid_pairs])
        
        ner_acc = accuracy_score(all_ner_true, all_ner_pred)
        ner_f1_macro = f1_score(all_ner_true, all_ner_pred, average='macro', zero_division=0)
        ner_f1_weighted = f1_score(all_ner_true, all_ner_pred, average='weighted', zero_division=0)
        
        print(f"\nBiLSTM NER Test Results:")
        print(f"  Accuracy: {ner_acc:.4f}")
        print(f"  Macro F1: {ner_f1_macro:.4f}")
        print(f"  Weighted F1: {ner_f1_weighted:.4f}")

# ============================================================================
# PART 7: COMPARISON TABLE
# ============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON SUMMARY")
print("=" * 80)

comparison_data = {
    'Model': ['Word2Vec (C3)', 'BiLSTM', 'BiLSTM-CRF', 'Transformer'],
    'Task': ['Embedding Quality', 'POS Tagging', 'NER', 'Text Classification'],
    'Accuracy': ['-', f'{pos_acc:.4f}' if pos_acc else '-', f'{ner_acc:.4f}' if ner_acc else '-', f'{transformer_acc:.4f}' if transformer_acc else '-'],
    'F1 (Weighted)': ['-', f'{pos_f1_weighted:.4f}' if pos_f1_weighted else '-', f'{ner_f1_weighted:.4f}' if ner_f1_weighted else '-', f'{transformer_f1_weighted:.4f}' if transformer_f1_weighted else '-'],
    'F1 (Macro)': ['-', f'{pos_f1_macro:.4f}' if pos_f1_macro else '-', f'{ner_f1_macro:.4f}' if ner_f1_macro else '-', f'{transformer_f1_macro:.4f}' if transformer_f1_macro else '-'],
    'Parameters': ['103,300', '~400K', '~400K', '932,229'],
    'Training Samples': ['335K pairs', '346 sent', '346 sent', '145 articles']
}

print("\n" + "-" * 110)
print(f"{'Model':<15} {'Task':<20} {'Accuracy':<10} {'F1 (W)':<10} {'F1 (M)':<10} {'Params':<12} {'Train Samples':<15}")
print("-" * 110)
for i in range(len(comparison_data['Model'])):
    print(f"{comparison_data['Model'][i]:<15} "
          f"{comparison_data['Task'][i]:<20} "
          f"{comparison_data['Accuracy'][i]:<10} "
          f"{comparison_data['F1 (Weighted)'][i]:<10} "
          f"{comparison_data['F1 (Macro)'][i]:<10} "
          f"{comparison_data['Parameters'][i]:<12} "
          f"{comparison_data['Training Samples'][i]:<15}")
print("-" * 110)

# ============================================================================
# PART 8: VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

os.makedirs(figuresDir, exist_ok=True)

# Confusion Matrix
if transformer_model:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(transformer_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=category_names, yticklabels=category_names)
    plt.title('Transformer Confusion Matrix (Test Set)', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_path = os.path.join(figuresDir, 'final_transformer_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {cm_path}")

# Attention Heatmaps
if transformer_model and len(transformer_correct_indices) >= 3:
    selected = transformer_attentions[:3]
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Attention Weights from Final Encoder Layer (Heads 1-4)', fontsize=14)
    
    for row_idx, sample in enumerate(selected):
        article_idx = sample['idx']
        article = test_json[article_idx]
        true_label = category_names[sample['true_label'] - 1]
        attention_weights = sample['attention']
        tokens = article['tokens'][:20]
        token_labels = [str(t) for t in tokens]
        
        for head in range(4):
            ax = axes[row_idx, head]
            attn = attention_weights[head, 1:21, 1:21]
            ax.imshow(attn, cmap='Blues', aspect='auto', vmin=0, vmax=attn.max())
            ax.set_title(f'Head {head+1} - {true_label}', fontsize=10)
            ax.set_xticks(range(min(20, len(token_labels))))
            ax.set_xticklabels(token_labels, rotation=90, fontsize=8)
            ax.set_yticks(range(min(20, len(token_labels))))
            ax.set_yticklabels(token_labels, fontsize=8)
    
    plt.tight_layout()
    attn_path = os.path.join(figuresDir, 'final_transformer_attention_heatmaps.png')
    plt.savefig(attn_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {attn_path}")
    
    print("\nArticles used for attention visualization:")
    for i, sample in enumerate(selected[:3]):
        article = test_json[sample['idx']]
        true_label = category_names[sample['true_label'] - 1]
        print(f"  {i+1}. {article['title'][:60]}... ({true_label})")

# Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))

models_for_plot = []
accs_for_plot = []
f1s_for_plot = []

if pos_model and pos_acc is not None:
    models_for_plot.append('BiLSTM\n(POS)')
    accs_for_plot.append(pos_acc)
    f1s_for_plot.append(pos_f1_weighted)

if ner_model and ner_acc is not None:
    models_for_plot.append('BiLSTM-CRF\n(NER)')
    accs_for_plot.append(ner_acc)
    f1s_for_plot.append(ner_f1_weighted)

if transformer_model:
    models_for_plot.append('Transformer\n(Classification)')
    accs_for_plot.append(transformer_acc)
    f1s_for_plot.append(transformer_f1_weighted)

if models_for_plot:
    x = np.arange(len(models_for_plot))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accs_for_plot, width, label='Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, f1s_for_plot, width, label='F1 (Weighted)', color='coral')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models_for_plot)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    comp_path = os.path.join(figuresDir, 'final_model_comparison.png')
    plt.savefig(comp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comp_path}")

# ============================================================================
# PART 9: SAVE REPORT
# ============================================================================

report = {
    'evaluation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
    'transformer': {
        'accuracy': float(transformer_acc) if transformer_acc else None,
        'f1_weighted': float(transformer_f1_weighted) if transformer_f1_weighted else None,
        'f1_macro': float(transformer_f1_macro) if transformer_f1_macro else None,
    },
    'bilstm_pos': {
        'accuracy': float(pos_acc) if pos_acc else None,
        'f1_weighted': float(pos_f1_weighted) if pos_f1_weighted else None,
        'f1_macro': float(pos_f1_macro) if pos_f1_macro else None,
    },
    'bilstm_ner': {
        'accuracy': float(ner_acc) if ner_acc else None,
        'f1_weighted': float(ner_f1_weighted) if ner_f1_weighted else None,
        'f1_macro': float(ner_f1_macro) if ner_f1_macro else None,
    }
}

report_path = os.path.join(resultsDir, 'final_evaluation_report.json')
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
print(f"\n  Saved report: {report_path}")

print("\n" + "=" * 80)
print("FINAL EVALUATION COMPLETE")
print("=" * 80)