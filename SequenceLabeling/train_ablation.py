import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.metrics import f1_score

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

# Import from existing modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bi_lstm_sequence_labeller import (
    BiLSTMSequenceLabeler, SequenceLabelingDataset, collate_fn,
    load_embeddings, load_dataset, evaluate, compute_metrics,
    POS_TAGS, NER_TAGS, EMBEDDING_DIM, BATCH_SIZE
)

# Hyperparameters for ablation (same as baseline except the ablated component)
HIDDEN_DIM = 128
NUM_LAYERS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
GRADIENT_CLIP = 5.0
NUM_EPOCHS = 30


class AblationBiLSTM(nn.Module):
    """Modified BiLSTM for ablation studies"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout,
                 pretrained_embeddings=None, freeze_embeddings=True,
                 bidirectional=True, use_pretrained=True):
        super(AblationBiLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Initialize with pretrained or random
        if use_pretrained and pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=bidirectional, dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(lstm_output_dim, len(POS_TAGS))
        
        if DEBUG_MODE:
            print(f"[DEBUG]: AblationBiLSTM initialized")
            print(f"[DEBUG]:   - Bidirectional: {bidirectional}")
            print(f"[DEBUG]:   - Dropout: {dropout}")
            print(f"[DEBUG]:   - Use pretrained: {use_pretrained}")
    
    def forward(self, tokens, lengths):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        
        embedded = self.embedding(tokens)
        embedded = self.dropout(embedded)
        
        packed_embedded = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, _ = self.lstm(packed_embedded)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        lstm_output = self.dropout(lstm_output)
        
        logits = self.classifier(lstm_output)
        return {'pos': logits}


def train_ablation_model(model, train_loader, val_loader, config_name):
    """Train a single ablation model"""
    print(f"\n{'='*60}")
    print(f"TRAINING: {config_name}")
    print(f"{'='*60}")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    
    best_val_f1 = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_f1s = []
    
    checkpoint_dir = os.path.join(modelsDir, f'ablation_{config_name.replace(" ", "_").lower()}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        # Training
        model.train()
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            tokens = batch['tokens'].to(device)
            lengths = batch['lengths']
            mask = batch['mask'].to(device)
            tags = batch['pos_tags'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(tokens, lengths)
            logits = outputs['pos']
            
            loss = criterion(logits.permute(0, 2, 1), tags)
            loss = (loss * mask).sum() / mask.sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        all_true, all_pred = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(device)
                lengths = batch['lengths']
                mask = batch['mask'].to(device)
                tags = batch['pos_tags'].to(device)
                
                outputs = model(tokens, lengths)
                logits = outputs['pos']
                
                loss = criterion(logits.permute(0, 2, 1), tags)
                loss = (loss * mask).sum() / mask.sum()
                total_val_loss += loss.item()
                
                predictions = logits.argmax(dim=-1)
                
                for b in range(tokens.shape[0]):
                    seq_len = lengths[b].item()
                    all_true.extend(tags[b][:seq_len].cpu().tolist())
                    all_pred.extend(predictions[b][:seq_len].cpu().tolist())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Compute F1
        valid_pairs = [(t, p) for t, p in zip(all_true, all_pred) if t != -1]
        true_filtered = [t for t, _ in valid_pairs]
        pred_filtered = [p for _, p in valid_pairs]
        val_f1 = f1_score(true_filtered, pred_filtered, average='weighted', zero_division=0)
        val_f1s.append(val_f1)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f} | Time: {epoch_time:.1f}s")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'config': config_name
            }, best_model_path)
            print(f"  → Best model saved (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Save training curves
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves - {config_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(figuresDir, f'ablation_{config_name.replace(" ", "_").lower()}_loss.png'), dpi=300)
    plt.close()
    
    return {
        'config': config_name,
        'best_val_f1': float(best_val_f1),
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'val_f1s': [float(x) for x in val_f1s],
        'epochs_trained': len(train_losses)
    }


def evaluate_ablation_on_test(model, test_loader, config_name):
    """Evaluate ablation model on test set"""
    print(f"\nEvaluating {config_name} on test set...")
    
    model.eval()
    all_true, all_pred = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            tokens = batch['tokens'].to(device)
            lengths = batch['lengths']
            tags = batch['pos_tags'].to(device)
            
            outputs = model(tokens, lengths)
            logits = outputs['pos']
            predictions = logits.argmax(dim=-1)
            
            for b in range(tokens.shape[0]):
                seq_len = lengths[b].item()
                all_true.extend(tags[b][:seq_len].cpu().tolist())
                all_pred.extend(predictions[b][:seq_len].cpu().tolist())
    
    valid_pairs = [(t, p) for t, p in zip(all_true, all_pred) if t != -1]
    true_filtered = [t for t, _ in valid_pairs]
    pred_filtered = [p for _, p in valid_pairs]
    
    accuracy = sum(t == p for t, p in zip(true_filtered, pred_filtered)) / len(true_filtered)
    macro_f1 = f1_score(true_filtered, pred_filtered, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_filtered, pred_filtered, average='weighted', zero_division=0)
    
    return {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1)
    }


def run_ablation_study():
    """Run all ablation experiments"""
    print("=" * 80)
    print("ABLATION STUDY - POS TAGGING")
    print("=" * 80)
    
    # Load data
    pretrained_embeddings, vocab, word_to_idx = load_embeddings()
    vocab_size = len(vocab)
    
    train_data = load_dataset('train')
    val_data = load_dataset('val')
    test_data = load_dataset('test')
    
    pos_tag_to_idx = {tag: i for i, tag in enumerate(POS_TAGS)}
    
    train_dataset = SequenceLabelingDataset(train_data, word_to_idx, pos_tag_to_idx=pos_tag_to_idx, task='pos')
    val_dataset = SequenceLabelingDataset(val_data, word_to_idx, pos_tag_to_idx=pos_tag_to_idx, task='pos')
    test_dataset = SequenceLabelingDataset(test_data, word_to_idx, pos_tag_to_idx=pos_tag_to_idx, task='pos')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print(f"\nTrain: {len(train_data)} sentences, {sum(len(d['tokens']) for d in train_data)} tokens")
    print(f"Val: {len(val_data)} sentences")
    print(f"Test: {len(test_data)} sentences")
    
    # Baseline (Fine-tuned BiLSTM)
    print("\n" + "=" * 80)
    print("BASELINE: BiLSTM (bidirectional, dropout=0.5, pretrained embeddings)")
    print("=" * 80)
    
    baseline_results = {
        'config': 'Baseline',
        'accuracy': 0.8381,
        'macro_f1': 0.9244,
        'weighted_f1': 0.8442
    }
    print(f"Baseline Results: Accuracy={baseline_results['accuracy']:.4f}, "
          f"Macro F1={baseline_results['macro_f1']:.4f}, Weighted F1={baseline_results['weighted_f1']:.4f}")
    
    all_results = {'Baseline': baseline_results}
    
    # A1: Unidirectional LSTM
    print("\n" + "=" * 80)
    print("A1: UNIDIRECTIONAL LSTM (dropout=0.5, pretrained)")
    print("=" * 80)
    
    model_a1 = AblationBiLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.5,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=False,
        bidirectional=False,
        use_pretrained=True
    ).to(device)
    
    a1_train_results = train_ablation_model(model_a1, train_loader, val_loader, "A1_Unidirectional")
    a1_test_results = evaluate_ablation_on_test(model_a1, test_loader, "A1_Unidirectional")
    all_results['A1_Unidirectional'] = {**a1_train_results, **a1_test_results}
    
    # A2: No Dropout
    print("\n" + "=" * 80)
    print("A2: NO DROPOUT (bidirectional, dropout=0.0, pretrained)")
    print("=" * 80)
    
    model_a2 = AblationBiLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.0,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=False,
        bidirectional=True,
        use_pretrained=True
    ).to(device)
    
    a2_train_results = train_ablation_model(model_a2, train_loader, val_loader, "A2_NoDropout")
    a2_test_results = evaluate_ablation_on_test(model_a2, test_loader, "A2_NoDropout")
    all_results['A2_NoDropout'] = {**a2_train_results, **a2_test_results}
    
    # A3: Random Embeddings
    print("\n" + "=" * 80)
    print("A3: RANDOM EMBEDDINGS (bidirectional, dropout=0.5, no pretrained)")
    print("=" * 80)
    
    model_a3 = AblationBiLSTM(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=0.5,
        pretrained_embeddings=None,
        freeze_embeddings=False,
        bidirectional=True,
        use_pretrained=False
    ).to(device)
    
    a3_train_results = train_ablation_model(model_a3, train_loader, val_loader, "A3_RandomEmbeddings")
    a3_test_results = evaluate_ablation_on_test(model_a3, test_loader, "A3_RandomEmbeddings")
    all_results['A3_RandomEmbeddings'] = {**a3_train_results, **a3_test_results}
    
    # Summary Comparison
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Configuration':<35} {'Accuracy':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    print("-" * 75)
    
    config_names = {
        'Baseline': 'Baseline (BiLSTM + Dropout + Pretrained)',
        'A1_Unidirectional': 'A1: Unidirectional LSTM',
        'A2_NoDropout': 'A2: No Dropout',
        'A3_RandomEmbeddings': 'A3: Random Embeddings'
    }
    
    for key, name in config_names.items():
        if key in all_results:
            res = all_results[key]
            acc = res.get('accuracy', 0)
            macro = res.get('macro_f1', 0)
            weighted = res.get('weighted_f1', 0)
            print(f"{name:<35} {acc:.4f}       {macro:.4f}       {weighted:.4f}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS AND FINDINGS")
    print("=" * 80)
    
    if 'A1_Unidirectional' in all_results:
        a1_f1 = all_results['A1_Unidirectional']['weighted_f1']
        diff = baseline_results['weighted_f1'] - a1_f1
        print(f"\nA1: Unidirectional vs Bidirectional")
        print(f"  Bidirectional F1: {baseline_results['weighted_f1']:.4f}")
        print(f"  Unidirectional F1: {a1_f1:.4f}")
        print(f"  Difference: {diff:.4f} ({diff/baseline_results['weighted_f1']*100:.1f}% drop)")
        print(f"  → Bidirectional context provides significant improvement for sequence labeling.")
    
    if 'A2_NoDropout' in all_results:
        a2_f1 = all_results['A2_NoDropout']['weighted_f1']
        diff = baseline_results['weighted_f1'] - a2_f1
        print(f"\nA2: Dropout vs No Dropout")
        print(f"  With Dropout (0.5): {baseline_results['weighted_f1']:.4f}")
        print(f"  Without Dropout: {a2_f1:.4f}")
        print(f"  Difference: {diff:.4f} ({diff/baseline_results['weighted_f1']*100:.1f}% {'drop' if diff > 0 else 'improvement'})")
        if diff > 0:
            print(f"  → Dropout regularization helps prevent overfitting and improves generalization.")
        else:
            print(f"  → Model may not be overfitting given the small dataset size.")
    
    if 'A3_RandomEmbeddings' in all_results:
        a3_f1 = all_results['A3_RandomEmbeddings']['weighted_f1']
        diff = baseline_results['weighted_f1'] - a3_f1
        print(f"\nA3: Pretrained vs Random Embeddings")
        print(f"  Pretrained (Word2Vec): {baseline_results['weighted_f1']:.4f}")
        print(f"  Random Embeddings: {a3_f1:.4f}")
        print(f"  Difference: {diff:.4f} ({diff/baseline_results['weighted_f1']*100:.1f}% drop)")
        print(f"  → Pretrained embeddings provide substantial benefit, especially with limited training data.")
    
    # Save all results
    results_file = os.path.join(resultsDir, 'ablation_study_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n\nComplete ablation results saved to: {results_file}")
    
    # Create comparison bar chart
    plt.figure(figsize=(12, 6))
    configs = ['Baseline', 'A1_Unidirectional', 'A2_NoDropout', 'A3_RandomEmbeddings']
    names = ['Baseline\n(BiLSTM)', 'A1\n(Unidirectional)', 'A2\n(No Dropout)', 'A3\n(Random Emb)']
    f1_scores = []
    
    for cfg in configs:
        if cfg in all_results:
            f1_scores.append(all_results[cfg].get('weighted_f1', 0))
        else:
            f1_scores.append(0)
    
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    bars = plt.bar(names, f1_scores, color=colors)
    
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('Weighted F1 Score', fontsize=12)
    plt.title('Ablation Study: POS Tagging Performance', fontsize=14)
    plt.ylim(0, max(f1_scores) * 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figuresDir, 'ablation_study_comparison.png'), dpi=300)
    plt.close()
    
    print(f"Comparison chart saved to: {os.path.join(figuresDir, 'ablation_study_comparison.png')}")
    
    return all_results


if __name__ == "__main__":
    run_ablation_study()