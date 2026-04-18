import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import json
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

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

# Hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 128  # LSTM hidden dimension (each direction)
NUM_LAYERS = 2
DROPOUT = 0.5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5
CHECKPOINT_INTERVAL = 3  # Save checkpoint every N epochs
GRADIENT_CLIP = 5.0

# POS and NER tag mappings
POS_TAGS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'CONJ', 'POST', 'NUM', 'PUNC', 'UNK']
NER_TAGS = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']


class SequenceLabelingDataset(Dataset):
    """Dataset for sequence labeling tasks"""
    
    def __init__(self, data, word_to_idx, pos_tag_to_idx=None, ner_tag_to_idx=None, task='both'):
        self.data = data
        self.word_to_idx = word_to_idx
        self.pos_tag_to_idx = pos_tag_to_idx or {tag: i for i, tag in enumerate(POS_TAGS)}
        self.ner_tag_to_idx = ner_tag_to_idx or {tag: i for i, tag in enumerate(NER_TAGS)}
        self.task = task  # 'pos', 'ner', or 'both'
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        
        # Convert tokens to indices
        token_indices = []
        for token in tokens:
            if token in self.word_to_idx:
                token_indices.append(self.word_to_idx[token])
            else:
                token_indices.append(0)  # <UNK> token
        
        result = {
            'tokens': token_indices,
            'length': len(token_indices),
            'raw_tokens': tokens
        }
        
        if self.task in ['pos', 'both']:
            pos_tags = item['pos_tags']
            pos_indices = [self.pos_tag_to_idx.get(tag, self.pos_tag_to_idx['UNK']) for tag in pos_tags]
            result['pos_tags'] = pos_indices
        
        if self.task in ['ner', 'both']:
            ner_tags = item['ner_tags']
            ner_indices = [self.ner_tag_to_idx.get(tag, self.ner_tag_to_idx['O']) for tag in ner_tags]
            result['ner_tags'] = ner_indices
        
        return result


def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    tokens = [torch.tensor(item['tokens'], dtype=torch.long) for item in batch]
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    
    # Pad sequences
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    
    result = {
        'tokens': padded_tokens,
        'lengths': lengths,
        'raw_tokens': [item['raw_tokens'] for item in batch],
        'mask': (padded_tokens != 0).float()
    }
    
    if 'pos_tags' in batch[0]:
        pos_tags = [torch.tensor(item['pos_tags'], dtype=torch.long) for item in batch]
        padded_pos = pad_sequence(pos_tags, batch_first=True, padding_value=-1)
        result['pos_tags'] = padded_pos
    
    if 'ner_tags' in batch[0]:
        ner_tags = [torch.tensor(item['ner_tags'], dtype=torch.long) for item in batch]
        padded_ner = pad_sequence(ner_tags, batch_first=True, padding_value=-1)
        result['ner_tags'] = padded_ner
    
    return result


class BiLSTMSequenceLabeler(nn.Module):
    """2-layer BiLSTM for sequence labeling"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout,
                 pretrained_embeddings=None, freeze_embeddings=True,
                 num_pos_tags=None, num_ner_tags=None, task='both'):
        super(BiLSTMSequenceLabeler, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.task = task
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Initialize with pretrained embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
                if DEBUG_MODE:
                    print("[DEBUG]: Embeddings frozen")
            elif DEBUG_MODE:
                print("[DEBUG]: Embeddings fine-tunable")
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layers
        lstm_output_dim = hidden_dim * 2  # Bidirectional
        
        if task in ['pos', 'both']:
            self.pos_classifier = nn.Linear(lstm_output_dim, num_pos_tags)
        
        if task in ['ner', 'both']:
            self.ner_classifier = nn.Linear(lstm_output_dim, num_ner_tags)
        
        if DEBUG_MODE:
            print(f"[DEBUG]: BiLSTM initialized")
            print(f"[DEBUG]:   - Vocab size: {vocab_size}")
            print(f"[DEBUG]:   - Embedding dim: {embedding_dim}")
            print(f"[DEBUG]:   - Hidden dim: {hidden_dim}")
            print(f"[DEBUG]:   - Num layers: {num_layers}")
            print(f"[DEBUG]:   - Task: {task}")
    
    def forward(self, tokens, lengths):
        """Forward pass - returns logits"""
        batch_size, seq_len = tokens.shape
        
        # Embedding
        embedded = self.embedding(tokens)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # Pack for LSTM
        packed_embedded = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # BiLSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        lstm_output = self.dropout(lstm_output)  # [batch_size, seq_len, hidden_dim*2]
        
        outputs = {}
        
        if self.task in ['pos', 'both']:
            pos_logits = self.pos_classifier(lstm_output)
            outputs['pos'] = pos_logits
        
        if self.task in ['ner', 'both']:
            ner_logits = self.ner_classifier(lstm_output)
            outputs['ner'] = ner_logits
        
        return outputs


class CRF(nn.Module):
    """Conditional Random Field for NER sequence decoding"""
    
    def __init__(self, num_tags):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        
        # Transition matrix: transitions[i][j] = score from tag j to tag i
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Constraint: never transition to START tag, never transition from END tag
        self.transitions.data[:, 0] = -10000  # START tag at index 0
        self.transitions.data[-1, :] = -10000  # END tag at last index
        
        if DEBUG_MODE:
            print(f"[DEBUG]: CRF initialized with {num_tags} tags")
    
    def forward(self, emissions, mask):
        """Compute log likelihood for CRF"""
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize forward variables
        alpha = torch.full((batch_size, num_tags), -10000.0, device=emissions.device)
        alpha[:, 0] = 0  # START tag
        
        for t in range(seq_len):
            alpha_t = []
            for tag in range(num_tags):
                emit_score = emissions[:, t, tag].unsqueeze(1)
                trans_score = self.transitions[tag, :].unsqueeze(0)
                next_tag_score = alpha + trans_score + emit_score
                alpha_t.append(torch.logsumexp(next_tag_score, dim=1))
            alpha = torch.stack(alpha_t, dim=1)
            
            # Apply mask
            mask_t = mask[:, t].unsqueeze(1)
            alpha = alpha * mask_t + (1 - mask_t) * alpha.detach()
        
        # Final transition to END tag
        alpha = alpha + self.transitions[-1, :].unsqueeze(0)
        log_likelihood = torch.logsumexp(alpha, dim=1)
        
        return log_likelihood
    
    def viterbi_decode(self, emissions, mask):
        """Viterbi decoding for inference"""
        batch_size, seq_len, num_tags = emissions.shape
        
        # Initialize backpointers and viterbi variables
        backpointers = []
        viterbi = torch.full((batch_size, num_tags), -10000.0, device=emissions.device)
        viterbi[:, 0] = 0  # START tag
        
        for t in range(seq_len):
            next_viterbi = []
            backpointer_t = []
            
            for tag in range(num_tags):
                next_tag_score = viterbi + self.transitions[tag, :].unsqueeze(0)
                best_score, best_tag = torch.max(next_tag_score, dim=1)
                
                emit_score = emissions[:, t, tag]
                next_viterbi.append(best_score + emit_score)
                backpointer_t.append(best_tag)
            
            viterbi = torch.stack(next_viterbi, dim=1)
            backpointers.append(torch.stack(backpointer_t, dim=1))
            
            # Apply mask
            mask_t = mask[:, t].unsqueeze(1)
            viterbi = viterbi * mask_t + (1 - mask_t) * viterbi.detach()
        
        # Add transition to END tag
        viterbi = viterbi + self.transitions[-1, :].unsqueeze(0)
        best_score, best_tag = torch.max(viterbi, dim=1)
        
        # Backtrack
        best_paths = []
        for b in range(batch_size):
            path = [best_tag[b].item()]
            for t in range(seq_len - 1, -1, -1):
                if mask[b, t] == 1:
                    path.append(backpointers[t][b, path[-1]].item())
                else:
                    path.append(0)
            path.reverse()
            best_paths.append(path[1:])  # Remove START tag
        
        return best_paths


class BiLSTMCRF(nn.Module):
    """BiLSTM with CRF for NER"""
    
    def __init__(self, bilstm, num_ner_tags):
        super(BiLSTMCRF, self).__init__()
        self.bilstm = bilstm
        self.crf = CRF(num_ner_tags)
        
    def forward(self, tokens, lengths, ner_tags=None, mask=None):
        outputs = self.bilstm(tokens, lengths)
        emissions = outputs['ner']
        
        # Create mask if not provided
        if mask is None:
            mask = (tokens != 0).float()
        
        if ner_tags is not None:
            # Training: compute loss
            log_likelihood = self.crf(emissions, mask)
            batch_size = tokens.shape[0]
            
            # Compute actual score for gold tags
            gold_score = torch.zeros(batch_size, device=emissions.device)
            for b in range(batch_size):
                prev_tag = 0  # START
                for t in range(lengths[b].item()):
                    tag = ner_tags[b, t].item()
                    emit_score = emissions[b, t, tag]
                    trans_score = self.crf.transitions[tag, prev_tag]
                    gold_score[b] += emit_score + trans_score
                    prev_tag = tag
                # Final transition to END
                gold_score[b] += self.crf.transitions[-1, prev_tag]
            
            loss = (log_likelihood - gold_score).mean()
            return loss
        else:
            # Inference: decode
            best_paths = self.crf.viterbi_decode(emissions, mask)
            return best_paths


def load_embeddings():
    """Load Word2Vec embeddings from C3"""
    embeddings_file = os.path.join(embeddingsDir, 'embeddings_w2v_old.npy')
    vocab_file = os.path.join(resultsDir, 'w2v_vocab_old.json')
    
    if not os.path.exists(embeddings_file) or not os.path.exists(vocab_file):
        raise FileNotFoundError("Word2Vec embeddings not found. Run C3 training first.")
    
    embeddings = np.load(embeddings_file)
    
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        vocab = vocab_data['vocab']
        word_to_idx = vocab_data['word_to_idx']
    
    return embeddings, vocab, word_to_idx


def load_dataset(split):
    """Load annotated dataset"""
    filepath = os.path.join(dataDir, f'{split}_annotated.json')
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_metrics(true_tags, pred_tags, tagset):
    """Compute precision, recall, and F1 score"""
    # Flatten lists and remove padding (-1)
    flat_true, flat_pred = [], []
    for t, p in zip(true_tags, pred_tags):
        for tt, pp in zip(t, p):
            if tt != -1:  # Not padding
                flat_true.append(tt)
                flat_pred.append(pp)
    
    f1 = f1_score(flat_true, flat_pred, average='weighted', zero_division=0)
    
    # Per-class F1
    per_class_f1 = f1_score(flat_true, flat_pred, average=None, labels=range(len(tagset)), zero_division=0)
    per_class_dict = {tag: float(f1) for tag, f1 in zip(tagset, per_class_f1)}
    
    return f1, per_class_dict


def train_epoch(model, dataloader, optimizer, criterion, task, crf_model=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        tokens = batch['tokens'].to(device)
        lengths = batch['lengths']
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        if task == 'ner' and crf_model is not None:
            ner_tags = batch['ner_tags'].to(device)
            loss = crf_model(tokens, lengths, ner_tags, mask)
        else:
            outputs = model(tokens, lengths)
            
            if task == 'pos':
                logits = outputs['pos']
                tags = batch['pos_tags'].to(device)
            else:  # ner without CRF
                logits = outputs['ner']
                tags = batch['ner_tags'].to(device)
            
            # Compute loss with mask
            loss = criterion(logits.permute(0, 2, 1), tags)
            loss = (loss * mask).sum() / mask.sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, task, crf_model=None):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    
    all_true_tags = []
    all_pred_tags = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            tokens = batch['tokens'].to(device)
            lengths = batch['lengths']
            mask = batch['mask'].to(device)
            
            if task == 'ner' and crf_model is not None:
                ner_tags = batch['ner_tags'].to(device)
                loss = crf_model(tokens, lengths, ner_tags, mask)
                predictions = crf_model(tokens, lengths, mask=mask)
                
                for b in range(tokens.shape[0]):
                    all_true_tags.append(batch['ner_tags'][b][:lengths[b]].tolist())
                    all_pred_tags.append(predictions[b][:lengths[b]])
            else:
                outputs = model(tokens, lengths)
                
                if task == 'pos':
                    logits = outputs['pos']
                    tags = batch['pos_tags'].to(device)
                else:
                    logits = outputs['ner']
                    tags = batch['ner_tags'].to(device)
                
                loss = criterion(logits.permute(0, 2, 1), tags)
                loss = (loss * mask).sum() / mask.sum()
                
                predictions = logits.argmax(dim=-1)
                
                for b in range(tokens.shape[0]):
                    all_true_tags.append(tags[b][:lengths[b]].cpu().tolist())
                    all_pred_tags.append(predictions[b][:lengths[b]].cpu().tolist())
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader), all_true_tags, all_pred_tags


def save_checkpoint(model, optimizer, epoch, val_f1, task, freeze_mode, checkpoint_path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1,
        'task': task,
        'freeze_mode': freeze_mode
    }
    torch.save(checkpoint, checkpoint_path)
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Checkpoint loaded from {checkpoint_path}")
        print(f"[DEBUG]:   - Resuming from epoch {checkpoint['epoch']} with F1: {checkpoint['val_f1']:.4f}")
    
    return checkpoint['epoch'], checkpoint['val_f1']


def plot_loss_curves(train_losses, val_losses, task, freeze_mode, save_path):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    plt.title(f'{task.upper()} Sequence Labeling - Loss Curves ({freeze_mode})', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Loss curves saved to {save_path}")


def train_model(task, freeze_embeddings=True, num_epochs=50, use_crf=False):
    """Main training function"""
    mode_str = "frozen" if freeze_embeddings else "fine-tuned"
    print("=" * 80)
    print(f"TRAINING BiLSTM FOR {task.upper()} - EMBEDDINGS: {mode_str.upper()}")
    print("=" * 80)
    
    # Load embeddings
    pretrained_embeddings, vocab, word_to_idx = load_embeddings()
    vocab_size = len(vocab)
    
    # Load datasets
    train_data = load_dataset('train')
    val_data = load_dataset('val')
    test_data = load_dataset('test')
    
    print(f"\nTrain sentences: {len(train_data)}")
    print(f"Val sentences: {len(val_data)}")
    print(f"Test sentences: {len(test_data)}")
    
    # Create datasets and dataloaders
    pos_tag_to_idx = {tag: i for i, tag in enumerate(POS_TAGS)}
    ner_tag_to_idx = {tag: i for i, tag in enumerate(NER_TAGS)}
    
    train_dataset = SequenceLabelingDataset(train_data, word_to_idx, pos_tag_to_idx, ner_tag_to_idx, task=task)
    val_dataset = SequenceLabelingDataset(val_data, word_to_idx, pos_tag_to_idx, ner_tag_to_idx, task=task)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    num_pos_tags = len(POS_TAGS)
    num_ner_tags = len(NER_TAGS)
    
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
    
    # CRF model for NER
    crf_model = None
    if task == 'ner' and use_crf:
        crf_model = BiLSTMCRF(model, num_ner_tags).to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    
    # Checkpoint paths
    checkpoint_dir = os.path.join(modelsDir, f'bilstm_{task}_{mode_str}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    
    # Check for existing checkpoint
    start_epoch = 0
    best_val_f1 = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    if os.path.exists(checkpoint_path):
        print("\nResuming from checkpoint...")
        start_epoch, best_val_f1 = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch += 1
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch + 1}...")
    print("-" * 60)
    epoch_useOut=0
    val_f1_useOut=0.0
    train_loss=0.0
    val_loss=0.0
    tagset=None
    val_f1=0.0
    per_class_f1=0.0
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model if crf_model is None else crf_model, 
                                train_loader, optimizer, criterion, task, crf_model)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss, true_tags, pred_tags = evaluate(model if crf_model is None else crf_model,
                                                   val_loader, task, crf_model)
        val_losses.append(val_loss)
        
        # Compute F1
        tagset = POS_TAGS if task == 'pos' else NER_TAGS
        val_f1, per_class_f1 = compute_metrics(true_tags, pred_tags, tagset)
        val_f1_useOut=val_f1
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | Time: {epoch_time:.1f}s")
        
        # Save checkpoint periodically
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(model, optimizer, epoch, val_f1, task, mode_str, checkpoint_path)
            print(f"  → Checkpoint saved at epoch {epoch+1}")
        epoch_useOut=epoch
        val_f1_useOut=val_f1
        
        # Early stopping based on validation F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            val_f1_useOut = val_f1

            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'per_class_f1': per_class_f1
            }, best_model_path)
            print(f"  → Best model saved (F1: {val_f1:.4f})")
            epoch_useOut=epoch
            val_f1_useOut=val_f1

        else:
            patience_counter += 1
            epoch_useOut=epoch
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                epoch_useOut=epoch
                val_f1_useOut=val_f1
                break
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, epoch_useOut, val_f1_useOut, task, mode_str, checkpoint_path)
    
    # Plot loss curves
    loss_curve_path = os.path.join(figuresDir, f'bilstm_{task}_{mode_str}_loss.png')
    plot_loss_curves(train_losses, val_losses, task, mode_str, loss_curve_path)
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    test_dataset = SequenceLabelingDataset(test_data, word_to_idx, pos_tag_to_idx, ner_tag_to_idx, task=task)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    test_loss, test_true, test_pred = evaluate(model if crf_model is None else crf_model,
                                                test_loader, task, crf_model)
    
    test_f1, test_per_class_f1 = compute_metrics(test_true, test_pred, tagset)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test F1 (weighted): {test_f1:.4f}")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    
    # Save results
    results = {
        'task': task,
        'freeze_embeddings': freeze_embeddings,
        'use_crf': use_crf,
        'best_val_f1': float(best_val_f1),
        'test_f1': float(test_f1),
        'test_per_class_f1': test_per_class_f1,
        'epochs_trained': len(train_losses),
        'hyperparameters': {
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY
        }
    }
    
    results_file = os.path.join(resultsDir, f'bilstm_{task}_{mode_str}_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return results


def main():
    """Main execution function"""
    print("=" * 80)
    print("BiLSTM SEQUENCE LABELER")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Configuration:")
    print(f"  - Embedding dim: {EMBEDDING_DIM}")
    print(f"  - Hidden dim: {HIDDEN_DIM}")
    print(f"  - Num layers: {NUM_LAYERS}")
    print(f"  - Dropout: {DROPOUT}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print("=" * 80)
    
    all_results = {}
    
    # Task 1: POS Tagging
    print("\n" + "#" * 80)
    print("# POS TAGGING")
    print("#" * 80)
    
    # POS - Frozen embeddings
    results_pos_frozen = train_model(task='pos', freeze_embeddings=True, num_epochs=50, use_crf=False)
    all_results['pos_frozen'] = results_pos_frozen
    
    # POS - Fine-tuned embeddings
    results_pos_finetuned = train_model(task='pos', freeze_embeddings=False, num_epochs=50, use_crf=False)
    all_results['pos_finetuned'] = results_pos_finetuned
    
    # Task 2: NER
    print("\n" + "#" * 80)
    print("# NAMED ENTITY RECOGNITION")
    print("#" * 80)
    
    # NER - Frozen embeddings
    results_ner_frozen = train_model(task='ner', freeze_embeddings=True, num_epochs=50, use_crf=True)
    all_results['ner_frozen'] = results_ner_frozen
    
    # NER - Fine-tuned embeddings
    results_ner_finetuned = train_model(task='ner', freeze_embeddings=False, num_epochs=50, use_crf=True)
    all_results['ner_finetuned'] = results_ner_finetuned
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - VALIDATION F1 SCORES")
    print("=" * 80)
    print(f"{'Task':<12} {'Embeddings':<15} {'Val F1':<10} {'Test F1':<10}")
    print("-" * 50)
    
    for key, results in all_results.items():
        task = results['task'].upper()
        mode = "Frozen" if results['freeze_embeddings'] else "Fine-tuned"
        val_f1 = results['best_val_f1']
        test_f1 = results['test_f1']
        print(f"{task:<12} {mode:<15} {val_f1:.4f}     {test_f1:.4f}")
    
    # Save complete summary
    summary_file = os.path.join(resultsDir, 'bilstm_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nComplete summary saved to: {summary_file}")
    
    return all_results


if __name__ == "__main__":
    main()