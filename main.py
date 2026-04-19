import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import math

# Import transformer components
from TransformerEncoder.positional_encoding import PositionalEncoding
from TransformerEncoder.transformer_encoder import TransformerEncoderBlock

# Load environment variables
load_dotenv()

DEBUG_MODE = int(os.getenv('DEBUG_MODE', 0))
PROCESSOR = os.getenv('PROCESSOR', 'cpu')
dataDir = os.getenv('dataDir', './data')
modelsDir = os.getenv('modelsDir', './models')
resultsDir = os.getenv('resultsDir', './results')
embeddingsDir = os.getenv('embeddingsDir', './embeddings')
figuresDir = os.getenv('figuresDir', './figures')
outputsDir = os.getenv('outputsDir', './outputs')

device = torch.device(PROCESSOR if torch.cuda.is_available() and PROCESSOR == "cuda" else "cpu")

# Hyperparameters
D_MODEL = 128
H = 4
D_K = 32
D_V = 32
D_FF = 512
NUM_LAYERS = 4
DROPOUT = 0.1
MAX_SEQ_LEN = 256
NUM_CLASSES = 5

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50
EPOCHS = 50
CHECKPOINT_INTERVAL = 5
GRADIENT_CLIP = 1.0


class TransformerDataset(Dataset):
    """Dataset for Transformer classification"""
    
    def __init__(self, data_path):
        # Load data from npz file
        data = np.load(data_path)
        self.tokens = torch.tensor(data['tokens'], dtype=torch.long)
        self.labels = torch.tensor(data['labels'], dtype=torch.long) - 1  # 0-indexed
        
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


class TransformerClassifier(nn.Module):
    """
    Complete Transformer Encoder for Text Classification.
    
    Architecture:
        - Token Embedding + Positional Encoding
        - Prepend learned [CLS] token
        - 4 stacked Transformer Encoder blocks (Pre-LN)
        - MLP Classification Head: 128 → 64 → 5
    """
    
    def __init__(self, vocab_size, d_model=128, h=4, d_k=32, d_v=32, d_ff=512, 
                 num_layers=4, max_seq_len=256, num_classes=5, dropout=0.1, pad_idx=0):
        super(TransformerClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.pad_idx = pad_idx
        
        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_seq_len + 1, dropout=dropout)
        
        # Learned [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Stacked Transformer Encoder Blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model, h=h, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final Layer Normalization
        self.final_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        # Classification Head (MLP)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        if DEBUG_MODE:
            print(f"[DEBUG]: TransformerClassifier initialized")
            print(f"[DEBUG]:   - Vocab size: {vocab_size}")
            print(f"[DEBUG]:   - d_model: {d_model}")
            print(f"[DEBUG]:   - num_layers: {num_layers}")
            print(f"[DEBUG]:   - num_classes: {num_classes}")
    
    def _initialize_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def create_padding_mask(self, tokens):
        """Create padding mask for attention"""
        # tokens shape: (batch_size, seq_len)
        # Add 1 for CLS token
        cls_tokens = torch.ones(tokens.size(0), 1, device=tokens.device)
        full_tokens = torch.cat([cls_tokens, tokens], dim=1)
        mask = (full_tokens != self.pad_idx)
        return mask
    
    def forward(self, tokens, return_attention=False):
        """
        Forward pass.
        
        Args:
            tokens: Input token indices of shape (batch_size, seq_len)
            return_attention: Whether to return attention weights
        
        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
            attention_weights: List of attention weights if return_attention=True
        """
        batch_size, seq_len = tokens.shape
        
        # Token Embedding
        x = self.token_embedding(tokens)  # (batch_size, seq_len, d_model)
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_len + 1, d_model)
        
        # Add Positional Encoding
        x = self.positional_encoding(x)
        
        # Create padding mask (includes CLS token)
        mask = self.create_padding_mask(tokens)  # (batch_size, seq_len + 1)
        
        # Pass through encoder blocks
        attention_weights = []
        for encoder_block in self.encoder_blocks:
            if return_attention:
                x, attn = encoder_block(x, mask=mask, return_attention=True)
                attention_weights.append(attn)
            else:
                x = encoder_block(x, mask=mask)
        
        # Final Layer Normalization
        x = self.final_norm(x)
        
        # Extract [CLS] token representation
        cls_output = x[:, 0, :]  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(cls_output)  # (batch_size, num_classes)
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits


class CosineWarmupScheduler:
    """
    Cosine learning rate scheduler with linear warmup.
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self):
        self.current_step += 1
        lr = self._compute_lr()
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = lr * base_lr / self.base_lrs[0]
        return lr
    
    def _compute_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.base_lrs[0] * 0.5 * (1 + math.cos(math.pi * progress))
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def train_epoch(model, dataloader, optimizer, criterion, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for tokens, labels in progress_bar:
        tokens = tokens.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(tokens)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion):
    """Evaluate on validation/test set"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for tokens, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            tokens = tokens.to(device)
            labels = labels.to(device)
            
            logits = model(tokens)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels


def save_checkpoint(model, optimizer, scheduler, epoch, train_losses, val_losses, 
                    train_accs, val_accs, best_val_acc, checkpoint_path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_current_step': scheduler.current_step if scheduler else 0,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'hyperparameters': {
            'd_model': D_MODEL,
            'h': H,
            'num_layers': NUM_LAYERS,
            'd_ff': D_FF,
            'dropout': DROPOUT,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'warmup_steps': WARMUP_STEPS
        }
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"  → Checkpoint saved to: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler:
        scheduler.current_step = checkpoint.get('scheduler_current_step', 0)
    
    print(f"  → Checkpoint loaded from: {checkpoint_path}")
    print(f"  → Resuming from epoch {checkpoint['epoch']}")
    print(f"  → Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    
    return (checkpoint['epoch'], checkpoint['train_losses'], checkpoint['val_losses'],
            checkpoint['train_accs'], checkpoint['val_accs'], checkpoint['best_val_acc'])


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training and validation curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Training curves saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  → Confusion matrix saved to: {save_path}")
    
    return cm


def main():
    """Main training function"""
    print("=" * 80)
    print("TRANSFORMER ENCODER - TEXT CLASSIFICATION")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Configuration:")
    print(f"  - d_model: {D_MODEL}")
    print(f"  - num_heads: {H}")
    print(f"  - num_layers: {NUM_LAYERS}")
    print(f"  - d_ff: {D_FF}")
    print(f"  - dropout: {DROPOUT}")
    print(f"  - batch_size: {BATCH_SIZE}")
    print(f"  - learning_rate: {LEARNING_RATE}")
    print(f"  - weight_decay: {WEIGHT_DECAY}")
    print(f"  - warmup_steps: {WARMUP_STEPS}")
    print(f"  - epochs: {EPOCHS}")
    print("=" * 80)
    
    # Load category mapping
    categories_file = os.path.join(resultsDir, 'transformer_categories.json')
    with open(categories_file, 'r', encoding='utf-8') as f:
        categories_data = json.load(f)
        class_names = [categories_data['categories'][str(i)] for i in range(1, NUM_CLASSES + 1)]
    
    print(f"\nCategories: {class_names}")
    
    # Load datasets
    print("\n" + "-" * 60)
    print("LOADING DATASETS")
    print("-" * 60)
    
    train_data_path = os.path.join(embeddingsDir, 'transformer_train.npz')
    val_data_path = os.path.join(embeddingsDir, 'transformer_val.npz')
    test_data_path = os.path.join(embeddingsDir, 'transformer_test.npz')
    
    train_dataset = TransformerDataset(train_data_path)
    val_dataset = TransformerDataset(val_data_path)
    test_dataset = TransformerDataset(test_data_path)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Load vocabulary to get vocab size
    vocab_file = os.path.join(resultsDir, 'w2v_vocab_old.json')
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        vocab_size = len(vocab_data['vocab'])
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Initialize model
    print("\n" + "-" * 60)
    print("INITIALIZING MODEL")
    print("-" * 60)
    
    model = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        h=H,
        d_k=D_K,
        d_v=D_V,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize scheduler
    total_steps = EPOCHS * len(train_loader)
    scheduler = CosineWarmupScheduler(optimizer, WARMUP_STEPS, total_steps)
    
    # Checkpoint paths
    checkpoint_dir = os.path.join(modelsDir, 'transformer_classifier')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    # Check for existing checkpoint
    start_epoch = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    if os.path.exists(checkpoint_path):
        print("\n" + "-" * 60)
        print("RESUMING FROM CHECKPOINT")
        print("-" * 60)
        start_epoch, train_losses, val_losses, train_accs, val_accs, best_val_acc = load_checkpoint(
            checkpoint_path, model, optimizer, scheduler
        )
        start_epoch += 1
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Starting from epoch {start_epoch + 1}/{EPOCHS}")
    
    # Store first epoch stats for comparison
    first_epoch_stats = None
    if start_epoch == 0 and len(train_losses) == 0:
        first_epoch_stats = None
    elif len(train_losses) > 0:
        first_epoch_stats = {
            'epoch': 1,
            'train_loss': train_losses[0],
            'val_loss': val_losses[0],
            'train_acc': train_accs[0],
            'val_acc': val_accs[0]
        }
    
    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else LEARNING_RATE
        
        print(f"\nEpoch {epoch + 1}/{EPOCHS} (LR: {current_lr:.6f})")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scheduler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save first epoch stats
        if epoch == 0:
            first_epoch_stats = {
                'epoch': 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'hyperparameters': {
                    'd_model': D_MODEL,
                    'h': H,
                    'num_layers': NUM_LAYERS,
                }
            }, best_model_path)
            print(f"  → Best model saved (Val Acc: {val_acc:.4f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_losses, val_losses, train_accs, val_accs,
                best_val_acc, checkpoint_path
            )
    
    # Save final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, EPOCHS - 1,
        train_losses, val_losses, train_accs, val_accs,
        best_val_acc, checkpoint_path
    )
    
    # Plot training curves
    print("\n" + "-" * 60)
    print("PLOTTING TRAINING CURVES")
    print("-" * 60)
    
    curves_path = os.path.join(figuresDir, 'transformer_training_curves.png')
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, curves_path)
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    # Load best model
    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='weighted', zero_division=0
    )
    
    print(f"\nTest Results:")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 56)
    per_class = precision_recall_fscore_support(test_labels, test_preds, average=None, zero_division=0)
    for i, name in enumerate(class_names):
        print(f"{name:<20} {per_class[0][i]:<12.4f} {per_class[1][i]:<12.4f} {per_class[2][i]:<12.4f}")
    
    # Plot confusion matrix
    cm_path = os.path.join(figuresDir, 'transformer_confusion_matrix.png')
    plot_confusion_matrix(test_labels, test_preds, class_names, cm_path)
    
    # Show progress comparison
    print("\n" + "=" * 60)
    print("TRAINING PROGRESS COMPARISON")
    print("=" * 60)
    
    if first_epoch_stats:
        last_epoch = len(train_losses) - 1
        print(f"\nFirst Epoch (Epoch 1):")
        print(f"  Train Loss: {first_epoch_stats['train_loss']:.4f} | Train Acc: {first_epoch_stats['train_acc']:.4f}")
        print(f"  Val Loss:   {first_epoch_stats['val_loss']:.4f} | Val Acc:   {first_epoch_stats['val_acc']:.4f}")
        
        print(f"\nLast Epoch (Epoch {last_epoch + 1}):")
        print(f"  Train Loss: {train_losses[-1]:.4f} | Train Acc: {train_accs[-1]:.4f}")
        print(f"  Val Loss:   {val_losses[-1]:.4f} | Val Acc:   {val_accs[-1]:.4f}")
        
        print(f"\nImprovement:")
        loss_improvement = first_epoch_stats['val_loss'] - val_losses[-1]
        acc_improvement = val_accs[-1] - first_epoch_stats['val_acc']
        print(f"  Val Loss:   {loss_improvement:+.4f} ({loss_improvement/first_epoch_stats['val_loss']*100:+.1f}%)")
        print(f"  Val Acc:    {acc_improvement:+.4f} ({acc_improvement/first_epoch_stats['val_acc']*100:+.1f}%)")
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save results
    results = {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'best_val_acc': float(best_val_acc),
        'train_losses': [float(x) for x in train_losses],
        'val_losses': [float(x) for x in val_losses],
        'train_accs': [float(x) for x in train_accs],
        'val_accs': [float(x) for x in val_accs],
        'hyperparameters': {
            'd_model': D_MODEL,
            'h': H,
            'num_layers': NUM_LAYERS,
            'd_ff': D_FF,
            'dropout': DROPOUT,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'warmup_steps': WARMUP_STEPS,
            'epochs': EPOCHS
        }
    }
    
    results_file = os.path.join(resultsDir, 'transformer_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()