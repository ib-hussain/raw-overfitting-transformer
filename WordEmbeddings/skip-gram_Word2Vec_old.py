import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import re
from collections import Counter
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()

DEBUG_MODE = int(os.getenv('DEBUG_MODE', 0))
PROCESSOR = os.getenv('PROCESSOR', 'cpu')
dataDir = os.getenv('dataDir', './data')
modelsDir = os.getenv('modelsDir', './models')
resultsDir = os.getenv('resultsDir', './results')
embeddingsDir = os.getenv('embeddingsDir', './embeddings')
outputsDir = os.getenv('outputsDir', './outputs')
figuresDir = os.getenv('figuresDir', './figures')

# Set device
device = torch.device(PROCESSOR if torch.cuda.is_available() and PROCESSOR == "cuda" else "cpu")

# Hyperparameters
EMBEDDING_DIM = 100  # d
WINDOW_SIZE = 5      # k
NEGATIVE_SAMPLES = 10  # K
BATCH_SIZE = 1024
EPOCHS = 50  # Can be adjusted based on training time
LEARNING_RATE = 0.001
MIN_FREQ = 5  # Minimum frequency to include in vocabulary


class SkipGramDataset(Dataset):
    """Dataset for Skip-gram Word2Vec"""
    def __init__(self, documents, word_to_idx, window_size=5):
        self.word_to_idx = word_to_idx
        self.window_size = window_size
        self.pairs = []
        
        if DEBUG_MODE:
            print("[DEBUG]: Creating Skip-gram training pairs...")
        
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            token_indices = []
            for token in tokens:
                if token in word_to_idx:
                    token_indices.append(word_to_idx[token])
                else:
                    token_indices.append(0)  # <UNK> token
            
            # Create center-context pairs
            for i, center_idx in enumerate(token_indices):
                start = max(0, i - window_size)
                end = min(len(token_indices), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_idx = token_indices[j]
                        self.pairs.append((center_idx, context_idx))
            
            if DEBUG_MODE and doc_idx % 50 == 0:
                print(f"[DEBUG]: Processed document {doc_idx}, total pairs: {len(self.pairs)}")
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Total training pairs created: {len(self.pairs)}")
    
    def _tokenize(self, text):
        """Simple tokenization"""
        text = text.replace('<SOS>', '').replace('<EOS>', '')
        tokens = text.split()
        return [t for t in tokens if t]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


class SkipGramWord2Vec(nn.Module):
    """Skip-gram Word2Vec model from scratch"""
    def __init__(self, vocab_size, embedding_dim=100):
        super(SkipGramWord2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Center embeddings (V)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Context embeddings (U)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self._initialize_embeddings()
        
        if DEBUG_MODE:
            print(f"[DEBUG]: SkipGramWord2Vec initialized")
            print(f"[DEBUG]:   - Vocab size: {vocab_size}")
            print(f"[DEBUG]:   - Embedding dim: {embedding_dim}")
            print(f"[DEBUG]:   - Center embeddings shape: {self.center_embeddings.weight.shape}")
            print(f"[DEBUG]:   - Context embeddings shape: {self.context_embeddings.weight.shape}")
    
    def _initialize_embeddings(self):
        """Initialize embeddings with uniform distribution"""
        init_range = 0.5 / self.embedding_dim
        nn.init.uniform_(self.center_embeddings.weight, -init_range, init_range)
        nn.init.uniform_(self.context_embeddings.weight, -init_range, init_range)
    
    def forward(self, center_words, context_words, negative_samples):
        """
        Forward pass for skip-gram with negative sampling
        
        Args:
            center_words: Tensor of center word indices [batch_size]
            context_words: Tensor of positive context word indices [batch_size]
            negative_samples: Tensor of negative sample indices [batch_size, K]
        
        Returns:
            loss: Binary cross-entropy loss
        """
        batch_size = center_words.size(0)
        
        # Get center embeddings [batch_size, embedding_dim]
        center_embeds = self.center_embeddings(center_words)
        
        # Get positive context embeddings [batch_size, embedding_dim]
        pos_context_embeds = self.context_embeddings(context_words)
        
        # Positive scores: σ(u_o^T v_c)
        pos_scores = torch.sum(center_embeds * pos_context_embeds, dim=1)
        pos_scores = torch.sigmoid(pos_scores)
        pos_loss = -torch.log(pos_scores + 1e-10)
        
        # Negative scores: σ(-u_k^T v_c)
        neg_context_embeds = self.context_embeddings(negative_samples)  # [batch_size, K, embedding_dim]
        center_embeds_expanded = center_embeds.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        neg_scores = torch.sum(center_embeds_expanded * neg_context_embeds, dim=2)  # [batch_size, K]
        neg_scores = torch.sigmoid(-neg_scores)
        neg_loss = -torch.sum(torch.log(neg_scores + 1e-10), dim=1)
        
        # Total loss
        loss = torch.mean(pos_loss + neg_loss)
        
        return loss
    
    def get_embeddings(self):
        """Get averaged final embeddings: 1/2(V + U)"""
        with torch.no_grad():
            center = self.center_embeddings.weight.data
            context = self.context_embeddings.weight.data
            return (center + context) / 2.0


class NegativeSampler:
    """Negative sampler using noise distribution Pn(w) ∝ f(w)^(3/4)"""
    def __init__(self, word_counts, vocab_size, vocab):
        self.vocab_size = vocab_size
        self.vocab = vocab
        
        # Compute noise distribution: Pn(w) ∝ f(w)^(3/4)
        # word_counts is a Counter object with word -> frequency mapping
        freqs = np.array([word_counts.get(word, 0) for word in vocab])
        freqs = freqs ** 0.75  # f(w)^(3/4)
        self.probs = freqs / freqs.sum()
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Negative sampler initialized")
            print(f"[DEBUG]:   - Noise distribution computed")
            print(f"[DEBUG]:   - Sum of probabilities: {self.probs.sum():.4f}")
            print(f"[DEBUG]:   - Max prob: {self.probs.max():.6f}, Min prob: {self.probs.min():.6f}")
    
    def sample(self, batch_size, K, positive_indices=None):
        """
        Sample K negative samples per positive pair
        
        Args:
            batch_size: Number of positive pairs
            K: Number of negative samples per positive pair
            positive_indices: Positive context indices to avoid sampling (not implemented)
        
        Returns:
            Tensor of negative samples [batch_size, K]
        """
        samples = np.random.choice(
            self.vocab_size, 
            size=(batch_size, K), 
            p=self.probs
        )
        return torch.tensor(samples, dtype=torch.long, device=device)


def load_documents(filepath):
    """Load and parse documents from cleaned.txt"""
    if DEBUG_MODE:
        print(f"[DEBUG]: Loading documents from {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    documents = []
    doc_pattern = r'\[\d+\]\s*\n'
    doc_texts = re.split(doc_pattern, content)
    
    for doc_text in doc_texts:
        if doc_text.strip():
            lines = doc_text.strip().split('\n')
            doc_content = ' '.join([line.strip() for line in lines if line.strip()])
            if doc_content:
                documents.append(doc_content)
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Loaded {len(documents)} documents")
    
    return documents


def build_vocabulary(documents, min_freq=5, max_vocab=10000):
    """Build vocabulary from documents"""
    if DEBUG_MODE:
        print(f"[DEBUG]: Building vocabulary (min_freq={min_freq}, max_vocab={max_vocab})")
    
    # Count tokens
    word_counts = Counter()
    for doc_idx, doc in enumerate(documents):
        text = doc.replace('<SOS>', '').replace('<EOS>', '')
        tokens = text.split()
        word_counts.update([t for t in tokens if t])
        
        if DEBUG_MODE and doc_idx % 50 == 0:
            print(f"[DEBUG]: Processed {doc_idx} documents for vocabulary")
    
    # Filter by frequency
    vocab = ['<UNK>']  # Unknown token at index 0
    for word, count in word_counts.most_common():
        if count >= min_freq and len(vocab) < max_vocab:
            vocab.append(word)
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Vocabulary built with {len(vocab)} tokens")
        print(f"[DEBUG]: Top 20 most frequent: {vocab[:20]}")
    
    return vocab, word_to_idx, word_counts


def plot_loss_curve(losses, save_path):
    """Plot and save training loss curve"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2, alpha=0.7)
    
    # Add moving average for smoother curve
    if len(losses) > 100:
        window = 100
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(losses)), moving_avg, 'r-', linewidth=2, label='Moving Average (100)')
        plt.legend()
    
    plt.title('Skip-gram Word2Vec Training Loss', fontsize=14)
    plt.xlabel('Batch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Loss curve saved to {save_path}")


def train_skipgram(model, dataloader, negative_sampler, optimizer, epochs, vocab_size, K=10):
    """Train the Skip-gram model"""
    if DEBUG_MODE:
        print(f"[DEBUG]: Starting training...")
        print(f"[DEBUG]:   - Epochs: {epochs}")
        print(f"[DEBUG]:   - Batches per epoch: {len(dataloader)}")
        print(f"[DEBUG]:   - Negative samples: {K}")
    
    model.train()
    all_losses = []
    epoch_losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (center_words, context_words) in enumerate(progress_bar):
            center_words = center_words.to(device)
            context_words = context_words.to(device)
            batch_size = center_words.size(0)
            
            # Sample negative words
            negative_samples = negative_sampler.sample(batch_size, K)
            
            # Forward pass
            loss = model(center_words, context_words, negative_samples)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record loss
            loss_val = loss.item()
            epoch_loss += loss_val
            all_losses.append(loss_val)
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss_val:.4f}'})
            
            if DEBUG_MODE and batch_idx % 100 == 0 and batch_idx > 0:
                print(f"[DEBUG]: Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss_val:.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} completed - Avg Loss: {avg_epoch_loss:.4f} - Time: {elapsed_time:.2f}s")
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Training completed")
        print(f"[DEBUG]:   - Total time: {time.time() - start_time:.2f}s")
        print(f"[DEBUG]:   - Final loss: {all_losses[-1]:.4f}")
    
    return all_losses, epoch_losses


def main():
    """Main execution function"""
    if DEBUG_MODE:
        print("[DEBUG]: ========================================")
        print("[DEBUG]: Skip-gram Word2Vec(Old) Module Started")
        print("[DEBUG]: ========================================")
        print(f"[DEBUG]: Configuration:")
        print(f"[DEBUG]:   - DEBUG_MODE: {DEBUG_MODE}")
        print(f"[DEBUG]:   - PROCESSOR: {PROCESSOR}")
        print(f"[DEBUG]:   - Device: {device}")
        print(f"[DEBUG]:   - Embedding dim: {EMBEDDING_DIM}")
        print(f"[DEBUG]:   - Window size: {WINDOW_SIZE}")
        print(f"[DEBUG]:   - Negative samples: {NEGATIVE_SAMPLES}")
        print(f"[DEBUG]:   - Batch size: {BATCH_SIZE}")
        print(f"[DEBUG]:   - Epochs: {EPOCHS}")
        print(f"[DEBUG]:   - Learning rate: {LEARNING_RATE}")
    
    # File paths
    cleaned_file = os.path.join(dataDir, 'cleaned.txt')
    model_file = os.path.join(modelsDir, 'skipgram_word2vec_old.pth')
    embeddings_file = os.path.join(embeddingsDir, 'embeddings_w2v_old.npy')
    
    # Load documents
    documents = load_documents(cleaned_file)
    
    # Build vocabulary
    vocab, word_to_idx, word_counts = build_vocabulary(documents, min_freq=MIN_FREQ)
    vocab_size = len(vocab)
    
    # Save vocabulary
    vocab_file = os.path.join(resultsDir, 'w2v_vocab_old.json')
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump({'vocab': vocab, 'word_to_idx': word_to_idx}, f, ensure_ascii=False, indent=2)
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Vocabulary saved to {vocab_file}")
    
    # Create dataset and dataloader
    dataset = SkipGramDataset(documents, word_to_idx, window_size=WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Initialize model
    model = SkipGramWord2Vec(vocab_size, embedding_dim=EMBEDDING_DIM).to(device)
    
    # Initialize negative sampler
    negative_sampler = NegativeSampler(word_counts, vocab_size, vocab)
    
    # Initialize optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    print("TRAINING SKIP-GRAM WORD2VEC")
    print("="*60)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training pairs: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    all_losses, epoch_losses = train_skipgram(
        model, dataloader, negative_sampler, optimizer, 
        epochs=EPOCHS, vocab_size=vocab_size, K=NEGATIVE_SAMPLES
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'word_to_idx': word_to_idx,
        'embedding_dim': EMBEDDING_DIM,
        'window_size': WINDOW_SIZE,
        'negative_samples': NEGATIVE_SAMPLES
    }, model_file)
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Model saved to {model_file}")
    
    # Get and save averaged embeddings
    embeddings = model.get_embeddings().cpu().numpy()
    np.save(embeddings_file, embeddings)
    
    print(f"\nEmbeddings saved to: {embeddings_file}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Plot loss curve
    loss_curve_file = os.path.join(figuresDir, 'w2v_loss_curve_old.png')
    plot_loss_curve(all_losses, loss_curve_file)
    print(f"Loss curve saved to: {loss_curve_file}")
    
    # Plot epoch losses
    epoch_loss_file = os.path.join(figuresDir, 'w2v_epoch_loss_old.png')
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, marker='o', linewidth=2)
    plt.title('Skip-gram Word2Vec Epoch Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(epoch_loss_file, dpi=300)
    plt.close()
    print(f"Epoch loss curve saved to: {epoch_loss_file}")
    
    # Save training statistics
    stats = {
        'vocab_size': vocab_size,
        'training_pairs': len(dataset),
        'epochs': EPOCHS,
        'final_loss': float(all_losses[-1]) if all_losses else None,
        'epoch_losses': [float(l) for l in epoch_losses],
        'hyperparameters': {
            'embedding_dim': EMBEDDING_DIM,
            'window_size': WINDOW_SIZE,
            'negative_samples': NEGATIVE_SAMPLES,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        }
    }
    
    stats_file = os.path.join(resultsDir, 'w2v_training_stats_old.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Training stats saved to {stats_file}")
        print("[DEBUG]: ========================================")
        print("[DEBUG]: Skip-gram Word2Vec(Old) Module Completed")
        print("[DEBUG]: ========================================")
    
    return embeddings, model


if __name__ == "__main__":
    main()