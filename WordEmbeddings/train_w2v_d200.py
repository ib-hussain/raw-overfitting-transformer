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

device = torch.device(PROCESSOR if torch.cuda.is_available() and PROCESSOR == "cuda" else "cpu")

# Hyperparameters - d=200
EMBEDDING_DIM = 200  # Doubled from 100
WINDOW_SIZE = 5
NEGATIVE_SAMPLES = 10
BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 0.001
MIN_FREQ = 5


class SkipGramDataset(Dataset):
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
                    token_indices.append(0)
            
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
        text = text.replace('<SOS>', '').replace('<EOS>', '')
        tokens = text.split()
        return [t for t in tokens if t]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


class SkipGramWord2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim=200):
        super(SkipGramWord2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self._initialize_embeddings()
        
        if DEBUG_MODE:
            print(f"[DEBUG]: SkipGramWord2Vec initialized")
            print(f"[DEBUG]:   - Vocab size: {vocab_size}")
            print(f"[DEBUG]:   - Embedding dim: {embedding_dim}")
    
    def _initialize_embeddings(self):
        init_range = 0.5 / self.embedding_dim
        nn.init.uniform_(self.center_embeddings.weight, -init_range, init_range)
        nn.init.uniform_(self.context_embeddings.weight, -init_range, init_range)
    
    def forward(self, center_words, context_words, negative_samples):
        center_embeds = self.center_embeddings(center_words)
        pos_context_embeds = self.context_embeddings(context_words)
        
        pos_scores = torch.sum(center_embeds * pos_context_embeds, dim=1)
        pos_scores = torch.sigmoid(pos_scores)
        pos_loss = -torch.log(pos_scores + 1e-10)
        
        neg_context_embeds = self.context_embeddings(negative_samples)
        center_embeds_expanded = center_embeds.unsqueeze(1)
        neg_scores = torch.sum(center_embeds_expanded * neg_context_embeds, dim=2)
        neg_scores = torch.sigmoid(-neg_scores)
        neg_loss = -torch.sum(torch.log(neg_scores + 1e-10), dim=1)
        
        return torch.mean(pos_loss + neg_loss)
    
    def get_embeddings(self):
        with torch.no_grad():
            center = self.center_embeddings.weight.data
            context = self.context_embeddings.weight.data
            return (center + context) / 2.0


class NegativeSampler:
    def __init__(self, word_counts, vocab_size, vocab):
        self.vocab_size = vocab_size
        self.vocab = vocab
        freqs = np.array([word_counts.get(word, 0) for word in vocab])
        freqs = freqs ** 0.75
        self.probs = freqs / freqs.sum()
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Negative sampler initialized")
    
    def sample(self, batch_size, K, positive_indices=None):
        samples = np.random.choice(self.vocab_size, size=(batch_size, K), p=self.probs)
        return torch.tensor(samples, dtype=torch.long, device=device)


def load_documents(filepath):
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
    if DEBUG_MODE:
        print(f"[DEBUG]: Building vocabulary (min_freq={min_freq})")
    
    word_counts = Counter()
    for doc_idx, doc in enumerate(documents):
        text = doc.replace('<SOS>', '').replace('<EOS>', '')
        tokens = text.split()
        word_counts.update([t for t in tokens if t])
        
        if DEBUG_MODE and doc_idx % 50 == 0:
            print(f"[DEBUG]: Processed {doc_idx} documents")
    
    vocab = ['<UNK>']
    for word, count in word_counts.most_common():
        if count >= min_freq and len(vocab) < max_vocab:
            vocab.append(word)
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Vocabulary built with {len(vocab)} tokens")
    
    return vocab, word_to_idx, word_counts


def train_skipgram(model, dataloader, negative_sampler, optimizer, epochs, K=10):
    if DEBUG_MODE:
        print(f"[DEBUG]: Starting training...")
    
    model.train()
    all_losses = []
    epoch_losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for center_words, context_words in progress_bar:
            center_words = center_words.to(device)
            context_words = context_words.to(device)
            batch_size = center_words.size(0)
            
            negative_samples = negative_sampler.sample(batch_size, K)
            loss = model(center_words, context_words, negative_samples)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            all_losses.append(loss_val)
            progress_bar.set_postfix({'loss': f'{loss_val:.4f}'})
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_epoch_loss:.4f} - Time: {elapsed_time:.2f}s")
    
    return all_losses, epoch_losses


def main():
    print("="*60)
    print("TRAINING SKIP-GRAM WORD2VEC WITH d=200 (C4)")
    print("="*60)
    
    cleaned_file = os.path.join(dataDir, 'cleaned.txt')
    documents = load_documents(cleaned_file)
    
    vocab, word_to_idx, word_counts = build_vocabulary(documents, min_freq=MIN_FREQ)
    vocab_size = len(vocab)
    
    # Save vocabulary
    vocab_file = os.path.join(resultsDir, 'w2v_d200_vocab.json')
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump({'vocab': vocab, 'word_to_idx': word_to_idx}, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary saved to: {vocab_file}")
    
    # Create dataset
    dataset = SkipGramDataset(documents, word_to_idx, window_size=WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Training pairs: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print("="*60 + "\n")
    
    # Initialize model
    model = SkipGramWord2Vec(vocab_size, embedding_dim=EMBEDDING_DIM).to(device)
    negative_sampler = NegativeSampler(word_counts, vocab_size, vocab)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train
    all_losses, epoch_losses = train_skipgram(
        model, dataloader, negative_sampler, optimizer,
        epochs=EPOCHS, K=NEGATIVE_SAMPLES
    )
    
    # Save embeddings
    embeddings = model.get_embeddings().cpu().numpy()
    embeddings_file = os.path.join(embeddingsDir, 'embeddings_w2v_d200.npy')
    np.save(embeddings_file, embeddings)
    print(f"\nEmbeddings saved to: {embeddings_file}")
    
    # Save model
    model_file = os.path.join(modelsDir, 'skipgram_word2vec_d200.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'word_to_idx': word_to_idx,
    }, model_file)
    print(f"Model saved to: {model_file}")
    
    # Save training stats
    stats = {
        'vocab_size': vocab_size,
        'training_pairs': len(dataset),
        'epochs': EPOCHS,
        'final_loss': float(all_losses[-1]),
        'epoch_losses': [float(l) for l in epoch_losses],
        'hyperparameters': {
            'embedding_dim': EMBEDDING_DIM,
            'window_size': WINDOW_SIZE,
            'negative_samples': NEGATIVE_SAMPLES,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'corpus': 'cleaned.txt'
        }
    }
    
    stats_file = os.path.join(resultsDir, 'w2v_d200_training_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"Training stats saved to: {stats_file}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(all_losses, alpha=0.5, linewidth=1, color='blue', label='Batch Loss')
    if len(all_losses) > 100:
        window = 100
        moving_avg = np.convolve(all_losses, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(all_losses)), moving_avg, 'r-', linewidth=2, label='Moving Avg (100)')
    plt.title('Skip-gram Word2Vec Training Loss (d=200)', fontsize=14)
    plt.xlabel('Batch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_curve_file = os.path.join(figuresDir, 'w2v_d200_loss_curve.png')
    plt.savefig(loss_curve_file, dpi=300)
    plt.close()
    print(f"Loss curve saved to: {loss_curve_file}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()