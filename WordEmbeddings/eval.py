import torch
import numpy as np
import json
import os
import sys
from dotenv import load_dotenv
from scipy.spatial.distance import cosine

# Add WordEmbeddings directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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


class WordEmbeddingEvaluator:
    """Comprehensive evaluator for word embeddings"""
    
    def __init__(self, embeddings, vocab, word_to_idx, embedding_type="Word2Vec"):
        self.embeddings = embeddings
        self.vocab = vocab
        self.word_to_idx = word_to_idx
        self.embedding_type = embedding_type
        self.vocab_size = len(vocab)
        self.embedding_dim = embeddings.shape[1]
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Evaluator initialized")
            print(f"[DEBUG]:   - Embedding type: {embedding_type}")
            print(f"[DEBUG]:   - Vocab size: {self.vocab_size}")
            print(f"[DEBUG]:   - Embedding dim: {self.embedding_dim}")
            print(f"[DEBUG]:   - Embeddings shape: {embeddings.shape}")
    
    def get_vector(self, word):
        """Get embedding vector for a word"""
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.embeddings[idx]
        return None
    
    def compute_similarity(self, word1, word2):
        """Compute cosine similarity between two words"""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)
        
        if vec1 is None or vec2 is None:
            return None
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def find_nearest_neighbors(self, query_word, top_k=10, exclude_query=True):
        """Find top-k nearest neighbors for a query word"""
        query_vec = self.get_vector(query_word)
        
        if query_vec is None:
            if DEBUG_MODE:
                print(f"[DEBUG]: Query word '{query_word}' not in vocabulary")
            return []
        
        similarities = []
        query_idx = self.word_to_idx.get(query_word, -1)
        
        for i, word in enumerate(self.vocab):
            if exclude_query and i == query_idx:
                continue
            
            vec = self.embeddings[i]
            norm1 = np.linalg.norm(query_vec)
            norm2 = np.linalg.norm(vec)
            
            if norm1 == 0 or norm2 == 0:
                sim = 0.0
            else:
                sim = np.dot(query_vec, vec) / (norm1 * norm2)
            
            similarities.append((word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def analogy(self, a, b, c, top_k=3):
        """
        Solve analogy: a : b :: c : ?
        Using vector arithmetic: v(b) - v(a) + v(c)
        """
        vec_a = self.get_vector(a)
        vec_b = self.get_vector(b)
        vec_c = self.get_vector(c)
        
        if vec_a is None or vec_b is None or vec_c is None:
            if DEBUG_MODE:
                print(f"[DEBUG]: One of analogy words not in vocab: {a}, {b}, {c}")
            return []
        
        # Target vector
        target_vec = vec_b - vec_a + vec_c
        
        # Find closest words (excluding a, b, c)
        exclude_set = {a, b, c}
        similarities = []
        
        for word in self.vocab:
            if word in exclude_set:
                continue
            
            vec = self.get_vector(word)
            if vec is None:
                continue
            
            norm1 = np.linalg.norm(target_vec)
            norm2 = np.linalg.norm(vec)
            
            if norm1 == 0 or norm2 == 0:
                sim = 0.0
            else:
                sim = np.dot(target_vec, vec) / (norm1 * norm2)
            
            similarities.append((word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def compute_mrr(self, word_pairs):
        """
        Compute Mean Reciprocal Rank for word pairs
        Each pair: (word1, word2) - they should be similar
        """
        ranks = []
        
        for word1, word2 in word_pairs:
            if word1 not in self.word_to_idx:
                if DEBUG_MODE:
                    print(f"[DEBUG]: Word '{word1}' not in vocab, skipping")
                continue
            
            # Find neighbors of word1
            neighbors = self.find_nearest_neighbors(word1, top_k=len(self.vocab))
            neighbor_words = [w for w, _ in neighbors]
            
            # Find rank of word2
            if word2 in neighbor_words:
                rank = neighbor_words.index(word2) + 1
                ranks.append(1.0 / rank)
            else:
                ranks.append(0.0)
        
        if not ranks:
            return 0.0
        
        return np.mean(ranks)


def load_ppmi_embeddings(filepath, vocab_file):
    """Load PPMI embeddings"""
    if DEBUG_MODE:
        print(f"[DEBUG]: Loading PPMI embeddings from {filepath}")
    
    embeddings = np.load(filepath)
    
    # Load vocabulary from the vocab file
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        vocab = vocab_data.get('vocabulary', [])
    
    # Ensure vocab size matches embeddings
    if len(vocab) > embeddings.shape[0]:
        vocab = vocab[:embeddings.shape[0]]
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    return embeddings, vocab, word_to_idx


def load_w2v_embeddings(filepath, vocab_file):
    """Load Word2Vec embeddings"""
    if DEBUG_MODE:
        print(f"[DEBUG]: Loading Word2Vec embeddings from {filepath}")
    
    embeddings = np.load(filepath)
    
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        vocab = vocab_data.get('vocab', [])
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    return embeddings, vocab, word_to_idx


def create_analogy_tests():
    """Create 10 analogy tests for evaluation - using words known to be in vocabulary"""
    analogies = [
        # Politics analogies - using words that should be in vocab
        ('عمران', 'خان', 'نواز', 'شریف'),  # Imran : Khan :: Nawaz : Sharif
        ('پاکستان', 'اسلام', 'انڈیا', 'ہندو'),  # Pakistan : Islam :: India : Hindu (approximate)
        ('وزیر', 'اعظم', 'صدر', 'مملکت'),  # Minister : Prime :: President : Republic
        
        # People analogies
        ('شہباز', 'شریف', 'مریم', 'نواز'),  # Shehbaz : Sharif :: Maryam : Nawaz
        
        # Location analogies
        ('پنجاب', 'لاہور', 'سندھ', 'کراچی'),  # Punjab : Lahore :: Sindh : Karachi
        ('پاکستان', 'اسلام آباد', 'انڈیا', 'دہلی'),  # Pakistan : Islamabad :: India : Delhi
        
        # Security analogies
        ('فوج', 'جرنیل', 'پولیس', 'افسر'),  # Army : General :: Police : Officer
        ('عدالت', 'جج', 'ہسپتال', 'ڈاکٹر'),  # Court : Judge :: Hospital : Doctor
        
        # Action analogies
        ('حکومت', 'وزیر', 'جماعت', 'رہنما'),  # Government : Minister :: Party : Leader
        ('جنگ', 'فوج', 'امن', 'سیاست'),  # War : Army :: Peace : Politics
    ]
    
    return analogies


def create_word_pairs():
    """Create 20 manually labeled word pairs for MRR evaluation"""
    pairs = [
        # Politics pairs
        ('پاکستان', 'اسلام آباد'),
        ('عمران', 'خان'),
        ('نواز', 'شریف'),
        ('حکومت', 'وزیر'),
        ('صدر', 'وزیر'),
        ('پنجاب', 'لاہور'),
        ('سندھ', 'کراچی'),
        
        # Security pairs
        ('فوج', 'پولیس'),
        ('عدالت', 'جج'),
        ('قتل', 'سزا'),
        ('حملہ', 'دفاع'),
        ('پولیس', 'افسر'),
        
        # People pairs
        ('شہباز', 'شریف'),
        ('مریم', 'نواز'),
        
        # Location pairs
        ('پاکستان', 'پنجاب'),
        ('انڈیا', 'دہلی'),
        ('امریکہ', 'واشنگٹن'),
        
        # Economy pairs
        ('روپے', 'ڈالر'),
        ('بینک', 'قرض'),
        
        # General
        ('ہسپتال', 'ڈاکٹر'),
    ]
    
    return pairs[:20]


def create_query_words():
    """Create query words for evaluation"""
    return {
        'pakistan': 'پاکستان',
        'hukumat': 'حکومت',
        'adalat': 'عدالت',
        'maeeshat': 'معیشت',
        'fauj': 'فوج',
        'sehat': 'صحت',
        'taleem': 'تعلیم',
        'aabadi': 'آبادی'
    }


def evaluate_condition(condition_id, condition_name, embeddings_file, vocab_file, 
                       embedding_type, output_prefix):
    """Evaluate a single condition"""
    print("\n" + "="*80)
    print(f"EVALUATING {condition_id}: {condition_name}")
    print("="*80)
    
    # Load embeddings
    if embedding_type == "PPMI":
        embeddings, vocab, word_to_idx = load_ppmi_embeddings(embeddings_file, vocab_file)
    else:
        embeddings, vocab, word_to_idx = load_w2v_embeddings(embeddings_file, vocab_file)
    
    # Initialize evaluator
    evaluator = WordEmbeddingEvaluator(embeddings, vocab, word_to_idx, embedding_type)
    
    results = {
        'condition_id': condition_id,
        'condition_name': condition_name,
        'embedding_type': embedding_type,
        'vocab_size': len(vocab),
        'embedding_dim': embeddings.shape[1]
    }
    
    # 1. Nearest neighbors for query words
    print("\n" + "-"*60)
    print("NEAREST NEIGHBORS")
    print("-"*60)
    
    query_words = create_query_words()
    neighbors_results = {}
    
    for eng_name, urdu_word in query_words.items():
        neighbors = evaluator.find_nearest_neighbors(urdu_word, top_k=5)
        neighbors_results[urdu_word] = [[w, float(s)] for w, s in neighbors]
        
        print(f"\n{urdu_word} ({eng_name}):")
        if neighbors:
            for i, (word, sim) in enumerate(neighbors, 1):
                print(f"  {i}. {word:25s} (sim: {sim:.4f})")
        else:
            print("  No neighbors found (word not in vocabulary)")
    
    results['nearest_neighbors'] = neighbors_results
    
    # 2. Analogy tests
    print("\n" + "-"*60)
    print("ANALOGY TESTS")
    print("-"*60)
    
    analogies = create_analogy_tests()
    analogy_results = []
    correct_count = 0
    
    for a, b, c, expected in analogies:
        predictions = evaluator.analogy(a, b, c, top_k=3)
        pred_words = [w for w, _ in predictions]
        is_correct = expected in pred_words if predictions else False
        
        if is_correct:
            correct_count += 1
            status = "✓"
        else:
            status = "✗"
        
        analogy_results.append({
            'analogy': f"{a} : {b} :: {c} : ?",
            'expected': expected,
            'predictions': [[w, float(s)] for w, s in predictions] if predictions else [],
            'correct': is_correct
        })
        
        print(f"\n{status} {a} : {b} :: {c} : ?")
        print(f"   Expected: {expected}")
        if predictions:
            print(f"   Top-3 predictions: {', '.join(pred_words)}")
        else:
            print(f"   Top-3 predictions: [words not in vocabulary]")
    
    results['analogy_tests'] = analogy_results
    results['analogy_accuracy'] = correct_count / len(analogies) if analogies else 0
    
    print(f"\nAnalogy Accuracy: {correct_count}/{len(analogies)} = {results['analogy_accuracy']:.2%}")
    
    # 3. MRR on word pairs
    print("\n" + "-"*60)
    print("MRR EVALUATION")
    print("-"*60)
    
    word_pairs = create_word_pairs()
    mrr = evaluator.compute_mrr(word_pairs)
    results['mrr'] = float(mrr)
    
    print(f"MRR on {len(word_pairs)} word pairs: {mrr:.4f}")
    
    # Save results
    output_file = os.path.join(resultsDir, f'{output_prefix}_evaluation.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    """Main evaluation function"""
    print("="*80)
    print("WORD EMBEDDINGS EVALUATION - FOUR CONDITION COMPARISON")
    print("="*80)
    
    all_results = {}
    
    # C1: PPMI Baseline
    print("\n" + "="*80)
    print("CONDITION C1: PPMI BASELINE")
    print("="*80)
    
    ppmi_file = os.path.join(embeddingsDir, 'ppmi_matrix.npy')
    ppmi_vocab = os.path.join(resultsDir, 'ppmi_vocab_categories.json')
    
    if os.path.exists(ppmi_file) and os.path.exists(ppmi_vocab):
        results_c1 = evaluate_condition(
            'C1', 'PPMI Baseline',
            ppmi_file, ppmi_vocab,
            'PPMI', 'c1_ppmi'
        )
        all_results['C1'] = results_c1
    else:
        print(f"PPMI files not found: {ppmi_file} or {ppmi_vocab}")
    
    # C3: Skip-gram on cleaned.txt (old - 50 epochs, batch 1024)
    print("\n" + "="*80)
    print("CONDITION C3: SKIP-GRAM ON cleaned.txt")
    print("="*80)
    
    cleaned_embeddings = os.path.join(embeddingsDir, 'embeddings_w2v_old.npy')
    cleaned_vocab = os.path.join(resultsDir, 'w2v_vocab_old.json')
    
    if os.path.exists(cleaned_embeddings) and os.path.exists(cleaned_vocab):
        results_c3 = evaluate_condition(
            'C3', 'Skip-gram on cleaned.txt',
            cleaned_embeddings, cleaned_vocab,
            'Word2Vec', 'c3_w2v_cleaned'
        )
        all_results['C3'] = results_c3
    else:
        print(f"Cleaned embeddings not found: {cleaned_embeddings} or {cleaned_vocab}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print("\nCondition | MRR | Analogy Accuracy | Embedding Dim | Vocab Size")
    print("-" * 75)
    
    for cond_id, results in all_results.items():
        mrr = results.get('mrr', 0.0)
        acc = results.get('analogy_accuracy', 0.0)
        dim = results.get('embedding_dim', 0)
        vocab = results.get('vocab_size', 0)
        print(f"{cond_id:9} | {mrr:.4f} | {acc:15.2%} | {dim:13} | {vocab}")
    
    # Save all results
    summary_file = os.path.join(resultsDir, 'embedding_evaluation_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nComplete summary saved to: {summary_file}")
    
    return all_results


if __name__ == "__main__":
    main()