import torch
import numpy as np
import json
import re
from collections import Counter
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.font_manager as fm
from scipy.spatial.distance import cosine

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

# Try to find a font that supports Urdu/Arabic script
try:
    # Try to use a font that supports Arabic script
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
except:
    pass


class PPMIVectorizer:
    """
    Positive Pointwise Mutual Information (PPMI) Vectorizer
    """
    def __init__(self, max_vocab_size=10000, window_size=5, unk_token='<UNK>'):
        self.max_vocab_size = max_vocab_size
        self.window_size = window_size
        self.unk_token = unk_token
        self.vocab = None
        self.word_to_idx = None
        self.cooccurrence_matrix = None
        self.ppmi_matrix = None
        self.vocab_size = 0
        
    def build_vocabulary(self, documents):
        """Build vocabulary from documents, keeping top max_vocab_size tokens"""
        if DEBUG_MODE:
            print(f"[DEBUG]: Building vocabulary from {len(documents)} documents")
            print(f"[DEBUG]: Max vocabulary size: {self.max_vocab_size}")
        
        # Count all tokens
        word_counts = Counter()
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            word_counts.update(tokens)
            if DEBUG_MODE and doc_idx % 50 == 0:
                print(f"[DEBUG]: Processed {doc_idx} documents for vocabulary")
        
        # Get most common tokens
        most_common = word_counts.most_common(self.max_vocab_size)
        self.vocab = [word for word, _ in most_common]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Vocabulary built with {len(self.vocab)} tokens")
            print(f"[DEBUG]: Top 20 most frequent tokens: {self.vocab[:20]}")
        
        return self.vocab
    
    def _tokenize(self, text):
        """Simple tokenization - split on whitespace"""
        # Remove special tokens like <SOS>, <EOS>
        text = text.replace('<SOS>', '').replace('<EOS>', '')
        # Split on whitespace
        tokens = text.split()
        # Remove empty tokens
        tokens = [t for t in tokens if t]
        return tokens
    
    def build_cooccurrence_matrix(self, documents):
        """Build word-word co-occurrence matrix with symmetric context window"""
        if DEBUG_MODE:
            print(f"[DEBUG]: Building co-occurrence matrix with window size {self.window_size}")
        
        # Initialize co-occurrence matrix
        self.cooccurrence_matrix = torch.zeros((self.vocab_size, self.vocab_size), device=device)
        
        total_tokens_processed = 0
        
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            
            # Convert tokens to indices (skip unknown tokens)
            token_indices = []
            for token in tokens:
                if token in self.word_to_idx:
                    token_indices.append(self.word_to_idx[token])
            
            # Build co-occurrence counts
            for i, center_idx in enumerate(token_indices):
                # Define context window
                start = max(0, i - self.window_size)
                end = min(len(token_indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:  # Don't count self co-occurrence
                        context_idx = token_indices[j]
                        self.cooccurrence_matrix[center_idx, context_idx] += 1
            
            total_tokens_processed += len(token_indices)
            
            if DEBUG_MODE and doc_idx % 50 == 0:
                print(f"[DEBUG]: Processed document {doc_idx}, total tokens: {total_tokens_processed}")
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Co-occurrence matrix built")
            print(f"[DEBUG]: Matrix shape: {self.cooccurrence_matrix.shape}")
            print(f"[DEBUG]: Total tokens processed: {total_tokens_processed}")
            print(f"[DEBUG]: Non-zero elements: {torch.count_nonzero(self.cooccurrence_matrix)}")
            print(f"[DEBUG]: Total co-occurrence sum: {self.cooccurrence_matrix.sum():.0f}")
        
        return self.cooccurrence_matrix
    
    def compute_ppmi(self):
        """Compute Positive Pointwise Mutual Information (PPMI) matrix"""
        if DEBUG_MODE:
            print(f"[DEBUG]: Computing PPMI matrix")
        
        # Get total co-occurrence sum
        total_sum = self.cooccurrence_matrix.sum()
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Total co-occurrence sum: {total_sum:.0f}")
        
        # Compute word probabilities (marginal probabilities)
        word_sums = self.cooccurrence_matrix.sum(dim=1)  # Sum of co-occurrences for each word
        word_probs = word_sums / total_sum
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Word probabilities computed")
            print(f"[DEBUG]: Min word prob: {word_probs.min():.6f}")
            print(f"[DEBUG]: Max word prob: {word_probs.max():.6f}")
        
        # Compute joint probabilities
        joint_probs = self.cooccurrence_matrix / total_sum
        
        # Compute PMI
        # PMI(w1, w2) = log2(P(w1, w2) / (P(w1) * P(w2)))
        self.ppmi_matrix = torch.zeros_like(self.cooccurrence_matrix, device=device)
        
        for i in range(self.vocab_size):
            for j in range(self.vocab_size):
                if joint_probs[i, j] > 0:
                    pmi = torch.log2(joint_probs[i, j] / (word_probs[i] * word_probs[j] + 1e-10))
                    # PPMI: max(0, PMI)
                    self.ppmi_matrix[i, j] = torch.max(torch.tensor(0.0, device=device), pmi)
            
            if DEBUG_MODE and i % 1000 == 0:
                print(f"[DEBUG]: Computed PPMI for row {i}/{self.vocab_size}")
        
        if DEBUG_MODE:
            print(f"[DEBUG]: PPMI matrix computed")
            print(f"[DEBUG]: PPMI matrix - non-zero elements: {torch.count_nonzero(self.ppmi_matrix)}")
            print(f"[DEBUG]: PPMI matrix - max value: {self.ppmi_matrix.max():.4f}")
            print(f"[DEBUG]: PPMI matrix - mean value: {self.ppmi_matrix.mean():.4f}")
        
        return self.ppmi_matrix
    
    def fit_transform(self, documents):
        """Build co-occurrence matrix and compute PPMI"""
        if DEBUG_MODE:
            print(f"[DEBUG]: Starting fit_transform on {len(documents)} documents")
            print(f"[DEBUG]: Device: {device}")
        
        # Build vocabulary
        self.build_vocabulary(documents)
        
        # Build co-occurrence matrix
        self.build_cooccurrence_matrix(documents)
        
        # Compute PPMI
        self.compute_ppmi()
        
        return self.ppmi_matrix


def load_documents(filepath):
    """Load and parse documents from cleaned.txt"""
    if DEBUG_MODE:
        print(f"[DEBUG]: Loading documents from {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by document markers [1], [2], etc.
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


def get_semantic_categories(vocab):
    """
    Assign semantic categories to words based on simple keyword matching
    """
    categories = {}
    
    # Define category keywords (in Urdu)
    category_keywords = {
        'Politics': ['پاکستان', 'حکومت', 'وزیر', 'صدر', 'الیکشن', 'جماعت', 'سیاست', 'پارلیمان', 'قومی', 'اسمبلی', 
                     'عمران', 'نواز', 'شریف', 'زرداری', 'بھٹو', 'تحریک', 'انصاف', 'مسلم', 'لیگ', 'پیپلز'],
        'Security': ['فوج', 'پولیس', 'حملہ', 'دہشت', 'سیکیورٹی', 'قتل', 'ہلاک', 'گرفتار', 'مقدمہ', 'عدالت',
                    'فائرنگ', 'بم', 'دھماکہ', 'خودکش', 'شدت', 'پسند', 'مسلح', 'جنگی', 'آپریشن', 'کارروائی'],
        'Sports': ['کرکٹ', 'میچ', 'ٹیم', 'کھلاڑی', 'ورلڈ', 'کپ', 'سیریز', 'پاکستان', 'انڈیا', 'فٹبال',
                  'کھیل', 'سٹیڈیم', 'چیمپئن', 'ٹرافی', 'رنز', 'وکٹ', 'بال', 'کیچ', 'فیلڈ', 'باؤلنگ'],
        'Economy': ['روپے', 'ڈالر', 'معیشت', 'کاروبار', 'سرمایہ', 'بینک', 'مارکیٹ', 'سٹاک', 'قیمت', 'مہنگائی',
                   'برآمد', 'درآمد', 'تجارت', 'صنعت', 'ترسیل', 'زر', 'بجٹ', 'ٹیکس', 'آمدنی', 'قرض'],
        'International': ['امریکہ', 'انڈیا', 'چین', 'افغانستان', 'ایران', 'سعودی', 'عرب', 'ترکی', 'روس', 'برطانیہ',
                         'عالمی', 'بین', 'الاقوامی', 'سفیر', 'خارجہ', 'اقوام', 'متحدہ', 'ممالک', 'سرحد', 'معاہدہ'],
        'Health': ['ہسپتال', 'ڈاکٹر', 'مریض', 'بیماری', 'علاج', 'صحت', 'دوائی', 'کینسر', 'کورونا', 'وائرس',
                  'سرجری', 'ایمرجنسی', 'میڈیکل', 'نرس', 'کلینک', 'ٹیسٹ', 'ویکسین', 'انجکشن', 'درد', 'زخمی'],
        'Education': ['سکول', 'کالج', 'یونیورسٹی', 'طلبہ', 'اساتذہ', 'تعلیم', 'امتحان', 'ڈگری', 'کلاس', 'کتاب',
                     'پڑھائی', 'مضمون', 'سائنس', 'ریاضی', 'تاریخ', 'ادب', 'شعبہ', 'فیکلٹی', 'سٹوڈنٹ', 'ایجوکیشن'],
        'Media': ['میڈیا', 'ٹی', 'وی', 'چینل', 'صحافی', 'خبر', 'سوشل', 'ٹوئٹر', 'فیس', 'بک', 'یوٹیوب',
                 'پوسٹ', 'وائرل', 'ویڈیو', 'تصویر', 'رپورٹ', 'انٹرویو', 'پروگرام', 'نشر', 'اخبار'],
        'Legal': ['عدالت', 'جج', 'وکیل', 'مقدمہ', 'قانون', 'سزا', 'ضمانت', 'گرفتار', 'جیل', 'فیصلہ',
                 'سپریم', 'کورٹ', 'ہائیکورٹ', 'سیشن', 'جج', 'مجرمان', 'الزام', 'ثبوت', 'گواہ', 'کارروائی'],
        'Religion': ['اللہ', 'اسلام', 'مسجد', 'نماز', 'قرآن', 'مذہب', 'علماء', 'امام', 'مولوی', 'دینی',
                    'عید', 'رمضان', 'حج', 'عمرہ', 'روزہ', 'مکہ', 'مدینہ', 'رسول', 'نبی', 'صحابہ'],
        'Accidents': ['حادثہ', 'آگ', 'دھماکہ', 'زلزلہ', 'سیلاب', 'طوفان', 'بارش', 'برفباری', 'ہلاکت', 'زخمی',
                     'ریسکیو', 'امدادی', 'تباہ', 'نقصان', 'متاثر', 'بچاؤ', 'امداد', 'سانحہ', 'المیہ', 'جاں', 'بحق'],
        'Transport': ['گاڑی', 'سڑک', 'ٹریفک', 'بس', 'رکشہ', 'موٹر', 'سائیکل', 'ہوائی', 'جہاز', 'پرواز',
                     'ایئرپورٹ', 'ٹرین', 'ریلوے', 'سفر', 'مسافر', 'ڈرائیور', 'پائلٹ', 'شاہراہ', 'موٹروے', 'حادثہ']
    }
    
    for word in vocab:
        assigned = False
        for category, keywords in category_keywords.items():
            if word in keywords:
                categories[word] = category
                assigned = True
                break
        if not assigned:
            categories[word] = 'Other'
    
    return categories


def compute_cosine_similarity(matrix, idx1, idx2):
    """Compute cosine similarity between two vectors"""
    vec1 = matrix[idx1].cpu().numpy()
    vec2 = matrix[idx2].cpu().numpy()
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return 1 - cosine(vec1, vec2)


def find_nearest_neighbors(ppmi_matrix, vectorizer, query_words, top_k=5):
    """Find top-k nearest neighbors by cosine similarity"""
    if DEBUG_MODE:
        print(f"[DEBUG]: Finding nearest neighbors for {len(query_words)} query words")
    
    results = {}
    
    for query in query_words:
        if query not in vectorizer.word_to_idx:
            if DEBUG_MODE:
                print(f"[DEBUG]: Query word '{query}' not in vocabulary")
            continue
        
        query_idx = vectorizer.word_to_idx[query]
        similarities = []
        
        for i in range(vectorizer.vocab_size):
            if i != query_idx:
                sim = compute_cosine_similarity(ppmi_matrix, query_idx, i)
                similarities.append((vectorizer.vocab[i], sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        results[query] = similarities[:top_k]
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Found neighbors for '{query}'")
    
    return results


def visualize_tsne(ppmi_matrix, vectorizer, n_words=200, output_file=f'{figuresDir}/tsne_visualization.png'):
    """Create t-SNE visualization of top n_words most frequent tokens"""
    if DEBUG_MODE:
        print(f"[DEBUG]: Creating t-SNE visualization for top {n_words} words")
    
    # Get top n_words and their vectors
    top_words = vectorizer.vocab[:n_words]
    top_vectors = ppmi_matrix[:n_words].cpu().numpy()
    
    # Get semantic categories
    categories = get_semantic_categories(top_words)
    
    # Apply t-SNE
    if DEBUG_MODE:
        print(f"[DEBUG]: Applying t-SNE...")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter_without_progress=1000)
    vectors_2d = tsne.fit_transform(top_vectors)
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Define colors for categories
    category_colors = {
        'Politics': "#000000",
        'Security': "#00FFEE",
        'Sports': "#FF8800",
        'Economy': "#00FF2A",
        'International': "#E5FF00",
        'Health': "#FF00B3",
        'Education': "#1100FF",
        'Media': "#FF0000",
        'Legal': "#E3A0FF",
        'Religion': "#137200",
        'Accidents': "#B85300",
        'Transport': "#640000",
        'Other': "#FFFFFF"
    }
    
    # Plot points by category
    for category in category_colors.keys():
        category_indices = [i for i, word in enumerate(top_words) if categories.get(word, 'Other') == category]
        if category_indices:
            plt.scatter(
                vectors_2d[category_indices, 0],
                vectors_2d[category_indices, 1],
                c=category_colors[category],
                label=category,
                alpha=0.6,
                s=50
            )
    
    # Add labels for some interesting points (every 10th word to avoid clutter)
    for i, word in enumerate(top_words):
        if i % 10 == 0 and categories.get(word) != 'Other':
            plt.annotate(
                word, 
                (vectors_2d[i, 0], vectors_2d[i, 1]),
                fontsize=8,
                alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5)
            )
    
    plt.title(f't-SNE Visualization of Top {n_words} Words (PPMI Embeddings)', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(resultsDir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if DEBUG_MODE:
        print(f"[DEBUG]: t-SNE visualization saved to {output_path}")
    
    plt.close()
    
    return output_path


def main():
    """Main execution function"""
    if DEBUG_MODE:
        print("[DEBUG]: ========================================")
        print("[DEBUG]: PPMI Weighting Module Started")
        print("[DEBUG]: ========================================")
        print(f"[DEBUG]: Configuration:")
        print(f"[DEBUG]:   - DEBUG_MODE: {DEBUG_MODE}")
        print(f"[DEBUG]:   - PROCESSOR: {PROCESSOR}")
        print(f"[DEBUG]:   - Device: {device}")
        print(f"[DEBUG]:   - dataDir: {dataDir}")
    
    # File paths
    cleaned_file = os.path.join(dataDir, 'cleaned.txt')
    output_file = os.path.join(embeddingsDir, 'ppmi_matrix.npy')
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Input file: {cleaned_file}")
        print(f"[DEBUG]: Output file: {output_file}")
    
    # Load documents
    documents = load_documents(cleaned_file)
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Total documents loaded: {len(documents)}")
    
    # Create and fit PPMI vectorizer
    vectorizer = PPMIVectorizer(max_vocab_size=10000, window_size=5)
    
    if DEBUG_MODE:
        print("[DEBUG]: Starting PPMI computation...")
    
    ppmi_matrix = vectorizer.fit_transform(documents)
    
    # Convert to numpy and save
    ppmi_numpy = ppmi_matrix.cpu().numpy()
    np.save(output_file, ppmi_numpy)
    
    if DEBUG_MODE:
        print(f"[DEBUG]: PPMI matrix saved to {output_file}")
        print(f"[DEBUG]: Matrix shape: {ppmi_numpy.shape}")
        print(f"[DEBUG]: Matrix size: {ppmi_numpy.nbytes / 1024 / 1024:.2f} MB")
    
    # Create t-SNE visualization
    print("\n" + "="*60)
    print("CREATING t-SNE VISUALIZATION")
    print("="*60)
    
    viz_path = visualize_tsne(ppmi_matrix, vectorizer, n_words=200)
    print(f"t-SNE visualization saved to: {viz_path}")
    
    # Find nearest neighbors for query words
    print("\n" + "="*60)
    print("TOP-5 NEAREST NEIGHBORS BY COSINE SIMILARITY")
    print("="*60)
    
    # Define query words (important words from the corpus)
    query_words = [
        'پاکستان',
        'حکومت',
        'وزیر',
        'کرکٹ',
        'پولیس',
        'عدالت',
        'امریکہ',
        'انڈیا',
        'عمران',
        'فوج',
        'معیشت',
        'تعلیم'
    ]
    
    neighbors = find_nearest_neighbors(ppmi_matrix, vectorizer, query_words, top_k=5)
    
    for query, neighbor_list in neighbors.items():
        print(f"\n[Query: {query}]")
        print("-" * 40)
        for i, (word, sim) in enumerate(neighbor_list, 1):
            print(f"  {i}. {word:20s} (similarity: {sim:.4f})")
    
    # Save nearest neighbors to file
    neighbors_file = os.path.join(resultsDir, 'ppmi_nearest_neighbors.json')
    with open(neighbors_file, 'w', encoding='utf-8') as f:
        json.dump({q: [[w, float(s)] for w, s in n] for q, n in neighbors.items()}, 
                 f, ensure_ascii=False, indent=2)
    
    if DEBUG_MODE:
        print(f"\n[DEBUG]: Nearest neighbors saved to {neighbors_file}")
        
        # Save vocabulary and semantic categories
        vocab_info = {
            'vocabulary': vectorizer.vocab[:200],
            'semantic_categories': get_semantic_categories(vectorizer.vocab[:200])
        }
        vocab_file = os.path.join(resultsDir, 'ppmi_vocab_categories.json')
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_info, f, ensure_ascii=False, indent=2)
        print(f"[DEBUG]: Vocabulary and categories saved to {vocab_file}")
        
        print("[DEBUG]: ========================================")
        print("[DEBUG]: PPMI Weighting Module Completed")
        print("[DEBUG]: ========================================")
    
    return ppmi_numpy, neighbors


if __name__ == "__main__":
    main()