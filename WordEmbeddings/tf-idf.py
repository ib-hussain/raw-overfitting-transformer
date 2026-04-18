import torch
import numpy as np
import json
import re
from collections import Counter
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DEBUG_MODE = int(os.getenv('DEBUG_MODE', 0))
PROCESSOR = os.getenv('PROCESSOR', 'cpu')
dataDir = os.getenv('dataDir', './data')
modelsDir = os.getenv('modelsDir', './models')
resultsDir = os.getenv('resultsDir', './results')
embeddingsDir = os.getenv('embeddingsDir', './embeddings')
outputsDir = os.getenv('outputsDir', './outputs')

# Set device
device = torch.device(PROCESSOR if torch.cuda.is_available() and PROCESSOR == "cuda" else "cpu")

class TFIDFVectorizer:
    """
    TF-IDF Vectorizer implemented from scratch in PyTorch
    """
    def __init__(self, max_vocab_size=10000, unk_token='<UNK>'):
        self.max_vocab_size = max_vocab_size
        self.unk_token = unk_token
        self.vocab = None
        self.word_to_idx = None
        self.idf = None
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
        self.vocab_size = len(self.vocab) + 1  # +1 for UNK
        
        if DEBUG_MODE:
            print(f"[DEBUG]: Vocabulary built with {len(self.vocab)} tokens")
            print(f"[DEBUG]: Top 20 most frequent tokens: {self.vocab[:20]}")
        
        return self.vocab
    
    def _tokenize(self, text):
        """Simple tokenization - split on whitespace and remove punctuation"""
        # Remove special tokens like <SOS>, <EOS> for vocabulary building
        text = text.replace('<SOS>', '').replace('<EOS>', '')
        # Split on whitespace
        tokens = text.split()
        # Remove empty tokens
        tokens = [t for t in tokens if t]
        return tokens
    
    def _get_token_ids(self, tokens):
        """Convert tokens to ids, mapping unknown tokens to UNK"""
        ids = []
        for token in tokens:
            if token in self.word_to_idx:
                ids.append(self.word_to_idx[token])
            else:
                ids.append(len(self.vocab))  # UNK token index
        return ids
    
    def compute_tf(self, documents):
        """Compute Term Frequency matrix"""
        if DEBUG_MODE:
            print(f"[DEBUG]: Computing TF matrix for {len(documents)} documents")
        
        tf_matrix = torch.zeros((len(documents), self.vocab_size), device=device)
        
        for doc_idx, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            token_ids = self._get_token_ids(tokens)
            
            # Count frequencies
            for token_id in token_ids:
                tf_matrix[doc_idx, token_id] += 1
            
            # Normalize by document length
            doc_length = len(token_ids)
            if doc_length > 0:
                tf_matrix[doc_idx] = tf_matrix[doc_idx] / doc_length
            
            if DEBUG_MODE and doc_idx % 50 == 0:
                print(f"[DEBUG]: Processed TF for document {doc_idx}")
        
        if DEBUG_MODE:
            print(f"[DEBUG]: TF matrix shape: {tf_matrix.shape}")
            print(f"[DEBUG]: TF matrix - non-zero elements: {torch.count_nonzero(tf_matrix)}")
        
        return tf_matrix
    
    def compute_idf(self, tf_matrix):
        """Compute Inverse Document Frequency"""
        if DEBUG_MODE:
            print(f"[DEBUG]: Computing IDF from TF matrix of shape {tf_matrix.shape}")
        
        N = tf_matrix.shape[0]  # Number of documents
        
        # Document frequency: number of documents containing each term
        df = torch.sum(tf_matrix > 0, dim=0).float()
        
        # IDF formula: log(N / (1 + df))
        idf = torch.log(N / (1 + df))
        
        if DEBUG_MODE:
            print(f"[DEBUG]: IDF computed for {len(idf)} terms")
            print(f"[DEBUG]: IDF range: [{idf.min():.4f}, {idf.max():.4f}]")
        
        self.idf = idf
        return idf
    
    def fit_transform(self, documents):
        """Fit the vectorizer and transform documents to TF-IDF matrix"""
        if DEBUG_MODE:
            print(f"[DEBUG]: Starting fit_transform on {len(documents)} documents")
            print(f"[DEBUG]: Device: {device}")
        
        # Build vocabulary
        self.build_vocabulary(documents)
        
        # Compute TF
        tf_matrix = self.compute_tf(documents)
        
        # Compute IDF
        idf = self.compute_idf(tf_matrix)
        
        # Compute TF-IDF
        tfidf_matrix = tf_matrix * idf.unsqueeze(0)
        
        if DEBUG_MODE:
            print(f"[DEBUG]: TF-IDF matrix computed")
            print(f"[DEBUG]: TF-IDF matrix shape: {tfidf_matrix.shape}")
            print(f"[DEBUG]: TF-IDF matrix - non-zero elements: {torch.count_nonzero(tfidf_matrix)}")
        
        return tfidf_matrix
    
    def transform(self, documents):
        """Transform new documents using fitted vocabulary and IDF"""
        if self.vocab is None or self.idf is None:
            raise ValueError("Vectorizer must be fitted before transform")
        
        tf_matrix = self.compute_tf(documents)
        tfidf_matrix = tf_matrix * self.idf.unsqueeze(0)
        
        return tfidf_matrix


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
    
    # Filter out empty documents and the initial header
    for doc_text in doc_texts:
        if doc_text.strip():
            # Each document may contain multiple lines with <SOS>...<EOS>
            # We'll combine all lines in the document
            lines = doc_text.strip().split('\n')
            doc_content = ' '.join([line.strip() for line in lines if line.strip()])
            if doc_content:
                documents.append(doc_content)
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Loaded {len(documents)} documents")
        print(f"[DEBUG]: First document sample: {documents[0][:200]}...")
    
    return documents


def load_metadata(filepath):
    """Load metadata.json and extract topics from titles"""
    if DEBUG_MODE:
        print(f"[DEBUG]: Loading metadata from {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        metadata_dict = json.load(f)
    
    # Convert to list and extract topics from titles
    # Since there's no explicit topic field, we'll infer topics from title keywords
    metadata_list = []
    
    # Sort by document ID to maintain order
    for key in sorted(metadata_dict.keys(), key=lambda x: int(x)):
        item = metadata_dict[key]
        
        # Infer topic from title
        title = item.get('title', '')
        topic = infer_topic_from_title(title)
        
        metadata_list.append({
            'doc_id': key,
            'title': title,
            'publish_date': item.get('publish_date', ''),
            'topic': topic
        })
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Loaded metadata for {len(metadata_list)} documents")
        topics = set(item['topic'] for item in metadata_list)
        print(f"[DEBUG]: Inferred topics: {topics}")
        # Show topic distribution
        topic_counts = {}
        for item in metadata_list:
            topic = item['topic']
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        print(f"[DEBUG]: Topic distribution: {topic_counts}")
    
    return metadata_list


def infer_topic_from_title(title):
    """Infer topic category from title keywords"""
    title_lower = title.lower()
    
    # Define topic keywords (in Urdu/English as they appear)
    if any(kw in title for kw in ['پاکستان', 'وزیر', 'حکومت', 'سیاست', 'عمران', 'نواز', 'زرداری', 'بلاول', 'الیکشن', 'اسمبلی']):
        return 'Politics'
    elif any(kw in title for kw in ['کرکٹ', 'کھیل', 'ورلڈ کپ', 'میچ', 'پلیئر', 'ٹیم', 'سیریز']):
        return 'Sports'
    elif any(kw in title for kw in ['دہشت', 'حملہ', 'فوج', 'آپریشن', 'سیکیورٹی', 'پولیس', 'قتل', 'خودکش', 'شدت پسند']):
        return 'Security'
    elif any(kw in title for kw in ['معیشت', 'معاشی', 'ڈالر', 'روپے', 'کاروبار', 'سٹاک', 'مارکیٹ', 'سرمایہ', 'برآمد', 'درآمد']):
        return 'Economy'
    elif any(kw in title for kw in ['امریکہ', 'انڈیا', 'چین', 'افغانستان', 'ایران', 'خارجہ', 'سفیر', 'بین الاقوامی']):
        return 'International'
    elif any(kw in title for kw in ['عدالت', 'مقدمہ', 'قانون', 'جج', 'سپریم کورٹ', 'فیصلہ']):
        return 'Legal'
    elif any(kw in title for kw in ['فلم', 'ڈرامہ', 'اداکار', 'گانا', 'ثقافت', 'موسیقی', 'سنیما']):
        return 'Entertainment'
    elif any(kw in title for kw in ['صحت', 'بیماری', 'ہسپتال', 'ڈاکٹر', 'کینسر', 'علاج']):
        return 'Health'
    elif any(kw in title for kw in ['تعلیم', 'یونیورسٹی', 'سکول', 'طلبہ', 'کالج']):
        return 'Education'
    elif any(kw in title for kw in ['موسم', 'بارش', 'برفباری', 'سردی', 'گرمی', 'طوفان']):
        return 'Weather'
    elif any(kw in title for kw in ['حادثہ', 'آتشزدگی', 'سانحہ', 'زلزلہ']):
        return 'Accidents'
    else:
        return 'General'


def get_top_discriminative_words(tfidf_matrix, vectorizer, metadata, top_k=10):
    """
    Identify top-k most discriminative words per topic category
    Using average TF-IDF score per topic
    """
    if DEBUG_MODE:
        print(f"[DEBUG]: Finding top {top_k} discriminative words per topic")
    
    # Group documents by topic
    topic_docs = {}
    for idx, item in enumerate(metadata):
        if idx >= tfidf_matrix.shape[0]:
            break
        topic = item.get('topic', 'General')
        if topic not in topic_docs:
            topic_docs[topic] = []
        topic_docs[topic].append(idx)
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Topics: {list(topic_docs.keys())}")
        for topic, indices in topic_docs.items():
            print(f"[DEBUG]: Topic '{topic}' has {len(indices)} documents")
    
    # Calculate average TF-IDF per topic
    topic_words = {}
    for topic, doc_indices in topic_docs.items():
        if len(doc_indices) == 0:
            continue
            
        # Get TF-IDF for documents in this topic
        topic_tfidf = tfidf_matrix[doc_indices]
        
        # Average TF-IDF across documents in this topic
        avg_tfidf = torch.mean(topic_tfidf, dim=0)
        
        # Get top-k words
        top_indices = torch.argsort(avg_tfidf, descending=True)[:top_k * 2]  # Get more to filter UNK
        
        top_words = []
        for idx in top_indices:
            idx_val = idx.item()
            if idx_val < len(vectorizer.vocab):
                word = vectorizer.vocab[idx_val]
                score = avg_tfidf[idx].item()
                if word != vectorizer.unk_token and score > 0:
                    top_words.append((word, score))
            if len(top_words) >= top_k:
                break
        
        topic_words[topic] = top_words
    
    return topic_words


def main():
    """Main execution function"""
    if DEBUG_MODE:
        print("[DEBUG]: ========================================")
        print("[DEBUG]: TF-IDF Weighting Module Started")
        print("[DEBUG]: ========================================")
        print(f"[DEBUG]: Configuration:")
        print(f"[DEBUG]:   - DEBUG_MODE: {DEBUG_MODE}")
        print(f"[DEBUG]:   - PROCESSOR: {PROCESSOR}")
        print(f"[DEBUG]:   - Device: {device}")
        print(f"[DEBUG]:   - dataDir: {dataDir}")
    
    # File paths
    cleaned_file = os.path.join(dataDir, 'cleaned.txt')
    metadata_file = os.path.join(dataDir, 'metadata.json')  # Fixed filename
    output_file = os.path.join(embeddingsDir, 'tfidf_matrix.npy')
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Input files:")
        print(f"[DEBUG]:   - Cleaned text: {cleaned_file}")
        print(f"[DEBUG]:   - Metadata: {metadata_file}")
        print(f"[DEBUG]: Output file: {output_file}")
    
    # Load documents
    documents = load_documents(cleaned_file)
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Total documents loaded: {len(documents)}")
    
    # Load metadata
    metadata = load_metadata(metadata_file)
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TFIDFVectorizer(max_vocab_size=10000)
    
    if DEBUG_MODE:
        print("[DEBUG]: Starting TF-IDF computation...")
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Convert to numpy and save
    tfidf_numpy = tfidf_matrix.cpu().numpy()
    np.save(output_file, tfidf_numpy)
    
    if DEBUG_MODE:
        print(f"[DEBUG]: TF-IDF matrix saved to {output_file}")
        print(f"[DEBUG]: Matrix shape: {tfidf_numpy.shape}")
        print(f"[DEBUG]: Matrix size: {tfidf_numpy.nbytes / 1024 / 1024:.2f} MB")
    
    # Get top discriminative words per topic
    topic_words = get_top_discriminative_words(tfidf_matrix, vectorizer, metadata, top_k=10)
    
    # Report results
    print("\n" + "="*60)
    print("TOP-10 MOST DISCRIMINATIVE WORDS PER TOPIC")
    print("="*60)
    
    for topic, words in topic_words.items():
        print(f"\n[Topic: {topic}]")
        print("-" * 40)
        for i, (word, score) in enumerate(words, 1):
            print(f"  {i:2d}. {word:30s} (score: {score:.6f})")
    
    # Save topic words to file
    topic_words_file = os.path.join(resultsDir, 'tf-idf_topic_words.json')
    with open(topic_words_file, 'w', encoding='utf-8') as f:
        # Convert to serializable format
        serializable = {topic: [[w, float(s)] for w, s in words] 
                       for topic, words in topic_words.items()}
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    
    if DEBUG_MODE:
        print(f"\n[DEBUG]: Topic words saved to {topic_words_file}")
        print("[DEBUG]: ========================================")
        print("[DEBUG]: TF-IDF Weighting Module Completed")
        print("[DEBUG]: ========================================")
    
    return tfidf_numpy, topic_words


if __name__ == "__main__":
    main()