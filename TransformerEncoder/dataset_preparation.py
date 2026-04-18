import json
import re
import os
import numpy as np
from collections import Counter, defaultdict
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

DEBUG_MODE = int(os.getenv('DEBUG_MODE', 0))
embeddingsDir = os.getenv('embeddingsDir', './embeddings')
dataDir = os.getenv('dataDir', './data')
resultsDir = os.getenv('resultsDir', './results')
outputsDir = os.getenv('outputsDir', './outputs')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# CATEGORY DEFINITIONS
# ============================================================================
CATEGORIES = {
    1: {
        'name': 'Politics',
        'keywords': [
            'election', 'government', 'minister', 'parliament', 'party', 'vote',
            'democracy', 'president', 'senate', 'assembly', 'political',
            'الیکشن', 'انتخابات', 'حکومت', 'وزیر', 'پارلیمان', 'جماعت', 'ووٹ',
            'جمہوریت', 'صدر', 'سینیٹ', 'اسمبلی', 'سیاسی', 'پاکستان', 'تحریک',
            'انصاف', 'مسلم', 'لیگ', 'پیپلز', 'پارٹی', 'عمران', 'نواز', 'شریف'
        ]
    },
    2: {
        'name': 'Sports',
        'keywords': [
            'cricket', 'match', 'team', 'player', 'score', 'tournament',
            'football', 'hockey', 'sports', 'stadium', 'champion', 'league',
            'کرکٹ', 'میچ', 'ٹیم', 'کھلاڑی', 'سکور', 'ٹورنامنٹ', 'فٹبال',
            'ہاکی', 'کھیل', 'سٹیڈیم', 'چیمپئن', 'لیگ', 'ورلڈ', 'کپ',
            'پاکستان', 'کرکٹ', 'بورڈ', 'پی', 'سی', 'بی'
        ]
    },
    3: {
        'name': 'Economy',
        'keywords': [
            'inflation', 'trade', 'bank', 'GDP', 'budget', 'economy',
            'stock', 'market', 'investment', 'export', 'import', 'finance',
            'مہنگائی', 'تجارت', 'بینک', 'جی', 'ڈی', 'پی', 'بجٹ', 'معیشت',
            'سٹاک', 'مارکیٹ', 'سرمایہ', 'کاری', 'برآمد', 'درآمد', 'مالیات',
            'روپے', 'ڈالر', 'قرض', 'آئی', 'ایم', 'ایف', 'معاشی'
        ]
    },
    4: {
        'name': 'International',
        'keywords': [
            'UN', 'treaty', 'foreign', 'bilateral', 'conflict', 'diplomatic',
            'ambassador', 'sanction', 'war', 'peace', 'alliance', 'global',
            'اقوام', 'متحدہ', 'معاہدہ', 'خارجہ', 'دو', 'طرفہ', 'تنازع', 'سفارتی',
            'سفیر', 'پابندی', 'جنگ', 'امن', 'اتحاد', 'عالمی', 'بین', 'الاقوامی',
            'امریکہ', 'انڈیا', 'چین', 'افغانستان', 'ایران', 'روس', 'برطانیہ'
        ]
    },
    5: {
        'name': 'Health_Society',
        'keywords': [
            'hospital', 'disease', 'vaccine', 'flood', 'education', 'health',
            'doctor', 'patient', 'school', 'college', 'university', 'student',
            'teacher', 'earthquake', 'disaster', 'relief', 'welfare',
            'ہسپتال', 'بیماری', 'ویکسین', 'سیلاب', 'تعلیم', 'صحت', 'ڈاکٹر',
            'مریض', 'سکول', 'کالج', 'یونیورسٹی', 'طلبہ', 'استاد', 'زلزلہ',
            'آفت', 'امداد', 'فلاح', 'بہبود', 'کینسر', 'علاج', 'دوائی'
        ]
    }
}

# Reverse mapping: keyword -> category_id
KEYWORD_TO_CATEGORY = {}
for cat_id, cat_info in CATEGORIES.items():
    for keyword in cat_info['keywords']:
        KEYWORD_TO_CATEGORY[keyword.lower()] = cat_id


# ============================================================================
# ARTICLE CATEGORIZATION
# ============================================================================
def categorize_article(title):
    """
    Assign an article to one of 5 categories based on keyword matching.
    
    Args:
        title: Article title from Metadata.json
    
    Returns:
        category_id: Integer 1-5 representing the category
        confidence: Number of matching keywords found
    """
    title_lower = title.lower()
    category_scores = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    # Check each keyword
    for keyword, cat_id in KEYWORD_TO_CATEGORY.items():
        if keyword in title_lower:
            category_scores[cat_id] += 1
    
    # Find category with highest score
    best_category = max(category_scores, key=category_scores.get)
    confidence = category_scores[best_category]
    
    # If no keywords match, try to infer from common patterns
    if confidence == 0:
        # Check for political figures
        if any(name in title for name in ['عمران', 'خان', 'نواز', 'شریف', 'زرداری', 'بھٹو']):
            best_category = 1  # Politics
        # Check for sports events
        elif any(term in title for term in ['کرکٹ', 'میچ', 'ورلڈ کپ']):
            best_category = 2  # Sports
        # Check for economic terms
        elif any(term in title for term in ['روپے', 'ڈالر', 'معیشت', 'قیمت']):
            best_category = 3  # Economy
        # Check for international relations
        elif any(term in title for term in ['امریکہ', 'انڈیا', 'چین', 'افغانستان']):
            best_category = 4  # International
        else:
            best_category = 5  # Health & Society (default)
    
    return best_category, confidence


def load_metadata(filepath):
    """Load and parse metadata.json"""
    if DEBUG_MODE:
        print(f"[DEBUG]: Loading metadata from {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    articles = []
    for doc_id, info in metadata.items():
        articles.append({
            'doc_id': int(doc_id),
            'title': info.get('title', ''),
            'publish_date': info.get('publish_date', '')
        })
    
    # Sort by doc_id
    articles.sort(key=lambda x: x['doc_id'])
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Loaded {len(articles)} articles from metadata")
    
    return articles


def categorize_all_articles(articles):
    """Categorize all articles and return statistics"""
    categorized = []
    category_counts = defaultdict(int)
    
    for article in articles:
        cat_id, confidence = categorize_article(article['title'])
        article['category_id'] = cat_id
        article['category_name'] = CATEGORIES[cat_id]['name']
        article['category_confidence'] = confidence
        categorized.append(article)
        category_counts[cat_id] += 1
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Category distribution:")
        for cat_id in range(1, 6):
            name = CATEGORIES[cat_id]['name']
            count = category_counts[cat_id]
            pct = count / len(articles) * 100
            print(f"[DEBUG]:   Category {cat_id} ({name}): {count} articles ({pct:.1f}%)")
    
    return categorized, dict(category_counts)


# ============================================================================
# TOKENIZATION AND SEQUENCE PREPARATION
# ============================================================================
def load_vocabulary():
    """Load vocabulary from Word2Vec (C3)"""
    vocab_file = os.path.join(resultsDir, 'w2v_vocab_old.json')
    
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
        vocab = vocab_data['vocab']
        word_to_idx = vocab_data['word_to_idx']
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Loaded vocabulary with {len(vocab)} tokens")
    
    return vocab, word_to_idx


def load_documents(filepath):
    """Load and parse documents from cleaned.txt"""
    if DEBUG_MODE:
        print(f"[DEBUG]: Loading documents from {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    documents = {}
    doc_pattern = r'\[(\d+)\]\s*\n'
    
    # Split by document markers
    parts = re.split(r'\[\d+\]\s*\n', content)
    matches = re.findall(doc_pattern, content)
    
    for i, doc_id in enumerate(matches):
        if i + 1 < len(parts):
            doc_text = parts[i + 1]
            # Extract all tokens from <SOS>...<EOS>
            tokens = []
            sent_pattern = r'<SOS>(.*?)<EOS>'
            sent_matches = re.findall(sent_pattern, doc_text, re.DOTALL)
            
            for sent in sent_matches:
                sent_tokens = sent.strip().split()
                tokens.extend([t for t in sent_tokens if t])
            
            documents[int(doc_id)] = tokens
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Loaded {len(documents)} documents")
    
    return documents


def tokenize_documents(documents, word_to_idx, max_length=256):
    """
    Convert documents to token-ID sequences, padded/truncated to max_length.
    
    Args:
        documents: Dict mapping doc_id to list of tokens
        word_to_idx: Word to index mapping
        max_length: Maximum sequence length (padding/truncation)
    
    Returns:
        tokenized: Dict mapping doc_id to token-ID sequence
    """
    tokenized = {}
    unk_count = 0
    total_tokens = 0
    
    for doc_id, tokens in documents.items():
        token_ids = []
        for token in tokens[:max_length]:  # Truncate to max_length
            if token in word_to_idx:
                token_ids.append(word_to_idx[token])
            else:
                token_ids.append(0)  # <UNK> token
                unk_count += 1
            total_tokens += 1
        
        # Pad if shorter than max_length
        if len(token_ids) < max_length:
            token_ids.extend([0] * (max_length - len(token_ids)))
        
        tokenized[doc_id] = token_ids
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Tokenized {len(tokenized)} documents")
        print(f"[DEBUG]: Max length: {max_length}")
        print(f"[DEBUG]: UNK rate: {unk_count}/{total_tokens} ({unk_count/total_tokens*100:.2f}%)")
    
    return tokenized


def create_dataset(articles, tokenized_documents):
    """
    Create final dataset by combining article metadata with tokenized content.
    
    Returns:
        dataset: List of dicts with 'doc_id', 'category_id', 'category_name', 'tokens'
    """
    dataset = []
    missing_docs = []
    
    for article in articles:
        doc_id = article['doc_id']
        
        if doc_id in tokenized_documents:
            dataset.append({
                'doc_id': doc_id,
                'category_id': article['category_id'],
                'category_name': article['category_name'],
                'title': article['title'],
                'publish_date': article['publish_date'],
                'tokens': tokenized_documents[doc_id]
            })
        else:
            missing_docs.append(doc_id)
    
    if DEBUG_MODE:
        print(f"[DEBUG]: Created dataset with {len(dataset)} articles")
        if missing_docs:
            print(f"[DEBUG]: Missing documents: {missing_docs}")
    
    return dataset


# ============================================================================
# STRATIFIED SPLIT
# ============================================================================
def stratified_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train/val/test stratified by category.
    
    Args:
        dataset: List of articles with category_id
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
    
    Returns:
        train_data, val_data, test_data
    """
    # Group by category
    category_groups = defaultdict(list)
    for item in dataset:
        category_groups[item['category_id']].append(item)
    
    train_data, val_data, test_data = [], [], []
    
    for cat_id, items in category_groups.items():
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        # Shuffle
        np.random.shuffle(items)
        
        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])
    
    # Shuffle each split
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)
    
    return train_data, val_data, test_data


def get_category_distribution(dataset):
    """Get category distribution for a dataset split"""
    counts = defaultdict(int)
    for item in dataset:
        counts[item['category_id']] += 1
    return dict(counts)


def report_distribution(train_data, val_data, test_data, category_counts):
    """Report class distribution across splits"""
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION REPORT")
    print("=" * 60)
    
    total = len(train_data) + len(val_data) + len(test_data)
    
    print(f"\nOverall Dataset: {total} articles")
    print("-" * 40)
    print(f"{'Category':<20} {'Count':<10} {'Percentage':<10}")
    print("-" * 40)
    for cat_id in range(1, 6):
        name = CATEGORIES[cat_id]['name']
        count = category_counts.get(cat_id, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"{name:<20} {count:<10} {pct:.1f}%")
    
    print(f"\n{'Split':<15} {'Train':<10} {'Val':<10} {'Test':<10}")
    print("-" * 45)
    
    train_counts = get_category_distribution(train_data)
    val_counts = get_category_distribution(val_data)
    test_counts = get_category_distribution(test_data)
    
    for cat_id in range(1, 6):
        name = CATEGORIES[cat_id]['name']
        t = train_counts.get(cat_id, 0)
        v = val_counts.get(cat_id, 0)
        te = test_counts.get(cat_id, 0)
        print(f"{name:<15} {t:<10} {v:<10} {te:<10}")
    
    print(f"\n{'Total':<15} {len(train_data):<10} {len(val_data):<10} {len(test_data):<10}")
    print(f"{'Percentage':<15} {len(train_data)/total*100:.1f}%      {len(val_data)/total*100:.1f}%      {len(test_data)/total*100:.1f}%")
    
    return {
        'overall': {CATEGORIES[cat_id]['name']: count for cat_id, count in category_counts.items()},
        'train': {CATEGORIES[cat_id]['name']: train_counts.get(cat_id, 0) for cat_id in range(1, 6)},
        'val': {CATEGORIES[cat_id]['name']: val_counts.get(cat_id, 0) for cat_id in range(1, 6)},
        'test': {CATEGORIES[cat_id]['name']: test_counts.get(cat_id, 0) for cat_id in range(1, 6)}
    }


def save_dataset(dataset, filepath):
    """Save dataset to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"Dataset saved to: {filepath}")


def save_as_npy(dataset, filepath):
    """Save token sequences as numpy array"""
    tokens_array = np.array([item['tokens'] for item in dataset], dtype=np.int32)
    labels_array = np.array([item['category_id'] for item in dataset], dtype=np.int32)
    
    np.savez(filepath, tokens=tokens_array, labels=labels_array)
    print(f"Numpy arrays saved to: {filepath}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """Main dataset preparation function"""
    print("=" * 80)
    print("TRANSFORMER ENCODER - DATASET PREPARATION")
    print("=" * 80)
    
    # File paths
    metadata_file = os.path.join(dataDir, 'metadata.json')
    cleaned_file = os.path.join(dataDir, 'cleaned.txt')
    
    # Step 1: Load and categorize articles
    print("\n" + "-" * 60)
    print("STEP 1: CATEGORIZING ARTICLES")
    print("-" * 60)
    
    articles = load_metadata(metadata_file)
    categorized_articles, category_counts = categorize_all_articles(articles)
    
    # Step 2: Load vocabulary and tokenize documents
    print("\n" + "-" * 60)
    print("STEP 2: TOKENIZING DOCUMENTS")
    print("-" * 60)
    
    vocab, word_to_idx = load_vocabulary()
    documents = load_documents(cleaned_file)
    tokenized_docs = tokenize_documents(documents, word_to_idx, max_length=256)
    
    # Step 3: Create dataset
    print("\n" + "-" * 60)
    print("STEP 3: CREATING DATASET")
    print("-" * 60)
    
    dataset = create_dataset(categorized_articles, tokenized_docs)
    print(f"Total articles in dataset: {len(dataset)}")
    
    # Step 4: Stratified split
    print("\n" + "-" * 60)
    print("STEP 4: STRATIFIED SPLIT (70/15/15)")
    print("-" * 60)
    
    train_data, val_data, test_data = stratified_split(dataset)
    
    # Step 5: Report distribution
    distribution = report_distribution(train_data, val_data, test_data, category_counts)
    
    # Step 6: Save datasets
    print("\n" + "-" * 60)
    print("STEP 5: SAVING DATASETS")
    print("-" * 60)
    
    # Save as JSON (for inspection)
    save_dataset(train_data, os.path.join(resultsDir, 'transformer_train.json'))
    save_dataset(val_data, os.path.join(resultsDir, 'transformer_val.json'))
    save_dataset(test_data, os.path.join(resultsDir, 'transformer_test.json'))
    save_dataset(dataset, os.path.join(resultsDir, 'transformer_full.json'))
    
    # Save as numpy arrays (for efficient loading)
    save_as_npy(train_data, os.path.join(embeddingsDir, 'transformer_train.npz'))
    save_as_npy(val_data, os.path.join(embeddingsDir, 'transformer_val.npz'))
    save_as_npy(test_data, os.path.join(embeddingsDir, 'transformer_test.npz'))
    
    # Save category mapping
    category_mapping = {
        'categories': {str(k): v['name'] for k, v in CATEGORIES.items()},
        'num_categories': len(CATEGORIES)
    }
    mapping_file = os.path.join(resultsDir, 'transformer_categories.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(category_mapping, f, indent=2)
    print(f"Category mapping saved to: {mapping_file}")
    
    # Save distribution statistics
    stats = {
        'total_articles': len(dataset),
        'train_articles': len(train_data),
        'val_articles': len(val_data),
        'test_articles': len(test_data),
        'max_sequence_length': 256,
        'vocab_size': len(vocab),
        'distribution': distribution
    }
    
    stats_file = os.path.join(resultsDir, 'transformer_dataset_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to: {stats_file}")
    
    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 80)
    
    # Return summary for README
    return {
        'total_articles': len(dataset),
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'category_distribution': distribution['overall'],
        'max_seq_length': 256,
        'vocab_size': len(vocab)
    }


if __name__ == "__main__":
    summary = main()