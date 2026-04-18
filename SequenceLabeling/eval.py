import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_recall_fscore_support
from collections import defaultdict, Counter
from tqdm import tqdm
import re


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

# Tag sets
POS_TAGS = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'CONJ', 'POST', 'NUM', 'PUNC', 'UNK']
NER_TAGS = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']

# Import from the BiLSTM module
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bi_lstm_sequence_labeller import (
    BiLSTMSequenceLabeler, BiLSTMCRF, SequenceLabelingDataset, 
    collate_fn, load_embeddings, load_dataset, evaluate, compute_metrics,
    POS_TAGS as POS_TAGS_FULL, NER_TAGS as NER_TAGS_FULL,
    EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, BATCH_SIZE
)


class SequenceLabelingEvaluator:
    """Comprehensive evaluator for sequence labeling models"""
    
    def __init__(self, model, test_loader, task, word_to_idx, idx_to_tag):
        self.model = model
        self.test_loader = test_loader
        self.task = task
        self.word_to_idx = word_to_idx
        self.idx_to_tag = idx_to_tag
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(idx_to_tag)}
        
    def get_predictions(self, crf_model=None):
        """Get model predictions on test set"""
        self.model.eval()
        all_true = []
        all_pred = []
        all_tokens = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                tokens = batch['tokens'].to(device)
                lengths = batch['lengths']
                mask = batch['mask'].to(device)
                raw_tokens = batch['raw_tokens']
                
                if self.task == 'ner' and crf_model is not None:
                    predictions = crf_model(tokens, lengths, mask=mask)
                    tags = batch['ner_tags']
                else:
                    outputs = self.model(tokens, lengths)
                    logits = outputs['pos'] if self.task == 'pos' else outputs['ner']
                    tags = batch['pos_tags'] if self.task == 'pos' else batch['ner_tags']
                    predictions = logits.argmax(dim=-1)
                
                for b in range(tokens.shape[0]):
                    seq_len = lengths[b].item()
                    true_tags = tags[b][:seq_len].cpu().tolist()
                    
                    if self.task == 'ner' and crf_model is not None:
                        pred_tags = predictions[b][:seq_len]
                    else:
                        pred_tags = predictions[b][:seq_len].cpu().tolist()
                    
                    all_true.extend(true_tags)
                    all_pred.extend(pred_tags)
                    all_tokens.extend(raw_tokens[b])
        
        return all_true, all_pred, all_tokens
    
    def compute_token_metrics(self, true_tags, pred_tags):
        """Compute token-level accuracy and macro-F1"""
        # Filter out padding (-1)
        valid_pairs = [(t, p) for t, p in zip(true_tags, pred_tags) if t != -1]
        true_filtered = [t for t, _ in valid_pairs]
        pred_filtered = [p for _, p in valid_pairs]
        
        accuracy = accuracy_score(true_filtered, pred_filtered)
        macro_f1 = f1_score(true_filtered, pred_filtered, average='macro', zero_division=0)
        
        return accuracy, macro_f1
    
    def plot_confusion_matrix(self, true_tags, pred_tags, save_path):
        """Plot confusion matrix for POS tags"""
        # Filter padding
        valid_pairs = [(t, p) for t, p in zip(true_tags, pred_tags) if t != -1]
        true_filtered = [t for t, _ in valid_pairs]
        pred_filtered = [p for _, p in valid_pairs]
        
        cm = confusion_matrix(true_filtered, pred_filtered, labels=range(len(self.idx_to_tag)))
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.idx_to_tag, yticklabels=self.idx_to_tag)
        plt.title(f'{self.task.upper()} Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        return cm
    
    def find_confused_pairs(self, cm, top_k=3):
        """Find most confused tag pairs from confusion matrix"""
        confused_pairs = []
        n_tags = len(self.idx_to_tag)
        
        for i in range(n_tags):
            for j in range(n_tags):
                if i != j:
                    confused_pairs.append((i, j, cm[i, j]))
        
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        return confused_pairs[:top_k]
    
    def get_example_sentences(self, true_tags, pred_tags, all_tokens, confused_pairs):
        """Get example sentences for confused tag pairs"""
        examples = defaultdict(list)
        
        for pair_idx, (tag_i, tag_j, _) in enumerate(confused_pairs):
            tag_i_name = self.idx_to_tag[tag_i]
            tag_j_name = self.idx_to_tag[tag_j]
            
            # Find sentences where tag_i was predicted as tag_j
            batch_examples = []
            current_sent = []
            current_true = []
            current_pred = []
            
            # Reconstruct sentences from flattened data
            sent_start = 0
            for batch in self.test_loader:
                lengths = batch['lengths']
                for b in range(len(lengths)):
                    seq_len = lengths[b].item()
                    
                    sent_true = true_tags[sent_start:sent_start + seq_len]
                    sent_pred = pred_tags[sent_start:sent_start + seq_len]
                    sent_tokens = all_tokens[sent_start:sent_start + seq_len]
                    
                    for t, p, tok in zip(sent_true, sent_pred, sent_tokens):
                        if t != -1 and t == tag_i and p == tag_j:
                            if len(examples[f"{tag_i_name}→{tag_j_name}"]) < 2:
                                examples[f"{tag_i_name}→{tag_j_name}"].append({
                                    'tokens': sent_tokens,
                                    'true': [self.idx_to_tag[x] if x != -1 else 'PAD' for x in sent_true],
                                    'pred': [self.idx_to_tag[x] if x != -1 else 'PAD' for x in sent_pred],
                                    'error_at': tok
                                })
                    
                    sent_start += seq_len
        
        return examples


class NEREvaluator:
    """NER-specific evaluator with conlleval-style metrics"""
    
    def __init__(self, idx_to_ner_tag):
        self.idx_to_ner_tag = idx_to_ner_tag
        self.ner_tag_to_idx = {tag: idx for idx, tag in enumerate(idx_to_ner_tag)}
        
    def bio_to_entities(self, tags, tokens):
        """Convert BIO tags to entity spans"""
        entities = []
        current_entity = None
        current_type = None
        start_idx = 0
        
        for i, (tag, token) in enumerate(zip(tags, tokens)):
            if tag == -1:  # Padding
                continue
                
            tag_name = self.idx_to_ner_tag[tag]
            
            if tag_name.startswith('B-'):
                if current_entity is not None:
                    entities.append((current_type, start_idx, i, current_entity))
                current_type = tag_name[2:]
                current_entity = [token]
                start_idx = i
            elif tag_name.startswith('I-'):
                if current_entity is not None and current_type == tag_name[2:]:
                    current_entity.append(token)
                else:
                    if current_entity is not None:
                        entities.append((current_type, start_idx, i, current_entity))
                    current_entity = None
                    current_type = None
            else:  # 'O'
                if current_entity is not None:
                    entities.append((current_type, start_idx, i, current_entity))
                    current_entity = None
                    current_type = None
        
        if current_entity is not None:
            entities.append((current_type, start_idx, len(tokens), current_entity))
        
        return entities
    
    def compute_entity_metrics(self, true_tags, pred_tags, all_tokens):
        """Compute entity-level precision, recall, and F1 per type"""
        # Reconstruct sentences
        true_entities_all = []
        pred_entities_all = []
        
        sent_start = 0
        for batch in self.test_loader:
            lengths = batch['lengths']
            for b in range(len(lengths)):
                seq_len = lengths[b].item()
                
                sent_true = true_tags[sent_start:sent_start + seq_len]
                sent_pred = pred_tags[sent_start:sent_start + seq_len]
                sent_tokens = all_tokens[sent_start:sent_start + seq_len]
                
                true_entities = self.bio_to_entities(sent_true, sent_tokens)
                pred_entities = self.bio_to_entities(sent_pred, sent_tokens)
                
                true_entities_all.extend([(e[0], ' '.join(e[3])) for e in true_entities])
                pred_entities_all.extend([(e[0], ' '.join(e[3])) for e in pred_entities])
                
                sent_start += seq_len
        
        # Compute metrics per entity type
        entity_types = ['PER', 'LOC', 'ORG', 'MISC']
        results = {}
        
        for etype in entity_types:
            true_set = set([e[1] for e in true_entities_all if e[0] == etype])
            pred_set = set([e[1] for e in pred_entities_all if e[0] == etype])
            
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[etype] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        # Overall metrics
        true_all = set([e[1] for e in true_entities_all])
        pred_all = set([e[1] for e in pred_entities_all])
        
        tp_all = len(true_all & pred_all)
        fp_all = len(pred_all - true_all)
        fn_all = len(true_all - pred_all)
        
        precision_all = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0
        recall_all = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0
        f1_all = 2 * precision_all * recall_all / (precision_all + recall_all) if (precision_all + recall_all) > 0 else 0
        
        results['overall'] = {
            'precision': precision_all,
            'recall': recall_all,
            'f1': f1_all,
            'tp': tp_all,
            'fp': fp_all,
            'fn': fn_all
        }
        
        return results, true_entities_all, pred_entities_all
    
    def error_analysis(self, true_entities, pred_entities, all_tokens, true_tags, pred_tags, num_examples=5):
        """Analyze false positives and false negatives"""
        true_set = set([e[1] for e in true_entities])
        pred_set = set([e[1] for e in pred_entities])
        
        false_positives = list(pred_set - true_set)
        false_negatives = list(true_set - pred_set)
        
        # Find context for errors
        fp_examples = []
        fn_examples = []
        
        # Reconstruct sentences to find errors
        sent_start = 0
        for batch in self.test_loader:
            lengths = batch['lengths']
            for b in range(len(lengths)):
                seq_len = lengths[b].item()
                
                sent_true = true_tags[sent_start:sent_start + seq_len]
                sent_pred = pred_tags[sent_start:sent_start + seq_len]
                sent_tokens = all_tokens[sent_start:sent_start + seq_len]
                
                true_ents = self.bio_to_entities(sent_true, sent_tokens)
                pred_ents = self.bio_to_entities(sent_pred, sent_tokens)
                
                true_ent_strs = set([' '.join(e[3]) for e in true_ents])
                pred_ent_strs = set([' '.join(e[3]) for e in pred_ents])
                
                sent_fp = pred_ent_strs - true_ent_strs
                sent_fn = true_ent_strs - pred_ent_strs
                
                for fp in sent_fp:
                    if len(fp_examples) < num_examples:
                        fp_examples.append({
                            'entity': fp,
                            'sentence': ' '.join(sent_tokens),
                            'true_tags': [self.idx_to_ner_tag[x] if x != -1 else 'PAD' for x in sent_true],
                            'pred_tags': [self.idx_to_ner_tag[x] if x != -1 else 'PAD' for x in sent_pred]
                        })
                
                for fn in sent_fn:
                    if len(fn_examples) < num_examples:
                        fn_examples.append({
                            'entity': fn,
                            'sentence': ' '.join(sent_tokens),
                            'true_tags': [self.idx_to_ner_tag[x] if x != -1 else 'PAD' for x in sent_true],
                            'pred_tags': [self.idx_to_ner_tag[x] if x != -1 else 'PAD' for x in sent_pred]
                        })
                
                sent_start += seq_len
                
                if len(fp_examples) >= num_examples and len(fn_examples) >= num_examples:
                    break
        
        return fp_examples[:num_examples], fn_examples[:num_examples]


def load_model_from_checkpoint(checkpoint_path, task, freeze_embeddings, use_crf=False):
    """Load a trained model from checkpoint"""
    vocab_size = 1033
    num_pos_tags = len(POS_TAGS_FULL)
    num_ner_tags = len(NER_TAGS_FULL)
    
    # Load embeddings
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
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    crf_model = None
    if task == 'ner' and use_crf:
        crf_model = BiLSTMCRF(model, num_ner_tags).to(device)
    
    return model, crf_model, word_to_idx


def evaluate_pos():
    """Evaluate POS tagging models"""
    print("=" * 80)
    print("POS TAGGING EVALUATION")
    print("=" * 80)
    
    # Load test data
    _, word_to_idx = load_embeddings()[1:]
    test_data = load_dataset('test')
    pos_tag_to_idx = {tag: i for i, tag in enumerate(POS_TAGS_FULL)}
    test_dataset = SequenceLabelingDataset(test_data, word_to_idx, pos_tag_to_idx=pos_tag_to_idx, task='pos')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    results = {}
    
    for mode in ['frozen', 'fine-tuned']:
        print(f"\n{'='*60}")
        print(f"POS - {mode.upper()} EMBEDDINGS")
        print(f"{'='*60}")
        
        checkpoint_path = os.path.join(modelsDir, f'bilstm_pos_{mode}', 'best_model.pth')
        freeze = (mode == 'frozen')
        
        model, _, _ = load_model_from_checkpoint(checkpoint_path, 'pos', freeze, use_crf=False)
        
        evaluator = SequenceLabelingEvaluator(model, test_loader, 'pos', word_to_idx, POS_TAGS_FULL)
        true_tags, pred_tags, all_tokens = evaluator.get_predictions()
        
        # Token-level metrics
        accuracy, macro_f1 = evaluator.compute_token_metrics(true_tags, pred_tags)
        print(f"\nToken-level Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        
        # Confusion matrix
        cm_path = os.path.join(figuresDir, f'pos_confusion_matrix_{mode}.png')
        cm = evaluator.plot_confusion_matrix(true_tags, pred_tags, cm_path)
        print(f"\nConfusion matrix saved to: {cm_path}")
        
        # Confused pairs
        confused_pairs = evaluator.find_confused_pairs(cm, top_k=3)
        print(f"\nTop 3 Most Confused Tag Pairs:")
        for tag_i, tag_j, count in confused_pairs:
            print(f"  {POS_TAGS_FULL[tag_i]} → {POS_TAGS_FULL[tag_j]}: {count} times")
        
        # Example sentences
        examples = evaluator.get_example_sentences(true_tags, pred_tags, all_tokens, confused_pairs)
        
        for pair_name, pair_examples in examples.items():
            print(f"\n  Examples for {pair_name}:")
            for i, ex in enumerate(pair_examples, 1):
                print(f"    Example {i}:")
                print(f"      Sentence: {' '.join(ex['tokens'])}")
                print(f"      Error at: '{ex['error_at']}'")
        
        results[mode] = {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'confused_pairs': [(POS_TAGS_FULL[ti], POS_TAGS_FULL[tj], int(c)) for ti, tj, c in confused_pairs],
            'examples': {k: v for k, v in examples.items()}
        }
    
    # Summary table
    print(f"\n{'='*60}")
    print("POS SUMMARY: FROZEN vs FINE-TUNED")
    print(f"{'='*60}")
    print(f"{'Mode':<15} {'Accuracy':<12} {'Macro F1':<12}")
    print("-" * 40)
    for mode, res in results.items():
        print(f"{mode:<15} {res['accuracy']:.4f}       {res['macro_f1']:.4f}")
    
    return results


def evaluate_ner():
    """Evaluate NER models"""
    print("\n" + "=" * 80)
    print("NER EVALUATION")
    print("=" * 80)
    
    # Load test data
    _, word_to_idx = load_embeddings()[1:]
    test_data = load_dataset('test')
    ner_tag_to_idx = {tag: i for i, tag in enumerate(NER_TAGS_FULL)}
    test_dataset = SequenceLabelingDataset(test_data, word_to_idx, ner_tag_to_idx=ner_tag_to_idx, task='ner')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    results = {}
    
    # Evaluate both frozen and fine-tuned with CRF
    for mode in ['frozen', 'fine-tuned']:
        print(f"\n{'='*60}")
        print(f"NER - {mode.upper()} EMBEDDINGS (with CRF)")
        print(f"{'='*60}")
        
        checkpoint_path = os.path.join(modelsDir, f'bilstm_ner_{mode}', 'best_model.pth')
        freeze = (mode == 'frozen')
        
        model, crf_model, _ = load_model_from_checkpoint(checkpoint_path, 'ner', freeze, use_crf=True)
        
        evaluator = SequenceLabelingEvaluator(model, test_loader, 'ner', word_to_idx, NER_TAGS_FULL)
        true_tags, pred_tags, all_tokens = evaluator.get_predictions(crf_model=crf_model)
        
        ner_eval = NEREvaluator(NER_TAGS_FULL)
        ner_eval.test_loader = test_loader
        
        entity_metrics, true_entities, pred_entities = ner_eval.compute_entity_metrics(true_tags, pred_tags, all_tokens)
        
        print(f"\nEntity-level Metrics:")
        print(f"{'Type':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-" * 44)
        for etype in ['PER', 'LOC', 'ORG', 'MISC', 'overall']:
            m = entity_metrics[etype]
            print(f"{etype:<8} {m['precision']:.4f}       {m['recall']:.4f}       {m['f1']:.4f}")
        
        # Error analysis
        fp_examples, fn_examples = ner_eval.error_analysis(true_entities, pred_entities, all_tokens, true_tags, pred_tags)
        
        print(f"\nFalse Positives (first 3):")
        for i, ex in enumerate(fp_examples[:3], 1):
            print(f"  {i}. Entity: '{ex['entity']}'")
            print(f"     Sentence: {ex['sentence']}")
        
        print(f"\nFalse Negatives (first 3):")
        for i, ex in enumerate(fn_examples[:3], 1):
            print(f"  {i}. Entity: '{ex['entity']}'")
            print(f"     Sentence: {ex['sentence']}")
        
        results[f"{mode}_crf"] = {
            'entity_metrics': entity_metrics,
            'fp_examples': fp_examples[:5],
            'fn_examples': fn_examples[:5]
        }
    
    # Also evaluate without CRF (fine-tuned only)
    print(f"\n{'='*60}")
    print(f"NER - FINE-TUNED EMBEDDINGS (without CRF)")
    print(f"{'='*60}")
    
    checkpoint_path = os.path.join(modelsDir, 'bilstm_ner_fine-tuned', 'best_model.pth')
    model, _, _ = load_model_from_checkpoint(checkpoint_path, 'ner', False, use_crf=False)
    
    evaluator = SequenceLabelingEvaluator(model, test_loader, 'ner', word_to_idx, NER_TAGS_FULL)
    true_tags, pred_tags, all_tokens = evaluator.get_predictions()
    
    ner_eval = NEREvaluator(NER_TAGS_FULL)
    ner_eval.test_loader = test_loader
    
    entity_metrics_nocrf, _, _ = ner_eval.compute_entity_metrics(true_tags, pred_tags, all_tokens)
    
    print(f"\nEntity-level Metrics (No CRF):")
    print(f"{'Type':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 44)
    for etype in ['PER', 'LOC', 'ORG', 'MISC', 'overall']:
        m = entity_metrics_nocrf[etype]
        print(f"{etype:<8} {m['precision']:.4f}       {m['recall']:.4f}       {m['f1']:.4f}")
    
    results['finetuned_nocrf'] = {'entity_metrics': entity_metrics_nocrf}
    
    # CRF vs No-CRF comparison
    print(f"\n{'='*60}")
    print("NER SUMMARY: CRF vs NO-CRF (Fine-tuned)")
    print(f"{'='*60}")
    print(f"{'Method':<12} {'PER F1':<10} {'LOC F1':<10} {'ORG F1':<10} {'Overall F1':<12}")
    print("-" * 56)
    
    crf_m = results['fine-tuned_crf']['entity_metrics']
    nocrf_m = results['finetuned_nocrf']['entity_metrics']
    
    print(f"{'With CRF':<12} {crf_m['PER']['f1']:.4f}     {crf_m['LOC']['f1']:.4f}     {crf_m['ORG']['f1']:.4f}     {crf_m['overall']['f1']:.4f}")
    print(f"{'Without CRF':<12} {nocrf_m['PER']['f1']:.4f}     {nocrf_m['LOC']['f1']:.4f}     {nocrf_m['ORG']['f1']:.4f}     {nocrf_m['overall']['f1']:.4f}")
    
    return results


def run_ablation_study():
    """Run ablation experiments"""
    print("\n" + "=" * 80)
    print("ABLATION STUDY")
    print("=" * 80)
    
    _, word_to_idx = load_embeddings()[1:]
    test_data = load_dataset('test')
    pos_tag_to_idx = {tag: i for i, tag in enumerate(POS_TAGS_FULL)}
    test_dataset = SequenceLabelingDataset(test_data, word_to_idx, pos_tag_to_idx=pos_tag_to_idx, task='pos')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    ablations = {
        'A1': {'name': 'Unidirectional LSTM', 'bidirectional': False, 'dropout': DROPOUT, 'pretrained': True},
        'A2': {'name': 'No Dropout', 'bidirectional': True, 'dropout': 0.0, 'pretrained': True},
        'A3': {'name': 'Random Embeddings', 'bidirectional': True, 'dropout': DROPOUT, 'pretrained': False},
    }
    
    results = {}
    
    for ablation_id, config in ablations.items():
        print(f"\n{'='*60}")
        print(f"{ablation_id}: {config['name']}")
        print(f"{'='*60}")
        
        # Note: This would require training models with these configurations
        # For now, we'll simulate or note that training is needed
        
        results[ablation_id] = {
            'name': config['name'],
            'config': config,
            'status': 'Requires training with modified configuration'
        }
        print(f"Status: Requires training with {config}")
    
    return results


def main():
    """Main evaluation function"""
    print("=" * 80)
    print("SEQUENCE LABELING EVALUATION")
    print("=" * 80)
    
    all_results = {}
    
    # 5.1 POS Evaluation
    pos_results = evaluate_pos()
    all_results['pos'] = pos_results
    
    # 5.2 NER Evaluation
    ner_results = evaluate_ner()
    all_results['ner'] = ner_results
    
    # 5.3 Ablation Study
    ablation_results = run_ablation_study()
    all_results['ablations'] = ablation_results
    
    # Save all results
    results_file = os.path.join(resultsDir, 'sequence_labeling_evaluation.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        # Convert to serializable format
        serializable = {}
        for key, value in all_results.items():
            if key == 'pos':
                serializable[key] = value
            elif key == 'ner':
                serializable[key] = {
                    k: {
                        'entity_metrics': v['entity_metrics'] if 'entity_metrics' in v else None,
                        'fp_examples_count': len(v.get('fp_examples', [])),
                        'fn_examples_count': len(v.get('fn_examples', []))
                    } for k, v in value.items()
                }
            else:
                serializable[key] = value
        
        json.dump(serializable, f, indent=2)
    
    print(f"\n\nComplete evaluation results saved to: {results_file}")
    
    return all_results


if __name__ == "__main__":
    main()