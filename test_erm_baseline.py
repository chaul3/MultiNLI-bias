"""
Test version of ERM baseline - runs with minimal data for testing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    BertForSequenceClassification, 
    BertTokenizer, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class MultiNLIDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):  # Reduced max_length for test
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        premise, hypothesis = self.texts[idx]
        
        encoding = self.tokenizer(
            premise,
            hypothesis,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def identify_spurious_groups(examples):
    """Identify spurious correlation groups based on negation words"""
    negation_words = ['no', 'not', 'never', 'none', "n't", "don't", "won't", "can't"]
    
    groups = []
    for premise, hypothesis, label in zip(examples['premise'], examples['hypothesis'], examples['label']):
        hypothesis_lower = hypothesis.lower()
        has_negation = any(neg_word in hypothesis_lower for neg_word in negation_words)
        
        if label == 0 and not has_negation:  # contradiction without negation
            group = 0
        elif label == 0 and has_negation:    # contradiction with negation  
            group = 1
        elif label == 1 and not has_negation:  # entailment without negation
            group = 2
        elif label == 1 and has_negation:    # entailment with negation
            group = 3
        else:  # neutral cases
            group = 4 if not has_negation else 5
            
        groups.append(group)
    
    return groups

def load_test_data():
    """Load a small subset of MultiNLI for testing"""
    print("Loading MultiNLI dataset (test subset)...")
    
    dataset = load_dataset("multi_nli")
    
    # Take only first 100 samples for testing
    train_subset = dataset['train'].select(range(100))
    val_subset = dataset['validation_matched'].select(range(50))
    
    train_texts = list(zip(train_subset['premise'], train_subset['hypothesis']))
    train_labels = train_subset['label']
    train_groups = identify_spurious_groups(train_subset)
    
    val_texts = list(zip(val_subset['premise'], val_subset['hypothesis']))
    val_labels = val_subset['label']
    val_groups = identify_spurious_groups(val_subset)
    
    return (train_texts, train_labels, train_groups), (val_texts, val_labels, val_groups)

def main():
    print("🧪 Testing ERM baseline implementation")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    train_data, val_data = load_test_data()
    print(f"Training samples: {len(train_data[0])}")
    print(f"Validation samples: {len(val_data[0])}")
    
    # Print group statistics
    train_groups = train_data[2]
    print("\nTraining group distribution:")
    for group_id in sorted(set(train_groups)):
        count = train_groups.count(group_id)
        percentage = 100 * count / len(train_groups)
        print(f"Group {group_id}: {count} samples ({percentage:.1f}%)")
    
    # Initialize model and tokenizer
    print("\n🤖 Initializing BERT-base-uncased model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=3
    )
    model.to(device)
    
    # Create test dataset and dataloader
    train_texts, train_labels, train_groups = train_data
    train_dataset = MultiNLIDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Small batch for test
    
    # Test forward pass
    print("\n🔄 Testing forward pass...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        print(f"✅ Forward pass successful!")
        print(f"Loss: {outputs.loss.item():.4f}")
        print(f"Logits shape: {outputs.logits.shape}")
    
    # Test one training step
    print("\n🏋️ Testing training step...")
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    
    optimizer.zero_grad()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    
    print(f"✅ Training step successful!")
    print(f"Loss after step: {loss.item():.4f}")
    
    # Test evaluation
    print("\n📊 Testing evaluation...")
    val_texts, val_labels, val_groups = val_data
    val_dataset = MultiNLIDataset(val_texts, val_labels, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    overall_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"✅ Evaluation successful!")
    print(f"Random accuracy: {overall_accuracy:.4f}")
    
    print("\n🎉 All tests passed! Implementation is working correctly.")
    print("You can now run the full training with:")
    print("python run_erm_baseline.py")

if __name__ == "__main__":
    main()
