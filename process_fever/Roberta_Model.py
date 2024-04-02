#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy
import copy


# In[29]:


train_path = '/home/ashrafs/projects/dragon/data/fever/orginal_splits/train.jsonl'
dev_path = '/home/ashrafs/projects/dragon/data/fever/orginal_splits/dev.jsonl'
test_path = '/home/ashrafs/projects/dragon/data/fever/orginal_splits/test.jsonl'

def load_dataset(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]



# In[30]:


config= {  
    "model" : 'roberta-large',
    "batch_size": 64,
    "learning_rate": 5e-5,
    "num_epochs": 10,
    "patience": 4,
    "patience_counter" : 0, 
    "max_token_length": 512,
    "dropout_rate": 0.5
    
}


# In[31]:


# Assuming you have a function to convert labels from text to integers
def label_to_int(label):
    mapping = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
    return mapping[label]

# Load and prepare the datasets
def prepare_data(data):
    texts = [item['claim'] for item in data]
    labels = [label_to_int(item['label']) for item in data]
    return texts, labels

train_data = load_dataset(train_path)
dev_data = load_dataset(dev_path)
test_data = load_dataset(test_path)


# In[14]:


# train_data = train_dataa[1:1000]
# dev_data = dev_dataa[1:100]
# test_data = test_dataa[1:100]


# In[33]:


train_texts, train_labels = prepare_data(train_data)
dev_texts, dev_labels = prepare_data(dev_data)
test_texts, test_labels = prepare_data(test_data)

# Continue with the previous script for encoding data
tokenizer = RobertaTokenizer.from_pretrained(config['model'])

class FeverDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def encode_data(tokenizer, texts, labels, max_length=config['max_token_length']):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return FeverDataset(encodings, labels)

train_dataset = encode_data(tokenizer, train_texts, train_labels)
dev_dataset = encode_data(tokenizer, dev_texts, dev_labels)
test_dataset = encode_data(tokenizer, test_texts, test_labels)


batch_size = config["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model
model = RobertaForSequenceClassification.from_pretrained(
    config['model'],
    num_labels=3,
    hidden_dropout_prob=config['dropout_rate'],  
    attention_probs_dropout_prob=config['dropout_rate'] 
    
) 

# Optimizer
optimizer = AdamW(model.parameters(), lr= config['learning_rate'])





# In[17]:


device = torch.device('cpu')
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
else:
    print("GPU not available.")
    
model.to(device)


# In[22]:


print(f"------Model Parameters******-----:\n {config}\n------------------------")


# In[23]:


def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_examples = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop('labels')
            outputs = model(**inputs)

            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item()

            batch_predictions = torch.argmax(logits, axis=-1)
            correct_predictions += (batch_predictions == labels).sum().item()
            total_examples += labels.size(0)

            predictions.extend(batch_predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_examples
    return avg_loss, accuracy, true_labels, predictions

best_model_wts = copy.deepcopy(model.state_dict())
best_val_loss = float('inf')
patience = config['patience']
patience_counter = config['patience_counter']
num_epochs = config['num_epochs']

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    correct_predictions = 0
    total_train_examples = 0

    for batch in train_loader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        logits = outputs.logits
        predictions = torch.argmax(logits, axis=-1)
        correct_predictions += (predictions == batch['labels']).sum().item()
        total_train_examples += batch['labels'].size(0)

    train_loss = total_train_loss / len(train_loader)
    train_accuracy = correct_predictions / total_train_examples

    val_loss, val_accuracy, _, _ = evaluate_model(model, dev_loader, device)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0
        print("Validation loss decreased. Saving the model.")
    else:
        patience_counter += 1
        print(f"Validation loss did not decrease. Patience counter: {patience_counter}")

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# Load best model weights
model.load_state_dict(best_model_wts)

# Evaluate on test data
_, _, test_true_labels, test_predictions = evaluate_model(model, test_loader, device)

# Print classification report
print('Test Set Performance:')
print(classification_report(test_true_labels, test_predictions, target_names=['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']))


# In[ ]:




