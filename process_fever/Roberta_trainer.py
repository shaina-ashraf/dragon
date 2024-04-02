#!/usr/bin/env python
# coding: utf-8

# In[5]:


import json
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from sklearn.metrics import classification_report
import numpy as np
import torch
from transformers import TrainerCallback


# In[22]:


train_path = '/home/ashrafs/projects/dragon/data/fever/orginal_splits/train.jsonl'
dev_path = '/home/ashrafs/projects/dragon/data/fever/orginal_splits/dev.jsonl'
test_path = '/home/ashrafs/projects/dragon/data/fever/orginal_splits/test.jsonl'

def load_dataset(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]

train_data = load_dataset(train_path)
dev_data = load_dataset(dev_path)
test_data = load_dataset(test_path)


# In[23]:


if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available.")


# In[24]:


tdata = train_data[1:1000]
ddata = dev_data[1:100]
ttdata = test_data[1:100]


# In[25]:


train_df = pd.DataFrame(tdata)
dev_df = pd.DataFrame(ddata)
test_df = pd.DataFrame(ttdata)

train_df.shape, test_df.shape, dev_df.shape


# In[26]:


"Train data", train_df['label'].value_counts(), "Dev Data: ", dev_df['label'].value_counts(), "Test Data:", test_df['label'].value_counts()


# In[27]:


class FeverousDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# In[28]:


tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

def encode_data(tokenizer, data):
    texts = [item['claim'] for item in data]  # Extract text data for tokenization
    labels = [1 if item['label'] == 'REFUTES' else 0 if item['label'] == 'SUPPORTS' else 2 for item in data]  # Convert labels to numeric
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    return FeverousDataset(encodings, labels)

train_dataset = encode_data(tokenizer, tdata)
dev_dataset = encode_data(tokenizer, ddata)
test_dataset = encode_data(tokenizer, ttdata)


# In[29]:


# Function to compute metrics, can be used with the Trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return classification_report(labels, predictions, output_dict=True)


# In[30]:


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=3):
        self.early_stopping_patience = early_stopping_patience
        self.best_loss = None
        self.patience_counter = 0

    def on_evaluate(self, args, state, control, **kwargs):
        # Example assumes evaluation loss is available in kwargs
        eval_loss = kwargs.get("eval_loss")

        if self.best_loss is None or eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.early_stopping_patience:
            print("Early stopping triggered")
            control.should_training_stop = True


# In[34]:


model = RobertaForSequenceClassification.from_pretrained('roberta-large', num_labels=3)



training_args = TrainingArguments(
    output_dir='fresults',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='flogs',
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,  # Evaluate every 500 steps
    #save_strategy="steps",
    #save_steps=500,  # Save checkpoint every 500 steps
    gradient_accumulation_steps=2,  # Accumulate gradients over 2 steps
    #early_stopping_patience=3,  # Number of evaluations with no improvement after which training will be stopped
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,  # Define a function for metrics (not shown here)
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)



# In[33]:


trainer.train()


# In[ ]:


# Evaluate the model
results = trainer.evaluate(test_dataset)
print("******Printing Test Results ******")
print(results)

