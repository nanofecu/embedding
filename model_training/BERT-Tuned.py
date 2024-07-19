import torch
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
from torch.utils.data import Dataset
import random
import numpy as np


# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

class ChemistryDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=32, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
        # Load the dataset from the specified file
        with open(filename, 'r', encoding='utf-8') as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        tokenized = self.tokenizer(line, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        inputs = tokenized['input_ids'].squeeze(0)
        labels = inputs.clone()
        attention_mask = tokenized['attention_mask'].squeeze(0)

        # Create labels for the MLM task
        rand = torch.rand(inputs.shape)
        mask_arr = (rand < self.mlm_probability) * (inputs != self.tokenizer.pad_token_id)
        selection = []

        for i in range(inputs.shape[0]):
            if mask_arr[i]:
                selection.append(i)

        labels[~mask_arr] = -100  # Only predict masked tokens
        inputs[selection] = self.tokenizer.mask_token_id  # Replace selected tokens with mask_token

        return {"input_ids": inputs, "labels": labels, "attention_mask": attention_mask}

# Specify the path to the dataset file (use a generic path)
data = ChemistryDataset("<Your Path to Input Text>", tokenizer) # e.g. 'data.txt'

# Configure training parameters
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=16,
    num_train_epochs=1,  # Adjust the number of epochs
    learning_rate=1e-5,  # Adjust the learning rate
    logging_dir='./logs',
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False
)

# Instantiate Trainer with PyTorch's AdamW optimizer
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)  # Adjust the learning rate of the optimizer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data,
    eval_dataset=data,
    optimizers=(optimizer, None)
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer to a directory
model.save_pretrained('./fine_tuned_bert_base')
tokenizer.save_pretrained('./fine_tuned_bert_base')


