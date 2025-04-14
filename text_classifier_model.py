import pandas as pd
from sklearn.model_selection import train_test_split

# Load metadata.csv
df = pd.read_csv("local_dataset/metadata.csv")

# Clean and map labels
df['label'] = df['label'].map({'control': 0, 'dementia': 1})
df = df[['transcript', 'label']].dropna()

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['transcript'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
import torch

class DementiaDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

train_dataset = DementiaDataset(train_encodings, train_labels)
val_dataset = DementiaDataset(val_encodings, val_labels)
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    logging_dir="./logs",
    logging_steps=10,
    do_eval=True,
    do_train=True,
    save_steps=500,
    save_total_limit=2
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
model.save_pretrained("cognitive_model")
tokenizer.save_pretrained("cognitive_model")
