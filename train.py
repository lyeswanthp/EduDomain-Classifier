from transformers import AutoModel, AutoTokenizer
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import faiss
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

model_name = "BMRetriever/BMRetriever-7B"
device="cuda"

model = AutoModel.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        embedding = last_hidden[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        embedding = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    return embedding


def tokenize(batch, train=True, max_length=512):
    # Process a batch of samples
    passages = [f'Represent this passage\npassage: {content}' for content in batch["content"]]

    # Tokenize the passage batch
    inputs = tokenizer(
        passages, 
        max_length=max_length - 1, 
        padding='max_length', 
        truncation=True, 
        return_tensors='pt'
    )

    
    # Add EOS token and extend attention mask
    batch_size = inputs['input_ids'].shape[0]
    eos_token_id = torch.full((batch_size, 1), tokenizer.eos_token_id, dtype=torch.long)
    attention_val = torch.ones(batch_size, 1, dtype=torch.long)
    inputs['input_ids'] = torch.cat([inputs['input_ids'], eos_token_id], dim=1)
    inputs['attention_mask'] = torch.cat([inputs['attention_mask'], attention_val], dim=1)

    titles = batch["title"]
    numerical_labels = [label2id[title] for title in titles]

    # Convert tensors to lists for compatibility with the datasets library
    inputs['input_ids'] = inputs['input_ids'].tolist()
    inputs['attention_mask'] = inputs['attention_mask'].tolist()

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": numerical_labels
    }

def save_embeddings(model, data_loader, split='train'):
    embeddings_all = torch.empty((0, 4097))
    model.eval()
    for idx, inputs in  enumerate(data_loader):
        with torch.no_grad():
            embeddings = model(inputs['input_ids'].to(model.device), attention_mask=inputs['attention_mask'].to(model.device))
            embeddings = last_token_pool(embeddings.last_hidden_state, attention_mask=inputs['attention_mask'].to(model.device))
            embeddings = torch.cat([embeddings.detach().cpu(), inputs['labels'].unsqueeze(1)], dim=1)
            embeddings_all = torch.cat([embeddings_all, embeddings], dim=0)
    torch.save(embeddings_all, f'embeddings_{split}.pth')
    
class DomainClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 2048)
        self.fc2 = nn.Linear(2048, 18)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
        
def get_torch_dataset(split):
    ds = split.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

ds = load_dataset("MedRAG/textbooks")
labels = ds["train"]["title"]
unique_titles = np.unique(labels) 
label2id = {title: idx for idx, title in enumerate(unique_titles)}
id2label = {idx: title for title, idx in label2id.items()}
train_indices, test_indices = train_test_split(
    range(len(labels)),
    test_size=0.2,
    stratify=labels,
    random_state=42
)
train_indices, val_indices = train_test_split(
    range(len(labels)),
    test_size=0.25,
    stratify=labels,
    random_state=42
)
train_split = ds["train"].select(train_indices)
val_split = ds['train'].select(val_indices)
test_split = ds["train"].select(test_indices)


ds = load_dataset("MedRAG/textbooks")
train_ds = get_torch_dataset(train_split)
val_ds = get_torch_dataset(val_split)
test_ds = get_torch_dataset(test_split)

train_dataloader = DataLoader(train_ds, batch_size=4)
val_dataloader = DataLoader(val_ds, batch_size=4)
test_dataloader = DataLoader(test_ds, batch_size=4)

class EmbeddingDataset(Dataset):
    def __init__(self, x):
        self.samples = x.shape[0]
        self.embeddings = x[:, 0: -1]
        self.labels = x[:, -1].to(torch.long)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

    def __len__(self):
        return self.samples

classifier = DomainClassifier(4096)
classifer = classifier.to(device)

def compute_validation_loss(model, data_loader, loss_fn, device='cuda'):
    val_loss = []
    with torch.no_grad():
        for valx, valy in data_loader:
            valx = valx.to(device)
            valy = valy.to(device)
            scores = model(valx)
            loss = loss_fn(scores, valy)
            val_loss.append(loss.item())
    return sum(val_loss)/len(val_loss)


class EarlyStopper():
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, model, validation_loss, path):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print('Saving model') 
            torch.save(model.state_dict(), f'{path}.pth')
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(model, train_data_loader, val_data_loader, training_params, save_path):
    epochs = training_params['epochs']
    optimizer = training_params['optimizer']
    loss_fn = training_params['loss']
    scheduler = training_params['scheduler']
    early_stopper = EarlyStopper(patience=15)
    train_loss = None
    val_loss = None
    for epoch in range(epochs):
        loss_at_epoch = []
        for x, y in tqdm(train_data_loader, desc=f"Epoch: {epoch}/{epochs} train_loss: {train_loss} val_loss: {val_loss}"):
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            loss = loss_fn(scores, y)
            loss_at_epoch.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = sum(loss_at_epoch)/len(loss_at_epoch)
        val_loss = compute_validation_loss(model, val_data_loader, loss_fn, device=device)
        scheduler.step(val_loss)
        stop = early_stopper.early_stop(model, val_loss, save_path)
        if stop:
            print(f'Stopping early at epoch {epoch}')
            break

optimizer = optim.AdamW(classifier.parameters(), lr=1e-4)
loss = nn.NLLLoss()
scheduler = ReduceLROnPlateau(optimizer, verbose=True)

train_embeddings = torch.load('./embeddings_train.pth', weights_only=True)
val_embeddings = torch.load('./embeddings_val.pth', weights_only=True)
test_embeddings = torch.load('./embeddings_test.pth', weights_only=True)

train_dataset = EmbeddingDataset(train_embeddings)
val_dataset = EmbeddingDataset(val_embeddings)
test_dataset = EmbeddingDataset(test_embeddings)

train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

train_params = {'epochs': 30, 'optimizer': optimizer, 'loss': loss, 'scheduler': scheduler}

# Save label mappings for inference
import json
with open('label_mappings.json', 'w') as f:
    json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)

#classifier.load_state_dict(torch.load('./domainclassifier.pth', weights_only=True))
train(classifier, train_loader, val_loader, train_params, './domainclassifier')