import os
import chardet
import torch
import random
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
import argparse

def parse_arge():
    parse=argparse.ArgumentParser(description='add parameter')
    parse.add_argument('--model',default='bert',help='bert or lstm')
    parse.add_argument('--lr',default=1e-5)
    parse.add_argument('--epoch',default=10)
    parse.add_argument('--batch_size',default=16)
    parse.add_argument('--dropout',default=0.1)
    parse.add_argument('--embedding_sizes',default=1000)
    parse.add_argument('--num_layers',default=2)
    parse.add_argument('--hidden_dim',default=2048)
    parse.add_argument('--train_percent',default=0.8)
    args=parse.parse_args()
    return args
args=parse_arge()

torch.cuda.empty_cache()

val_accs=[]
val_ps=[]
val_rs=[]
val_fs=[]
num_epochs=args.epoch

class LSTMClassifier(torch.nn.Module):
    def __init__(self, embedding_size, hidden_dim, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=tokenizer.vocab_size, embedding_dim=embedding_size)
        self.lstm = torch.nn.LSTM(input_size=embedding_size, hidden_dim=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embeddings = self.embedding(x)
        lstm_output, _ = self.lstm(embeddings)
        last_hidden_state = lstm_output[:, -1, :]
        logits = self.fc(last_hidden_state)
        return logits

def train(model, optimizer, scheduler, train_data_loader, val_data_loader, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_predictions = []
        total_labels = []

        for batch in train_data_loader:
            batch_texts, batch_labels = batch
            batch_texts = batch_texts.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            if args.model=='bert':
                outputs = model(batch_texts, labels=batch_labels)
                loss = outputs.loss
                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
            elif args.model=='lstm':
                outputs = model(batch_texts)
                loss = criterion(outputs, batch_labels)
                _, predictions = torch.max(outputs, 1)
                total_loss += loss.item() * batch_texts.size(0)
                
            total_predictions.extend(predictions.cpu().tolist())
            total_labels.extend(batch_labels.cpu().tolist())

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_data_loader)
        print(f'Epoch {epoch + 1}: Average Loss = {avg_loss:.4f}')

        val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_data_loader, device)
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        print(f'Validation Precision: {val_precision:.4f}')
        print(f'Validation Recall: {val_recall:.4f}')
        print(f'Validation F1-Score: {val_f1:.4f}')
        val_accs.append(val_accuracy)
        val_ps.append(val_precision)
        val_rs.append(val_recall)
        val_fs.append(val_f1)

        total_predictions = torch.tensor(total_predictions)
        total_labels = torch.tensor(total_labels)

        train_accuracy = accuracy_score(total_labels, total_predictions)
        train_precision = precision_score(total_labels, total_predictions, average='weighted',zero_division=1)
        train_recall = recall_score(total_labels, total_predictions, average='weighted')
        train_f1 = f1_score(total_labels, total_predictions, average='weighted')

        print(f'Train Accuracy: {train_accuracy:.4f}')
        print(f'Train Precision: {train_precision:.4f}')
        print(f'Train Recall: {train_recall:.4f}')
        print(f'Train F1-Score: {train_f1:.4f}')

    return model

def evaluate(model, data_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            batch_texts, batch_labels = batch
            batch_texts = batch_texts.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_texts)
            if args.model=='bert':
                predictions = torch.argmax(outputs.logits, dim=1)
            elif args.model=='lstm':
                _, predictions = torch.max(outputs, 1)
                
            all_labels.extend(batch_labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return accuracy, precision, recall, f1

def collate_fn(batch):
    texts, labels = zip(*batch)
    tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    max_length = max(len(tokens) for tokens in tokenized_texts)
    padded_texts = [tokens + [tokenizer.pad_token_id] * (max_length - len(tokens)) for tokens in tokenized_texts]
    return torch.tensor(padded_texts), torch.tensor(labels)

random.seed(102)
torch.manual_seed(102)
torch.cuda.manual_seed_all(102)

out_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
train_tag = {}
train_txt = {}
failed = []
test_id = []
test_txt = {}
test_tag = {}

with open('dataset/train.txt', encoding='utf-8') as f:
    f.readline()
    for line in f.readlines():
        lst = line.strip().split(',')
        train_tag[lst[0]] = out_dict[lst[1]]

with open('dataset/test_without_label.txt', encoding='utf-8') as f:
    f.readline()
    for line in f.readlines():
        test_id.append(line.strip().split(',')[0])

path = "dataset/data"
print('Begin reading articles...')
directory = os.listdir(path)
for file in directory:
    if file.endswith('.txt'):
        pos = os.path.join(path, file)
        with open(pos, 'rb') as f:
            content = f.read()
            result = chardet.detect(content)
            encoding = result['encoding']
        try:
            with open(pos, 'r', encoding=encoding) as f:
                guid = file.strip('.txt')
                if guid in train_tag.keys():
                    train_txt[guid] = f.read()
                elif guid in test_id:
                    test_txt[guid] = f.read()
        except UnicodeDecodeError as e:
            failed.append(file)
print('failed:', len(failed))
print('test:', len(test_id), ' ', len(test_txt))

model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

if args.model=='bert':
    model = BertForSequenceClassification.from_pretrained(model_name,num_labels=3)
elif args.model=='lstm':
    model = LSTMClassifier(args.embedding_sizes, args.hidden_dim, args.num_layers, 3)

train_texts = list(train_txt.values())
train_labels = [train_tag[guid] for guid in train_txt.keys()]
test_texts = list(test_txt.values())

train_val_dataset = list(zip(train_texts, train_labels))
random.shuffle(train_val_dataset)

train_size = int(args.train_percent * len(train_val_dataset))
train_dataset = train_val_dataset[:train_size]
val_dataset = train_val_dataset[train_size:]
test_dataset = [(text, 5) for text in test_texts]

max_length_train = max(len(tokenizer.encode(text, add_special_tokens=True)) for text, _ in train_dataset)
max_length_test = max(len(tokenizer.encode(text, add_special_tokens=True)) for text in test_texts)
max_length = max(max_length_train, max_length_test)
train_dataset = [(tokenizer.encode(text, add_special_tokens=True, padding='max_length', max_length=max_length), label) for text, label in train_dataset]
val_dataset = [(tokenizer.encode(text, add_special_tokens=True, padding='max_length', max_length=max_length), label) for text, label in val_dataset]
test_dataset = [(tokenizer.encode(text, add_special_tokens=True, padding='max_length', max_length=max_length), label) for text, label in test_dataset]

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

best_val_accs=[]
best_val_ps=[]
best_val_rs=[]
best_val_fs=[]

learning_rates=[args.lr]
dropouts=[args.dropout]
for lr in learning_rates:
    for dropout_rate in dropouts:
        print('learning_rate:',lr,' dropout:',dropout_rate)
        val_accs=[]
        val_ps=[]
        val_rs=[]
        val_fs=[]

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        total_steps = len(train_data_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        if args.model=='bert':
            model.dropout.p = dropout_rate

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return_model=train(model, optimizer, scheduler,  train_data_loader, val_data_loader, device)

        if len(best_val_fs) == 0 or best_val_fs[-1] < val_fs[-1]:
            print('Best model:--learning_rate:',lr,' --dropout:',dropout_rate)
            best_val_accs=val_accs
            best_val_ps=val_ps
            best_val_rs=val_rs
            best_val_fs=val_fs
            torch.save(return_model.state_dict(), 'best_model.pt')

model.load_state_dict(torch.load('best_model.pt'))

test_predictions = []
model.eval()
with torch.no_grad():
    for batch in test_data_loader:
        batch_texts = batch[0].to(device)
        outputs = model(batch_texts)
        if args.model=='bert':
            predictions = torch.argmax(outputs.logits, dim=1)
        elif args.model=='lstm':
            _, predictions = torch.max(outputs, 1)
        test_predictions.extend(predictions.cpu().tolist())

with open('test_predictions.txt', 'w', encoding='utf-8') as f:
    f.write('guid,tag\n')
    for guid, prediction in zip(test_txt.keys(), test_predictions):
        line = f'{guid},{prediction}\n'
        f.write(line)

plt.figure(figsize=(24, 3))
plt.subplot(1, 4, 1)
plt.plot(range(1,num_epochs+1,1), best_val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 4, 2)
plt.plot(range(1,num_epochs+1,1), best_val_ps, label='Val Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

plt.subplot(1, 4, 3)
plt.plot(range(1,num_epochs+1,1), best_val_rs, label='Val Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(range(1,num_epochs+1,1), best_val_fs, label='Val F1')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.legend()

plt.tight_layout()
plt.savefig('train_metrics.png')
plt.show()