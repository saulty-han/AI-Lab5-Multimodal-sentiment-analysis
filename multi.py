import os
import chardet
import torch
import random
from transformers import BertTokenizer, BertForSequenceClassification
from torchvision.models import resnet50,resnet101
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
from multi_model import MultimodalSentimentAnalysisModel_add,MultimodalSentimentAnalysisModel_atten,MultimodalSentimentAnalysisModel_cat_direct,MultimodalSentimentAnalysisModel_cat_trans
import argparse

def parse_arge():
    parse=argparse.ArgumentParser(description='add parameter')
    parse.add_argument('--model',default='add',help='cat_direct or add or cat_trans or cat_atten')
    parse.add_argument('--lr',default=1e-5)
    parse.add_argument('--epoch',default=1)
    parse.add_argument('--batch_size',default=16)
    parse.add_argument('--dropout',default=0.1)
    parse.add_argument('--num_heads',default=16)
    parse.add_argument('--train_percent',default=0.8)
    parse.add_argument('--step_size',default=1)
    parse.add_argument('--scheduler_gamma',default=0.1)
    parse.add_argument('--hidden_dim',default=2048)
    args=parse.parse_args()
    return args
args=parse_arge()

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
num_epochs=args.epoch
val_accs=[]
val_ps=[]
val_rs=[]
val_fs=[]

data_folder = 'dataset'
image_folder = os.path.join(data_folder, 'data')
text_folder = os.path.join(data_folder, 'data')
test_id_file = os.path.join(data_folder, 'test_without_label.txt')
train_file = os.path.join(data_folder, 'train.txt')
out_dict = {'negative': 0, 'neutral': 1, 'positive': 2}

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_id_list = []
with open(test_id_file, 'r') as f:
    f.readline()
    for line in f.readlines():
        test_id_list.append(line.strip().split(',')[0])

train_id = []
train_tag = {}
with open(train_file, 'r') as f:
    f.readline()
    for line in f.readlines():
        id, tag = line.strip().split(',')
        train_id.append(id)
        train_tag[id] = int(out_dict[tag])

model_name = 'bert-large-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

resnet_model = resnet101(pretrained=True)

if args.model=='cat_direct':
    model = MultimodalSentimentAnalysisModel_cat_direct(bert_model, resnet_model, num_labels=3, hidden_dim=args.hidden_dim)
elif args.model=='add':
    model = MultimodalSentimentAnalysisModel_add(bert_model, resnet_model, num_labels=3, hidden_dim=args.hidden_dim)
elif args.model=='cat_trans':
    model = MultimodalSentimentAnalysisModel_cat_trans(bert_model, resnet_model, num_labels=3, hidden_dim=args.hidden_dim)
elif args.model=='cat_atten':
    model = MultimodalSentimentAnalysisModel_atten(bert_model, resnet_model, num_labels=3, hidden_dim=args.hidden_dim, num_heads=args.num_heads, dropout=args.dropout)

random.shuffle(train_id)
train_size = int(args.train_percent * len(train_id))
train_dataset = train_id[:train_size]
val_dataset = train_id[train_size:]

train_texts = []
train_images = []
train_labels = []
failed_txt = []
for id in train_dataset:
    text_file = os.path.join(text_folder, f"{id}.txt")
    image_file = os.path.join(image_folder, f"{id}.jpg")

    with open(text_file, 'rb') as f:
            content = f.read()
            result = chardet.detect(content)
            encoding = result['encoding']
    try:
        with open(text_file, 'r', encoding=encoding) as f:
            text = f.read()
    except UnicodeDecodeError as e:
        failed_txt.append(id)

    image = Image.open(image_file).convert("RGB")
    image = image_transform(image)

    train_texts.append(text)
    train_images.append(image)
    train_labels.append(train_tag[id])

val_texts = []
val_images = []
val_labels = []
for id in val_dataset:
    text_file = os.path.join(text_folder, f"{id}.txt")
    image_file = os.path.join(image_folder, f"{id}.jpg")

    with open(text_file, 'rb') as f:
        content = f.read()
        result = chardet.detect(content)
        encoding = result['encoding']
    try:
        with open(text_file, 'r', encoding=encoding) as f:
            text = f.read()
    except UnicodeDecodeError as e:
        failed_txt.append(id)

    image = Image.open(image_file).convert("RGB")
    image = image_transform(image)

    val_texts.append(text)
    val_images.append(image)
    val_labels.append(train_tag[id])

train_dataset = list(zip(train_texts, train_images, train_labels))
val_dataset = list(zip(val_texts, val_images, val_labels))

def collate_fn(batch):
    texts, images, labels = zip(*batch)
    tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    max_length = max(len(tokens) for tokens in tokenized_texts)
    padded_texts = [tokens + [tokenizer.pad_token_id] * (max_length - len(tokens)) for tokens in tokenized_texts]
    return torch.tensor(padded_texts), torch.stack(images), torch.tensor(labels)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
total_steps = len(train_data_loader) * num_epochs
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.scheduler_gamma)
total_steps = len(train_data_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

best_val_accuracy = 0.0

for epoch in range(num_epochs):
    print('Epoch: ',epoch+1)
    model.train()
    total_loss = 0.0

    for batch in train_data_loader:
        batch_texts, batch_images, batch_labels = batch
        batch_texts = batch_texts.to(device)
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)
        # 将目标标签转换为one-hot编码
        # batch_labels_one_hot = F.one_hot(batch_labels, num_classes=4).float().to(torch.int64)
        # batch_labels_one_hot=batch_labels_one_hot.to(device)

        optimizer.zero_grad()
        logits = model(batch_texts, batch_images)
        # loss = loss_fn(logits, batch_labels_one_hot)
        loss = loss_fn(logits, batch_labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    model.eval()
    val_predictions = []
    val_true_labels = []

    with torch.no_grad():
        for batch in val_data_loader:
            batch_texts, batch_images, batch_labels = batch
            batch_texts = batch_texts.to(device)
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_texts, batch_images)
            predictions = torch.argmax(logits, dim=1)

            val_predictions.extend(predictions.cpu().tolist())
            val_true_labels.extend(batch_labels.cpu().tolist())

    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    val_precision = precision_score(val_true_labels, val_predictions, average='weighted')
    val_recall = recall_score(val_true_labels, val_predictions, average='weighted')
    val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pt')

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation F1-Score: {val_f1:.4f}")

    val_accs.append(val_accuracy)
    val_ps.append(val_precision)
    val_rs.append(val_recall)
    val_fs.append(val_f1)

model.load_state_dict(torch.load('best_model.pt'))

test_data = []
for id in test_id_list:
    text_file = os.path.join(text_folder, f"{id}.txt")
    image_file = os.path.join(image_folder, f"{id}.jpg")

    with open(text_file, 'rb') as f:
        content = f.read()
        result = chardet.detect(content)
        encoding = result['encoding']
    try:
        with open(text_file, 'r', encoding=encoding) as f:
            text = f.read()
    except UnicodeDecodeError as e:
        failed_txt.append(id)

    image = Image.open(image_file).convert("RGB")
    image = image_transform(image)

    test_data.append((text, image))

test_predictions = []
model.eval()

with torch.no_grad():
    for text, image in test_data:
        text = tokenizer.encode(text, add_special_tokens=True)
        text = torch.tensor(text).unsqueeze(0).to(device)
        image = image.unsqueeze(0).to(device)

        logits = model(text, image)
        predictions = torch.argmax(logits, dim=1)

        test_predictions.append(predictions.item())

with open('test_predictions.txt', 'w') as f:
    f.write('guid,tag\n')
    for id, prediction in zip(test_id_list, test_predictions):
        f.write(f"{id},{prediction}\n")

plt.figure(figsize=(24, 3))
plt.subplot(1, 4, 1)
plt.plot(range(1,num_epochs+1,1), val_accs, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 4, 2)
plt.plot(range(1,num_epochs+1,1), val_ps, label='Val Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

plt.subplot(1, 4, 3)
plt.plot(range(1,num_epochs+1,1), val_rs, label='Val Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

plt.subplot(1, 4, 4)
plt.plot(range(1,num_epochs+1,1), val_fs, label='Val F1')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.legend()

plt.tight_layout()
plt.savefig('train_metrics.png')
plt.show()