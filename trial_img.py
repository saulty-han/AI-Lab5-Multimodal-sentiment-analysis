import os
import chardet
import torch
import random
from torchvision.models import resnet50,resnet101
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import tqdm
import argparse

def parse_arge():
    parse=argparse.ArgumentParser(description='add parameter')
    parse.add_argument('--lr',default=1e-5)
    parse.add_argument('--epoch',default=15)
    parse.add_argument('--batch_size',default=16)
    parse.add_argument('--train_percent',default=0.8)
    args=parse.parse_args()
    return args
args=parse_arge()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_folder = 'dataset'
image_folder = os.path.join(data_folder, 'data')
test_id_file = os.path.join(data_folder, 'test_without_label.txt')
train_file = os.path.join(data_folder, 'train.txt')
out_dict = {'negative': 0, 'neutral': 1, 'positive': 2}
val_accs=[]
val_ps=[]
val_rs=[]
val_fs=[]

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

random.shuffle(train_id)
train_size = int(args.train_percent * len(train_id))
train_dataset = train_id[:train_size]
val_dataset = train_id[train_size:]

train_images = []
train_labels = []
for id in train_dataset:
    image_file = os.path.join(image_folder, f"{id}.jpg")

    image = Image.open(image_file).convert("RGB")
    image = image_transform(image)

    train_images.append(image)
    train_labels.append(train_tag[id])

val_images = []
val_labels = []
for id in val_dataset:
    image_file = os.path.join(image_folder, f"{id}.jpg")

    image = Image.open(image_file).convert("RGB")
    image = image_transform(image)

    val_images.append(image)
    val_labels.append(train_tag[id])
random.shuffle(train_id)
train_size = int(0.8 * len(train_id))
train_dataset = train_id[:train_size]
val_dataset = train_id[train_size:]
num_classes = len(out_dict)

model = resnet101(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.to(device)

learning_rate = args.lr
num_epochs = args.epoch
batch_size = args.batch_size

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    print('Epoch: ', epoch+1)
    model.train()
    torch.cuda.empty_cache()

    train_images = []
    train_labels = []
    for id in train_dataset:
        image_file = os.path.join(image_folder, f"{id}.jpg")

        image = Image.open(image_file).convert("RGB")
        image = image_transform(image)

        train_images.append(image)
        train_labels.append(train_tag[id])

    train_images = torch.stack(train_images)
    train_labels = torch.tensor(train_labels)

    num_batches = len(train_images) // batch_size
    for batch_idx in tqdm.tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = start_idx + batch_size

        batch_images = train_images[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]
        batch_images = batch_images.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    val_images = []
    val_labels = []
    for id in tqdm.tqdm(val_dataset):
        image_file = os.path.join(image_folder, f"{id}.jpg")

        image = Image.open(image_file).convert("RGB")
        image = image_transform(image)

        val_images.append(image)
        val_labels.append(train_tag[id])

    val_images = torch.stack(val_images)
    val_labels = torch.tensor(val_labels)
    val_images = val_images.to(device)
    val_labels = val_labels.to(device)

    with torch.no_grad():
        val_outputs = model(val_images)
        val_predictions = torch.argmax(val_outputs, dim=1)

    val_accuracy = accuracy_score(val_labels.cpu().tolist(), val_predictions.cpu().tolist())
    val_precision = precision_score(val_labels.cpu().tolist(), val_predictions.cpu().tolist(), average='macro')
    val_recall = recall_score(val_labels.cpu().tolist(), val_predictions.cpu().tolist(), average='macro')
    val_f1 = f1_score(val_labels.cpu().tolist(), val_predictions.cpu().tolist(), average='macro')

    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation F1-Score: {val_f1:.4f}")

    val_accs.append(val_accuracy)
    val_ps.append(val_precision)
    val_rs.append(val_recall)
    val_fs.append(val_f1)

test_images = []
for id in test_id_list:
    image_file = os.path.join(image_folder, f"{id}.jpg")

    image = Image.open(image_file).convert("RGB")
    image = image_transform(image)

    test_images.append((image))

test_predictions = []
model.eval()

with torch.no_grad():
    for image in test_images:
        image = image.unsqueeze(0).to(device)

        logits = model(image)
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