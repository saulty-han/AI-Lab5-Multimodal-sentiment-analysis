import torch

class MultimodalSentimentAnalysisModel_atten(torch.nn.Module):
    def __init__(self, bert_model, resnet_model, num_labels, hidden_dim=2048, num_heads=16, dropout=0.5):
        super(MultimodalSentimentAnalysisModel_atten, self).__init__()
        self.bert_model = bert_model
        self.resnet_model = resnet_model
        self.fc1 = torch.nn.Linear(1003, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.attention = torch.nn.MultiheadAttention(hidden_dim, num_heads)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.batch_norm = torch.nn.BatchNorm1d(hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_labels)
        self.activation = torch.nn.Sigmoid()

    def forward(self, texts, images):
        bert_outputs = self.bert_model(texts)
        text_embeddings = bert_outputs.logits
        image_embeddings = self.resnet_model(images)
        combined_embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)
        combined_embeddings = self.fc1(combined_embeddings)
        combined_embeddings = self.relu(combined_embeddings)

        combined_embeddings = combined_embeddings.unsqueeze(0).transpose(0, 1)

        combined_embeddings, _ = self.attention(combined_embeddings, combined_embeddings, combined_embeddings)

        combined_embeddings = combined_embeddings.transpose(0, 1).squeeze(0)
        combined_embeddings = self.fc2(combined_embeddings)
        combined_embeddings = self.relu(combined_embeddings)

        combined_embeddings = self.dropout(combined_embeddings)
        combined_embeddings = self.batch_norm(combined_embeddings)
        combined_embeddings = self.activation(combined_embeddings)

        logits = self.fc3(combined_embeddings)
        return logits
    
class MultimodalSentimentAnalysisModel_cat_trans(torch.nn.Module):
    def __init__(self, bert_model, resnet_model, num_labels, hidden_dim=2048):
        super(MultimodalSentimentAnalysisModel_cat_trans, self).__init__()
        self.bert_model = bert_model
        self.resnet_model = resnet_model
        self.fc_txt = torch.nn.Linear(3, hidden_dim//2)
        self.fc_img = torch.nn.Linear(1000, hidden_dim//2)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, num_labels)

    def forward(self, texts, images):
        bert_outputs = self.bert_model(texts)
        text_embeddings = bert_outputs.logits
        image_embeddings = self.resnet_model(images)
        text_embeddings = self.fc_txt(text_embeddings)
        image_embeddings = self.fc_img(image_embeddings)
        text_embeddings = self.relu(text_embeddings)
        image_embeddings = self.relu(image_embeddings)
        combined_embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)
        combined_embeddings = self.fc1(combined_embeddings)
        combined_embeddings = self.relu(combined_embeddings)

        logits = self.fc2(combined_embeddings)
        return logits
    

class MultimodalSentimentAnalysisModel_cat_direct(torch.nn.Module):
    def __init__(self, bert_model, resnet_model, num_labels, hidden_dim=2048):
        super(MultimodalSentimentAnalysisModel_cat_direct, self).__init__()
        self.bert_model = bert_model
        self.resnet_model = resnet_model
        self.fc1 = torch.nn.Linear(1003, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, num_labels)

    def forward(self, texts, images):
        bert_outputs = self.bert_model(texts)
        text_embeddings = bert_outputs.logits
        image_embeddings = self.resnet_model(images)
        combined_embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)
        combined_embeddings = self.fc1(combined_embeddings)
        combined_embeddings = self.relu(combined_embeddings)

        logits = self.fc2(combined_embeddings)
        return logits
    
class MultimodalSentimentAnalysisModel_add(torch.nn.Module):
    def __init__(self, bert_model, resnet_model, num_labels, hidden_dim=2048):
        super(MultimodalSentimentAnalysisModel_add, self).__init__()
        self.bert_model = bert_model
        self.resnet_model = resnet_model
        self.fc_txt = torch.nn.Linear(3, hidden_dim//2)
        self.fc_img = torch.nn.Linear(1000, hidden_dim//2)
        self.fc1 = torch.nn.Linear(hidden_dim//2, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, num_labels)

    def forward(self, texts, images):
        bert_outputs = self.bert_model(texts)
        text_embeddings = bert_outputs.logits
        image_embeddings = self.resnet_model(images)
        text_embeddings = self.fc_txt(text_embeddings)
        image_embeddings = self.fc_img(image_embeddings)
        text_embeddings = self.relu(text_embeddings)
        image_embeddings = self.relu(image_embeddings)

        combined_embeddings = torch.add(text_embeddings, image_embeddings)
        combined_embeddings = self.fc1(combined_embeddings)
        combined_embeddings = self.relu(combined_embeddings)

        logits = self.fc2(combined_embeddings)
        return logits