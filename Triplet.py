import torch
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from efficientnet_pytorch import EfficientNet
import pandas as pd
import os
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, pairwise_distances

train_data_path = 'C:/Master/Thesis/Pycharm/pytorch/img_40/train'
test_data_path = 'C:/Master/Thesis/Pycharm/pytorch/img_40/test'
valid_data_path = 'C:/Master/Thesis/Pycharm/pytorch/img_40/valid'
csv_paths = ['C:/Master/Thesis/Pycharm/pytorch/img_40/train_40.csv',
             'C:/Master/Thesis/Pycharm/pytorch/img_40/test_40.csv',
             'C:/Master/Thesis/Pycharm/pytorch/img_40/valid_40.csv']

# Map class labels to numerical labels
class_to_label = {'healthy': 0, 'leaf_rust': 1, 'powdery_mildew': 2, 'seedlings': 3, 'septoria': 4, 'stem_rust': 5,
                  'yellow_rust': 6}


class TripletDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.df['label'] = self.df.iloc[:, 1:].idxmax(axis=1)
        self.df['label'] = self.df['label'].map(class_to_label)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Select a random row index for positive and negative samples
        anchor_img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        anchor_image = Image.open(anchor_img_name).convert('RGB')
        anchor_label = torch.tensor(self.df.loc[idx, 'label'], dtype=torch.long)

        # Find pos image
        positive_idx = torch.randint(0, len(self.df), (1,))
        while self.df.loc[positive_idx.item(), 'label'] != anchor_label:
            positive_idx = torch.randint(0, len(self.df), (1,))
        positive_img_name = os.path.join(self.root_dir, self.df.iloc[positive_idx.item(), 0])
        positive_image = Image.open(positive_img_name).convert('RGB')

        # Find neg image
        negative_idx = torch.randint(0, len(self.df), (1,))
        while self.df.loc[negative_idx.item(), 'label'] == anchor_label:
            negative_idx = torch.randint(0, len(self.df), (1,))
        negative_img_name = os.path.join(self.root_dir, self.df.iloc[negative_idx.item(), 0])
        negative_image = Image.open(negative_img_name).convert('RGB')

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image, anchor_label

class SiameseNetwork(nn.Module):
    def __init__(self, base_model, dropout_prob=0.5):
        super(SiameseNetwork, self).__init__()
        self.base_model = base_model
        self.base_model.fc = nn.Identity()
        self.fc = nn.Linear(1000, 7)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward_once(self, x):
        output = self.base_model(x)
        output = self.dropout(output)
        output = self.fc(output)
        return output

    def forward(self, anchor, positive, negative):
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)

        return anchor_output, positive_output, negative_output


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_output, positive_output, negative_output):
        dist_positive_sq = torch.sum((anchor_output - positive_output) ** 2, dim=1)
        dist_negative_sq = torch.sum((anchor_output - negative_output) ** 2, dim=1)

        # Compute the difference between distances and add the margin
        # loss_values = dist_positive_sq - dist_negative_sq + self.margin
        loss_value = torch.relu(dist_positive_sq - dist_negative_sq + self.margin)
        # Compute the mean of the losses
        loss_mean = torch.mean(loss_value)

        return loss_mean

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

batch_size = 16

train_dataset = TripletDataset(root_dir=train_data_path, csv_file=csv_paths[0], transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TripletDataset(root_dir=test_data_path, csv_file=csv_paths[1], transform=data_transform)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = TripletDataset(root_dir=valid_data_path, csv_file=csv_paths[2], transform=data_transform)
valid_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

base_model = EfficientNet.from_pretrained('efficientnet-b0')
siamese_net = SiameseNetwork(base_model, dropout_prob=0.5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
siamese_net.to(device)
optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)
num_epochs = 20
triplet_loss = TripletLoss()

train_loss_list = []
valid_loss_list = []
for epoch in range(num_epochs):
    siamese_net.train()
    total_loss = 0.0

    for anchor, positive, negative, idx_label in train_loader:
        optimizer.zero_grad()
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        anchor_output, positive_output, negative_output = siamese_net(anchor, positive, negative)
        loss = triplet_loss(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss_list.append(total_loss)
    siamese_net.eval()
    with torch.no_grad():
        valid_loss = 0.0
        correct = 0
        total = 0
        for anchor, positive, negative, indel_labels in valid_loader:
            anchor, positive, negative, indel_labels = anchor.to(device), positive.to(device), negative.to(
                device), indel_labels.to(device)
            anchor_output, positive_output, negative_output = siamese_net(anchor, positive, negative)
            loss = triplet_loss(anchor_output, positive_output, negative_output)
            valid_loss += loss.item()
        valid_loss_list.append(valid_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f},' 
          f' Valid Loss: {valid_loss / len(valid_loader):.4f}')

torch.save(siamese_net, 'siamese_40.pth')