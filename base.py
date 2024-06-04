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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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

        if self.transform:
            anchor_image = self.transform(anchor_image)

        return anchor_image, anchor_label


train_data_path = 'C:/Master/Thesis/Pycharm/pytorch/img_40/train'
test_data_path = 'C:/Master/Thesis/Pycharm/pytorch/img_40/test'
valid_data_path = 'C:/Master/Thesis/Pycharm/pytorch/img_40/valid'
csv_paths = ['C:/Master/Thesis/Pycharm/pytorch/img_40/train_40.csv',
             'C:/Master/Thesis/Pycharm/pytorch/img_40/test_40.csv',
             'C:/Master/Thesis/Pycharm/pytorch/img_40/valid_40.csv']

class_to_label = {'healthy': 0, 'leaf_rust': 1, 'powdery_mildew': 2, 'seedlings': 3, 'septoria': 4, 'stem_rust': 5,
                  'yellow_rust': 6}

data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

batch_size = 10

train_dataset = TripletDataset(root_dir=train_data_path, csv_file=csv_paths[0], transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TripletDataset(root_dir=test_data_path, csv_file=csv_paths[1], transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = TripletDataset(root_dir=valid_data_path, csv_file=csv_paths[2], transform=data_transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

base_model = EfficientNet.from_pretrained('efficientnet-b0')
base_model.fc = nn.Identity()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)
optimizer = optim.Adam(base_model.parameters(), lr=0.001)
num_epochs = 20
cross_loss = nn.CrossEntropyLoss()

train_loss_list = []
valid_loss_list = []

anchor_output_train = []
labels_train = []

for epoch in range(num_epochs):
    base_model.train()
    total_loss = 0.0

    for anchor, idx_label in train_loader:
        optimizer.zero_grad()
        anchor = anchor.to(device)
        anchor_output = base_model(anchor)
        loss = cross_loss(anchor_output, idx_label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss_list.append(total_loss)

    base_model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        correct = 0
        total = 0
        for anchor, indel_labels in valid_loader:
            anchor, indel_labels = anchor.to(device), indel_labels.to(device)
            anchor_output = base_model(anchor)
            loss = cross_loss(anchor_output, indel_labels)
            valid_loss += loss.item()
        valid_loss_list.append(valid_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {total_loss / len(train_loader):.4f},'
          f' Valid Loss: {valid_loss / len(valid_loader):.4f}')

torch.save(base_model, 'base_model_40.pth')
