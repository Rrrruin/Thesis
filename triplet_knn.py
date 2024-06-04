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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, davies_bouldin_score
from cross_knn import SiameseNetwork
import joblib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


train_data_path = './img_30/train'
test_data_path = './img_30/test'
valid_data_path = './img_30/valid'
csv_paths = ['./img_30/train_30.csv', './img_30/test_30.csv', './img_30/valid_30.csv']

# Map class labels to numerical labels
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

batch_size = 16

train_dataset = TripletDataset(root_dir=train_data_path, csv_file=csv_paths[0], transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TripletDataset(root_dir=test_data_path, csv_file=csv_paths[1], transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
siamese_net = torch.load('siamese_30.pth')
anchor_output_train = []
labels_train = []

for anchor, positive, negative, indel_labels in train_loader:
    anchor, positive, negative, indel_labels = anchor.to(device), positive.to(device), negative.to(
        device), indel_labels.to(device)
    anchor_output, positive_output, negative_output = siamese_net(anchor, positive, negative)

    anchor_output_train.append(anchor_output.detach().cpu().numpy())
    labels_train.append(indel_labels.detach().cpu().numpy())

X_train = np.concatenate(anchor_output_train)
y_train = np.concatenate(labels_train)

# anchor_output_valid = []
# labels_valid = []
#
# for anchor, positive, negative, indel_labels in valid_loader:
#     anchor, positive, negative, indel_labels = anchor.to(device), positive.to(device), negative.to(
#         device), indel_labels.to(device)
#     anchor_output, positive_output, negative_output = siamese_net(anchor, positive, negative)
#
#     anchor_output_valid.append(anchor_output.detach().cpu().numpy())
#     labels_valid.append(indel_labels.detach().cpu().numpy())
#
# X_valid = np.concatenate(anchor_output_valid)
# y_valid = np.concatenate(labels_valid)

anchor_output_test = []
labels_test = []

for anchor, positive, negative, indel_labels in test_loader:
    anchor, positive, negative, indel_labels = anchor.to(device), positive.to(device), negative.to(
        device), indel_labels.to(device)
    anchor_output, positive_output, negative_output = siamese_net(anchor, positive, negative)

    anchor_output_test.append(anchor_output.detach().cpu().numpy())
    labels_test.append(indel_labels.detach().cpu().numpy())

X_test = np.concatenate(anchor_output_test)
y_test = np.concatenate(labels_test)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

#joblib.dump(knn, 'trip_knn_80.pkl')
knn = joblib.load('trip_knn_30.pkl')



y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)


# true_positives = {class_name: 0 for class_name in class_to_label}
# false_positives = {class_name: 0 for class_name in class_to_label}
# false_negatives = {class_name: 0 for class_name in class_to_label}
#
# # Iterate through predictions and ground truth labels
# for pred_label, true_label in zip(y_pred, y_test):
#     pred_class = list(class_to_label.keys())[list(class_to_label.values()).index(pred_label)]
#     true_class = list(class_to_label.keys())[list(class_to_label.values()).index(true_label)]
#
#     # Update true positives, false positives, and false negatives counts
#     if pred_label == true_label:
#         true_positives[pred_class] += 1
#     else:
#         false_positives[pred_class] += 1
#         false_negatives[true_class] += 1
#
# # Compute precision, recall, and F1 score for each class
# print("F1 Score Results:")
# for class_name in class_to_label:
#     precision = true_positives[class_name] / (true_positives[class_name] + false_positives[class_name])
#     recall = true_positives[class_name] / (true_positives[class_name] + false_negatives[class_name])
#     f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
#     print(f"Class: {class_name}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

# # Initialize t-SNE object
# tsne = TSNE(n_components=2, random_state=42)
#
# # Fit and transform the predicted labels for visualization
# X_embedded_pred = tsne.fit_transform(X_test)
# dbi = davies_bouldin_score(X_embedded_pred, y_pred)
# print("Davies–Bouldin index:", dbi)
#
# # Plot the clusters for predicted labels
# unique_class_names = list(class_to_label.keys())
#
# # Plot the clusters for predicted labels
# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X_embedded_pred[:, 0], X_embedded_pred[:, 1], c=y_pred, cmap='viridis', s=10)
# plt.title('t-SNE Visualization of Cross loss')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
#
# # Customize colorbar ticks and labels
# cbar = plt.colorbar(scatter, ticks=range(len(unique_class_names)))
# cbar.ax.set_yticklabels(unique_class_names)
#
# plt.show()
