import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class ChangeDetectionDataset(Dataset):
    def __init__(self, model_paths, gt_path, image_indices, transform=None):
        self.model_paths = model_paths  # List of paths to different model output folders
        self.gt_path = gt_path  # Path to the ground truth labels
        self.image_indices = image_indices  # List of image indices to use
        self.transform = transform  # Any data transformations
        self.num_models = len(model_paths)  # Number of models being used

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        image_idx = self.image_indices[idx]
        inputs = []

        # Load model outputs for this image index
        for model_path in self.model_paths:
            img_path = os.path.join(model_path, f'image_{image_idx}.png')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32) / 255.0  # Normalize image to [0, 1]
            inputs.append(img)

        # Stack the model outputs to form the input tensor (shape: [num_models, 512, 512])
        inputs = np.stack(inputs, axis=0)

        # Load the ground truth label
        gt_path = os.path.join(self.gt_path, f'image_{image_idx}.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = (gt / 255).astype(np.float32)  # Normalize ground truth to [0, 1] for binary classification

        if self.transform:
            inputs = self.transform(inputs)
            gt = self.transform(gt)

        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(gt, dtype=torch.float32)

class FusionNet(nn.Module):
    def __init__(self, num_models):
        super(FusionNet, self).__init__()
        self.conv1 = nn.Conv2d(num_models, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # To get output in the range [0, 1]

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))  # Output a single channel with binary values
        return x

# Set paths
model_folders = model_folders = [
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/Single-Partition-Average',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/Single-Partition/TTP/final/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/Single-Partition/Changer-mit-b0/final/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/Single-Partition/Changer-mit-b1/final/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/Single-Partition/CGNet/final/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/Average', 
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/Single/TTP/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/Single/Changer-mit-b0/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/Single/Changer-mit-b1/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/Single/CGNet/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/Single/BAN-vit-b16-int21k-mit-b2/vis_data/vis_image'
] 

gt_folder = '/nas/datasets/lzy/RS-ChangeDetection/CGWX/val/label'

# Image indices (1 to 100)
image_indices = list(range(1, 101))

# Split data into training (80 images) and validation (20 images)
train_indices, val_indices = train_test_split(image_indices, test_size=0.2, random_state=42)

# Create datasets and dataloaders
train_dataset = ChangeDetectionDataset(model_folders, gt_folder, train_indices)
val_dataset = ChangeDetectionDataset(model_folders, gt_folder, val_indices)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize model, loss function, and optimizer
model = FusionNet(num_models=len(model_folders))
criterion = nn.BCELoss()  # Binary Cross Entropy loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 20
best_val_loss = float('inf')
best_model_path = './best_fusion_model.pth'

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, gt in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, gt.unsqueeze(1))  # Add channel dimension to gt
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, gt in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, gt.unsqueeze(1))
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f'Saved best model at epoch {epoch+1}')

# Load the best model for testing or further use
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Example of using the trained model for prediction on a new image
test_image_index = 1  # Example image index
test_dataset = ChangeDetectionDataset(model_folders, gt_folder, [test_image_index])
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    for inputs, gt in test_loader:
        output = model(inputs)
        # Convert the output back to binary image (0 or 255)
        prediction = (output.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8) * 255

        # Save the predicted image
        cv2.imwrite(f'predicted_image_{test_image_index}.png', prediction[0])
