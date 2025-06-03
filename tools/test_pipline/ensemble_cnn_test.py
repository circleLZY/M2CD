import torch
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 加载数据集的类 (保持和训练时相同)
class ChangeDetectionDataset(Dataset):
    def __init__(self, model_paths, gt_path, image_indices, transform=None):
        self.model_paths = model_paths
        self.gt_path = gt_path
        self.image_indices = image_indices
        self.transform = transform
        self.num_models = len(model_paths)

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        image_idx = self.image_indices[idx]
        inputs = []

        # Load model outputs for this image index
        for model_path in self.model_paths:
            img_path = os.path.join(model_path, f'image_{image_idx}.png')
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Check if image is None (i.e., failed to load)
            if img is None:
                print(f"Warning: Failed to load image at {img_path}. Skipping this data point.")
                return None

            img = img.astype(np.float32) / 255.0  # Normalize image to [0, 1]
            inputs.append(img)

        # Stack the model outputs to form the input tensor (shape: [num_models, 512, 512])
        inputs = np.stack(inputs, axis=0)

        # Load the ground truth label
        gt_path = os.path.join(self.gt_path, f'image_{image_idx}.png')
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if gt is None:
            print(f"Warning: Failed to load ground truth at {gt_path}. Skipping this data point.")
            return None

        gt = (gt / 255).astype(np.float32)  # Normalize ground truth to [0, 1] for binary classification

        if self.transform:
            inputs = self.transform(inputs)
            gt = self.transform(gt)

        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(gt, dtype=torch.float32)


# 定义 FusionNet (与训练中使用的相同)
class FusionNet(nn.Module):
    def __init__(self, num_models):
        super(FusionNet, self).__init__()
        self.conv1 = nn.Conv2d(num_models, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # 输出范围[0,1]

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x

# 设置路径
model_folders = [
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/Single-Partition-Average',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/Single-Partition/TTP/final/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/Single-Partition/Changer-mit-b0/final/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/Single-Partition/Changer-mit-b1/final/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/Single-Partition/CGNet/final/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/Average', 
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/Single/TTP/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/Single/Changer-mit-b0/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/Single/Changer-mit-b1/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/Single/CGNet/vis_data/vis_image',
    '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/Single/BAN-vit-b16-int21k-mit-b2/vis_data/vis_image'
] 

gt_folder = '/nas/datasets/lzy/RS-ChangeDetection/CGWX/test/label'

# 图片索引 (1到100)
image_indices = list(range(1, 1001))

# 加载训练好的模型
model = FusionNet(num_models=len(model_folders))
best_model_path = '/nas/datasets/lzy/RS-ChangeDetection/Figures/CNN/best_fusion_model.pth'
model.load_state_dict(torch.load(best_model_path))
model.eval()  # 切换到评估模式

# 创建测试数据集和DataLoader
test_dataset = ChangeDetectionDataset(model_folders, gt_folder, image_indices)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 设置输出路径
output_folder = '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/CNN'
os.makedirs(output_folder, exist_ok=True)

# 开始测试并保存结果
with torch.no_grad():
    for i, (inputs, _) in enumerate(test_loader):
        # 推理
        output = model(inputs)

        # 将输出转化为二值图像 (0 或 255)
        prediction = (output.squeeze(1).cpu().numpy() > 0.5).astype(np.uint8) * 255

        # 保存预测结果
        image_index = image_indices[i]
        output_path = os.path.join(output_folder, f'image_{image_index}.png')
        cv2.imwrite(output_path, prediction[0])  # 保存每张预测结果

        print(f'Saved predicted image {image_index} to {output_path}')
