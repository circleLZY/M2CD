import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# 1. Dataset Class
class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir):
        self.a_dir = os.path.join(root_dir, 'A')
        self.b_dir = os.path.join(root_dir, 'B')
        self.label_dir = os.path.join(root_dir, 'label')
        self.image_filenames = os.listdir(self.a_dir)
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        a_image_path = os.path.join(self.a_dir, self.image_filenames[idx])
        b_image_path = os.path.join(self.b_dir, self.image_filenames[idx])
        label_image_path = os.path.join(self.label_dir, self.image_filenames[idx])
        
        a_image = Image.open(a_image_path).convert('RGB')
        b_image = Image.open(b_image_path).convert('RGB')
        label_image = Image.open(label_image_path).convert('L')
        
        a_image = self.transform(a_image)
        b_image = self.transform(b_image)
        label_image = self.transform(label_image)
        
        # 计算变化区域的比例 (label中255的比例)
        change_ratio = torch.sum(label_image == 1) / (512 * 512)
        
        # 将A和B图像拼接起来作为输入
        input_image = torch.cat((a_image, b_image), dim=0)
        
        return input_image, change_ratio

# 2. 网络结构
class ChangeRatioNet(nn.Module):
    def __init__(self):
        super(ChangeRatioNet, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 128 * 128, 512)
        self.fc2 = nn.Linear(512, 1)  # 输出变化比例
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 128 * 128)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 使用sigmoid输出比例在0到1之间
        return x

# 3. 准确率计算
def compute_accuracy(outputs, targets, threshold=0.01):
    """ 计算预测变化区域比例的准确率
    outputs: 模型预测的变化比例
    targets: 真实的变化比例
    threshold: 预测值与真实值差异的容忍范围
    """
    correct = torch.abs(outputs - targets) < threshold
    return torch.mean(correct.float()).item()

# 4. 训练函数
def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=10, checkpoint_dir='./checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    best_acc = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            accuracy = compute_accuracy(outputs, targets)
            running_accuracy += accuracy
            
            # 每1000次迭代保存一次checkpoint
            if i % 100 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_batch_{i}.pth')
                torch.save(model.state_dict(), checkpoint_path)
        
        # 每个epoch结束后计算平均loss和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_accuracy / len(train_loader)
        
        # 在验证集上测试
        val_loss, val_accuracy = validate_model(val_loader, model, criterion)
        print(f'Epoch {epoch}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}')
        print(f'Epoch {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        
        # 保存最好的模型
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model_path = os.path.join(checkpoint_dir, f'best_model_{epoch}.pth')
            torch.save(model.state_dict(), best_model_path)

# 5. 验证函数
def validate_model(val_loader, model, criterion):
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs).squeeze()
            
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            accuracy = compute_accuracy(outputs, targets)
            val_accuracy += accuracy
    
    return val_loss / len(val_loader), val_accuracy / len(val_loader)

# 6. 主函数
def main():
    train_dir = '/nas/datasets/lzy/RS-ChangeDetection/CGWX/train'
    val_dir = '/nas/datasets/lzy/RS-ChangeDetection/CGWX/val'
    checkpoint_dir = './checkpoints'  # 可以自定义存放checkpoint的路径
    
    train_dataset = ChangeDetectionDataset(train_dir)
    val_dataset = ChangeDetectionDataset(val_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    model = ChangeRatioNet().cuda()
    criterion = nn.MSELoss()  # 回归问题用均方误差
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
    train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=20, checkpoint_dir=checkpoint_dir)

if __name__ == "__main__":
    main()
