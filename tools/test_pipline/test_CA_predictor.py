import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
from train_CA_predictor import ChangeRatioNet

# 1. 测试集Dataset类
class ChangeDetectionTestDataset(Dataset):
    def __init__(self, root_dir):
        self.a_dir = os.path.join(root_dir, 'A')
        self.b_dir = os.path.join(root_dir, 'B')
        self.image_filenames = os.listdir(self.a_dir)
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        a_image_path = os.path.join(self.a_dir, self.image_filenames[idx])
        b_image_path = os.path.join(self.b_dir, self.image_filenames[idx])
        
        a_image = Image.open(a_image_path).convert('RGB')
        b_image = Image.open(b_image_path).convert('RGB')
        
        a_image = self.transform(a_image)
        b_image = self.transform(b_image)
        
        # 将A和B图像拼接起来作为输入
        input_image = torch.cat((a_image, b_image), dim=0)
        
        return self.image_filenames[idx], input_image

# 2. 加载模型
def load_model(checkpoint_path):
    model = ChangeRatioNet().cuda()  # 模型需要与训练时保持一致
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()  # 切换到评估模式
    return model

# 3. 测试函数
def test_model(test_loader, model):
    results = []
    with torch.no_grad():
        for image_name, inputs in test_loader:
            inputs = inputs.cuda()
            outputs = model(inputs).squeeze()  # 获取预测的变化区域比例
            change_ratio = outputs.item()  # 转换为Python标量
            results.append((image_name[0], change_ratio))
    
    return results

# 4. 输出测试结果
def print_results(results):
    for image_name, change_ratio in results:
        print(f"{image_name}: {change_ratio:.4f}")

# 5. 主函数
def main():
    test_dir = '/nas/datasets/lzy/RS-ChangeDetection/CGWX/val'  # 测试集目录
    checkpoint_path = './checkpoints/best_model.pth'  # 指定checkpoint路径
    
    # 加载测试集
    test_dataset = ChangeDetectionTestDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # 加载模型
    model = load_model(checkpoint_path)
    
    # 进行测试并输出结果
    results = test_model(test_loader, model)
    print_results(results)

if __name__ == "__main__":
    main()
