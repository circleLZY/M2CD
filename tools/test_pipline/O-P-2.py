import os
import subprocess
import numpy as np
from PIL import Image
import shutil
from pathlib import Path
import tqdm

class ModelEvaluator:
    def __init__(self):
        # 基础路径配置
        self.base_dir = "/nas/datasets/lzy/RS-ChangeDetection"
        self.config_file = "/home/lzy/proj/rmcd-kd/configs/distill-changeformer-sysucd/changeformer_mit-b0_512x512_200k_cgwx.py"
        self.ckpt_base = f"{self.base_dir}/Best_ckpt-SYSU-CD/ChangeFormer-MiT-b0"
        self.output_base = f"{self.base_dir}/Figures-SYSUCD/TEST/Two-Teachers/ChangeFormer-MiT-b0"
        
        # 创建final目录结构
        self.final_dir = os.path.join(self.output_base, "final", "vis_data", "vis_image")
        os.makedirs(self.final_dir, exist_ok=True)

        # 模型配置 - 移除了medium模型
        self.models = {
            'initial': 'best_mIoU_iter_11000.pth',
            'large-2': 'best_mIoU_iter_82000.pth',
            'small-2': 'best_mIoU_iter_11000.pth'
        }

        # CAR阈值
        self.car_threshold = 0.1  # 新的CAR阈值

    def run_tests(self):
        """运行所有模型的测试"""
        for model_type, ckpt_name in self.models.items():
            print(f"\n测试 {model_type} 模型...")
            
            # 构建完整的checkpoint路径
            ckpt_path = os.path.join(self.ckpt_base, model_type, ckpt_name)
            output_dir = os.path.join(self.output_base, model_type)
            
            # 构建并运行测试命令
            cmd = [
                "CUDA_VISIBLE_DEVICES=0",
                "python tools/test.py",
                self.config_file,
                ckpt_path,
                "--show-dir", output_dir
            ]
            
            try:
                subprocess.run(" ".join(cmd), shell=True, check=True)
                print(f"{model_type} 模型测试完成")
            except subprocess.CalledProcessError as e:
                print(f"{model_type} 模型测试失败: {e}")

    def calculate_car(self, image_path):
        """计算单张图片的CAR值"""
        img = np.array(Image.open(image_path))
        total_pixels = img.size
        change_pixels = np.sum(img == 255)
        return change_pixels / total_pixels

    def select_and_copy_images(self):
        """基于CAR值选择并复制图片"""
        # 获取initial文件夹中的所有图片
        initial_dir = os.path.join(self.output_base, "initial", "vis_data", "vis_image")
        if not os.path.exists(initial_dir):
            raise FileNotFoundError(f"Initial directory not found: {initial_dir}")

        # 获取所有图片并排序
        image_files = sorted(os.listdir(initial_dir))
        
        print("\n开始处理图片...")
        for image_name in tqdm.tqdm(image_files):
            # 计算CAR值
            initial_image_path = os.path.join(initial_dir, image_name)
            car = self.calculate_car(initial_image_path)
            
            # 根据新的CAR阈值选择源文件夹
            if car < self.car_threshold:
                source_type = "small-2"
            else:
                source_type = "large-2"
            
            # 构建源文件路径并复制文件
            source_path = os.path.join(
                self.output_base,
                source_type,
                "vis_data",
                "vis_image",
                image_name
            )
            
            if os.path.exists(source_path):
                shutil.copy2(source_path, os.path.join(self.final_dir, image_name))
            else:
                print(f"Warning: Source file not found: {source_path}")

    def verify_results(self):
        """验证处理结果"""
        # 定义initial目录路径
        initial_dir = os.path.join(self.output_base, "initial", "vis_data", "vis_image")
        
        # 计算图片数量
        initial_count = len(os.listdir(initial_dir))
        final_count = len(os.listdir(self.final_dir))
        
        print("\n验证结果:")
        print(f"Initial图片数量: {initial_count}")
        print(f"Final图片数量: {final_count}")
        if initial_count == final_count:
            print("验证通过: 图片数量匹配")
        else:
            print("警告: 图片数量不匹配")
            
        # 统计选择情况
        small_count = 0
        large_count = 0
        for image_name in os.listdir(initial_dir):
            initial_image_path = os.path.join(initial_dir, image_name)
            car = self.calculate_car(initial_image_path)
            if car < self.car_threshold:
                small_count += 1
            else:
                large_count += 1
                
        print(f"按CAR阈值（{self.car_threshold}）分类结果:")
        print(f"小于阈值（small）图片数量: {small_count} ({small_count/initial_count*100:.2f}%)")
        print(f"大于等于阈值（large）图片数量: {large_count} ({large_count/initial_count*100:.2f}%)")


def main():
    evaluator = ModelEvaluator()
    
    # 1. 运行所有模型的测试
    print("开始运行模型测试...")
    evaluator.run_tests()
    
    # 2. 基于CAR值选择并复制图片
    print("\n开始选择并复制图片...")
    evaluator.select_and_copy_images()
    
    # 3. 验证结果
    evaluator.verify_results()

if __name__ == "__main__":
    main()