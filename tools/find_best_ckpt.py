import os
import subprocess
import re
import json
from pathlib import Path

class CheckpointEvaluator:
    def __init__(self):
        # 基础配置
        self.checkpoint_dir = "/nas/datasets/lzy/RS-ChangeDetection/checkpoints_distill/ChangeFormer-mit-b0/distill/0.001"
        self.config_file = "/home/lzy/proj/rmcd-kd/configs/distill-changeformer/changeformer_mit-b0_512x512_200k_cgwx.py"
        self.show_dir = "/nas/datasets/lzy/RS-ChangeDetection/Figures-KD/TEST/Three-Teachers/ChangeFormer-mit-b0/distill"
        self.results = {}  # 存储所有结果

    def get_checkpoints(self):
        """获取所有checkpoint文件并按迭代次数排序"""
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.endswith('.pth'):
                # 使用正则表达式提取迭代次数
                match = re.search(r'iter_(\d+)', file)
                if match:
                    iter_num = int(match.group(1))
                    checkpoints.append((iter_num, os.path.join(self.checkpoint_dir, file)))
        
        # 按迭代次数排序
        return sorted(checkpoints, key=lambda x: x[0])

    def run_test(self, checkpoint_path):
        """运行测试命令"""
        cmd = [
            "CUDA_VISIBLE_DEVICES=1",
            "python", "tools/test.py",
            self.config_file,
            checkpoint_path,
            "--show-dir", self.show_dir
        ]
        
        try:
            subprocess.run(" ".join(cmd), shell=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"测试失败: {e}")
            return False

    def calculate_miou(self):
        """运行mIOU计算脚本"""
        cmd = [
            "python",
            "/nas/datasets/lzy/RS-ChangeDetection/tools/cal_mIOU.py",
            "--gt_folder", "/nas/datasets/lzy/RS-ChangeDetection/CGWX-Original/test/label",
            "--result_folder", f"{self.show_dir}/vis_data/vis_image"
        ]
        
        try:
            result = subprocess.run(" ".join(cmd), 
                                  shell=True, 
                                  check=True,
                                  capture_output=True,
                                  text=True)
            
            # 从输出中提取mIOU值
            image_miou = float(re.search(r"Average Image mIOU: ([\d.]+)", 
                                       result.stdout).group(1))
            pixel_miou = float(re.search(r"Average Pixel mIOU: ([\d.]+)", 
                                       result.stdout).group(1))
            
            return image_miou, pixel_miou
        except subprocess.CalledProcessError as e:
            print(f"mIOU计算失败: {e}")
            return None, None

    def evaluate_all_checkpoints(self):
        """评估所有checkpoints并保存结果"""
        checkpoints = self.get_checkpoints()
        best_miou = 0
        best_checkpoint = None
        
        # 创建结果目录
        results_dir = Path("checkpoint_evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        for iter_num, checkpoint_path in checkpoints:
            print(f"\n评估checkpoint: {checkpoint_path}")
            
            # 运行测试
            if self.run_test(checkpoint_path):
                # 计算mIOU
                image_miou, pixel_miou = self.calculate_miou()
                
                if image_miou is not None:
                    self.results[iter_num] = {
                        'checkpoint': checkpoint_path,
                        'image_miou': image_miou,
                        'pixel_miou': pixel_miou
                    }
                    
                    # 更新最佳结果
                    if image_miou > best_miou:
                        best_miou = image_miou
                        best_checkpoint = checkpoint_path
                    
                    # 保存当前结果
                    self.save_results()
            
        return best_checkpoint, best_miou

    def save_results(self):
        """保存评估结果"""
        output_file = Path("checkpoint_evaluation_results/changeformer-0.0001.json")
        
        # 将结果转换为更易读的格式
        formatted_results = {
            'all_results': self.results,
            'best_result': {
                'checkpoint': max(self.results.items(), 
                                key=lambda x: x[1]['image_miou'])[1]['checkpoint'],
                'image_miou': max(self.results.items(), 
                                key=lambda x: x[1]['image_miou'])[1]['image_miou']
            }
        }
        
        # 保存为JSON文件
        with open(output_file, 'w') as f:
            json.dump(formatted_results, f, indent=4)
        
        print(f"\n结果已保存至: {output_file}")

def main():
    evaluator = CheckpointEvaluator()
    best_checkpoint, best_miou = evaluator.evaluate_all_checkpoints()
    
    print("\n评估完成!")
    print(f"最佳checkpoint: {best_checkpoint}")
    print(f"最佳image mIOU: {best_miou:.2f}")

if __name__ == "__main__":
    main()