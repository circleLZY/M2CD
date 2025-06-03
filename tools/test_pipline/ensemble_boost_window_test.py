import numpy as np
import xgboost as xgb
import cv2
import os

# 设置路径
output_folder = '/nas/datasets/lzy/RS-ChangeDetection/Figures/Boost/win5'  # 保存模型、特征等数据的位置
model_path = os.path.join(output_folder, 'win5.model')  # 模型路径
features_path = os.path.join(output_folder, 'features.npy')  # 特征路径
gt_path = os.path.join(output_folder, 'gt_images.npy')  # ground truth 路径
num_models = 11  # 模型数量
window_size = 5  # 窗口大小
image_size = 512  # 图片尺寸
num_images = 100  # 图片总数

# 加载 ground truth 和特征矩阵
gt_images = np.load(gt_path)  # (100, 512, 512)
print('successfully load gt_images!')

features = np.load(features_path)  # (num_images * 512 * 512, window_size * window_size * num_models)
print('successfully load features!')

# 加载 xgboost 模型
bst = xgb.Booster()
bst.load_model(model_path)
print('successfully load xgboost model!')

# 准备预测输出路径
prediction_output_folder = os.path.join(output_folder, 'predictions')
os.makedirs(prediction_output_folder, exist_ok=True)

def process_single_image(image_index):
    # 从features中提取单张图片的特征
    start_idx = image_index * image_size * image_size
    end_idx = (image_index + 1) * image_size * image_size
    image_features = features[start_idx:end_idx]  # (512*512, window_size * window_size * num_models)

    # 预测
    dtest = xgb.DMatrix(image_features)
    preds = bst.predict(dtest)
    preds_binary = (preds > 0.5).astype(int)  # 二值化

    # 将预测结果恢复为图片格式
    pred_image = preds_binary.reshape(image_size, image_size) * 255  # 转换为0和255的格式

    # 保存预测结果
    output_path = os.path.join(prediction_output_folder, f'image_{image_index + 1}.png')
    cv2.imwrite(output_path, pred_image)
    print(f'Prediction for image_{image_index + 1}.png saved at {output_path}')

# 示例：处理第1张图片
for i in range(100):
    process_single_image(i)  # 处理image_1.png
    print(f'successfully process image{i+1}!')

# 如果需要处理其他图片，可以更改image_index的值
# 比如：process_single_image(1) 处理image_2.png，依此类推
