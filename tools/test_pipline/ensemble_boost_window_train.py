import numpy as np
import xgboost as xgb
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 设置路径
gt_folder = '/nas/datasets/lzy/RS-ChangeDetection/CGWX/val/label'
model_folders = [
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
]  # 添加所有模型文件夹名
output_folder = '/nas/datasets/lzy/RS-ChangeDetection/Figures/Boost/win5'  # 设置输出目录
num_models = len(model_folders)
num_images = 100  # 根据你的图像数量

# 设置窗口大小
window_size = 5
half_window = window_size // 2

# 读取 ground truth
gt_images = []
for i in range(1, num_images + 1):
    gt_path = os.path.join(gt_folder, f'image_{i}.png')
    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_images.append(gt_image)

gt_images = np.array(gt_images)  # 形状 (100, 512, 512)

np.save(f'{output_folder}/gt_images.npy', gt_images)
print('gt_images.npy successfully saved')

# 准备特征矩阵：每个像素点用邻域窗口表示
features = np.zeros((num_images, 512, 512, window_size * window_size * num_models))  # 形状 (100, 512, 512, window_size * window_size * num_models)

# 读取每个模型的输出
for model_idx, model_folder in enumerate(model_folders):
    print(model_folder)
    for i in range(1, num_images + 1):
        model_path = os.path.join(model_folder, f'image_{i}.png')
        model_image = cv2.imread(model_path, cv2.IMREAD_GRAYSCALE)

        # 为每个模型的输出添加边界填充，防止窗口超出边界
        padded_image = np.pad(model_image, ((half_window, half_window), (half_window, half_window)), mode='reflect')

        # 生成特征矩阵：对于每个像素，取 11x11 窗口
        for x in range(512):
            for y in range(512):
                window = padded_image[x:x + window_size, y:y + window_size]
                features[i - 1, x, y, model_idx * window_size * window_size:(model_idx + 1) * window_size * window_size] = window.flatten()

# 将特征矩阵展平成 (num_images * 512 * 512, window_size * window_size * num_models)
features = features.reshape(-1, window_size * window_size * num_models)

np.save(f'{output_folder}/features.npy', features)
print('features.npy successfully saved')

# 将 ground truth 转为0/1格式并展平为 (num_images * 512 * 512,)
labels = (gt_images > 0).astype(int).flatten()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 创建 DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# 设置参数
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.1,
    'max_depth': 3,
    'tree_method': 'gpu_hist'  # 使用 GPU 加速
}

# 训练模型
bst = xgb.train(params, dtrain, num_boost_round=100)

bst.save_model(f'{output_folder}/win5.model') 
print('boost model successfully saved')

# 预测
preds = bst.predict(dtest)
preds_binary = (preds > 0.5).astype(int)  # 二值化

# 计算准确率
accuracy = accuracy_score(y_test, preds_binary)
print(f'Accuracy: {accuracy}')

# 如果需要将预测结果转换为图像并保存
os.makedirs(output_folder, exist_ok=True)  # 创建输出目录（如果不存在）

# 预测并保存每张图像的结果
preds_full = bst.predict(xgb.DMatrix(features))  # 针对所有图像进行预测
preds_binary_full = (preds_full > 0.5).astype(int)

# 还原为图像格式并保存
for i in range(num_images):
    pred_image = preds_binary_full[i * 512 * 512: (i + 1) * 512 * 512].reshape(512, 512) * 255  # 将0/1转换为0/255
    output_path = os.path.join(output_folder, f'image_{i+1}.png')  # 生成输出文件路径
    cv2.imwrite(output_path, pred_image)  # 保存预测图像
