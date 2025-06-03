import os
import numpy as np
from PIL import Image
import itertools

# 定义输入文件夹路径的根目录和输出文件夹路径
root_folder = "/nas/datasets/lzy/RS-ChangeDetection/Figures-Augment"
best_output_folder = os.path.join(root_folder, "Average/Average-new")

# 定义需要处理的模型子文件夹名称
model_folders = [
    # '/nas/datasets/lzy/RS-ChangeDetection/Figures-Augment/Average/Average-78.44',
    # '/nas/datasets/lzy/RS-ChangeDetection/Figures-Augment/Average/Average-78.59',
    "/nas/datasets/lzy/RS-ChangeDetection/Figures-Augment/Changer-mit-b1/Initial/Distill-2-0.001/vis_data/vis_image",
    "/nas/datasets/lzy/RS-ChangeDetection/Figures-Augment/Changer-mit-b1/Initial/Augment-Distill-2-0.001/vis_data/vis_image",
    
    # "/nas/datasets/lzy/RS-ChangeDetection/Figures-Augment/Changer-mit-b1/Initial/Distill-3-0.001/vis_data/vis_image",
    "/nas/datasets/lzy/RS-ChangeDetection/Figures-Augment/Changer-mit-b1/Initial/Augment-Distill-3-0.001/vis_data/vis_image",
    
    # "/nas/datasets/lzy/RS-ChangeDetection/Figures-Augment/Changer-mit-b1/Initial/Distill-3-0.0001/vis_data/vis_image",
    # "/nas/datasets/lzy/RS-ChangeDetection/Figures-Augment/Changer-mit-b1/Initial/Augment-Distill-3-0.0001/vis_data/vis_image",
    
    # "/nas/datasets/lzy/RS-ChangeDetection/Figures-Augment/Changer-mit-b1/Initial/Initial/vis_data/vis_image",
    # "/nas/datasets/lzy/RS-ChangeDetection/Figures-Augment/Changer-mit-b1/Initial/Initial-Augment/vis_data/vis_image",
]

# 定义每个文件夹中图片数量估计值（不用于限制）
gt_folder = "/nas/datasets/lzy/RS-ChangeDetection/CGWX-Augment/val/label"

# 创建输出文件夹
os.makedirs(best_output_folder, exist_ok=True)

# 全局变量用于累计TP、FP、FN、TN
TP_cnt, FP_cnt, FN_cnt, TN_cnt = 0, 0, 0, 0

# 获取图片文件名列表

def get_image_filenames(folder_path):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or -1))

# 计算mIOU的函数
def calculate_mIOU(gt_image, result_image):
    global TP_cnt, FP_cnt, FN_cnt, TN_cnt
    gt_array = np.array(gt_image)
    result_array = np.array(result_image)

    gt_array = np.where(gt_array == 255, 1, 0)
    result_array = np.where(result_array == 255, 1, 0)

    TP = np.sum((gt_array == 1) & (result_array == 1))
    FP = np.sum((gt_array == 0) & (result_array == 1))
    FN = np.sum((gt_array == 1) & (result_array == 0))
    TN = np.sum((gt_array == 0) & (result_array == 0))

    TP_cnt += TP
    FP_cnt += FP
    FN_cnt += FN
    TN_cnt += TN

    if TP + FP + FN == 0 or TN + FP + FN == 0:
        return 0

    mIOU = 0.5 * TP / (TP + FP + FN) + 0.5 * TN / (TN + FP + FN)
    return mIOU

# 计算所有图片的平均mIOU
def calculate_average_mIOU(gt_folder, result_folder):
    global TP_cnt, FP_cnt, FN_cnt, TN_cnt
    TP_cnt, FP_cnt, FN_cnt, TN_cnt = 0, 0, 0, 0

    gt_images = get_image_filenames(gt_folder)
    result_images = get_image_filenames(result_folder)

    assert len(gt_images) == len(result_images), "GT文件夹和结果文件夹中的图片数量不一致"

    mIOU_list = []

    for gt_image_name, result_image_name in zip(gt_images, result_images):
        gt_image_path = os.path.join(gt_folder, gt_image_name)
        result_image_path = os.path.join(result_folder, result_image_name)

        gt_image = Image.open(gt_image_path).convert('L')
        result_image = Image.open(result_image_path).convert('L')

        mIOU = calculate_mIOU(gt_image, result_image)
        mIOU_list.append(mIOU)

    average_mIOU = np.mean(mIOU_list)
    return average_mIOU

# 遍历所有可能的组合
best_mIOU = 0
best_combination = []

temp_output_folder = "/nas/datasets/lzy/RS-ChangeDetection/Figures-Augment/tmp/temp_result_images"
os.makedirs(temp_output_folder, exist_ok=True)

for r in range(1, len(model_folders) + 1):
    for combination in itertools.combinations(model_folders, r):
        for filename in get_image_filenames(os.path.join(root_folder, model_folders[0])):
            images = []

            for model in combination:
                vis_image_folder = os.path.join(root_folder, model)
                image_path = os.path.join(vis_image_folder, filename)

                if os.path.exists(image_path):
                    image = Image.open(image_path).convert('L')
                    images.append(np.array(image))
                else:
                    print(f"Warning: {image_path} does not exist and will be skipped.")

            if images:
                mean_image = np.mean(np.stack(images, axis=0), axis=0)
                result_image = np.where(mean_image >= 127.5, 255, 0).astype(np.uint8)
                output_image_path = os.path.join(temp_output_folder, filename)
                Image.fromarray(result_image).save(output_image_path)

        average_mIOU = calculate_average_mIOU(gt_folder, temp_output_folder)
        print(f"Combination {combination} has mIOU {average_mIOU}")

        if average_mIOU > best_mIOU:
            best_mIOU = average_mIOU
            best_combination = combination

        for f in os.listdir(temp_output_folder):
            os.remove(os.path.join(temp_output_folder, f))

if best_combination:
    print(f"Best combination is {best_combination} with mIOU {best_mIOU}")
    for filename in get_image_filenames(os.path.join(root_folder, model_folders[0])):
        images = []
        for model in best_combination:
            vis_image_folder = os.path.join(root_folder, model)
            image_path = os.path.join(vis_image_folder, filename)
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('L')
                images.append(np.array(image))

        if images:
            mean_image = np.mean(np.stack(images, axis=0), axis=0)
            result_image = np.where(mean_image >= 127.5, 255, 0).astype(np.uint8)
            output_image_path = os.path.join(best_output_folder, filename)
            Image.fromarray(result_image).save(output_image_path)
else:
    print("No valid combination found.")
