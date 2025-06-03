import numpy as np
import xgboost as xgb
import cv2
import os

# Set paths
model_path = '/nas/datasets/lzy/RS-ChangeDetection/Figures/Boost/pixelwise/pixelwise.model'
gt_folder = '/nas/datasets/lzy/RS-ChangeDetection/CGWX/test/label'
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

output_folder = '/nas/datasets/lzy/RS-ChangeDetection/Figures/TEST/Boost'
# Load the model
bst = xgb.Booster()
bst.load_model(model_path)
print('Model loaded successfully.')
for image_index in range(1,1000+1):

    # Initialize a list to store model predictions
    all_model_predictions = []

    # Prepare feature matrix for the specific image
    features = np.zeros((512 * 512, len(model_folders)))  # Initialize feature array

    for model_idx, model_folder in enumerate(model_folders):
        model_path = os.path.join(model_folder, f'image_{image_index}.png')
        model_image = cv2.imread(model_path, cv2.IMREAD_GRAYSCALE)

        if model_image is not None:
            # Ensure the model image is the correct size
            if model_image.shape != (512, 512):
                print(f'Image {model_path} is not of shape (512, 512)')
                continue

            # Fill the features array with flattened model images
            features[:, model_idx] = model_image.flatten()  # Use the column for each model

    # Convert features to a DMatrix
    dtest = xgb.DMatrix(features)

    # Perform prediction
    preds_full = bst.predict(dtest)
    preds_binary_full = (preds_full > 0.5).astype(int)  # Binarize predictions

    # Reshape binary predictions into an image
    final_prediction = preds_binary_full.reshape(512, 512) * 255  # Convert to 0/255 format

    # Save the final prediction as image_1.png
    output_path = os.path.join(output_folder, f'image_{image_index}.png')
    cv2.imwrite(output_path, final_prediction)
    print(f'Final prediction saved as {output_path}.')
