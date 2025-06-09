import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

train_path = r"C:\Users\neelr\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray\val"

norm_dir = train_path + r"\NORMAL" 
pnem_dir = train_path + r"\PNEUMONIA"

list_norm = os.listdir(norm_dir)
list_pnem = os.listdir(pnem_dir)

norm_img_dir = []
pnem_img_dir = []

for x in range(0, len(norm_dir)):
    norm_img_dir.append(os.path.join(norm_dir, list_norm[x]))
    pnem_img_dir.append(os.path.join(pnem_dir, list_pnem[x]))

processed_imgs = []
labels = [] # 0 stands for normal image; 1 stands for pnem image

for img in norm_img_dir:
    processed_imgs.append((cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (128, 128)))/255)
    labels.append(0)

for img in pnem_img_dir:
    processed_imgs.append((cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (128, 128)))/255)
    labels.append(1)

A = np.array(processed_imgs)
B = np.array(labels)

print(f"Shape of X (processed images): {X.shape}")
print(f"Shape of Y (labels): {Y.shape}")
print(f"Data type of X: {X.dtype}")
print(f"First few labels: {Y[:10]}")