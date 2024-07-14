"""
在使用上面的代码时，请确保以下几点：

annotation_file 是包含COCO标注的JSON文件的路径。
depth_image_dir 是包含相对深度图像的文件夹路径，此处假设深度图像与标注图像的文件名相同但扩展名为 .png。
annotations.json 文件符合COCO标注格式。
深度图像存储在32位浮点格式的 PNG 文件中。
该代码将计算每个目标物体的中心位置的深度值，并将这些深度值作为数据点一个直方图绘制，在0到1的相对深度范围内分成多个等级来统计每个等级中的目标数目
"""


import json
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

def read_coco_annotations(json_file):
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    return annotations

def calculate_center(bbox):
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    return int(center_x), int(center_y)
    
def get_depth_from_image(depth_image, center_point):
    x, y = center_point
    return depth_image[y, x]

def plot_histogram(depth_values, bins, title, x_title, y_title):
    plt.hist(depth_values, bins=bins, edgecolor='black')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.show()
    
def main(annotation_file, depth_image_dir, sample_size=100):
    annotations = read_coco_annotations(annotation_file)
    
    # Get a list of image_info
    image_info_list = annotations['images']
    
    # Randomly sample image_info entries
    sampled_images = random.sample(image_info_list, min(sample_size, len(image_info_list)))
    
    depth_values = []
    
    # Process each sampled image
    for image_info in sampled_images:
        image_id = image_info['id']
        image_filename = os.path.basename(image_info['file_name'])
        depth_image_path = os.path.join(depth_image_dir, image_filename.replace('.jpg', '.png'))
        
        # Open depth image
        depth_image = np.array(Image.open(depth_image_path), dtype=np.float32)
        
        for ann in annotations['annotations']:
            if ann['image_id'] == image_id:
                bbox = ann['bbox']
                center = calculate_center(bbox)
                depth_value = get_depth_from_image(depth_image, center)
                depth_values.append(depth_value)

    # Plot histogram
    plot_histogram(depth_values, bins=np.linspace(0, 1, 21), title='Depth Histogram',
                   x_title='Relative Depth', y_title='Number of Objects')
    
if __name__ == "__main__":
    annotation_file = "/opt/data/private/fcf/mmdetection/data/HazyDet-365k/train/train_coco.json"
    depth_image_dir = "path/to/your/depth/images"
    main(annotation_file, depth_image_dir, sample_size=100)
    
