import json
import matplotlib.pyplot as plt
import numpy as np


def load_coco_data(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def analyze_image_resolutions(data):
    image_dimensions = {}
    for image in data['images']:
        image_dimensions[image['id']] = (image['width'], image['height'])
    
    resolution_count = {}
    for dimensions in image_dimensions.values():
        if dimensions in resolution_count:
            resolution_count[dimensions] += 1
        else:
            resolution_count[dimensions] = 1
    
    total_images = sum(resolution_count.values())

    fig, ax = plt.subplots(figsize=(12, 9))
    for dimensions, count in resolution_count.items():
        ax.scatter(dimensions[0], dimensions[1], s=count*5)
        # ax.text(dimensions[0], dimensions[1] + np.sqrt(count*10), f"{count}", fontsize=10, ha='center', va='bottom')
    
    ax.set_xlabel('Width', fontsize=16)
    ax.set_ylabel('Height', fontsize=16)
    ax.set_title('Real Set Image Resolution Analysis', fontsize=16, fontweight='bold')
    plt.figtext(0.99, 0.01, f'Total images: {total_images}', horizontalalignment='right', fontsize=12, fontweight='bold')

    # 设置刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=14)

    # 移除上边和右边的边线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.show()

def analyze_object_sizes(data, max_samples=1000):
    object_dimensions = {}
    sample_count = 0
    for annotation in data['annotations']:
        if sample_count >= max_samples:
            break
        bbox = annotation['bbox']
        width = bbox[2]
        height = bbox[3]
        dimensions = (width, height)
        if dimensions in object_dimensions:
            object_dimensions[dimensions] += 1
        else:
            object_dimensions[dimensions] = 1
        sample_count += 1
    
    fig, ax = plt.subplots(figsize=(12, 9))
    for dimensions, count in object_dimensions.items():
        ax.scatter(dimensions[0], dimensions[1], s=count*3)
        # ax.text(dimensions[0], dimensions[1] + np.sqrt(count*10), f"{count}", fontsize=10, ha='center', va='bottom') 

    ax.set_xlabel('Object Width', fontsize=16)
    ax.set_ylabel('Object Height', fontsize=16)
    ax.set_title('Sampled Object Size Distribution in Real Annotations', fontsize=16, fontweight='bold')
    plt.figtext(0.99, 0.01, f'Sampled objects: {max_samples}', horizontalalignment='right', fontsize=12, fontweight='bold')

    # 设置刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=14)

    # 移除上边和右边的边线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.show()

def analyze_category_distribution(data):
    # Count occurrences of each category
    category_counts = {}
    for annotation in data['annotations']:
        category_id = annotation['category_id']
        if category_id in category_counts:
            category_counts[category_id] += 1
        else:
            category_counts[category_id] = 1

    # Map category IDs to their names
    category_names = {}
    for category in data['categories']:
        category_names[category['id']] = category['name']

    # Prepare data for the pie chart
    labels = [category_names[id] for id in category_counts.keys()]
    sizes = [category_counts[id] for id in category_counts.keys()]
    total = sum(sizes)

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 12))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}%\n({int(p*total/100)})',
                                      startangle=90, pctdistance=0.85, textprops={'fontsize': 16})

    # Draw a circle at the center of pie to make it look like a donut
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')  
    plt.title('Distribution of Test Object Categories', fontsize=16, fontweight='bold')
    
    # 设置刻度字体大小（虽然饼图没有轴刻度，但我们保持代码一致性）
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.show()

# 加载数据
data = load_coco_data('/home/feng/MMCV/mmdetection-3.2/data/UAV-HAZY/Synthetic_Haze/test/test_coco.json')

# 进行分析
analyze_image_resolutions(data)
analyze_object_sizes(data, max_samples=1000)  # Analyze only 1000 samples for performance reasons
analyze_category_distribution(data)  # New function to analyze category distribution
