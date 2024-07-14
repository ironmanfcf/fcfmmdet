import json
from collections import defaultdict


def count_categories_in_coco_json(json_file):
    # 创建一个默认字典来记录每个类别的数量
    category_counts = defaultdict(int)

    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)

        # 假设'categories'和'annotations'字段都存在
    # 并且每个'annotation'都有一个'category_id'字段
    categories = {cat['id']: cat['name'] for cat in data['categories']}

    # 遍历所有标注
    for annotation in data['annotations']:
        category_id = annotation['category_id']
        # 使用类别ID作为键，增加对应类别的计数
        category_counts[categories[category_id]] += 1

        # 返回类别及其数量
    return category_counts


# 使用函数并打印结果
json_file = 'D:\\ALLHaveMlNoTc\\ALL\\all.json'  # 替换为你的COCO标注JSON文件路径
category_counts = count_categories_in_coco_json(json_file)
for category, count in category_counts.items():
    print(f"{category}: {count}")