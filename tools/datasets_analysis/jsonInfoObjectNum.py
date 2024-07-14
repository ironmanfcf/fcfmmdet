import json


def analyze_coco_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # 获取图像数量
    num_images = len(annotations['images'])

    # 获取目标数量
    num_objects = len(annotations['annotations'])

    # 计算每张图片的目标数量
    if num_images != 0:
        avg_objects_per_image = num_objects / num_images
    else:
        avg_objects_per_image = 0
    name = 'test'
    print(f"Number of {name} images: {num_images}")
    print(f"Number of {name} objects: {num_objects}")
    print(f"Average objects per image: {avg_objects_per_image:.2f}")


# 使用示例
annotation_file = 'D:\\ALLHaveMlNoTc\\ALL\\test.json'
analyze_coco_annotations(annotation_file)