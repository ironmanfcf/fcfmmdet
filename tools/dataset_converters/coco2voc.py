import os
import json
import shutil
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

def create_voc_annotation(file_basename, bboxes, labels, size, label_map):
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'JPGImages'

    filename = ET.SubElement(annotation, 'filename')
    filename.text = file_basename

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    size_node = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size_node, 'width')
    width.text = str(size[0])
    height = ET.SubElement(size_node, 'height')
    height.text = str(size[1])
    depth = ET.SubElement(size_node, 'depth')
    depth.text = str(size[2])

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    for bbox, label in zip(bboxes, labels):
        obj = ET.SubElement(annotation, 'object')
        name = ET.SubElement(obj, 'name')
        name.text = label_map[label]
        pose = ET.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(obj, 'truncated')
        truncated.text = '0'
        difficult = ET.SubElement(obj, 'difficult')
        difficult.text = '0'

        bndbox = ET.SubElement(obj, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(bbox[0]))
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(bbox[1]))
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(bbox[0] + bbox[2]))
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(bbox[1] + bbox[3]))

    return ET.ElementTree(annotation)

def process_image(image_info, annotation_list, categories, image_dir, voc_image_dir, voc_annotation_dir):
    file_name = image_info['file_name']
    image_path = os.path.join(image_dir, file_name)
    voc_image_path = os.path.join(voc_image_dir, file_name)

    # Copy image to voc_image_dir
    if not os.path.exists(voc_image_path):
        shutil.copy(image_path, voc_image_path)

    voc_annotation_path = os.path.join(voc_annotation_dir, os.path.splitext(file_name)[0] + '.xml')
    
    size = (image_info['width'], image_info['height'], 3)
    bboxes = [annotation['bbox'] for annotation in annotation_list]
    labels = [annotation['category_id'] for annotation in annotation_list]
    
    annotation_tree = create_voc_annotation(file_name, bboxes, labels, size, categories)
    annotation_tree.write(voc_annotation_path)

    return os.path.splitext(file_name)[0]

def convert_coco_to_voc(coco_json, image_dir, voc_image_dir, voc_annotation_dir, split_file):
    with open(coco_json, 'r') as f:
        coco = json.load(f)

    images = {image['id']: image for image in coco['images']}
    categories = {category['id']: category['name'] for category in coco['categories']}
    
    img_list = []
    
    annotations_by_image = {}
    for annotation in coco['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_image = {executor.submit(process_image, images[image_id], annotation_list, categories, image_dir, voc_image_dir, voc_annotation_dir): image_id for image_id, annotation_list in annotations_by_image.items()}

        for future in as_completed(future_to_image):
            img_list.append(future.result())

    with open(split_file, 'w') as f:
        for img in img_list:
            f.write(img + '\n')

def main(train_coco, val_coco, test_coco, train_image_dir, val_image_dir, test_image_dir):
    base_dir = '/opt/data/private/fcf/mmdetection/data/HazyDetdevkit'
    voc_image_dir = os.path.join(base_dir, 'JPGImages')
    voc_annotation_dir = os.path.join(base_dir, 'HazyDet', 'Bbox')
    voc_splits_dir = os.path.join(base_dir, 'Splits')

    os.makedirs(voc_image_dir, exist_ok=True)
    os.makedirs(voc_annotation_dir, exist_ok=True)
    os.makedirs(voc_splits_dir, exist_ok=True)
    
    convert_coco_to_voc(train_coco, train_image_dir, voc_image_dir, voc_annotation_dir, os.path.join(voc_splits_dir, 'train.txt'))
    convert_coco_to_voc(val_coco, val_image_dir, voc_image_dir, voc_annotation_dir, os.path.join(voc_splits_dir, 'val.txt'))
    convert_coco_to_voc(test_coco, test_image_dir, voc_image_dir, voc_annotation_dir, os.path.join(voc_splits_dir, 'test.txt'))

# Example usage
main('/opt/data/private/fcf/mmdetection/data/HazyDet-365k/train/train_coco.json', 
     '/opt/data/private/fcf/mmdetection/data/HazyDet-365k/val/val_coco.json',
     '/opt/data/private/fcf/mmdetection/data/HazyDet-365k/test/test_coco.json',
     '/opt/data/private/fcf/mmdetection/data/HazyDet-365k/train/hazy_images/', 
     '/opt/data/private/fcf/mmdetection/data/HazyDet-365k/val/hazy_images/', 
     '/opt/data/private/fcf/mmdetection/data/HazyDet-365k/test/hazy_images/')