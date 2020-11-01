import xml.etree.ElementTree as ET
from albumentations.core.composition import Compose
from torch.utils.data import Dataset
import glob
import cv2
import os
import numpy as np
import torch
from src.utils.constants import murad_classes_mod


class MuradPavementDataset(Dataset):
    def __init__(self,
                mode: str = 'train',
                base_path: str = 'output_train',
                transforms: Compose = None):
        """
        Args:
            dataframe: dataframe with image id and bboxes
            mode: train/val/test
            transforms: albumentations
        """
        self.mode = mode
        self.base_path = base_path
        self.data = self.generate_annotations_dict()
        self.image_ids = list(self.data.keys())
        self.transforms = transforms

    def read_content(self, xml_file: str):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        bounding_boxes = []
        labels = []
        filename = root.find('filename').text
        for boxes in root.iter('object'):
            ymin, xmin, ymax, xmax = None, None, None, None
            
            for box in boxes.findall("bndbox"):
                ymin = float(box.find("ymin").text)
                xmin = float(box.find("xmin").text)
                ymax = float(box.find("ymax").text)
                xmax = float(box.find("xmax").text)

                label = boxes.find("name").text
                label_id = murad_classes_mod[label]
                labels.append(label_id)
                list_with_single_boxes = [xmin, ymin, xmax, ymax]
                bounding_boxes.append(list_with_single_boxes)
        return filename, bounding_boxes, labels

    def generate_annotations_dict(self):
        data = {}
        for annot_file in glob.glob(self.base_path + '/' + '*.xml'):
            filename, bboxes, labels = self.read_content(annot_file)
            img_id = filename.replace(' ', '_').replace('.jpg', '').lower()
            filename = filename.replace(' ', '_').lower()
            data[img_id] = {
                'bboxes': bboxes,
                'labels': labels,
                'filename': os.path.join(self.base_path, filename)
            }
        return data

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = self.data[image_id]['filename']

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # normalization.
        # TO DO: refactor preprocessing
        # image /= 255.0

        # # test dataset must have some values so that transforms work.
        target = {'labels': torch.as_tensor([[0]], dtype=torch.float32),
                'boxes': torch.as_tensor([[0, 0, 0, 0]], dtype=torch.float32)}

        # for train and valid test create target dict.
        # if self.mode != 'test':

        boxes = self.data[image_id]['bboxes']
        boxes = np.array(boxes)
        labels = self.data[image_id]['labels']

        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        num_objs = len(labels)
        target['boxes'] = boxes
        target['labels'] = torch.from_numpy(np.array(labels))
        target['area'] = areas
        target['image_id'] = torch.tensor([idx])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int32)
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']
            
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            target['boxes'] = target['boxes'].type(torch.float32)
        return image, target, image_id

    def __len__(self) -> int:
        return len(self.image_ids)