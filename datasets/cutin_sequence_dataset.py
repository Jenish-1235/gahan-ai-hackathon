import os
import xml.etree.ElementTree as ET
from PIL import Image
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CutInSequenceDataset(Dataset):
    def __init__(self, root_dir: str, sequence_length: int = 5, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.samples = self._collect_sequences()

    def _collect_sequences(self) -> List[Tuple[str, str, List[str]]]:
        sequences = []
        print("[INFO] Starting to collect sequences...")

        for recording in sorted(os.listdir(self.root_dir)):
            anno_dir = os.path.join(self.root_dir, recording, 'Annotations')
            if not os.path.isdir(anno_dir):
                continue

            print(f"[INFO] Processing recording folder: {recording}")
            
            frames = sorted(
                [f for f in os.listdir(anno_dir) if f.endswith('.xml')],
                key=lambda x: int(x.replace("frame_", "").replace(".xml", ""))
            )

            for i in range(len(frames) - self.sequence_length + 1):
                frame_seq = frames[i:i + self.sequence_length]
                sequences.append((recording, anno_dir, frame_seq))

            print(f"[INFO] {len(frames)} frames found. {len(frames) - self.sequence_length + 1} sequences added.")

        print(f"[INFO] Total recordings processed: {len(os.listdir(self.root_dir))}")
        print(f"[INFO] Total sequences collected: {len(sequences)}")
        return sequences


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        recording, anno_dir, frame_seq = self.samples[idx]
        images = []
        annotations = []

        for xml_file in frame_seq:
            frame_id = xml_file.replace('.xml', '')
            img_path = os.path.join(anno_dir, f"{frame_id}.JPG")
            xml_path = os.path.join(anno_dir, xml_file)

            max_attempts = len(self.samples)
            attempts = 0

            while attempts < max_attempts:
                try:
                    recording, anno_dir, frame_seq = self.samples[idx]
                    images, annotations = [], []

                    for xml_file in frame_seq:
                        frame_id = xml_file.replace('.xml', '')
                        img_path = os.path.join(anno_dir, f"{frame_id}.JPG")
                        xml_path = os.path.join(anno_dir, xml_file)

                        image = Image.open(img_path).convert('RGB')
                        image = self.transform(image)
                        ann = self._parse_xml(xml_path)

                        images.append(image)
                        annotations.append(ann)

                    return torch.stack(images), annotations

                except (FileNotFoundError, ET.ParseError, OSError) as e:
                    print(f"[WARN] Skipping corrupted sample idx={idx}: {e}")
                    idx = (idx + 1) % len(self.samples)
                    attempts += 1

            raise RuntimeError("All samples failed in __getitem__")


            images.append(image)
            annotations.append(ann)

        return torch.stack(images), annotations

    def _parse_xml(self, xml_path: str) -> List[Dict]:
        objects = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            cutting = False
            attributes = obj.find('attributes')
            if attributes is not None:
                for attr in attributes.findall('attribute'):
                    name_node = attr.find('name')
                    value_node = attr.find('value')
                    if name_node is None or value_node is None:
                        continue
                    if name_node.text == 'Cutting':
                        cutting = value_node.text.lower() == 'true'

            objects.append({
                'label': name,
                'bbox': [xmin, ymin, xmax, ymax],
                'cutting': cutting
            })
        return objects
