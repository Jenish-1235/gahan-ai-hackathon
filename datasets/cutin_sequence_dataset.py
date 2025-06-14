import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np


class CutInSequenceDataset(Dataset):
    def __init__(self, root_dir: str, sequence_length: int = 5, transform=None, 
                 roi_filter=True, balance_classes=True, augment=True):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.roi_filter = roi_filter
        self.balance_classes = balance_classes
        self.augment = augment
        
        # Define ROI (Region of Interest) - focus area for lane cutting
        # Bottom center area where lane cutting is most relevant
        self.roi = [480, 540, 1440, 1080]  # [xmin, ymin, xmax, ymax] for 1920x1080
        
        # Enhanced transforms with augmentation
        if transform is None:
            base_transforms = [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])  # ImageNet normalization
            ]
            
            if augment:
                aug_transforms = [
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomHorizontalFlip(p=0.3),  # Careful with driving scenes
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ]
                self.transform = transforms.Compose(aug_transforms + base_transforms)
            else:
                self.transform = transforms.Compose(base_transforms)
        else:
            self.transform = transform
            
        self.samples = self._collect_sequences()
        
        # Class balancing: oversample positive cutting sequences
        if balance_classes:
            self.samples = self._balance_classes()

    def _is_in_roi(self, bbox):
        """Check if bounding box intersects with ROI"""
        xmin, ymin, xmax, ymax = bbox
        roi_xmin, roi_ymin, roi_xmax, roi_ymax = self.roi
        
        return not (xmax < roi_xmin or xmin > roi_xmax or 
                   ymax < roi_ymin or ymin > roi_ymax)

    def _collect_sequences(self) -> List[Tuple[str, str, List[str]]]:
        sequences = []
        cutting_sequences = []
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
                sequence_tuple = (recording, anno_dir, frame_seq)
                
                # Check if this sequence contains cutting behavior
                has_cutting = self._sequence_has_cutting(anno_dir, frame_seq)
                
                sequences.append(sequence_tuple)
                if has_cutting:
                    cutting_sequences.append(sequence_tuple)

            print(f"[INFO] {len(frames)} frames found. {len(frames) - self.sequence_length + 1} sequences added.")

        print(f"[INFO] Total sequences collected: {len(sequences)}")
        print(f"[INFO] Cutting sequences found: {len(cutting_sequences)}")
        return sequences

    def _sequence_has_cutting(self, anno_dir, frame_seq):
        """Check if any frame in sequence has cutting behavior in ROI"""
        for xml_file in frame_seq:
            xml_path = os.path.join(anno_dir, xml_file)
            try:
                objects = self._parse_xml(xml_path)
                for obj in objects:
                    if obj['cutting'] and (not self.roi_filter or self._is_in_roi(obj['bbox'])):
                        return True
            except:
                continue
        return False

    def _balance_classes(self):
        """Oversample sequences with cutting behavior"""
        cutting_sequences = []
        normal_sequences = []
        
        for seq in self.samples:
            recording, anno_dir, frame_seq = seq
            if self._sequence_has_cutting(anno_dir, frame_seq):
                cutting_sequences.append(seq)
            else:
                normal_sequences.append(seq)
        
        # Oversample cutting sequences to balance classes
        if len(cutting_sequences) > 0:
            oversample_factor = min(10, len(normal_sequences) // len(cutting_sequences))
            cutting_sequences = cutting_sequences * oversample_factor
        
        balanced_samples = normal_sequences + cutting_sequences
        random.shuffle(balanced_samples)
        
        print(f"[INFO] Balanced dataset: {len(normal_sequences)} normal + {len(cutting_sequences)} cutting sequences")
        return balanced_samples

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        max_attempts = 10
        attempts = 0

        while attempts < max_attempts:
            try:
                recording, anno_dir, frame_seq = self.samples[idx]
                images = []
                annotations = []

                for xml_file in frame_seq:
                    frame_id = xml_file.replace('.xml', '')
                    img_path = os.path.join(anno_dir, f"{frame_id}.JPG")
                    xml_path = os.path.join(anno_dir, xml_file)

                    # Load and transform image
                    image = Image.open(img_path).convert('RGB')
                    image = self.transform(image)
                    
                    # Parse annotations with ROI filtering
                    ann = self._parse_xml(xml_path)
                    if self.roi_filter:
                        ann = [obj for obj in ann if self._is_in_roi(obj['bbox'])]

                    images.append(image)
                    annotations.append(ann)

                return torch.stack(images), annotations

            except (FileNotFoundError, ET.ParseError, OSError, Exception) as e:
                print(f"[WARN] Skipping corrupted sample idx={idx}: {e}")
                idx = (idx + 1) % len(self.samples)
                attempts += 1

        # If all attempts fail, return a dummy sample
        print(f"[ERROR] All attempts failed for idx={idx}, returning dummy sample")
        dummy_images = torch.zeros(self.sequence_length, 3, 224, 224)
        dummy_annotations = [[] for _ in range(self.sequence_length)]
        return dummy_images, dummy_annotations

    def _parse_xml(self, xml_path: str) -> List[Dict]:
        objects = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions for normalization
        size_elem = root.find('size')
        img_width = float(size_elem.find('width').text) if size_elem is not None else 1920
        img_height = float(size_elem.find('height').text) if size_elem is not None else 1080
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Normalize bounding box coordinates
            xmin_norm = xmin / img_width
            ymin_norm = ymin / img_height
            xmax_norm = xmax / img_width
            ymax_norm = ymax / img_height

            cutting = False
            track_id = 0
            attributes = obj.find('attributes')
            if attributes is not None:
                for attr in attributes.findall('attribute'):
                    name_node = attr.find('name')
                    value_node = attr.find('value')
                    if name_node is None or value_node is None:
                        continue
                    if name_node.text == 'Cutting':
                        cutting = value_node.text.lower() == 'true'
                    elif name_node.text == 'track_id':
                        try:
                            track_id = int(value_node.text)
                        except:
                            track_id = 0

            objects.append({
                'label': name,
                'bbox': [xmin, ymin, xmax, ymax],  # Original coordinates
                'bbox_norm': [xmin_norm, ymin_norm, xmax_norm, ymax_norm],  # Normalized
                'cutting': cutting,
                'track_id': track_id
            })
        return objects
