import os
import xml.etree.ElementTree as ET
from PIL import Image
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class FastCutInDataset(Dataset):
    """Ultra-fast dataset loader with caching and minimal preprocessing"""
    
    def __init__(self, root_dir: str, sequence_length: int = 5, transform=None, 
                 cache_dir=None, max_samples=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.max_samples = max_samples
        
        # Simple, fast transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Cache directory for preprocessed data
        self.cache_dir = cache_dir or f"{root_dir}_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load or create sample list
        self.samples = self._load_or_create_samples()
        
        # Limit samples if specified
        if max_samples and len(self.samples) > max_samples:
            self.samples = random.sample(self.samples, max_samples)
            print(f"[INFO] Limited to {max_samples} samples for faster training")

    def _load_or_create_samples(self):
        """Load cached samples or create new ones"""
        cache_file = os.path.join(self.cache_dir, 'samples_cache.pkl')
        
        if os.path.exists(cache_file):
            print("[INFO] Loading cached samples...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("[INFO] Creating sample cache...")
        samples = self._collect_sequences_fast()
        
        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(samples, f)
        
        return samples

    def _collect_sequences_fast(self):
        """Fast sequence collection with minimal processing"""
        sequences = []
        
        recordings = [d for d in os.listdir(self.root_dir) 
                     if os.path.isdir(os.path.join(self.root_dir, d))]
        
        print(f"[INFO] Processing {len(recordings)} recordings...")
        
        for recording in recordings[:20]:  # Limit recordings for speed
            anno_dir = os.path.join(self.root_dir, recording, 'Annotations')
            if not os.path.isdir(anno_dir):
                continue
            
            # Get frame files
            frames = [f for f in os.listdir(anno_dir) if f.endswith('.xml')]
            frames.sort(key=lambda x: int(x.replace("frame_", "").replace(".xml", "")))
            
            # Create sequences
            for i in range(0, len(frames) - self.sequence_length + 1, 2):  # Skip every other
                frame_seq = frames[i:i + self.sequence_length]
                sequences.append((recording, anno_dir, frame_seq))
        
        print(f"[INFO] Created {len(sequences)} sequences")
        return sequences

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Fast item retrieval with minimal error handling"""
        try:
            recording, anno_dir, frame_seq = self.samples[idx]
            images = []
            annotations = []

            for xml_file in frame_seq:
                frame_id = xml_file.replace('.xml', '')
                img_path = os.path.join(anno_dir, f"{frame_id}.JPG")
                xml_path = os.path.join(anno_dir, xml_file)

                # Fast image loading
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
                
                # Fast annotation parsing
                ann = self._parse_xml_fast(xml_path)
                
                images.append(image)
                annotations.append(ann)

            return torch.stack(images), annotations

        except Exception as e:
            # Return dummy data on error
            dummy_images = torch.zeros(self.sequence_length, 3, 224, 224)
            dummy_annotations = [[] for _ in range(self.sequence_length)]
            return dummy_images, dummy_annotations

    def _parse_xml_fast(self, xml_path: str) -> List[Dict]:
        """Ultra-fast XML parsing with minimal validation"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = []
            
            for obj in root.findall('object')[:10]:  # Limit objects per frame
                name_elem = obj.find('name')
                bbox_elem = obj.find('bndbox')
                
                if name_elem is None or bbox_elem is None:
                    continue
                
                name = name_elem.text
                
                # Fast bbox extraction
                xmin = float(bbox_elem.find('xmin').text)
                ymin = float(bbox_elem.find('ymin').text)
                xmax = float(bbox_elem.find('xmax').text)
                ymax = float(bbox_elem.find('ymax').text)
                
                # Normalize (assume 1920x1080)
                xmin_norm = xmin / 1920.0
                ymin_norm = ymin / 1080.0
                xmax_norm = xmax / 1920.0
                ymax_norm = ymax / 1080.0
                
                # Fast cutting detection
                cutting = False
                attributes = obj.find('attributes')
                if attributes is not None:
                    for attr in attributes.findall('attribute'):
                        name_node = attr.find('name')
                        value_node = attr.find('value')
                        if (name_node is not None and value_node is not None and 
                            name_node.text == 'Cutting'):
                            cutting = value_node.text.lower() == 'true'
                            break
                
                objects.append({
                    'label': name,
                    'bbox': [xmin, ymin, xmax, ymax],
                    'bbox_norm': [xmin_norm, ymin_norm, xmax_norm, ymax_norm],
                    'cutting': cutting,
                    'track_id': 0
                })
            
            return objects
            
        except Exception:
            return []

# Optimized collate function
def fast_collate_fn(batch):
    """Ultra-fast collate function"""
    images, annotations = zip(*batch)
    images = torch.stack(images)
    
    targets = []
    class_mapping = {'Car': 1, 'MotorBike': 2, 'EgoVehicle': 3}
    
    for frame_annotations in annotations:
        all_objects = []
        for frame_ann in frame_annotations:
            all_objects.extend(frame_ann[:20])  # Limit objects
        
        if not all_objects:
            targets.append({
                'labels': torch.tensor([], dtype=torch.long),
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'cutin': torch.tensor([], dtype=torch.float32)
            })
            continue
        
        # Vectorized processing
        labels = torch.tensor([class_mapping.get(obj['label'], 0) for obj in all_objects], 
                            dtype=torch.long)
        boxes = torch.tensor([obj['bbox_norm'] for obj in all_objects], 
                           dtype=torch.float32)
        cutin = torch.tensor([1.0 if obj['cutting'] else 0.0 for obj in all_objects], 
                           dtype=torch.float32)
        
        targets.append({
            'labels': labels,
            'boxes': boxes,
            'cutin': cutin
        })
    
    return images, targets 