import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms

class CutInSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=5):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.sequences = self._load_sequences()
        
    def _load_sequences(self):
        sequences = []
        for recording in os.listdir(self.root_dir):
            recording_path = os.path.join(self.root_dir, recording)
            if os.path.isdir(recording_path):
                annotations_path = os.path.join(recording_path, "Annotations")
                if os.path.exists(annotations_path):
                    # Get all frame files
                    frames = []
                    for file in os.listdir(annotations_path):
                        if file.endswith('.JPG') or file.endswith('.jpg') or file.endswith('.png'):
                            frame_num = int(file.split('_')[-1].split('.')[0])
                            frames.append((frame_num, file))
                    
                    # Sort by frame number
                    frames.sort(key=lambda x: x[0])
                    
                    # Create sequences
                    for i in range(0, len(frames), self.sequence_length):
                        seq_frames = frames[i:i+self.sequence_length]
                        if len(seq_frames) >= 1:  # At least 1 frame
                            sequences.append((recording_path, seq_frames))
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        recording_path, frame_list = self.sequences[idx]
        annotations_path = os.path.join(recording_path, "Annotations")
        
        images = []
        annotations = []
        
        for frame_num, frame_file in frame_list:
            # Load image
            img_path = os.path.join(annotations_path, frame_file)
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Create dummy image if loading fails
                if self.transform:
                    dummy_img = torch.zeros(3, 224, 224)
                else:
                    dummy_img = Image.new('RGB', (224, 224))
                images.append(dummy_img)
            
            # Load annotation
            xml_file = frame_file.replace('.JPG', '.xml').replace('.jpg', '.xml').replace('.png', '.xml')
            xml_path = os.path.join(annotations_path, xml_file)
            
            frame_annotations = []
            if os.path.exists(xml_path):
                try:
                    frame_annotations = self._parse_xml(xml_path)
                except Exception as e:
                    print(f"Error parsing XML {xml_path}: {e}")
            
            annotations.append(frame_annotations)
        
        # Ensure we have at least one frame
        if not images:
            dummy_img = torch.zeros(3, 224, 224) if self.transform else Image.new('RGB', (224, 224))
            images = [dummy_img]
            annotations = [[]]
        
        # Convert to tensor if needed
        if len(images) == 1:
            images = images[0].unsqueeze(0)  # Add time dimension
        else:
            images = torch.stack(images)
        
        return images, annotations
    
    def _parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotations = []
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            
            bbox_elem = obj.find('bndbox')
            if bbox_elem is not None:
                xmin = float(bbox_elem.find('xmin').text)
                ymin = float(bbox_elem.find('ymin').text)
                xmax = float(bbox_elem.find('xmax').text)
                ymax = float(bbox_elem.find('ymax').text)
                
                # Parse attributes for cutting behavior
                cutting = False
                attributes = obj.find('attributes')
                if attributes is not None:
                    for attr in attributes.findall('attribute'):
                        attr_name = attr.find('name')
                        attr_value = attr.find('value')
                        if attr_name is not None and attr_value is not None:
                            if attr_name.text in ['Cutting', 'LaneChanging', 'OverTaking']:
                                if attr_value.text.lower() == 'true':
                                    cutting = True
                
                annotations.append({
                    'label': name,
                    'bbox': [xmin, ymin, xmax, ymax],
                    'cutting': cutting
                })
        
        return annotations
