import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import numpy as np

class CutInDataset(Dataset):
    def __init__(self, root_dir, config, mode='Train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., /content/distribution/).
            config (object): Configuration object with dataset and model parameters.
            mode (str): 'Train', 'Val', or 'Test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = os.path.join(root_dir, getattr(config, f"{mode.lower()}_dir"))
        self.config = config
        self.transform = transform
        self.sequence_length = config.sequence_length
        self.class_mapping = config.class_mapping
        self.image_extension = config.image_extension
        self.xml_extension = config.xml_extension
        self.annotation_folder_name = config.annotation_folder_name

        self.sequences = self._load_sequences()

    def _load_sequences(self):
        sequences = []
        if not os.path.exists(self.data_path):
            print(f"Warning: Data path not found: {self.data_path}")
            return sequences

        for rec_folder_name in sorted(os.listdir(self.data_path)):
            rec_path = os.path.join(self.data_path, rec_folder_name)
            if not os.path.isdir(rec_path):
                continue

            annotations_dir = os.path.join(rec_path, self.annotation_folder_name)
            if not os.path.isdir(annotations_dir):
                continue

            frame_files = sorted([
                f for f in os.listdir(annotations_dir)
                if f.lower().endswith(self.image_extension.lower())
            ])

            if not frame_files:
                continue

            # Group frames into sequences
            for i in range(0, len(frame_files), 1): # Overlapping sequences can be considered later if needed
                seq_frame_names = frame_files[i : i + self.sequence_length]
                if len(seq_frame_names) < 1: # Ensure at least one frame, collate_fn will handle padding
                    continue

                current_sequence_frames = []
                for frame_name in seq_frame_names:
                    base_name = frame_name.rsplit('.', 1)[0]
                    img_path = os.path.join(annotations_dir, frame_name)
                    xml_path = os.path.join(annotations_dir, base_name + self.xml_extension)

                    if os.path.exists(img_path) and os.path.exists(xml_path):
                        current_sequence_frames.append({'img_path': img_path, 'xml_path': xml_path})
                    else:
                        # If any file is missing, this frame in sequence is invalid.
                        # Depending on strategy, we could skip sequence or pad later.
                        # For now, if a frame is incomplete, we'll have a shorter list here.
                        # The collate_fn will be crucial for handling varied actual sequence lengths.
                        pass
                
                if current_sequence_frames: # Only add if we have at least one valid frame pair
                    sequences.append(current_sequence_frames)
        
        print(f"Loaded {len(sequences)} sequences for mode {self.mode}.")
        return sequences

    def _parse_xml(self, xml_path, img_width, img_height):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        labels = []
        boxes = []
        cutting_flags = []

        for obj_node in root.findall('object'):
            obj_name = obj_node.find('name').text
            if obj_name not in self.class_mapping:
                continue # Skip objects not in our defined classes

            label = self.class_mapping[obj_name]

            bndbox_node = obj_node.find('bndbox')
            xmin = float(bndbox_node.find('xmin').text)
            ymin = float(bndbox_node.find('ymin').text)
            xmax = float(bndbox_node.find('xmax').text)
            ymax = float(bndbox_node.find('ymax').text)

            # Normalize and convert to [cx, cy, w, h]
            # DETR expects normalized [0, 1] coordinates
            box_w = xmax - xmin
            box_h = ymax - ymin
            cx = (xmin + box_w / 2) / img_width
            cy = (ymin + box_h / 2) / img_height
            norm_w = box_w / img_width
            norm_h = box_h / img_height
            
            # Clamp normalized values to [0, 1] to avoid issues with slight out-of-bounds boxes
            cx = np.clip(cx, 0.0, 1.0)
            cy = np.clip(cy, 0.0, 1.0)
            norm_w = np.clip(norm_w, 0.0, 1.0)
            norm_h = np.clip(norm_h, 0.0, 1.0)

            is_cutting = False
            attributes_node = obj_node.find('attributes')
            if attributes_node is not None:
                for attr_node in attributes_node.findall('attribute'):
                    attr_name_node = attr_node.find('name')
                    attr_value_node = attr_node.find('value')
                    if attr_name_node is not None and attr_value_node is not None:
                        attr_name = attr_name_node.text
                        attr_value = attr_value_node.text.lower()
                        # Consider 'Cutting', 'LaneChanging', 'OverTaking' as cutting events
                        if attr_name in ['Cutting', 'LaneChanging', 'OverTaking'] and attr_value == 'true':
                            is_cutting = True
                            break
            
            labels.append(label)
            boxes.append([cx, cy, norm_w, norm_h])
            cutting_flags.append(is_cutting)

        return {
            'labels': torch.tensor(labels, dtype=torch.int64),
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'cutting_flags': torch.tensor(cutting_flags, dtype=torch.bool)
        }

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_frames_info = self.sequences[idx]
        
        images_sequence = []
        targets_sequence = []

        for frame_info in sequence_frames_info:
            img_path = frame_info['img_path']
            xml_path = frame_info['xml_path']

            try:
                image = Image.open(img_path).convert('RGB')
                original_width, original_height = image.size
            except FileNotFoundError:
                print(f"Warning: Image file not found {img_path}. Skipping frame.")
                # Create a dummy image if file not found, or handle differently
                image = Image.new('RGB', (self.config.image_size, self.config.image_size), color='grey')
                original_width, original_height = image.size


            if self.transform:
                # Pass original_width, original_height if needed by transform,
                # but typical ToTensor, Resize, Normalize don't need them.
                # For custom transforms that might do aspect-ratio aware padding, they might be useful.
                # For now, assume standard transforms.
                processed_image = self.transform(image)
            else:
                # Basic ToTensor if no transform provided
                processed_image = F.to_tensor(image)


            try:
                # Get image dimensions from XML for normalization if not using original_width/height from PIL
                # For simplicity, using PIL's reported size before transform.
                # If XML size is more reliable, parse it here.
                # The provided XML has <size><width>1920</width><height>1080</height></size>
                # We'll use original_width, original_height from PIL for now.
                target = self._parse_xml(xml_path, original_width, original_height)
            except FileNotFoundError:
                print(f"Warning: XML file not found {xml_path}. Creating empty target.")
                target = {
                    'labels': torch.empty(0, dtype=torch.int64),
                    'boxes': torch.empty(0, 4, dtype=torch.float32),
                    'cutting_flags': torch.empty(0, dtype=torch.bool)
                }
            except ET.ParseError:
                print(f"Warning: XML file {xml_path} is corrupted. Creating empty target.")
                target = {
                    'labels': torch.empty(0, dtype=torch.int64),
                    'boxes': torch.empty(0, 4, dtype=torch.float32),
                    'cutting_flags': torch.empty(0, dtype=torch.bool)
                }


            images_sequence.append(processed_image)
            targets_sequence.append(target)
        
        # At this point, images_sequence is a list of (C,H,W) tensors
        # and targets_sequence is a list of dictionaries.
        # The collate_fn will stack images and handle padding for targets.
        # If images_sequence is empty due to all frames failing, collate_fn needs to handle it.
        if not images_sequence: # Should not happen if _load_sequences filters properly
            # Create a dummy sequence if all frames failed
            dummy_img = torch.zeros((3, self.config.image_size, self.config.image_size))
            images_sequence = [dummy_img] * self.sequence_length # Pad to expected length
            dummy_target = {
                'labels': torch.empty(0, dtype=torch.int64),
                'boxes': torch.empty(0, 4, dtype=torch.float32),
                'cutting_flags': torch.empty(0, dtype=torch.bool)
            }
            targets_sequence = [dummy_target] * self.sequence_length


        return images_sequence, targets_sequence