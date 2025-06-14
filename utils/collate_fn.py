import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    """Custom collate function to handle variable sequence lengths"""
    images_list = []
    annotations_list = []
    
    max_seq_len = 0
    
    # Find maximum sequence length in batch
    for images, annotations in batch:
        if isinstance(images, torch.Tensor):
            seq_len = images.shape[0] if len(images.shape) == 4 else 1
        else:
            seq_len = len(images)
        max_seq_len = max(max_seq_len, seq_len)
    
    # Pad sequences to max length
    for images, annotations in batch:
        # Handle images
        if isinstance(images, torch.Tensor):
            if len(images.shape) == 4:  # (T, C, H, W)
                seq_len = images.shape[0]
                if seq_len < max_seq_len:
                    # Pad by repeating last frame
                    padding_needed = max_seq_len - seq_len
                    last_frame = images[-1:].repeat(padding_needed, 1, 1, 1)
                    images = torch.cat([images, last_frame], dim=0)
            else:  # Single image (C, H, W)
                images = images.unsqueeze(0).repeat(max_seq_len, 1, 1, 1)
        else:
            # Convert list to tensor if needed
            if len(images) < max_seq_len:
                # Pad with last image
                last_img = images[-1] if images else torch.zeros(3, 224, 224)
                images.extend([last_img] * (max_seq_len - len(images)))
            images = torch.stack(images[:max_seq_len])
        
        images_list.append(images)
        
        # Handle annotations - pad with empty annotations
        if len(annotations) < max_seq_len:
            last_ann = annotations[-1] if annotations else []
            annotations.extend([last_ann] * (max_seq_len - len(annotations)))
        annotations_list.append(annotations[:max_seq_len])
    
    # Stack all images
    try:
        images_batch = torch.stack(images_list)  # (B, T, C, H, W)
    except Exception as e:
        print(f"Error stacking images: {e}")
        # Fallback: ensure all have same shape
        target_shape = (max_seq_len, 3, 224, 224)
        fixed_images = []
        for img in images_list:
            if img.shape != target_shape:
                img = F.interpolate(img, size=(224, 224), mode='bilinear')
                if img.shape[0] != max_seq_len:
                    img = img[:max_seq_len] if img.shape[0] > max_seq_len else torch.cat([img, img[-1:].repeat(max_seq_len - img.shape[0], 1, 1, 1)])
            fixed_images.append(img)
        images_batch = torch.stack(fixed_images)
    
    return images_batch, annotations_list