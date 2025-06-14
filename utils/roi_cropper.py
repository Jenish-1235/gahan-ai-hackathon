import torch
import torch.nn.functional as F

class ROICropper:
    def __init__(self, crop_size=(224, 224), margin=0.1):
        self.crop_size = crop_size
        self.margin = margin
        
    def crop_around_objects(self, images, annotations):
        """Crop images around detected objects using normalized coordinates"""
        batch_crops = []
        
        for img, ann in zip(images, annotations):
            if not ann:  # No objects in this image
                # Return center crop
                batch_crops.append(self._center_crop(img))
                continue
                
            # Find bounding box that encompasses all objects
            all_bboxes = [obj['bbox'] for obj in ann]
            min_x = min([bbox[0] for bbox in all_bboxes])
            min_y = min([bbox[1] for bbox in all_bboxes])
            max_x = max([bbox[2] for bbox in all_bboxes])
            max_y = max([bbox[3] for bbox in all_bboxes])
            
            # Add margin
            width = max_x - min_x
            height = max_y - min_y
            min_x = max(0, min_x - width * self.margin)
            min_y = max(0, min_y - height * self.margin)
            max_x = min(img.shape[-1], max_x + width * self.margin)
            max_y = min(img.shape[-2], max_y + height * self.margin)
            
            # Crop and resize
            cropped = img[:, int(min_y):int(max_y), int(min_x):int(max_x)]
            resized = F.interpolate(
                cropped.unsqueeze(0), size=self.crop_size, mode='bilinear'
            ).squeeze(0)
            
            batch_crops.append(resized)
            
        return torch.stack(batch_crops)
    
    def _center_crop(self, img):
        """Center crop when no objects detected"""
        h, w = img.shape[-2:]
        crop_h, crop_w = self.crop_size
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        cropped = img[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
        return F.interpolate(cropped.unsqueeze(0), size=self.crop_size, mode='bilinear').squeeze(0)