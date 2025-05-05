import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

class ShiftedImageDataset(Dataset):
    def __init__(self, base_dataset, image_size=(32, 32)):
        self.base_dataset = base_dataset
        self.image_size = image_size
        
    def __len__(self):
        return len(self.base_dataset)
    
    @staticmethod
    def shift_batch(x, shift_h, shift_w):
        """Apply specified shift to all images in batch"""
        return torch.roll(x, shifts=(shift_h, shift_w), dims=(-2, -1))
    
    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        image = image + torch.randn_like(image) * 0.06
        return image, label

    @staticmethod
    def collate_fn(batch):
        # First use default collate to stack the images/labels
        images, labels = zip(*batch)
        images = default_collate(images)
        labels = default_collate(labels)
        
        # Generate one random shift for the whole batch
        H, W = images.shape[-2:]
        shift_h = torch.randint(0, H, (1,)).item()
        shift_w = torch.randint(0, W, (1,)).item()
        
        # Apply same shift to all images
        shifted_images = ShiftedImageDataset.shift_batch(images, shift_h, shift_w)
        
        return images, shifted_images, labels

