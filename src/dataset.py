import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class IncisionDataset(Dataset):
    """
    Custom Dataset for loading surgical incision images and 
    their corresponding normalized keypoint coordinates.
    """
    def __init__(self, image_files, label_files, resize_size=240):
        self.image_files = image_files
        self.label_files = label_files
        self.resize_size = resize_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #EfficientNet data distribution
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.image_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.resize_size, self.resize_size))
        
        # Load label (x, y)
        with open(self.label_files[idx], 'r') as f:
            line = f.readline().split()
            x_norm = float(line[0])
            y_norm = float(line[1])

        # Tensor's return
        img_tensor = self.transform(img)
        label_tensor = torch.tensor([x_norm, y_norm], dtype=torch.float32)

        return img_tensor, label_tensor
