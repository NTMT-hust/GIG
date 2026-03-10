from torch.utils.data import Dataset
import AwareAugmentation
from PIL import Image

class ImbalancedImageDataset(Dataset):
    def __init__(self, image_paths, labels, class_counts,
                transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_counts = class_counts
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)

        return img, label