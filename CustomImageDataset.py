import os
from torchvision.io import read_image
from torch.utils.data import Dataset



# 自定义数据集类
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = self._load_labels()

    def _load_labels(self):
        img_labels = []
        for img_name in os.listdir(self.img_dir):
            if img_name.endswith('.png'):
                digit, _, author = img_name.split('_')
                img_labels.append((img_name, int(digit), author))
        return img_labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, digit, author = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path).float() / 255.0
        if self.transform:
            image = self.transform(image)
        return image, digit, author
