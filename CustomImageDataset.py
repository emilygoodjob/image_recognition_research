import os
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset



# Custom dataset class to handle loading and processing of image data
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = self._load_labels()
        self.author_to_idx = self._create_author_to_idx()

    def _load_labels(self):
        img_labels = []
        # Iterate through all files in the image directory
        for img_name in os.listdir(self.img_dir):
            if img_name.endswith('.jpg'):
                parts = img_name.split('_')
                if len(parts) == 3:
                    digit, _, author = parts
                    # Remove file extension
                    author = author.split('.')[0]
                    img_labels.append((img_name, int(digit), author))
        print(f"Found {len(img_labels)} images in {self.img_dir}\n")
        return img_labels

    def _create_author_to_idx(self):
        authors = set([label[2] for label in self.img_labels])
        return {author: idx for idx, author in enumerate(authors)}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, digit, author = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Normalize to 0-1 and ensure RGB
        image = read_image(img_path, mode=ImageReadMode.RGB).float() / 255.0
        if self.transform:
            image = self.transform(image)
        author_idx = self.author_to_idx[author]
        return image, digit, author_idx
