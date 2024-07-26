import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Custom dataset class to handle loading and processing of image data
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = self._load_labels()

    def _load_labels(self):
        img_labels = []
        # Iterate through all files in the image directory
        for img_name in os.listdir(self.img_dir):
            if img_name.endswith('.jpg'):
                parts = img_name.split('_')
                if len(parts) == 3:
                    digit, _, author = parts
                    author = author.split('.')[0]  # Remove file extension
                    img_labels.append((img_name, int(digit), author))
        print(f"Found {len(img_labels)} images in {self.img_dir}")
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

# Function to load and preprocess data
def load_and_preprocess_data(img_dir):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((28, 28)),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = CustomImageDataset(img_dir, transform=transform)
    if len(dataset) == 0:
        raise ValueError(f"No images found in directory: {img_dir}")
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

# Improved model definition with more layers and BatchNorm
class ImprovedNN(nn.Module):
    def __init__(self):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Main function to execute the training and testing process
def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    img_dir = 'image'  # Set the path to your image directory
    train_loader, test_loader = load_and_preprocess_data(img_dir)
    
    model = ImprovedNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    
    for epoch in range(1, 31):  # Increase number of epochs
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    
    torch.save(model.state_dict(), "improved_mnist_model.pt")

    # Load the model for evaluation
    model.load_state_dict(torch.load("improved_mnist_model.pt"))
    test(model, device, test_loader)

if __name__ == "__main__":
    main()
