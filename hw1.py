import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ImprovedNN import ImprovedNN
from CustomImageDataset import CustomImageDataset



# Function to load and preprocess data
def load_and_preprocess_data(img_dir):
    transform = transforms.Compose([
        # Resize to 64x64
        transforms.Resize((64, 64)),
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
            # Sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')

# Main function to execute the training and testing process
def main():
    # Check if CUDA is available and use it if possible
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Set the path to your image directory
    img_dir = 'image'
    train_loader, test_loader = load_and_preprocess_data(img_dir)
    
    model = ImprovedNN().to(device)
    # Adjust learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    
    # Increase number of epochs
    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
    
    torch.save(model.state_dict(), "improved_mnist_model.pt")

    # Load the model for evaluation
    model.load_state_dict(torch.load("improved_mnist_model.pt"))
    test(model, device, test_loader)

if __name__ == "__main__":
    main()
