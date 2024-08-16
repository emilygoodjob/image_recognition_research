'''
Digital identification tasks

The first number of these numbers indicates which number the image is, 
the second digit represents the number of the picture under the number, 
and the third digit is who wrote the picture.

Task: Task1. What is the design network recognition image; 
Task2. Who wrote the design network recognition image?
'''

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torch.utils.data import DataLoader
from MultiTaskNN import MultiTaskNN
from CustomImageDataset import CustomImageDataset
import pandas as pd



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
    
    return train_loader, test_loader, dataset.author_to_idx

# Training function
def train_together(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, digit_target, author_target) in enumerate(train_loader):
        data, digit_target, author_target = data.to(device), digit_target.to(device), author_target.to(device)
        optimizer.zero_grad()
        digit_output, author_output = model(data)
        digit_loss = F.nll_loss(digit_output, digit_target)
        author_loss = F.nll_loss(author_output, author_target)
        loss = 0.7 * digit_loss + 0.3 * author_loss
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


# Function to train the model for digit classification
def digit_train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, digit_target, author_target) in enumerate(train_loader):
        data, digit_target = data.to(device), digit_target.to(device)
        optimizer.zero_grad()
        digit_output, _ = model(data)
        digit_loss = F.nll_loss(digit_output, digit_target)
        digit_loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Digit Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {digit_loss.item():.6f}')

# Function to train the model for author classification
def author_train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, digit_target, author_target) in enumerate(train_loader):
        data, author_target = data.to(device), author_target.to(device)
        optimizer.zero_grad()
        _, author_output = model(data)
        author_loss = F.nll_loss(author_output, author_target)
        author_loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Author Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {author_loss.item():.6f}')

# Testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct_digit = 0
    correct_author = 0
    with torch.no_grad():
        for data, digit_target, author_target in test_loader:
            data, digit_target, author_target = data.to(device), digit_target.to(device), author_target.to(device)
            digit_output, author_output = model(data)
            test_loss += F.nll_loss(digit_output, digit_target, reduction='sum').item()
            test_loss += F.nll_loss(author_output, author_target, reduction='sum').item()
            digit_pred = digit_output.argmax(dim=1, keepdim=True)
            author_pred = author_output.argmax(dim=1, keepdim=True)
            correct_digit += digit_pred.eq(digit_target.view_as(digit_pred)).sum().item()
            correct_author += author_pred.eq(author_target.view_as(author_pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    digit_accuracy = 100. * correct_digit / len(test_loader.dataset)
    author_accuracy = 100. * correct_author / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Digit Accuracy: {correct_digit}/{len(test_loader.dataset)} ({digit_accuracy:.0f}%), Author Accuracy: {correct_author}/{len(test_loader.dataset)} ({author_accuracy:.0f}%)')
    
    return digit_accuracy, author_accuracy

# Main function to execute the training and testing process
def main():
    # Check if CUDA is available and use it if possible
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Set the path to your image directory
    img_dir = 'image'
    train_loader, test_loader, author_to_idx = load_and_preprocess_data(img_dir)
    
    model = MultiTaskNN(len(author_to_idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    
    # run only one of the three model when execute main, comment out the other, try one by one
    # Digit
    # accuracies = {'Epoch': [], 'Digit Accuracy (Digit)': [], 'Author Accuracy (Digit)': []}
    # for epoch in range(1, 21):
    #     # train(model, device, train_loader, optimizer, epoch)
    #     # run only one of the three model when execute main, comment out the other, try one by one
    #     digit_train(model, device, train_loader, optimizer, epoch)
    #     digit_accuracy, author_accuracy = test(model, device, test_loader)
    #     scheduler.step()
        
    #     accuracies['Epoch'].append(epoch)
    #     accuracies['Digit Accuracy (Digit)'].append(digit_accuracy)
    #     accuracies['Author Accuracy (Digit)'].append(author_accuracy)
    # df = pd.DataFrame(accuracies)
    # df.to_excel('accuracies_digit.xlsx', index=False)
    
    # Author
    # accuracies = {'Epoch': [], 'Digit Accuracy (Author)': [], 'Author Accuracy (Author)': []}
    # for epoch in range(1, 21):
    #     # train(model, device, train_loader, optimizer, epoch)
    #     # run only one of the three model when execute main, comment out the other, try one by one
    #     author_train(model, device, train_loader, optimizer, epoch)
    #     digit_accuracy, author_accuracy = test(model, device, test_loader)
    #     scheduler.step()
        
    #     accuracies['Epoch'].append(epoch)
    #     accuracies['Digit Accuracy (Author)'].append(digit_accuracy)
    #     accuracies['Author Accuracy (Author)'].append(author_accuracy)
    # df = pd.DataFrame(accuracies)
    # df.to_excel('accuracies_author.xlsx', index=False)
    
    # Combined
    accuracies = {'Epoch': [], 'Digit Accuracy (Combined)': [], 'Author Accuracy (Combined)': []}
    for epoch in range(1, 21):
        # train(model, device, train_loader, optimizer, epoch)
        # run only one of the three model when execute main, comment out the other, try one by one
        train_together(model, device, train_loader, optimizer, epoch)
        digit_accuracy, author_accuracy = test(model, device, test_loader)
        scheduler.step()
        accuracies['Epoch'].append(epoch)
        accuracies['Digit Accuracy (Combined)'].append(digit_accuracy)
        accuracies['Author Accuracy (Combined)'].append(author_accuracy)
    # Save accuracies to Excel file
    df = pd.DataFrame(accuracies)
    df.to_excel('accuracies_combined.xlsx', index=False)
    
    torch.save(model.state_dict(), "multi_task_model.pt")

    model.load_state_dict(torch.load("multi_task_model.pt"))
    test(model, device, test_loader)

if __name__ == "__main__":
    main()
