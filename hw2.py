import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from HandwritingLSTM import HandwritingLSTM
from HandwritingRNN import HandwritingRNN
from hw1 import load_and_preprocess_data
import pandas as pd

# Training function for handwriting recognition using RNN
def train_rnn(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, _, author_target) in enumerate(train_loader):
        # 将图像数据转换为适合 RNN（循环神经网络）和 LSTM（长短时记忆网络）处理的格式
        data = data.view(data.size(0), 64, -1)  # Flatten to line by line pixel values
        # data.size(0) : 每次训练，模型处理的数量
        # 64 图像的宽度/行数
        # -1：这个维度的大小由系统自动计算，以确保重新调整形状后张量的总元素数量与原始张量保持一致
        
        data, author_target = data.to(device), author_target.to(device)
        optimizer.zero_grad()
        author_output = model(data)
        author_loss = F.cross_entropy(author_output, author_target) # 计算交叉熵损失（Cross Entropy Loss）
        author_loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'RNN Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {author_loss.item():.6f}')

# Training function for handwriting recognition using LSTM
def train_lstm(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, _, author_target) in enumerate(train_loader):
        data = data.view(data.size(0), 64, -1)  # Flatten to line by line pixel values
        data, author_target = data.to(device), author_target.to(device)
        optimizer.zero_grad()
        author_output = model(data)
        author_loss = F.cross_entropy(author_output, author_target)
        author_loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'LSTM Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {author_loss.item():.6f}')

# Testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct_author = 0
    with torch.no_grad():
        for data, _, author_target in test_loader:
            data = data.view(data.size(0), 64, -1)  # Reshaping to (batch_size, sequence_length, input_size)
            data, author_target = data.to(device), author_target.to(device)
            author_output = model(data)
            test_loss += F.cross_entropy(author_output, author_target, reduction='sum').item()
            author_pred = author_output.argmax(dim=1, keepdim=True)
            correct_author += author_pred.eq(author_target.view_as(author_pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    author_accuracy = 100. * correct_author / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Author Accuracy: {correct_author}/{len(test_loader.dataset)} ({author_accuracy:.0f}%)')
    
    return author_accuracy

# Main function to execute the training and testing process
def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    img_dir = 'image'
    train_loader, test_loader, author_to_idx = load_and_preprocess_data(img_dir)
    
    input_size = 192 # 彩色图像 (channels, height, width), 一般rgb channels为3. 每一行展成一个向量 3 * 64 = 192
    hidden_size = 128 # 隐藏层，一般默认128
    output_size = len(author_to_idx) # 等于分类的数量
    num_layers = 2 #一层不能处理复杂情况，太多层过拟合 overfitting - 先用两层试试
    
    # model = HandwritingRNN(input_size, hidden_size, output_size, num_layers).to(device)
    model = HandwritingLSTM(input_size, hidden_size, output_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    
    accuracies = {'Epoch': [], 'Author Accuracy': []}
    for epoch in range(1, 21):
        # train_rnn(model, device, train_loader, optimizer, epoch)
        train_lstm(model, device, train_loader, optimizer, epoch)
        author_accuracy = test(model, device, test_loader)
        scheduler.step()
        
        accuracies['Epoch'].append(epoch)
        accuracies['Author Accuracy'].append(author_accuracy)
    df = pd.DataFrame(accuracies)
    # df.to_excel('handwriting_RNN.xlsx', index=False)
    df.to_excel('handwriting_LSTM.xlsx', index=False)
    
    torch.save(model.state_dict(), "handwriting_model.pt")

    model.load_state_dict(torch.load("handwriting_model.pt"))
    test(model, device, test_loader)

if __name__ == "__main__":
    main()
