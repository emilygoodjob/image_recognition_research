import torch
import torch.nn as nn
import torch.nn.functional as F



# Improved model definition with more layers and BatchNorm
class ImprovedNN(nn.Module):
    def __init__(self):
        super(ImprovedNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        # Max pooling layer with a 2x2 window
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Apply convolutional layers, ReLU activation, and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the output from the convolutional layers
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        # Apply dropout
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
