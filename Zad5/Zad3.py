"""
ZADANIE 3: Klasyfikacja FashionMNIST
Model: Convolutional Neural Network (CNN)
Opis: Klasyfikacja 10 rodzajów odzieży na obrazach 28x28 w skali szarości.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 10)
        )
    def forward(self, x):
        return self.net(x)

def run_fashion_task():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    model = FashionCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("Trening FashionMNIST...")
    for epoch in range(2):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
    print("Zadanie 3 zakończone.")

if __name__ == "__main__":
    run_fashion_task()