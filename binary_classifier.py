import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

# Custom Dataset to filter only English letters (A-Z, a-z) and digits (0-9)
class CustomBinaryDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.filtered_data = []
        self.filtered_targets = []

        # Filter dataset to include only letters and digits
        for data, target in zip(self.dataset.data, self.dataset.targets):
            if (0 <= target <= 9) or (10 <= target <= 35):  # Digits: 0-9, Letters: A-Z, a-z
                self.filtered_data.append(data)
                # Label: 0 for digits (0-9), 1 for letters (A-Z, a-z)
                self.filtered_targets.append(0 if target <= 9 else 1)

        self.filtered_data = torch.stack(self.filtered_data)
        self.filtered_targets = torch.tensor(self.filtered_targets)

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        img, label = self.filtered_data[idx], self.filtered_targets[idx]
        # Convert the tensor image to a PIL Image
        img = to_pil_image(img)  # Converts torch.Tensor to PIL Image for transforms
        if self.transform:
            img = self.transform(img)
        return img, label


# Define the binary classification model
class BinaryNet(nn.Module):
    def __init__(self):
        super(BinaryNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = None  # Placeholder for the first fully connected layer
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        # Initialize fc1 dynamically if not already initialized
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


# Training function
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                  f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

    train_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    return train_loss, accuracy


# Testing function
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} "
          f"({accuracy:.0f}%)\n")

    return test_loss, accuracy


# Main function
def main():
    # Generate a random seed
    random_seed = random.randint(0, 100000)
    print(f"Using random seed: {random_seed}")

    # Set the seed for reproducibility
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    # Command-line arguments
    args = argparse.Namespace(
        batch_size=64,
        test_batch_size=128,
        epochs=10,
        lr=0.01,
        gamma=0.7,
        no_cuda=False,
        seed=random_seed,
        log_interval=10,
        save_model=True
    )

    # Device setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load datasets
    print("Loading datasets...")
    train_dataset_raw = datasets.EMNIST(root='../data', split='byclass', train=True, download=True)
    test_dataset_raw = datasets.EMNIST(root='../data', split='byclass', train=False, download=True)

    train_dataset = CustomBinaryDataset(train_dataset_raw, transform=transform)
    test_dataset = CustomBinaryDataset(test_dataset_raw, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # Model and optimizer
    model = BinaryNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train(args, model, device, train_loader, optimizer, epoch)
        test_loss, test_accuracy = test(model, device, test_loader)
        scheduler.step()

    # Save the model
    if args.save_model:
        torch.save(model.state_dict(), "binary_classifier.pt")
        print("Model saved as binary_classifier.pt")


if __name__ == "__main__":
    main()