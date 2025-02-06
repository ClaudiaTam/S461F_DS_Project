import sys
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self): # have parameter
        super(Net, self).__init__() # Calls the constructor of the parent class (nn.Module).
        self.conv1 = nn.Conv2d(1, 32, 3, 1) # input channels:1 (grayscale), output channels:32, Kernel Size: 3, Stride: 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # output channels:64
        self.dropout1 = nn.Dropout(0.25) # 25% chance of dropping each neuron
        self.dropout2 = nn.Dropout(0.5) # 50% chance of dropping each neuron
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10) # 128 inputs and outputs 10 classes (for the digits 0-9).

    def forward(self, x): # no parameter
        x = self.conv1(x) # Applies the first convolutional layer.
        x = F.relu(x) # Applies the ReLU (Rectified Linear Unit) activation function, introducing non-linearity.
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # Flattens the output into a 1D tensor, preserving the batch size
        x = self.fc1(x) # Fully Connected Layer with Activation
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x) # Output Layer
        output = F.log_softmax(x, dim=1) # Applies the log softmax function with a negative output
        return output
        
# Training Function
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    train_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    return train_loss, accuracy
    
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')

    return test_loss, accuracy
    
# Train and Evaluate
def train_and_evaluate(args, model, device, train_loader, test_loader, optimizer, scheduler):
    test_losses = []
    accuracies = []
    train_losses = []
    train_accuracies = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train(args, model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        scheduler.step()

    return train_losses, train_accuracies, test_losses, accuracies
    
# Plot graphs
def plot_results(train_losses, train_accuracies, test_losses, accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Train Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Train Loss')
    plt.plot(epochs, test_losses, 'o-', label='Test Loss')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'o-', label='Train Accuracy')
    plt.plot(epochs, accuracies, 'o-', label='Test Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
# Main function
def main():
    # Running in a Jupyter notebook
    args = argparse.Namespace(
        batch_size=64,
        test_batch_size=1000,
        epochs=14,
        lr=1.0,
        gamma=0.7, # Factor for adjusting the learning rate in the scheduler, reducing it over time.
        no_cuda=False, # can use the GPU if available
        no_mps=False, #  can use the MPS if available
        dry_run=False, # If True, performs a quick single pass to test the setup.
        seed=42,  # Remove fixed seed for more randomness
        log_interval=10, # Determines how often to log training progress (e.g., every 10 batches).
        save_model=True
    )

    # Device Setup
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Data Loading
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Data Loading with Augmentation for Training
    transform_train = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Transform for testing (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Dataset and DataLoader
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform_train)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform_test)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Model and Optimizer Initialization
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr) # optimizer: Adadelta / SGD / Adam
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Train and Evaluate
    train_losses, train_accuracies, test_losses, accuracies = train_and_evaluate(args, model, device, train_loader, test_loader, optimizer, scheduler)

    # Plot Results
    plot_results(train_losses, train_accuracies, test_losses, accuracies)

    # Model Saving
    if args.save_model:
        model_filename = f"mnist_cnn({accuracies[-1]:.2f}%).pt"
        try:
            torch.save(model.state_dict(), model_filename)
            print("Model is successfully saved")
        except Exception as e:
            print(f"Error saving model: {e}")

if __name__ == '__main__':
    main()           