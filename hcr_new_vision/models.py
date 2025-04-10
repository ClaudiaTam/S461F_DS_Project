import torch
from torchvision import transforms

# Define the digit classification model architecture (10 classes: 0-9)
class DigitNet(torch.nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, 1)
        self.conv3 = torch.nn.Conv2d(128, 256, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.4)
        self.fc1 = torch.nn.Linear(256 * 5 * 5, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

# Define the letter classification model architecture (26 classes: A-Z)
class LetterNet(torch.nn.Module):
    def __init__(self):
        super(LetterNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, 1)
        self.conv3 = torch.nn.Conv2d(128, 256, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.4)
        self.fc1 = torch.nn.Linear(256 * 5 * 5, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 26)  # 26 classes for letters A-Z

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

# Define the binary classification model architecture (2 classes: digit or letter)
class BinaryNet(torch.nn.Module):
    def __init__(self):
        super(BinaryNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, 3, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, 1)
        self.conv3 = torch.nn.Conv2d(128, 256, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.4)
        self.fc1 = torch.nn.Linear(256 * 5 * 5, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 2)  # 2 outputs: digit (0) or letter (1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

# Image transformation for MNIST compatibility
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def load_models():
    # Load the pre-trained digit classification model
    digit_model = DigitNet()
    digit_model_file = "/Users/ryan/Desktop/DL/FYP/Combined_Model_seed_14598/combined_cnn_epoch:9_test-accuracy:99.5481_test-loss:0.0128.pt"
    try:
        digit_model.load_state_dict(torch.load(digit_model_file, map_location=torch.device('cpu')))
        digit_model.eval()
    except Exception as e:
        print(f"Error loading digit model: {e}")
        exit()

    # Load the pre-trained letter classification model
    letter_model = LetterNet()
    letter_model_file = "/Users/ryan/Desktop/DL/FYP/Capitals_Model_seed_13692/capitals_cnn_epoch_10_test-accuracy_97.6128_test-loss_0.0016.pt"
    try:
        letter_model.load_state_dict(torch.load(letter_model_file, map_location=torch.device('cpu')))
        letter_model.eval()
    except Exception as e:
        print(f"Error loading letter model: {e}")
        exit()

    # Load the pre-trained binary classification model
    binary_model = BinaryNet()
    binary_model_file = "/Users/ryan/Desktop/DL/FYP/Binary_Model_seed_67675/binary_cnn_epoch:54_test-accuracy:99.9306_test-loss:0.0046.pt"
    try:
        binary_model.load_state_dict(torch.load(binary_model_file, map_location=torch.device('cpu')))
        binary_model.eval()
    except Exception as e:
        print(f"Error loading binary model: {e}")
        exit()

    return digit_model, letter_model, binary_model