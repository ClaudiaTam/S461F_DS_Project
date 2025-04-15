import os
import torch
from PIL import Image
from torchvision import transforms

from models import DigitNet, LetterNet, BinaryNet

# Image transformation for MNIST compatibility
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def load_models():
    """Load pre-trained digit, letter, and binary classification models.

    Returns:
        tuple: (digit_model, letter_model, binary_model)
    """
    # Paths to pre-trained models
    digit_model_file = "/Users/guest0701/Downloads/HCR_APP/digitnet.pt"
    letter_model_file = "/Users/guest0701/Downloads/HCR_APP/capitals_cnn.pt"
    binary_model_file = "/Users/guest0701/Downloads/HCR_APP/binary_cnn.pt"

    # Initialize models
    digit_model = DigitNet()
    letter_model = LetterNet()
    binary_model = BinaryNet()

    try:
        digit_model.load_state_dict(torch.load(digit_model_file, map_location=torch.device('cpu')))
        digit_model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading digit model: {e}")

    try:
        letter_model.load_state_dict(torch.load(letter_model_file, map_location=torch.device('cpu')))
        letter_model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading letter model: {e}")

    try:
        binary_model.load_state_dict(torch.load(binary_model_file, map_location=torch.device('cpu')))
        binary_model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading binary model: {e}")

    return digit_model, letter_model, binary_model

def classify_image(image_path, digit_model, letter_model, binary_model):
    """Classify an image as a digit or letter using the appropriate model.

    Args:
        image_path: Path to the image file.
        digit_model: Pre-trained digit classification model.
        letter_model: Pre-trained letter classification model.
        binary_model: Pre-trained binary classification model.

    Returns:
        tuple: (result, binary_label) where result is the predicted character
               and binary_label is 'Digit' or 'Letter'.
    """
    try:
        image = Image.open(image_path).convert("L")
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            # Binary classification first
            binary_output = binary_model(image_tensor)
            binary_pred = binary_output.argmax(dim=1, keepdim=True).item()
            binary_label = "Digit" if binary_pred == 0 else "Letter"

            # Route to appropriate model
            if binary_pred == 0:  # Digit
                digit_output = digit_model(image_tensor)
                pred = digit_output.argmax(dim=1, keepdim=True).item()
                result = str(pred)
            else:  # Letter
                letter_output = letter_model(image_tensor)
                pred = letter_output.argmax(dim=1, keepdim=True).item()
                result = chr(65 + pred)  # Convert 0-25 to A-Z

        return result, binary_label
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None