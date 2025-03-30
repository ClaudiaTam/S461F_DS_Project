import torch
from torchvision import transforms
from PIL import Image
import os
from tkinter import filedialog, messagebox
from models import CharNet

# Image transformation for MNIST compatibility
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def load_model(model_path="C:/Users/guest0701/Downloads/hcr2/emnist_cnn_epoch_14_test-accuracy_93.1792_test-loss_0.1858.pt"):
    """Load the pre-trained character classification model."""
    model = CharNet()
    if not os.path.exists(model_path):
        model_path = filedialog.askopenfilename(
            title="Select Trained Model File",
            filetypes=[("PyTorch Model Files", "*.pt")],
            initialdir=os.getcwd()
        )
        if not model_path:
            messagebox.showerror("Error", "No model file selected. Exiting.")
            exit()
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        messagebox.showerror("Error", f"Error loading model: {e}")
        exit()
    return model

def classify_characters(mnist_images, model, output_dir="C:/Users/guest0701"):
    """Classify a list of MNIST-formatted images."""
    char_results = []
    for i, mnist_img in enumerate(mnist_images):
        char_filename = os.path.join(output_dir, f"char_{i+1}_mnist.png")
        img = Image.fromarray(mnist_img).convert("L")
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1, keepdim=True).item()
            result = str(pred) if pred < 10 else chr(55 + pred)  # 0-9, then A-Z (10-35 -> 65-90)
            char_results.append(result)
        os.remove(char_filename)
    return char_results

model = load_model()