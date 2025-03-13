# inference_mnist.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the same neural network architecture used for training
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

# Function to preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Function to load model and make prediction
def predict_digit(image_path, model_path):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = Net().to(device)
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess the image
    image = preprocess_image(image_path).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1, keepdim=True)

    return prediction.item()

# Example usage
model_path = "mnist_model.pth"  # Path to your saved model
image_path = "../datasets/mnist-datasets/test"  # Path to the image you want to classify

# If you want to test with multiple images
test_images = ["../datasets/mnist-datasets/test/image1.jpg", "../datasets/mnist-datasets/test/image2.jpg", "../datasets/mnist-datasets/test/image3.jpg"]
result = []
for img in test_images:
    digit = predict_digit(img, model_path)
    result.append((img, digit))
print(f"Results: : {result}")
with open("mnist-results.txt", "w") as f:
    for img, digit in result:
        f.write(f"{img}: {digit}\n")