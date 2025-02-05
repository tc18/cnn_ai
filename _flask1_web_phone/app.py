import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

# Load your trained PyTorch model
# model = torch.jit.load("epoc5_cnn_mnist.pth")  # Ensure you have a trained model saved


# Load your trained CNN model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load trained model
model = CNN()
model.load_state_dict(torch.load("epoc5_cnn_mnist.pth", map_location=torch.device("cpu")))
model.eval()

def preprocess_image(image):
    """ Convert uploaded image to 28x28 grayscale format for model prediction """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

@app.route("/")
def index():
    return render_template("index.html")  # Serves the web page

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")  # Convert to RGB
    processed_image = preprocess_image(image)

    # Run model prediction
    with torch.no_grad():
        output = model(processed_image)
        prediction = output.argmax(dim=1).item()

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)  # Accessible via phone
