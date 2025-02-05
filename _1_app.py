import torch
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)
model = CNN()
model.load_state_dict(torch.load("cnn_mnist.pth"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file).convert("L")
    transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    image = transform(image).unsqueeze(0)
    
    output = model(image)
    _, predicted = torch.max(output, 1)
    
    return jsonify({"prediction": predicted.item()})

if __name__ == "__main__":
    app.run(port=5000)
