import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

######## STEP 1 ########
# 1️⃣ Define Data Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

######## STEP 2 ########
# 2️⃣ Load MNIST Dataset
train_dataset = torchvision.datasets.MNIST(root="./mnist_data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./mnist_data", train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

######## STEP 3 ########
# 3️⃣ Define CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

######## STEP 4 ########
# 4️⃣ Initialize Model, Loss Function, and Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("----------Train the Model----------")
######## STEP 5 ########
# 5️⃣ Train the Model
epochs = 5
for epoch in range(epochs):
    for images, labels in train_loader:
        print(len(labels))
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

######## STEP 6 ########
# 6️⃣ Test the Model
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")

######## STEP 7 ########
# 7️⃣ Visualize Some Predictions
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
model.eval()
with torch.no_grad():
    for i in range(5):
        image, label = test_dataset[i]
        image = image.unsqueeze(0).to(device)
        output = model(image)
        _, predicted = torch.max(output, 1)
        
        axes[i].imshow(image.cpu().squeeze(), cmap="gray")
        axes[i].set_title(f"Pred: {predicted.item()}")
        axes[i].axis("off")

plt.show()

######## STEP 8 ########
#$ Save the model
if True:
    torch.save(model.state_dict(), "cnn_mnist.pth")

## Load the model
if False:
    model.load_state_dict(torch.load("cnn_mnist.pth"))
    model.eval()  # Set model to evaluation mode

########################
##### Future Stuff #####
'''
- If the accuracy is low, increase epochs = 10 or tune the learning rate.
- Try adding Dropout layers to reduce overfitting.
- Experiment with different architectures (e.g., ResNet, MobileNet).
- Deploy the trained model in Flask or FastAPI.

If your model's accuracy isn't great, try:
✅ Increasing epochs (e.g., epochs = 10 instead of 5)
✅ Adding Dropout (nn.Dropout(0.5)) to prevent overfitting
✅ Experiment with different architectures (e.g., ResNet, MobileNet).
✅ Using Batch Normalization (nn.BatchNorm2d())
✅ Trying a deeper architecture (e.g., adding more Conv layers)
✅ Deploy the trained model in Flask or FastAPI.
'''