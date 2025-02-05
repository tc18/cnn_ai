import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

######## STEP 1 ########
# Define transformations (convert to tensor & normalize)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

######## STEP 2 ########
# Automatically download and load MNIST
train_dataset = torchvision.datasets.MNIST(root="./mnist_data", train=True, transform=transform, download=False)
test_dataset = torchvision.datasets.MNIST(root="./mnist_data", train=False, transform=transform, download=False)

##########
'''
✅ MNIST Dataset Files:

Training Images (train-images-idx3-ubyte.gz)
Contains 60,000 grayscale images (28×28 pixels each).
Training Labels (train-labels-idx1-ubyte.gz)
Contains the corresponding 60,000 labels (digits 0-9).
Test Images (t10k-images-idx3-ubyte.gz)
Contains 10,000 grayscale images (28×28 pixels each).
Test Labels (t10k-labels-idx1-ubyte.gz)
Contains the corresponding 10,000 labels.
'''

######## STEP 3 ########
# Show a sample image
image, label = train_dataset[2]
plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.show()


########################
##### Future Stuff #####
'''
✅ Other Handwritten Digit Datasets
If you're looking for more datasets like MNIST:

- EMNIST: Extended version of MNIST with letters.
- KMNIST: Kuzushiji MNIST (Japanese characters).
- Fashion-MNIST: Clothing images instead of digits.
'''