import torch
import torchvision
import torchvision.transforms as transforms
from safetensors.torch import save_file
from torch.utils.data import DataLoader

# Define a transformation to convert PIL images to PyTorch tensors
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Converts image to a PyTorch tensor (0.0 to 1.0)
        # Optional: Normalize the data if needed (e.g., mean=0.5, std=0.5 for all pixels)
        # transforms.Normalize((0.5,), (0.5,))
    ]
)

# Load the training dataset
train_dataset = torchvision.datasets.MNIST(
    root="./",  # Root directory to store the dataset
    train=True,  # Specifies training data
    download=True,  # Downloads the dataset if not already present
    transform=transform,  # Apply the defined transform
)

# Load the test dataset
test_dataset = torchvision.datasets.MNIST(
    root="./",  # Same root directory
    train=False,  # Specifies test data
    download=True,  # Downloads the dataset if not already present
    transform=transform,  # Apply the defined transform
)

# print(type(train_dataset[0][0]))

train_images = []
train_labels = []
test_images = []
test_labels = []

for image, label in train_dataset:
    train_images.append(image)
    one_hot = torch.zeros(10).long()
    one_hot[label] = 1
    train_labels.append(one_hot)

train_images = torch.stack(train_images)
train_labels = torch.stack(train_labels)

for image, label in test_dataset:
    test_images.append(image)
    one_hot = torch.zeros(10).long()
    one_hot[label] = 1
    test_labels.append(one_hot)

test_images = torch.stack(test_images)
test_labels = torch.stack(test_labels)

save_file(
    {
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
    },
    "train_test_data_mnist.safetensors",
)
