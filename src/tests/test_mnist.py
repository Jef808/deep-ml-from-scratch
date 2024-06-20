#!/usr/bin/env python

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data loading
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

2
# Network construction
class Net(nn.Module):
    """Simple neural network with one hidden layer"""
    def __init__(self):
        """Network constructor"""
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Forward pass of the network"""
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1
)

def visualize_inferences(model, data_loader):
    """Visualize model inferences on images from a DataLoader"""
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for data, target in data_loader:
            output = model(data)  # Get model output
            pred = output.argmax(dim=1, keepdim=True)  # Get the predicted class

            # Process each image in the batch
            for i in range(data.size(0)):
                img = data[i].numpy()  # Convert the tensor to numpy array
                img = np.squeeze(img)  # Remove unnecessary dimensions
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)  # Normalize the image for display
                img = img.astype(np.uint8)  # Convert the image to uint8
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR (OpenCV uses BGR by default)

                # Display the image and prediction
                cv2.putText(img, f'Pred: {pred[i].item()}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('Inference', img)
                cv2.waitKey(0)  # Wait for a key press to proceed to the next image
            cv2.destroyAllWindows()

def main():
    model = Net()

    # Training
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.NLLLoss()

    model.train()
    for _ in range(10):  # loop over the dataset multiple times
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Testing
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f'\nTest set: Accuracy: \
        {100. * correct / len(test_loader.dataset):.2f}\n')


if __name__ == "__main__":
    main()
