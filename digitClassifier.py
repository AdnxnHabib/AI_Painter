import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        # first convolutional layer -> 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
        self.dropout2 = nn.Dropout(0.25)
    
    def forward(self, x):
        # first conv block
        x = F.relu(self.conv1(x)) 
        x = self.pool1(x) # -> 32 feature maps of 14x14
        
        # second conv block
        x = F.relu(self.conv2(x))
        x = self.pool2(x) # -> 64 feature maps of 7x7
        
        # third conv block
        x = F.relu(self.conv3(x)) # -> 128 feature maps of 7x7
        
        # flatten the output 
        x = x.view(-1, 128 * 7 * 7) # -> we go from ()
        
        # fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


def load_data(batch_size=64):
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST dataset mean and std
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        
        loss = F.nll_loss(output, target)
        
        loss.backward()
        optimizer.step()
        
        # Update statistics
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}'
                  f'\tAccuracy: {100. * correct / total:.2f}%')
    
    return train_loss / len(train_loader), correct / total

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            
            # Get predictions
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({100. * accuracy:.2f}%)\n')
    
    return test_loss, accuracy


def visualize_predictions(model, device, test_loader, num_samples=10):
    model.eval()
    
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    

    images = images[:num_samples]
    labels = labels[:num_samples]

    with torch.no_grad():
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1).cpu().numpy()
    
    plt.figure(figsize=(15, 4))
    for i in range(num_samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].squeeze().numpy(), cmap='gray')
        plt.title(f"True: {labels[i]}, Pred: {predictions[i]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_accs)
    plt.plot(test_accs)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.show()

def save_model(model, filename="mnist_cnn_model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")


def load_model(filename="mnist_cnn_model.pth", device=torch.device("cpu")):
    model = MNISTClassifier().to(device)
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    return model

def predict_digit(model, image, device=torch.device("cpu")):
    """
    Predict digit from a single image
    
    Parameters:
    - model: trained CNN model
    - image: 28x28 numpy array or PIL image
    - device: device to run prediction on
    
    Returns:
    - predicted digit and confidence scores
    """

    model.eval()
    
    # Convert to tensor if needed
    if isinstance(image, np.ndarray):
        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image = transform(image.astype(np.float32))
    
    # Add batch dimension if needed
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # Move to device and predict
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
        
    # Get prediction and confidence scores
    predicted_digit = output.argmax(dim=1).item()
    confidence_scores = probabilities[0].cpu().numpy()
    
    return predicted_digit, confidence_scores


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, test_loader = load_data(batch_size=64)
    
    model = MNISTClassifier().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # Train model
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    
    # Plot training history
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    # Visualize predictions
    visualize_predictions(model, device, test_loader)
    
    save_model(model)
    
    return model

if __name__ == "__main__":
    main()