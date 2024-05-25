import torch
from model import FavConvolutionalNetwork
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader

# Define the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Define the mean and std for normalization
mean = [0.4330, 0.3819, 0.2964]
std = [0.2545, 0.2044, 0.2163]

# Define the test transform
test_transform = Compose([
    Resize((112, 112)),
    ToTensor(),
    Normalize(mean, std)
])

# Load the test data
ROOT = "/Users/maciek/cnn_data"
test_data = datasets.Flowers102(root=ROOT, split='test', download=True, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=512, shuffle=False)

# Load the model
model = FavConvolutionalNetwork().to(DEVICE)
model.eval()

# Load the model weights
model_weights = torch.load('model_weights.pth')
model.load_state_dict(model_weights)

# Run the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
        y_val = model(X_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum().item()
        total += y_test.size(0)

# Calculate accuracy
accuracy = correct / total
print(f'Accuracy of the model on the test set: {accuracy * 100:.2f}%')
