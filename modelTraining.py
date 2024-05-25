import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import *
import matplotlib.pyplot as plt
import numpy as np
from model import FavConvolutionalNetwork

EPOCHS = 150
EPOCH_PATIENCE = 11

BATCH_SIZE = 64

LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.00001

L1_NORM = 0.0001
ROOT = "/Users/maciek/cnn_data"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

mean = [0.4330, 0.3819, 0.2964]
std = [0.2545, 0.2044, 0.2163]

train_transform_1 = Compose([
    Resize((112,112)),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(30),
    ToTensor(),
    Normalize(mean, std)
])
train_transform_2 = Compose([
    Resize((112, 112)),
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    RandomPerspective(distortion_scale=0.4, p=0.6),
    ToTensor(),
    Normalize(mean, std)
])

test_transform = Compose([
    Resize((112, 112)),
    ToTensor(),
    Normalize(mean, std)
])

train_data_1 = datasets.Flowers102(root=ROOT, split='train', download=True, transform=test_transform)
train_data_2 = datasets.Flowers102(root=ROOT, split='train', download=True, transform=train_transform_1)
train_data_3 = datasets.Flowers102(root=ROOT, split='train', download=True, transform=train_transform_2)
train_data = train_data_1 + train_data_2 + train_data_3

val_data = datasets.Flowers102(root=ROOT, split='val', download=True, transform=test_transform)
test_data = datasets.Flowers102(root=ROOT, split='test', download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

model = FavConvolutionalNetwork().to(DEVICE)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

def l1_norm(model):
    l1_regularization = torch.tensor(0.).to(DEVICE)
    for param in model.parameters():
        l1_regularization += torch.norm(param, 1)
    return l1_regularization

# TRAINING AND VALIDATION LOOP

start_time = time.time()

# Create Variables To Tracks Things
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# Early Stopping Parameters
patience = EPOCH_PATIENCE  # How many epochs to wait after val loss has stopped improving
min_val_loss = float('inf')
stale_epochs = 0  # Counter for epochs without improvement

for i in range(EPOCHS):
    trn_corr = 0
    tst_corr = 0

    # Train
    model.train()
    running_loss = 0
    for b, (X_train, y_train) in enumerate(train_loader, 1):
        X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
        y_pred = model(X_train)  # get predicted values from the training set. Not flattened 2D
        loss = criterion(y_pred, y_train)  # Compare the predictions to correct answers in y_train

        predicted = torch.max(y_pred.data, 1)[
            1]  # add up the number of correct predictions. Indexed off the first point
        batch_corr = (predicted == y_train).sum()  # how many we got correct from this batch. True = 1, False=0, sum those up
        trn_corr += batch_corr  # keep track as we go along in training.

        l1_loss = L1_NORM * l1_norm(model)
        total_loss = l1_loss + loss
        running_loss += total_loss.item() * X_train.size(0)
        # Update our parameters
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_correct.append(trn_corr)

    # Test
    model.eval()
    total = 0
    with torch.no_grad():
        val_loss = 0
        for b, (X_test, y_test) in enumerate(val_loader):
            X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()
            total += y_test.size(0)
            val_loss += criterion(y_val, y_test).item()  # Sum up the loss from each batch

    avg_val_loss = val_loss / len(val_loader)  # Calculate the average loss

    test_losses.append(avg_val_loss)
    test_correct.append(tst_corr)

    scheduler.step(avg_val_loss)

    # Early Stopping Check
    if avg_val_loss < min_val_loss:
        min_val_loss = avg_val_loss
        stale_epochs = 0  # Reset the stale epochs counter
        best_model_state = model.state_dict()
    else:
        stale_epochs += 1  # Increment the stale epochs counter
        if stale_epochs >= patience:
            print(f'Stopping early at epoch {i} due to overfitting.')
            model.load_state_dict(best_model_state)
            break  # Break out of the loop

    print(f'[{i}/{EPOCHS}] train_loss: {avg_train_loss:.3f} val_loss: {avg_val_loss:.3f} val_acc: {(tst_corr/total)*100:.1f}%')

current_time = time.time()
total = current_time - start_time
print(f'Training Took: {total / 60} minutes!')

# Convert GPU tensors to CPU tensors, detach them from the computation graph, and then to NumPy arrays
train_losses = [tl for tl in train_losses]
test_losses = [tl for tl in test_losses]

# Now you can plot using matplotlib
#plt.plot(train_losses, label="Training Loss")
#plt.plot(test_losses, label="Validation Loss")
#plt.title("Loss at Epoch")
#plt.legend()
#plt.show(block=True)

#plt.plot([t.cpu()/30 for t in train_correct], label="Training Accuracy")
#plt.plot([t.cpu()/10 for t in test_correct], label="Validation Accuracy")
#plt.title("Accuracy at the end of each Epoch")
#plt.legend()

from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
# Load the best model state
test_load_everything = DataLoader(test_data, batch_size=512, shuffle=False)

# Initialize lists to store predictions and true labels for evaluation
final_predictions = []
final_true_labels = []

# Evaluation loop
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for X_test, y_test in test_load_everything:
        X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
        y_val = model(X_test)
        predicted = torch.max(y_val.data, 1)[1]
        correct += (predicted == y_test).sum().item()
        total += y_test.size(0)

        # Store predictions and true labels
        final_predictions.extend(predicted.view(-1).cpu().numpy())
        final_true_labels.extend(y_test.view(-1).cpu().numpy())

# Convert lists to numpy arrays
predictions = np.array(final_predictions)
true_labels = np.array(final_true_labels)

# Calculate precision, recall, and F1 score
precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
accuracy = correct / total

print(f'Accuracy of the model on the test set: {accuracy * 100:.2f}%')
print(f"Final Precision: {precision}")
print(f"Final Recall: {recall}")
print(f"Final F1 Score: {f1}")

# SAVING THE MODEL WEIGHTS
torch.save(model.state_dict(), 'model_weights.pth')
