# STD
import time
# Pytoch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# Other
import matplotlib.pyplot as plt
# Project tooling
from fashion import Fashion


# --- --- --- TO BE COMPLETED --- --- ---

# Define the classifier model using ConvFeatureExtractor for MNIST classification (10 classes)
class Classifier(nn.Module):
    def __init__(self, feature_output_dim:int, num_classes:int):
        super(Classifier, self).__init__()
        self.feature_extractor = ConvFeatureExtractor(output_dim=feature_output_dim)
        self.fc = nn.Linear(feature_output_dim, num_classes)  # Fully connected layer for classification

    def forward(self, x):
        #
        x = self.feature_extractor(x)
        x = F.gelu(x)
        #
        x = self.fc(x)
        return x
# End of Classifier


# --- --- --- PROVIDED CODE --- --- ---

class ConvFeatureExtractor(nn.Module):
    def __init__(self, output_dim):
        super(ConvFeatureExtractor, self).__init__()
        # Prepare the layer that we will use in the forward pass
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1)      # 28x28x1 (black and white image: 1 channel per pixel) -> 14x14x4 (reduces the image height and width, output more channel per "pixel")
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1)      # 14x14x4-> 7x7x8
        self.fcout = nn.Linear(7 * 7 * 8, output_dim)                                                  # Configurable output dimension
        
    def forward(self, x):
        # Convolutional layers followed by non lineairity (GELU function)
        x = self.conv1(x)
        x = F.gelu(x)
        #
        x = self.conv2(x)
        x = F.gelu(x)
        #
        # Flatten Batch,3D [BSIZE, 8, 7, 7] to Batch,1D [BSIZE, 8*7*7 = 392] so we can apply a fully connected layer [392, output_dim]
        # Keep the batch size (BSIZE obtained from x.size(0)), flatten the rest
        x = x.view(x.size(0), -1)
        x = self.fcout(x)
        return x
# End of ConvFeatureExtractor





# --- --- --- Test/Eval/train classifier
        
def eval_classifier(classifier:Classifier, data_loader, criterion)->tuple[float, float]:
    """Classifier evaluation; data_loader produces (batch of images, batch of labels). Return (eval loss, eval accuracy)"""
    # Classifier in eval mode
    classifier.eval()
    eval_loss = 0.0
    eval_correct:int = 0
    total = 0
    # torch.no_grad() to disable gradient tracking
    with torch.no_grad():
        for images, labels in data_loader:
            # Forward pass
            outputs = classifier(images)
            eval_loss += criterion(outputs, labels).item()*labels.size(0)
            # Accuracy
            _, predicted_idx = torch.max(outputs.data, 1)
            eval_correct += (predicted_idx == labels).sum().item()
            #
            total += labels.size(0)
    #
    eval_loss /= total
    eval_acc = 100.0 * eval_correct/total
    return eval_loss, eval_acc
#


def train_classifier(classifier:Classifier, fashion:Fashion, batch_size:int, lr:float, nepochs:int):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    
    # Get train set
    train_loader = fashion.get_train_loader(batch_size=batch_size)

    # Get eval set
    eval_loader = fashion.get_eval_loader(batch_size=batch_size)
    
    # train loop
    start_loop = time.time()
    for epoch in range(nepochs):
        start = time.time()

        # --- TRAIN
        classifier.train()
        train_loss = 0.0
        train_correct:int = 0
        total = 0
        for images, labels in train_loader:
            # Forward pass
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            # Backward pass and optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Loss information: scale by size of batch
            train_loss += loss.item()*labels.size(0)
            # Accuracy information
            _, predicted_idx = torch.max(outputs.data, 1)
            train_correct += (predicted_idx == labels).sum().item()
            #
            total += labels.size(0)
        #
        train_loss /= total # Average loss over the epoch -- all samples
        train_acc = 100.0 * train_correct / total

        # --- EVAL
        eval_loss, eval_acc = eval_classifier(classifier, eval_loader, criterion)

        # --- PRINT
        print(f'Epoch {epoch + 1:2d}/{nepochs} {time.time() - start:.2f}s | Train loss: {train_loss:.4f} acc: {train_acc:.2f}% | Eval loss: {eval_loss:.4f} acc: {eval_acc:.2f}%')
    #
    print(f"Total Time: {time.time() - start_loop:.2f}s")
#


def test_classifier(classifier:Classifier, data_loader:DataLoader):
    criterion = nn.CrossEntropyLoss()
    start = time.time()
    loss, acc = eval_classifier(classifier, data_loader, criterion)
    print(f"Test loss: {loss:.4f} acc: {acc:.2f}%, Test time: {time.time() - start:.2f}s")
#


def show_classification(classifier:Classifier, data_loader:DataLoader):
    class_names:list[str] = data_loader.dataset.classes
    nbclass = len(class_names)
    classifier.eval()
    # Gather one sample per class
    samples_per_class = {}
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = classifier(images)
            _, predicted_index = torch.max(outputs, 1)
            #
            for idx, label in enumerate(labels):
                label = label.item()
                if label not in samples_per_class:
                    samples_per_class[label] = (images[idx], predicted_index[idx].item())
            #
            if len(samples_per_class) == nbclass:
                break
    # Display
    fig, axes = plt.subplots(5, 2, figsize=(10, 10))
    axes = axes.flatten()
    for i in range(nbclass):
        ax = axes[i]
        img = samples_per_class[i][0]
        pred = samples_per_class[i][1]
        color = 'green' if pred == i else 'red'
        ax.imshow(img.squeeze(), cmap='binary')
        ax.set_title(f'Pred: {class_names[pred]} {pred} | Actual: {class_names[i]} {i}', color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
#