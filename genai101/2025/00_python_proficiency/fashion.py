# STD
import random
# Pytorch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import transforms
from torchvision.datasets import FashionMNIST
# Other
import matplotlib.pyplot as plt

class Fashion:
    
    def __init__(self, root='./data', train_eval:float=0.8, download=True):
        # Load the Fashion-MNIST training and test datasets
        transform = transforms.ToTensor() # this transform normalises images between 0 and 1
        raw_train_dataset = FashionMNIST(root=root, train=True, transform=transform, download=download)
        # Split the dataset into train and validation
        ltrain = len(raw_train_dataset)
        train_size = int(train_eval * ltrain)
        eval_size = ltrain - train_size
        self.train_dataset, self.eval_dataset = random_split(raw_train_dataset, [train_size, eval_size])
        # Test dataset
        self.test_dataset = FashionMNIST(root=root, train=False, transform=transform, download=download)


    def _get_random_subset(self, dataset, ratio:float):
        if ratio <= 0 or ratio > 1:
            raise Exception("Invalid ratio; must be >0 and <=1")
        elif ratio == 1:
            return dataset
        else:
            # Generate a list of random indices for the subset
            dataset_size = len(dataset)
            subsample_size = int(ratio * dataset_size)
            indices = random.sample(range(dataset_size), subsample_size)
            if len(indices) == 0:
                assert "Invalid ratio (too small); no data left"
            return Subset(dataset, indices)


    def get_train_loader(self, batch_size, sample_ratio:float=1, shuffle=True):
        d = self._get_random_subset(self.train_dataset, sample_ratio)
        return DataLoader(d, batch_size=batch_size, shuffle=shuffle, num_workers=2)


    def get_eval_loader(self, batch_size, sample_ratio:float=1, shuffle=False):
        d = self._get_random_subset(self.eval_dataset, sample_ratio)
        return DataLoader(d, batch_size=batch_size, shuffle=shuffle, num_workers=2)


    def get_test_loader(self, batch_size, sample_ratio:float=1, shuffle=False):
        d = self._get_random_subset(self.test_dataset, sample_ratio)
        return DataLoader(d, batch_size=batch_size, shuffle=shuffle, num_workers=2)


    def show_samples(self):
        # Create a DataLoader with batch size 1 to fetch one sample at a time
        ds = self.train_dataset
        data_loader = self.get_train_loader(batch_size=1)
        class_names = ds.dataset.classes

        # Iterate through the data_loader to gather one sample per class
        samples_per_class = {}
        for images, labels in data_loader: # images and labels are tensors of size [1, 1, 28, 28] and [1] respectively
            label = labels.item()
            if label not in samples_per_class:
                samples_per_class[label] = images[0]
            if len(samples_per_class) == len(class_names):
                break

        # Create a plot to display the images
        plt.figure(figsize=(10, 5))

        for i in range(10):
            plt.subplot(2, 5, i + 1)
            image = samples_per_class[i]
            # Operation per subplot: display, title, axis off
            plt.imshow(image.squeeze(), cmap='binary')
            plt.title(class_names[i] + f' ({i})')
            plt.axis('off')

        plt.tight_layout()
        plt.show() 
# End of Fashion


# --- --- --- Main

if __name__ == '__main__':
    f = Fashion()
    f.show_samples()