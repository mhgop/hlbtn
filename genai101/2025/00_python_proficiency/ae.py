# STD
import time
# Pytoch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
# Other
import matplotlib.pyplot as plt
# Project tooling
from fashion import Fashion
from tools import show_image
from classifier import ConvFeatureExtractor


# --- --- --- TO BE COMPLETED --- --- ---

class Encoder(nn.Module):

    def __init__(self, input_dim:int, hidden_dim:int, latent_dim:int):
        # COMPLETE ME!
        pass

    def forward(self, x):
        # COMPLETE ME!
        pass
# End of Encoder

        
class Decoder(nn.Module):

    def __init__(self, latent_dim:int, hidden_dim:int, output_dim:int):
        # COMPLETE ME!
        pass

    def forward(self, x):
        # COMPLETE ME!
        pass
# End of Decoder
    

class AutoEncoder(nn.Module):

    def __init__(self, feature_ouput_dim:int, encoder_hdim:int, latent_dim:int, decoder_hdim:int):
        # COMPLETE ME!
        pass

    def forward(self, x):
        """input of shape [-1, 1, 28, 28] -> output of shape [-1, 1, 28, 28]"""
        encoded = self.encode(x)
        reconstructed = self.decode(encoded)
        return reconstructed

    def encode(self, x):
        """input of shape [-1, 1, 28, 28] -> output of shape [-1, self.latent_dim]"""
        # COMPLETE ME!
        pass

    def decode(self, x):
        """input of shape [-1, self.latent_dim] -> output of shape [-1, 1, 28, 28]"""
        # COMPLETE ME!
        pass
        # Hint: use view to reshape the tensor

    def loss(self, reconstructed_x, x) -> Tensor:
        """Reconstruction loss: expect two tensors of same shape"""
        return F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
# End of AutoEncoder


def eval_ae(ae:AutoEncoder, data_loader)->float:
    """ Loss evaluation for the AutoEncoder; data_loader produces (batch of images, batch of labels) -> return eval loss """
    ae.eval()
    eval_loss = 0.0
    # COMPLETE ME!
    return eval_loss
#


# --- --- --- PROVIDED CODE --- --- ---

def train_ae(ae:AutoEncoder, fashion:Fashion, batch_size:int, lr:float, nepochs:int):
    # Create an optimizer, filtering out frozen parameters (actually: keep non frozen ones)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, ae.parameters()), lr=lr)
    
    # Get train set
    train_loader = fashion.get_train_loader(batch_size=batch_size)

    # Get eval set
    eval_loader = fashion.get_eval_loader(batch_size=batch_size)
    
    # train loop   
    start_loop = time.time()
    for epoch in range(nepochs):
        start = time.time()

        # --- TRAIN
        ae.train()
        train_loss = 0.0
        total = 0
        for images, _labels in train_loader:
            # Forward pass
            reconstructed_image = ae(images)
            # Calculate AE loss, flattening the input X to match the output shape
            loss = ae.loss(reconstructed_image, images)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Train loss (scale by the size of the batch)
            train_loss += loss.item()*images.size(0)
            total += images.size(0)
        #
        train_loss /= total # Average loss over the epoch -- all samples

        # --- EVAL
        eval_loss = eval_ae(ae, eval_loader)
        
        # --- PRINT
        print(f'Epoch {epoch + 1:2d}/{nepochs} {time.time() - start:.2f}s | Train loss: {train_loss:.4f} | Eval loss: {eval_loss:.4f}')
    #
    print(f"Total Time: {time.time() - start_loop:.2f}s")
#


def test_ae(ae:AutoEncoder, data_loader:DataLoader):
    start = time.time()
    loss = eval_ae(ae, data_loader)
    print(f"Test loss: {loss:.4f}, Test time: {time.time() - start:.2f}s")
#


# --- --- --- Latent Space Analysis

def decode_latent_space(ae:AutoEncoder, latent_points:Tensor)->Tensor:
    """Decode a tensor [-1, 2] to an image tensor [-1, 1, 28, 28]"""
    assert latent_points.shape[1:] == torch.Size([2])
    ae.eval()
    with torch.inference_mode():
        return ae.decode(latent_points)


def encode_to_latent_space(ae:AutoEncoder, images:Tensor)->Tensor:
    """Encode a tensor [-1, 1, 28, 28] to a latent tensor [-1, 2]"""
    assert images.shape[1:] == (1, 28, 28)
    ae.eval()
    with torch.inference_mode():
            return ae.encode(images)


def show_latent_space(ae:AutoEncoder, fashion:Fashion, batch_size, grid_size:int=19):
    latent_scale:float = 1.0

    classes_names = fashion.test_dataset.classes
    num_classes = len(classes_names)

    # --- --- --- Create latent space grid --- --- ---

    # Create a 2D grid of latent points
    latent_range = torch.linspace(-latent_scale, latent_scale, grid_size)  # Uniformly spaced points in the latent space
    latent_points = torch.cartesian_prod(latent_range, latent_range)       # Create a grid of points [grid_size x grid_size, 2]

    # Decode the latent points into images
    generated_images = decode_latent_space(ae, latent_points)


    # --- --- --- Setup Matlplotlib --- --- ---

    # Adjust image size to fit well in the scaled latent space
    image_size = (latent_range[1]-latent_range[0]).item()
    limit = latent_scale + image_size

    # Create a unified canvas for the images
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    
    # --- --- --- Show latent space --- --- ---

    # Plot each image at its corresponding position in the latent space
    for idx, (x, y) in enumerate(latent_points):
        show_image(ax, image_size, generated_images[idx], x.item(), y.item(), cmap='bone')


    # --- --- --- Show test data in the latent space --- --- ---
    # Dictionary to store latent space coordinates per class (init with list for appending later)
    class_latent_coords = {i: [] for i in range(num_classes)}

    # Dictionary to store one exemplar image per class
    class_exemplar_images = {i: [] for i in range(num_classes)}

    for images, labels in fashion.get_test_loader(batch_size=batch_size):
        # Encode the batch into the latent space
        encoded = encode_to_latent_space(ae, images)

        # Store per class
        for i, label in enumerate(labels):
            l = label.item()
            class_latent_coords[l].append(encoded[i])
            class_exemplar_images[l].append(images[i])

    # Calculate the average latent coordinates for each class
    class_centers = {c: torch.stack(coords).mean(dim=0).tolist() for c, coords in class_latent_coords.items()}
    
    # Show class center with exemplar image and label
    for l, center in class_centers.items():
        cname = classes_names[l]
        x,y = center

        # Plot the exemplar closest to the center
        reference = Tensor(center)
        distances = [torch.dist(tensor, reference) for tensor in class_latent_coords[l]]
        closest_index = torch.argmin(torch.tensor(distances))
        img = class_exemplar_images[l][closest_index]
        show_image(ax, image_size, img, x, y, cmap='afmhot_r') # Coordinate of the center, not the actual image

        # Plot the label under the image
        ax.text(x, y-image_size/1.3, f"{cname}({l})", fontsize=8, ha='center', va='center', color='black', backgroundcolor='orange')
        


    # --- --- --- Show average of two classes --- --- ---
    # Center of Trouser (1) and Shirt (6)
    center_trouser_shirt=(Tensor(class_centers[1]) + Tensor(class_centers[6]))/2
    ax.text(center_trouser_shirt[0].item(), center_trouser_shirt[1].item(), f"Trouser - Shirt", fontsize=8, ha='center', va='center', color='black', backgroundcolor='azure')
    
    # --- --- --- Display the plot --- --- ---
    plt.axis('on')
    plt.show()