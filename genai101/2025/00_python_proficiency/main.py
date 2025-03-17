# --- --- --- Imports --- --- ---
# STD
import os
# LIB
# PROJECT
from fashion import Fashion
from classifier import Classifier, train_classifier, test_classifier, show_classification
from ae import AutoEncoder, train_ae, test_ae, show_latent_space
from tools import save_model, load_model, print_model_parameters


# --- --- --- Configuration --- --- ---


# Training configuration - shared by both classifier and AE training
BSIZE = 64                  # Batch Size: how many are sent per training step.
                            # Tip: use 128 or above for faster training while coding, but lesser training (default 64)
                            
SAMPLE_RATIO = 1            # Ratio of the train set considered.
                            # Tip: use 0.5 or belove for faster training while coding (default 1)
                            
LR = 0.001                  # Learning Rate (default 0.001): how 'fast' parameters are updated. Too high creates instability, too low impedes learning.

NEPOCHS = 10                # How many time the training set is used while learning.
                            # Tip: use less for faster training while coding (default 10)

# Feature Extractor
FEATURE_OUTPUT_DIM = 96     # Number of features extracted by the ConvFeatureExtractor module (default 96)

# AutoEncoder
ENC_HDIM = 64               # Size fo the hidden layer the Encoder (default 64)
LATENT_DIM = 2              # Dimension of the latent space (DO NOT CHANGE - we want to "see" it as an image, so 2D -- MUST BE 2)
DEC_HDIM = 128              # Size of the hidden layer in the Decoder (default 128)

# Paths
CLASSIFIER_PATH = 'classifier.pth'  # Path to save the classifier model
AE_PATH = 'ae.pth'                  # Path to save the autoencoder model

# Execution - train flags
FORCE_TRAIN_CLASSIFIER = False      # Force training of the classifier, enven if CLASSIFIER_PATH exists
FORCE_TRAIN_AE = False              # Force training of the autoencoder, enven if AE_PATH exists

# Execution - display flags. Turn off after you saw the result to prevent execution from stopping
DISPLAY_FASHION = True
DISPLAY_CLASSIFICATION = True
DISPLAY_LATENT_SPACE = True


# --- --- --- Main script --- --- ---


if __name__ == '__main__':

    # Load dataset and display samples if activated
    fashion = Fashion()
    if DISPLAY_FASHION:
        fashion.show_samples()
        exit(0)
    
    NUM_CLASSES = len(fashion.test_dataset.classes) # Number of classes in the dataset: this is 10 for FashionMNIST


    # --- --- --- Classifier --- --- ---


    # Create a blank model and try it: accuracy should be random, around 10% (1/10 classes, i.e. 1 chance out of 10)
    classifier = Classifier(feature_output_dim=FEATURE_OUTPUT_DIM, num_classes=NUM_CLASSES)
    print("Create blank classifier model.")
    print_model_parameters(classifier)
    print()

    # Train the model if does not exist or forced. Else, load it.
    if not os.path.exists(CLASSIFIER_PATH) or FORCE_TRAIN_CLASSIFIER:
        print("Test classifier before training:")
        print()
        test_classifier(classifier, fashion.get_test_loader(batch_size=BSIZE))
        # Train the model
        train_classifier(classifier, fashion, BSIZE, LR, NEPOCHS)
        save_model(classifier, CLASSIFIER_PATH)
    else:
        load_model(classifier, CLASSIFIER_PATH)
    print()

    # Test model
    test_classifier(classifier, fashion.get_test_loader(batch_size=BSIZE))
    print()
    if DISPLAY_CLASSIFICATION:
        show_classification(classifier, fashion.get_test_loader(batch_size=BSIZE, shuffle=True))
        exit(0)


    # --- --- --- AutoEncoder --- --- ---
    
    # Initialize with the same feature output dimension as the Classifier: we are going to reuse the feature extractor!
    ae = AutoEncoder(feature_ouput_dim=FEATURE_OUTPUT_DIM, latent_dim=LATENT_DIM, encoder_hdim=ENC_HDIM, decoder_hdim=DEC_HDIM)
    print("Create blank AutoEncoder model.")
    print_model_parameters(ae)
    print()
    
    # Copy the pretrained weights into ae's feature extractor and freeze them: we don't want to train the feature extractor, just use it.
    ae.feature_extractor.load_state_dict(classifier.feature_extractor.state_dict())
    for param in ae.feature_extractor.parameters():
        param.requires_grad = False
            
    # Train the model if does not exist or forced. Else, load it.
    if not os.path.exists(AE_PATH) or FORCE_TRAIN_AE:
        train_ae(ae, fashion, BSIZE, LR, NEPOCHS)
        save_model(ae, AE_PATH)
    else:
        load_model(ae, AE_PATH)
    print()

    # Test model
    test_ae(ae, fashion.get_test_loader(batch_size=BSIZE))
    print()

    # Show latent space
    if DISPLAY_LATENT_SPACE:
        show_latent_space(ae, fashion, BSIZE)
    