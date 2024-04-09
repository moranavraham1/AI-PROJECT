from fastai import *
from fastai.vision.all import *
from fastai.metrics import error_rate
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Set the path to the directory containing the image data
path = Path('C:\\Users\\Avraham\\Desktop\\project\\p')

# Load the image data using Fastai's ImageDataLoaders
data = ImageDataLoaders.from_folder(path, train='.', valid_pct=0.2,
                                   item_tfms=Resize(300),
                                   batch_tfms=aug_transforms(),
                                   num_workers=4,
                                   normalize=imagenet_stats)

# Display information about the training and validation datasets
print("Train:", data.train_ds)
print("Valid:", data.valid_ds)

# Show a batch of images
data.show_batch(nrows=3, figsize=(7,6))

# Print data information
print(data)
print(data.train_ds.vocab)
print(len(data.train_ds.vocab))

# Create a vision learner using a ResNet18 model
learn = vision_learner(data, models.resnet18, metrics=[accuracy], model_dir=Path('C:\\Users\\Avraham\\Desktop\\project\\p'), path=Path("."))
print(learn)

# Find an appropriate learning rate
learn.lr_find()

# Set learning rate ranges
lr1 = 1e-3
lr2 = 1e-1

# Train the model using a one-cycle learning rate schedule
learn.fit_one_cycle(40, slice(lr1, lr2))

# Unfreeze the model and train further
learn.unfreeze()
learn.fit_one_cycle(20, slice(1e-4, 1e-3))

# Validate the model and print the loss and accuracy
loss, acc = learn.validate()
print(f'Loss: {loss}, Accuracy: {acc}')

# Plot the training loss
learn.recorder.plot_loss()

# Generate and display the confusion matrix
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# Display the top losses
interp
