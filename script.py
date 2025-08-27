# In this script we will be training a classification model to classify images of an asset into 2 categories: 
# 1. Normal
# 2. Corroded
# The model will be trained on a dataset of images of the asset.
# THe machine learning model that will be used is a Convolutional Neural Network (CNN).
# CNN is used because it is good at image classification.
# The dataset will be a collection of images of the asset.
# The dataset is already split into training and test sets. The dataset is in the folder "dataset".
# The training set is in the folder "dataset/train" and the test set is in the folder "dataset/test".
# The images for training are in the folder "dataset/train/normal" and "dataset/train/corroded".
# The images for testing are in the folder "dataset/test/normal" and "dataset/test/corroded".
# The images will be resized to 224x224 pixels.
# The model will be trained for 10 epochs.
# The model will be saved to a file.
# The model will be loaded from the file and used to classify new images.
# The model will be evaluated on a test set of images.
# After the model is trained, we will use it to classify new images.
# Model Performance will be evaluated using the test set.
# Performance will be evaluated using the accuracy, F1 score, recall, ROC AUC score, and precision of the model.
# In the end, the trained model will be wrapped in a Flask API. The API will be able to classify new images, with the following endpoints:
# 1. /classify/
# The API will receive a POST request with the image in the body of the request.
# The API will return the classification of the image.
# The API will return the probability of the image being corroded.
# The API will return the probability of the image being normal.
