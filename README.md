**Task 1: Image Classification + OOP**

This task focuses on building three classification models using the MNIST dataset:
Random Forest
Feed-Forward Neural Network
Convolutional Neural Network
Each model is implemented as a separate class following the MnistClassifierInterface, which includes train and predict methods. The models are encapsulated within the MnistClassifier class, which takes an algorithm name (cnn, rf, nn) as input and provides predictions using a unified interface.

More details can be found in the Task_1_Image_Classification folder.

**Task 2: Named Entity Recognition + Image Classification**

This task involves building an ML pipeline that combines Natural Language Processing (NLP) and Computer Vision to verify textual descriptions of images.

Process:
A user provides a text input like "There is a cow in the picture." along with an image.
The pipeline determines whether the description matches the image and returns True or False.
To achieve this, the pipeline consists of:

A NER model to extract animal names from text (using a transformer-based approach).
An image classification model trained on a dataset with at least 10 animal classes.

More details can be found in the Task_2_Image_Classification folder.
