"""
Implementation of Convolutional Neural Network (CNN) for MNIST classification
Implements MnistClassifierInterface
"""

import tensorflow as tf
from .interface import MnistClassifierInterface

class ConvolutionalMnistClassifier(MnistClassifierInterface):
    """
    A CNN architecture with:
    - Two convolutional layers
    - Max pooling
    - Batch normalization
    - Dropout regularization
    - Fully connected layers
    """

    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        """
        Initialize the CNN model architecture
        Args:
            input_shape (tuple): Shape of input images (height, width, channels)
            num_classes (int): Number of output classes (digits 0-9)
        """
        self.model = tf.keras.Sequential([
            # Explicit input layer to define the input shape
            tf.keras.layers.Input(shape=input_shape),
            
            # First convolutional layer with 32 filters and 3x3 kernel
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            
            # Batch normalization to stabilize training
            tf.keras.layers.BatchNormalization(),
            
            # Max pooling to reduce spatial dimensions
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Dropout layer to prevent overfitting
            tf.keras.layers.Dropout(0.25),
            
            # Second convolutional layer with 64 filters and 3x3 kernel
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            
            # Batch normalization
            tf.keras.layers.BatchNormalization(),
            
            # Max pooling
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Dropout layer
            tf.keras.layers.Dropout(0.25),
            
            # Flatten the output to feed into dense layers
            tf.keras.layers.Flatten(),
            
            # Fully connected layer with 256 units and ReLU activation
            tf.keras.layers.Dense(256, activation='relu'),
            
            # Batch normalization
            tf.keras.layers.BatchNormalization(),
            
            # Dropout layer
            tf.keras.layers.Dropout(0.5),
            
            # Output layer with softmax activation for classification
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model with Adam optimizer and sparse categorical crossentropy loss
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, epochs=10, batch_size=128):
        """
        Train the CNN model on the provided dataset
        Args:
            X_train (np.ndarray): Training images (n_samples, 28, 28, 1)
            y_train (np.ndarray): Training labels (n_samples,)
            epochs (int): Number of training iterations
            batch_size (int): Number of samples per gradient update
        """
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

    def predict(self, X_test):
        """
        Generate predictions for the test dataset
        Args:
            X_test (np.ndarray): Test images (n_samples, 28, 28, 1)
        Returns:
            np.ndarray: Predicted class labels (n_samples,)
        """
        # Predict class probabilities and return the class with the highest probability
        return self.model.predict(X_test).argmax(axis=1)
        