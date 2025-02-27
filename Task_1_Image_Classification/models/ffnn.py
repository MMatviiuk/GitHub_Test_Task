"""
Implementation of Feed-Forward Neural Network (FFNN) for MNIST classification
Implements MnistClassifierInterface
"""

import tensorflow as tf
from .interface import MnistClassifierInterface

class FeedForwardMnistClassifier(MnistClassifierInterface):
    """
    A simple feed-forward neural network with:
    - Input layer
    - Two hidden layers with ReLU activation
    - Output layer with softmax activation
    """

    def __init__(self, input_shape=(28, 28), num_classes=10):
        """
        Initialize the FFNN model architecture
        Args:
            input_shape (tuple): Shape of input images (height, width)
            num_classes (int): Number of output classes (digits 0-9)
        """
        self.model = tf.keras.Sequential([
            # Explicit input layer to define the input shape
            tf.keras.layers.Input(shape=input_shape),
            
            # Flatten the 2D image into a 1D vector
            tf.keras.layers.Flatten(),
            
            # First hidden layer with 128 units and ReLU activation
            tf.keras.layers.Dense(128, activation='relu'),
            
            # Second hidden layer with 64 units and ReLU activation
            tf.keras.layers.Dense(64, activation='relu'),
            
            # Output layer with softmax activation for classification
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model with Adam optimizer and sparse categorical crossentropy loss
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, epochs=5, batch_size=32):
        """
        Train the FFNN model on the provided dataset
        Args:
            X_train (np.ndarray): Training images (n_samples, 28, 28)
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
            X_test (np.ndarray): Test images (n_samples, 28, 28)
        Returns:
            np.ndarray: Predicted class labels (n_samples,)
        """
        # Predict class probabilities and return the class with the highest probability
        return self.model.predict(X_test).argmax(axis=1)
        