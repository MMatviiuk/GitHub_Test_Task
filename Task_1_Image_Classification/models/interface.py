"""
Abstract Base Class (ABC) definition for MNIST classifiers
Ensures consistent interface across all models
"""
from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    """
    Interface contract for MNIST classification models
    All concrete implementations must adhere to this interface
    """
    
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train model on provided dataset
        Args:
            X_train: Training data features (numpy array)
            y_train: Training labels (numpy array)
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Generate predictions for input data
        Args:
            X_test: Test data features (numpy array)
        Returns:
            Array of predicted labels
        """
        pass