"""
Random Forest implementation for MNIST classification
Implements MnistClassifierInterface
"""
from sklearn.ensemble import RandomForestClassifier
from .interface import MnistClassifierInterface

class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    Random Forest classifier for MNIST dataset
    Implements sklearn's RandomForestClassifier under the hood
    """
    
    def __init__(self, n_estimators=300, random_state=42):
        """
        Initialize Random Forest model
        Args:
            n_estimators: Number of decision trees in the forest
            random_state: Seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
    
    def train(self, X_train, y_train):
        """
        Train Random Forest model
        Args:
            X_train: Flattened images (n_samples, 784)
            y_train: Target labels (n_samples,)
        """
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        """
        Generate class predictions
        Args:
            X_test: Flattened test images (n_samples, 784)
        Returns:
            Predicted labels (n_samples,)
        """
        return self.model.predict(X_test)
        