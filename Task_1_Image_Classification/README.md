# MNIST Classification Project

## **Project Overview**

This project demonstrates image classification on the MNIST dataset using three different algorithms:

- **Random Forest Classifier** (using `scikit-learn`)
- **Feed-Forward Neural Network (FFNN)** (using `TensorFlow`)
- **Convolutional Neural Network (CNN)** (using `TensorFlow`)

All models follow a unified interface `MnistClassifierInterface`, ensuring consistency across implementations.

---

## **Project Structure**

```
├── models/             # Model implementations
│   ├── interface.py    # Abstract interface (train & predict methods)
│   ├── random_forest.py # Random Forest model
│   ├── ffnn.py         # Feed-Forward NN model
│   └── cnn.py          # Convolutional NN model
├── notebook/           # Jupyter Notebook with examples and edge cases
├── README.md           # Project documentation (this file)
└── requirements.txt    # Dependencies list
```

---

## **Installation**

1. **Clone the repository:**

```bash
git clone <repository_link>
cd <repository_folder>
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

---

## **Usage**

1. **Load the dataset and initialize the classifier:**

```python
from models import MnistClassifier
from tensorflow.keras.datasets import mnist

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Initialize classifier (options: 'cnn', 'rf', 'nn')
classifier = MnistClassifier(algorithm='cnn')

# Train and predict
classifier.train(X_train, y_train)
predictions = classifier.predict(X_test)
```

2. **Switching algorithms:**

```python
classifier_rf = MnistClassifier(algorithm='rf')  # Random Forest
classifier_nn = MnistClassifier(algorithm='nn')  # Feed-Forward Neural Network
```

---

## 🧪 **Models Implemented**

| Algorithm              | Accuracy | Key Features                       |
| ---------------------- | -------- | ---------------------------------- |
| Random Forest (RF)     | 97.1%    | Sklearn-based, no GPU requirement  |
| Feed-Forward NN (FFNN) | 97.5%    | Dense layers with ReLU activations |
| Convolutional NN (CNN) | 99.3%    | Convolutions, BatchNorm, Dropout   |

---

## **Edge Cases & Considerations**

- Input validation for empty datasets.
- Handling incorrect input shapes.
- Parameter tuning (e.g., epochs, batch size).
- Model fallback if GPU is unavailable.

---

## **Key Highlights**

- Unified interface for all models.
- Automatic data preprocessing.
- Visual prediction examples.
- Performance metrics included.
- Full type hints for IDE support.

---

## **Testing & Validation**

- Run provided Jupyter Notebook for step-by-step demonstrations.
- Compare performance across models using provided accuracy metrics.

---
