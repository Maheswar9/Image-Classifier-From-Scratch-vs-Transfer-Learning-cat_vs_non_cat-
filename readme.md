# Cat vs. Non-Cat Image Classifier: A Two-Part Project

This repository contains two complete projects for building an image classifier to distinguish between cats and non-cats.

1.  **Part 1: Classifier from Scratch**: A logistic regression model built using only NumPy to understand the core mechanics of neural networks.
2.  **Part 2: Classifier with Transfer Learning**: A high-performance model built using Keras and a pre-trained VGG16 network to demonstrate the power of transfer learning.

---

## Part 1: Classifier from Scratch with NumPy

### Methodology
This model was built from the ground up to demonstrate the fundamental mathematics of a simple neural network.
- **Architecture**: A single-neuron logistic regression model.
- **Core Components**: All functions, including the sigmoid activation, forward propagation, backward propagation (gradient calculation), and the gradient descent optimization loop, were implemented manually using NumPy.

### Results
- **Final Test Accuracy**: 80%

This result is a strong baseline and demonstrates a functional understanding of the core learning algorithms and hyperparameter tuning.

---

## Part 2: Classifier with Transfer Learning (VGG16)

### Methodology
This model leverages a pre-trained, state-of-the-art neural network to achieve high accuracy with minimal training.
- **Base Model**: Used the VGG16 model, pre-trained on the ImageNet dataset, as a fixed feature extractor. The base model's layers were "frozen" so their weights would not update.
- **Custom Head**: A new classifier head was added on top of the frozen base, consisting of several `Dense` and `Dropout` layers.
- **Training**: Only the custom head was trained on the cat vs. non-cat dataset.

### Results
- **Final Test Accuracy**: 94%

This significant improvement over the first model highlights the efficiency and power of the transfer learning technique.

---

## Dataset

The dataset used for both projects is the `train_catvnoncat.h5` and `test_catvnoncat.h5` set, which contains 64x64 pixel color images of cats and non-cats.

## Technologies Used
* Python 3 (in a Jupyter Notebook)
* NumPy
* h5py
* TensorFlow & Keras
* Matplotlib
* Scikit-learn

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Maheswar9/Image-Classifier-From-Scratch-vs-Transfer-Learning-cat_vs_non_cat-](https://github.com/Maheswar9/Image-Classifier-From-Scratch-vs-Transfer-Learning-cat_vs_non_cat-)
    cd your-repo-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Notebooks:**
    Open the project folder in Visual Studio Code. Open the `.ipynb` file for either Part 1 or Part 2 and run the cells sequentially.
