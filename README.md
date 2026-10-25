# 🎗️ Breast Cancer Detection

A machine learning-based web application built with **Python and Streamlit** to classify breast tumors as **malignant or benign** using a neural network.

## 📌 Project Overview

This project converts the original `Breast_Cancer_Detection.ipynb` notebook into an interactive Streamlit web application. It follows the notebook's data preprocessing, neural network architecture, and prediction workflow.

The application uses the Breast Cancer Wisconsin dataset available through Scikit-learn.

## ⚙️ Model Workflow

The application follows these steps:

1. **Dataset Loading:** Loads the dataset using `sklearn.datasets.load_breast_cancer()`.
2. **Data Preparation:** Converts the dataset into a DataFrame and prepares the target labels.
3. **Train-Test Split:** Splits the dataset using:

   * `test_size=0.2`
   * `random_state=2`
4. **Feature Scaling:** Standardizes all 30 input features using `StandardScaler`.
5. **Neural Network Architecture:**

   * Input layer: 30 features
   * Hidden layer: 20 neurons with ReLU activation
   * Output layer: 2 neurons with sigmoid activation
6. **Model Compilation:** Uses the Adam optimizer and sparse categorical cross-entropy loss.
7. **Model Training:** Trains for 10 epochs with `validation_split=0.1`.
8. **Prediction:** Uses `argmax` to select the predicted class:

   * `0` — Malignant
   * `1` — Benign

## 🛠️ Technologies Used

* **Python** — Core programming language
* **Streamlit** — Interactive web application
* **TensorFlow/Keras** — Neural network development and training
* **Scikit-learn** — Dataset loading, feature scaling, and data splitting
* **Pandas** — Data manipulation
* **NumPy** — Numerical computations
* **Jupyter Notebook** — Model experimentation and development

## 📂 Project Structure

```text
Breast-Cancer-Detection/
├── app.py
├── Breast_Cancer_Detection.ipynb
├── data.csv
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

* Python installed on your system
* pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/Samuel-dev67/Breast-Cancer-Detection.git
```

### 2. Navigate to the Project Directory

```bash
cd Breast-Cancer-Detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application trains the neural network on its first run and caches the model for the current Streamlit session.

## 📊 Dataset Information

The application loads its training dataset directly from:

`sklearn.datasets.load_breast_cancer()`

Although `data.csv` is included in the repository, it is not used as the application's training data. The CSV is retained unchanged for reference.

## ⚠️ Important Disclaimer

This project is intended for **educational and demonstration purposes only**. Its predictions are not a substitute for professional medical diagnosis, laboratory testing, or advice from a qualified healthcare professional.

## 🔮 Future Improvements

* Evaluate the model using precision, recall, F1-score, and a confusion matrix.
* Improve model performance through hyperparameter tuning.
* Add visualizations for prediction probabilities.
* Save the trained model to avoid retraining between application sessions.
* Improve the interface and input validation.

## 👨‍💻 Author

**Samuel Selvaraj**

GitHub: [@Samuel-dev67](https://github.com/Samuel-dev67)

---

⭐ If you find this project interesting, consider starring the repository!
