import streamlit as st
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🎗️",
    layout="centered"
)

# -----------------------------
# Load and prepare the dataset
# Same data source used in the notebook
# -----------------------------
@st.cache_resource
def load_model():
    breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

    data_frame = pd.DataFrame(
        breast_cancer_dataset.data,
        columns=breast_cancer_dataset.feature_names
    )
    data_frame["label"] = breast_cancer_dataset.target

    X = data_frame.drop("label", axis=1)
    Y = data_frame["label"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2
    )

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Same neural-network architecture as the supplied notebook
    tf.random.set_seed(3)
    model = keras.Sequential([
        keras.layers.Input(shape=(30,)),
        keras.layers.Dense(20, activation="relu"),
        keras.layers.Dense(2, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Same training settings as the notebook
    model.fit(
        X_train_std,
        Y_train,
        validation_split=0.1,
        epochs=10,
        verbose=0
    )

    loss, accuracy = model.evaluate(X_test_std, Y_test, verbose=0)
    return model, scaler, accuracy


# -----------------------------
# UI
# -----------------------------
st.title("🎗️ Breast Cancer Detection")
st.write(
    "Enter the tumor measurement values below to predict whether "
    "the tumor is **Benign** or **Malignant**."
)

st.info(
    "This is an educational machine-learning project and is not a medical diagnosis."
)

feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

default_values = [
    11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657,
    0.01115, 0.1495, 0.05888, 0.4062, 1.21, 2.635, 28.47,
    0.005857, 0.009758, 0.01168, 0.007445, 0.02406, 0.001769,
    12.98, 25.72, 82.98, 516.5, 0.1085, 0.08615, 0.05523,
    0.03715, 0.2433, 0.06563
]

st.subheader("Enter Tumor Measurements")

# Keep the same simple multi-column style as the IPL predictor,
# while grouping the 30 inputs so the page remains usable.
groups = [
    ("Mean Measurements", 0, 10),
    ("Standard Error Measurements", 10, 20),
    ("Worst Measurements", 20, 30),
]

values = []

for title, start, end in groups:
    st.markdown(f"### {title}")
    cols = st.columns(2)

    for i in range(start, end):
        with cols[(i - start) % 2]:
            value = st.number_input(
                feature_names[i].title(),
                value=float(default_values[i]),
                format="%.6f",
                key=f"feature_{i}"
            )
            values.append((i, value))

if st.button("🔍 Predict", use_container_width=True):
    model, scaler, accuracy = load_model()

    # Restore original feature order
    input_data = [value for _, value in sorted(values)]

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Same standardization step as the notebook
    input_data_std = scaler.transform(input_data_reshaped)

    prediction = model.predict(input_data_std, verbose=0)

    # Same argmax approach as the notebook
    prediction_label = int(np.argmax(prediction[0]))

    malignant_probability = float(prediction[0][0])
    benign_probability = float(prediction[0][1])

    st.divider()
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Malignant", f"{malignant_probability * 100:.2f}%")

    with col2:
        st.metric("Benign", f"{benign_probability * 100:.2f}%")

    st.write("### Model Prediction")

    if prediction_label == 0:
        st.error("⚠️ The model predicts: **Malignant**")
    else:
        st.success("✅ The model predicts: **Benign**")

    st.progress(max(malignant_probability, benign_probability))

    st.caption(
        f"Test accuracy on the notebook's 20% test split: "
        f"{accuracy * 100:.2f}%"
    )

    st.caption(
        "Prediction probabilities are model outputs and should not be "
        "interpreted as clinical certainty."
    )

with st.expander("About this model"):
    st.write(
        "This application follows the supplied Breast_Cancer_Detection.ipynb "
        "workflow: the scikit-learn Breast Cancer dataset is loaded, split "
        "with test_size=0.2 and random_state=2, standardized using "
        "StandardScaler, and classified using a Keras neural network."
    )
    st.code(
        """Flatten/Input: 30 features
Dense: 20 neurons, ReLU
Dense: 2 neurons, Sigmoid
Optimizer: Adam
Loss: sparse_categorical_crossentropy
Epochs: 10
Random seed: 3"""
    )
