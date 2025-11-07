# 🛰️ Automated Crop Disease Diagnosis from Hyperspectral Imagery

> A complete end-to-end project to diagnose crop diseases (Healthy vs. Diseased) from hyperspectral image (HSI) patches using a 3D Convolutional Neural Network (CNN) built with PyTorch.

This project includes data simulation, preprocessing, model training, evaluation, and a simple web-based interface for inference.

---

## 🚀 Features

* **HSI Data Simulation:** Includes a script to generate a synthetic dataset of HSI cubes and their corresponding disease masks.
* **3D-CNN Model:** Implements a 3D-CNN in PyTorch, which learns both spatial (shape) and spectral (reflectance) features.
* **Training Pipeline:** A full training and validation pipeline that saves the best-performing model.
* **Evaluation:** Generates a classification report (Accuracy, Precision, Recall, F1-score) and a visual confusion matrix.
* **Inference Script:** A command-line script to diagnose a new, unseen HSI cube and generate a visual disease map.
* **Web Interface:** A simple, interactive web app built with Streamlit to upload an HSI file and get an instant diagnosis.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Deep Learning:** PyTorch
* **Data Science:** NumPy, Scikit-learn, Pandas
* **Image/Data Processing:** Scikit-image, SciPy
* **Visualization:** Matplotlib, Seaborn
* **Web App:** Streamlit

---

## 📁 Project Structure
