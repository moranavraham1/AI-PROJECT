A project of image classification using 2 ai methods (cnn and svm)
# 📷 Image Classification using CNN & SVM

## 🔍 Overview
This project implements **image classification** using two powerful **AI methods**:
- **Convolutional Neural Networks (CNN) 🧠** – Deep learning-based approach.
- **Support Vector Machines (SVM) ⚡** – Traditional machine learning technique.

The goal is to compare both methods on the same dataset and analyze their performance.

## 📂 Project Structure
```
project/
├── dataset/           # Folder containing images organized by class
├── models/            # Saved trained models
├── notebooks/         # Jupyter notebooks for training & evaluation
├── src/               # Python scripts
│   ├── cnn_model.py   # CNN implementation
│   ├── svm_model.py   # SVM implementation
│   ├── preprocess.py  # Image preprocessing
│   ├── train.py       # Training script
│   ├── evaluate.py    # Model evaluation
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation 
```

## 🛠 Installation
To set up the project, follow these steps:

1️⃣ **Clone the repository:**
```bash
git clone https://github.com/yourusername/image-classification-ai.git
cd image-classification-ai
```

2️⃣ **Create a virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3️⃣ **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 📊 Dataset
Ensure your dataset is organized in the following structure:
```
dataset/
├── class1/
│   ├── img1.jpg
│   ├── img2.jpg
├── class2/
│   ├── img1.jpg
│   ├── img2.jpg
```
You can use any dataset, such as **CIFAR-10** or **custom images**.

## 🚀 Training
To train the CNN model:
```bash
python src/cnn_model.py
```

To train the SVM model:
```bash
python src/svm_model.py
```

## 🎯 Evaluation
After training, evaluate both models:
```bash
python src/evaluate.py
```
This will print accuracy, confusion matrix, and other performance metrics.

## 📈 Results
The evaluation compares CNN and SVM based on:
- Accuracy 🎯
- Precision & Recall 🔬
- Training time ⏳
- Computational cost 💻

A **comparison report** will be generated in `results/` folder.

## ✨ Example Prediction
To classify a new image:
```python
from src.cnn_model import predict_cnn
from src.svm_model import predict_svm

image_path = 'path/to/image.jpg'
print("CNN Prediction:", predict_cnn(image_path))
print("SVM Prediction:", predict_svm(image_path))
```

## 🏆 Conclusion
CNN is generally more powerful for image classification 📷🔍, but SVM is a good alternative for small datasets with fewer computational resources ⚡.



