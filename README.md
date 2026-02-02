# DeepTrace

Deepfake Detection using Deep Learning

DeepTrace is a deep learning–based project focused on detecting deepfake videos using computer vision techniques and neural networks. The project leverages the Deepfake Detection Challenge (DFDC) sample dataset, OpenCV for video processing, and PyTorch for building and experimenting with deep learning models.

The implementation is provided as a Jupyter Notebook and is designed to run smoothly on Google Colab.

# Overview

With the rise of synthetic media, detecting manipulated videos has become increasingly important. DeepTrace explores how deep learning can be applied to identify deepfake content by processing video data and preparing it for model training and evaluation.

Repository Structure
DeepTrace/
├── DeepTrace.ipynb
└── README.md

# Technologies Used

Python 3

PyTorch & TorchVision

OpenCV

NumPy

Matplotlib

Google Colab

# Dataset

This project uses the Deepfake Detection Challenge (DFDC) – Sample Dataset, which includes:

Real and fake video samples (.mp4)

Metadata describing video authenticity

Separate training and testing videos

The dataset is downloaded and extracted automatically within the notebook.

# Setup & Usage
Run on Google Colab (Recommended)

Open the notebook in Google Colab and run the cells sequentially. All required dependencies are installed inside the notebook.

Run Locally

Clone the repository:

git clone https://github.com/Abinav-Kishore/DeepTrace.git
cd DeepTrace


Install dependencies:

pip install opencv-python matplotlib torch torchvision kaggle


Launch the notebook:

jupyter notebook DeepTrace.ipynb
