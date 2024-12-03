# UNET_Brain_Tumor_Detection

This project implements a deep learning model for **brain tumor segmentation** using the **U-Net architecture**. The model is trained and evaluated on the **BRATS2020 dataset**, which contains multi-modal MRI scans. The goal is to accurately detect and segment brain tumors from these images.

## Key Highlights:
- **Dataset**: Utilized the **BRATS2020 dataset** for training and evaluation, which includes high-resolution MRI scans with labeled ground truth data for tumor regions.
- **Model Architecture**: Implemented a **U-Net** architecture designed for pixel-level segmentation tasks. U-Net is known for its high performance in biomedical image segmentation tasks.
- **Performance**: Achieved **96% accuracy** and **84% Mean Union Overlap (MUO)** for tumor segmentation, making it a highly accurate model for brain tumor detection.

## Features:
- **MRI Preprocessing**: The project includes preprocessing steps such as skull stripping, intensity normalization, and resizing to fit the model input requirements.
- **Model Training**: The U-Net model is trained with techniques like **data augmentation**, **batch normalization**, and **early stopping** to prevent overfitting and improve model generalization.
- **Evaluation**: The model's performance is evaluated using common metrics such as accuracy and Mean Union Overlap (MUO).

## Technologies Used:
- **Python**
- **TensorFlow/Keras**
- **NumPy**
- **OpenCV**
- **Matplotlib**

## Requirements:
To run this project, ensure you have the following installed:
- Python 3.x
- TensorFlow >= 2.0
- Keras
- NumPy
- OpenCV
- Matplotlib

## Installation:

1. Clone the repository:
   ```bash
   git clone https://github.com/amnydv17/UNET_Brain_Tumor_Detection.git
