# Automated Textile Defect Detection using CNN

## Overview
This project aims to automate the detection of fabric defects in textile manufacturing using a Convolutional Neural Network (CNN) built with the TensorFlow Keras Sequential API.  
Traditionally, defect detection in fabric production is done manually, which is time-consuming and prone to human error. Our system classifies fabric patches as “good” or “damaged” directly from camera images, improving consistency and production efficiency.

## Project Goal
The objective is to:
- Train a CNN to identify fabric defects from image patches.  
- Use the MVTec Anomaly Detection (MVTec AD) dataset for supervised learning.  
- Mitigate dataset imbalance and overfitting using data augmentation and class weighting.  
- Lay the foundation for multi-class classification of specific defect types in later phases.

## Dataset
- **Source:** [MVTec Anomaly Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/belkhirnacim/textiledefectdetection)  
- **Structure:** Fabric images split into training and test sets, containing six categories:  
  `good`, `color`, `cut`, `hole`, `thread`, `metal_contamination`.  
- **Simplification:** For the baseline, all defect types are merged into a single “damaged” class.  
- **Augmentation:** Each patch (32×32 or 64×64) is rotated at eight fixed angles for rotation-robust learning.

## Model Architecture
- Built with TensorFlow Keras Sequential API.  
- Basic CNN structure with convolution, pooling, and fully connected layers.  
- Evaluation metrics: accuracy, precision, recall, F1-score.  
- Planned enhancements:
  - Data augmentation for balancing.  
  - Regularization to reduce overfitting.  
  - Exploration of deeper or transfer-learning models such as ResNet or VGG.

## Experimental Evaluation
Baseline experiments achieved high overall accuracy but a decent F1-score for the minority “good” class due to imbalance.  
The next phase focuses on balancing techniques and model optimization to ensure fair and robust performance across both classes.

## How to Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/Chuck953/MachineLearning-UVM.git
   cd MachineLearning-UVM
   ```
2. Download and extract the dataset:  
   [Textile Defect Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/belkhirnacim/textiledefectdetection)
3. Edit the `path_to_data` variable in the notebook to point to your dataset folder.  
4. Run the Jupyter Notebook to train and evaluate the CNN.

## Contributors
- Marko Tatic  
- James Bouchat  
- Alexander  
- Kyrylo Kolesnichenko  
