Deep Learning Assignment 2 - CNN and Transfer Learning
This repository contains a Jupyter Notebook (dl-a2.ipynb) implementing a custom CNN and fine-tuned VGG19 model on the iNaturalist 12K dataset, with Weights & Biases for hyperparameter tuning.
Repository Structure
DL-Assignment2/
├── dl-a2.ipynb              # Jupyter Notebook with implementation
├── inaturalist_12K/         # iNaturalist 12K dataset
│   ├── train/               # Training images by class
│   └── val/                 # Test images by class
├── best_model.pth           # Pre-trained weights for custom CNN
└── README.md                # This file

Prerequisites

Python 3.8+
Jupyter Notebook
CUDA-enabled GPU (optional)
Weights & Biases account

Installation

Clone the repository:git clone https://github.com/your-username/DL-Assignment2.git
cd DL-Assignment2


Create and activate a virtual environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install torch torchvision wandb pillow numpy tqdm matplotlib


Log in to Weights & Biases:wandb login



Running the Notebook

Start Jupyter Notebook:jupyter notebook


Open dl-a2.ipynb.
Ensure inaturalist_12K/ contains train/ and val/ folders, and best_model.pth is in the root directory.
Update dataset paths in cells 4, 6, and 8 if not using /kaggle/input/inaturalist12k/inaturalist_12K/.

Part A: Custom CNN

Question 1 (Build CNN): Run cells 1–3 to set up W&B, dataset, and define BasicCNN.
Question 2 (Train CNN): Run cells 4–5 to train with W&B sweeps. Adjust count in cell 5 for fewer runs (e.g., count=5).
Question 4 (Test CNN): Run cell 6 to evaluate on test dataset using best_model.pth and visualize predictions.

Part B: Fine-Tune VGG19

Question 1 (Fine-Tune VGG19): Run cells 1–2, 7–9 to set up dataset, load VGG19, and fine-tune with W&B sweeps. Adjust sweep parameters in cell 9 for fewer runs.

Note: Use a GPU for faster training. Monitor results on wandb.ai under the DL_A2 project.
