# Deep Learning Assignment 2 - Sudhanva S (DA24M023)

This repository is my codebase for DA6401 Deep Learning Assignment 2 and contains `dl-a2.ipynb` implementing a custom CNN (Part A) and fine-tuned VGG19 model (Part B) on the iNaturalist 12K dataset, using Wandb for hyperparameter tuning.

## Repository Structure

```
DL-Assignment2/
├── dl-a2.ipynb              # Jupyter Notebook with full implementation
├── inaturalist_12K/         # iNaturalist 12K dataset
│   ├── train/               # Training images by class
│   └── val/                 # Test images by class
├── best_model.pth           # Pre-trained weights for custom CNN
└── README.md                # This file
```

## Prerequisites

- Python
- Jupyter Notebook
- CUDA-enabled GPU (optional)
- Wandb account

## Running the Notebook

1. Open `dl-a2.ipynb`.

2. Ensure `inaturalist_12K/` contains `train/` and `val/` folders, and `best_model.pth` is in the root directory.

3. Update dataset paths in all cells where data is loaded.

### Part A: Custom CNN

- **Question 1 (Build CNN)**: Run cells 1–3 to set up W&B, dataset, and define `BasicCNN`.
- **Question 2 (Train CNN)**: Run cells 4–5 to train with W&B sweeps. Adjust `count` in cell 5 for fewer runs (e.g., `count=5`).
- **Question 3 (Insights)**: This is in the wandb report
- **Question 4 (Test CNN)**: Run cell 6 to evaluate on test dataset using `best_model.pth` and visualize predictions.

### Part B: Fine-Tune VGG19

- **Question 1 (Fine-Tune VGG19)**: Run cells 1–2, 7–9 to set up dataset, load VGG19, and fine-tune with W&B sweeps. Adjust sweep parameters in cell 9 for fewer runs.
- **Question 2 and 3**: These are in the wandb report
