# Analyzing Biases in Loan Approval Machine Learning Models

This repository contains code and analysis for examining biases in loan approval machine learning models trained on the Home Mortgage Disclosure Act (HMDA) dataset. This was a final project exploring how different ML models perform in terms of both accuracy and fairness when making mortgage approval decisions.

## Overview

We examine several machine learning models trained on HMDA data to predict mortgage loan approvals:
- Random Forest
- XGBoost 
- Logistic Regression
- Deep Neural Network

The analysis focuses on:
- Model performance metrics (accuracy, F1 score)
- Impact of including/excluding sensitive features like race and location
- Disparities in approval rates across different racial groups
- Feature importance and potential sources of bias

## Key Findings

- XGBoost achieved the highest predictive accuracy but showed significant bias
- Logistic Regression provided the best balance between fairness and performance
- Including race data directly didn't significantly improve model accuracy, suggesting racial information may be implicitly encoded in other features
- Financial features were most important for prediction when race was excluded, but race became the dominant feature when included

## Repository Structure

- `HDMA/`: Contains HMDA dataset files
- `data_exploration/`: Jupyter notebooks for initial data analysis
- `main.ipynb`: Main analysis and model training code for non-neural network models
- `nn_model_race.ipynb`: Neural Network (including race feature) training and evaluation code
- `nn_model.ipynb`: Neural Network (no race feature) training and evaluation code
- `saved_output_images/`: Visualizations and results
- `*.npy` files: saved outputs from neural network code that are then used by `main.ipynb` for analysis.

## Requirements

The code uses the following main libraries:
- Python
- Pandas 
- Scikit-learn
- XGBoost
- PyTorch
- Matplotlib

## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/rgg1/aidms-loan-approval.git
cd aidms-loan-approval
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download the HMDA dataset:
   - Visit the HMDA website to download the 2017 dataset
   - Place the downloaded files in the `/HDMA/` directory
   - Required files (can add more if you want to explore more than just the 2017 data):
     - hmda_2017_transmittal_sheet.csv
     - hmda_2017_panel.csv
     - hmda_2017_nationwide_all-records_labels.csv (in `.gitignore` due to size)

## Running the Code

1. Data Exploration:
```bash
jupyter notebook data_exploration/data_Exploration.ipynb
```
This notebook contains initial data analysis and visualizations of the HMDA dataset.

2. Train Models:

The project includes two main model training notebooks:
```bash
jupyter notebook nn_model.ipynb        # Neural Network training
jupyter notebook nn_model_race.ipynb   # Neural Network with race features
jupyter notebook main.ipynb            # Main analysis and model training code
```

3. View Results:
- Model performance metrics and visualizations will be output in Jupyter Notebooks, I chose to save the important ones to `saved_output_images`.
- Key outputs include:
  - Feature importance plots
  - Approval rate disparities across races
  - Model accuracy comparisons
  - F1 score comparisons

## Model Configurations

- Random Forest: 100 binary classification trees
- XGBoost: 100 trees with gradient descent optimization
- Neural Network: 4 hidden layers (max 1024 neurons)
- Logistic Regression: Standard configuration with logistic activation

The non-neural network models were compared after being trained on three feature sets:
- Baseline (financial features only)
- Baseline + Location
- Baseline + Location + Race
