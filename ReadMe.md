# LSTNet Industry-Specific Employment Forecasting

This project implements the LSTNet (Long- and Short-term Time-series Network) model for time series forecasting on industry-specific data, with a focus on predicting employment across different industries.

## Overview

The system allows you to:
1. Train separate models for each industry
2. Train a model for a specific industry
3. Train one combined model with all data
4. Evaluate models and generate visualizations
5. Compare employment predictions across different industries

## Data Structure

The data is expected to be in a CSV file with the following structure:
- `date`: Date column
- `industry`: Industry identifier
- `y`: Target variable (employment)
- Additional features can be included as needed

## Setup

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure your data is in the correct format and saved as `data/data.csv`

## Workflow

The workflow is separated into training and evaluation steps:

### Quick Start

For convenience, you can run the entire workflow using:

```
python run.py [--all] [--industry INDUSTRY] [--train] [--evaluate] [--predict]
```

Arguments:
- `--all`: Process all industries (recommended)
- `--industry`: Process a specific industry
- `--train`: Train models
- `--evaluate`: Evaluate existing models
- `--predict`: Generate predictions

### Step 1: Training Models

To train the models separately:

```
python train.py
```

This script:
- Loads and preprocesses the data
- Creates sequences for training
- Trains the LSTNet model
- Saves the trained models to `results/models/[industry_name]/`

You'll be presented with the following options:
1. Train a separate model for each industry
2. Train a model for a specific industry
3. Train one combined model with all data

### Step 2: Evaluating Models

After training, evaluate the models with:

```
python evaluate.py [--all] [--industry INDUSTRY]
```

Arguments:
- `--all`: Evaluate all industries and create comparison visualizations
- `--industry`: Evaluate a specific industry

This script:
- Loads the trained models
- Makes predictions on the data
- Calculates performance metrics
- Generates visualizations in `visualizations/`
- Creates industry comparison charts

## Model Architecture

The LSTNet model combines CNN, RNN, and autoregression components to capture both short-term local dependency patterns and long-term patterns for time series forecasting of employment data.