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
python run.py [--all] [--industry INDUSTRY] [--train-only] [--evaluate-only]
```

Arguments:
- `--all`: Process all industries (recommended)
- `--industry`: Process a specific industry
- `--train-only`: Only train models, skip evaluation
- `--evaluate-only`: Only evaluate existing models, skip training

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
- Generates visualizations in `results/visualizations/`
- Creates industry comparison charts

## Project Structure

```
├── config.py                     # Configuration settings
├── train.py                      # Script for training models
├── evaluate.py                   # Script for evaluating models and generating visualizations
├── run.py                        # Script to run the entire workflow
├── data/
│   └── data.csv                  # Input data
├── results/
│   ├── models/                   # Saved models
│   │   ├── [industry_name]/      # Industry-specific models
│   │   │   ├── lstnet_model.pth  # Saved model
│   │   │   └── scaler.joblib     # Saved scaler
│   ├── visualizations/           # Output visualizations
│   │   ├── [industry_name]/      # Industry-specific visualizations
│   │   │   └── employment_predictions.png  # Employment predictions
│   │   ├── industry_comparisons/ # Industry comparison visualizations
│   │   │   ├── industry_comparison_MSE_top10.png
│   │   │   └── industry_comparison_MAPE_top10.png
│   │   └── predictions/          # Detailed prediction visualizations
├── src/
│   ├── data/                     # Data processing modules
│   │   └── data_processor.py
│   └── visualization/            # Visualization modules
│       └── plotter.py
```

## Model Architecture

The LSTNet model combines CNN, RNN, and autoregression components to capture both short-term local dependency patterns and long-term patterns for time series forecasting of employment data.

## Generated Visualizations

1. Employment prediction plots showing actual vs. predicted values
2. Industry comparison plots showing best and worst performing industries
3. Detailed prediction plots for top performing industries

## Performance Metrics

The system calculates the following metrics for employment predictions:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

For industry comparison, a summary table is provided showing these metrics for employment across all industries.