"""
Main script for training industry-specific LSTNet models.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import config
from src.data.data_processor import load_data, create_sequences, split_data, save_scaler, get_industries
from OLD_FILES.LSTNet import LSTNet
import time

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the LSTNet model.
    
    Args:
        model (LSTNet): The model to train
        X_train (torch.Tensor): Training input sequences
        y_train (torch.Tensor): Training target sequences
        X_val (torch.Tensor): Validation input sequences
        y_val (torch.Tensor): Validation target sequences
        
    Returns:
        tuple: (train_losses, val_losses)
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAINING_CONFIG['learning_rate'])
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config.TRAINING_CONFIG['epochs']):
        # Training phase
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), config.TRAINING_CONFIG['batch_size']):
            batch_X = X_train[i:i+config.TRAINING_CONFIG['batch_size']]
            batch_y = y_train[i:i+config.TRAINING_CONFIG['batch_size']]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        
        # Record losses
        train_losses.append(total_loss / len(X_train))
        val_losses.append(val_loss.item())
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{config.TRAINING_CONFIG["epochs"]}], '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}')
    
    return train_losses, val_losses

def save_model(model, feature_names, industry=None):
    """
    Save the trained model.
    
    Args:
        model (LSTNet): The trained model
        feature_names (list): List of feature names
        industry (str, optional): Industry name for industry-specific models
    """
    models_dir = Path(config.DATA_CONFIG['models_dir'])
    
    # If industry-specific, save in industry-specific folder
    if industry is not None:
        models_dir = models_dir / industry
    
    models_dir.mkdir(exist_ok=True, parents=True)
    
    model_path = models_dir / 'lstnet_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_names': feature_names
    }, model_path)
    print(f"Model saved to {model_path}")

def train_industry_model(industry=None):
    """
    Train a model for a specific industry or an overall model.
    
    Args:
        industry (str, optional): Industry to train on. If None, trains on all data.
        
    Returns:
        tuple: (model, feature_names) - The trained model and feature names
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Industry info for display
    industry_display = industry if industry else "All Industries"
    print(f"\n{'='*50}")
    print(f"Training model for: {industry_display}")
    print(f"{'='*50}")
    
    # Step 1: Load and preprocess data
    print(f"\nStep 1: Loading and preprocessing data for {industry_display}...")
    dates, scaled_data, scaler, feature_names = load_data(industry=industry)
    print(f"Dataset shape: {scaled_data.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Step 2: Create sequences
    print("\nStep 2: Creating sequences...")
    X, y = create_sequences(scaled_data)
    print(f"Sequence shape X: {X.shape}, y: {y.shape}")
    
    # Step 3: Split data
    print("\nStep 3: Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Step 4: Initialize model
    print("\nStep 4: Initializing model...")
    # Set the num_features dynamically
    model = LSTNet(
        num_features=scaled_data.shape[1],
        device=device
    )
    
    # Step 5: Train model
    print("\nStep 5: Training model...")
    train_losses, val_losses = train_model(model, X_train, y_train, X_val, y_val)
    
    # Step 6: Save model and scaler
    print("\nStep 6: Saving model and scaler...")
    save_model(model, feature_names, industry)
    save_scaler(scaler, feature_names, industry)
    
    print(f"\nTraining complete for {industry_display}! Model has been saved.")
    return model, feature_names

def main():
    start_time = time.time()
    
    # Ensure models directory exists
    models_dir = Path(config.DATA_CONFIG['models_dir'])
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Get list of all industries
    industries = get_industries()
    print(f"Found {len(industries)} industries: {industries}")
    
    # Ask user if they want to train all models or a specific one
    print("\nOptions:")
    print("1. Train a separate model for each industry")
    print("2. Train a model for a specific industry")
    print("3. Train one combined model with all data")
    
    try:
        choice = int(input("Enter your choice (1/2/3): "))
    except ValueError:
        choice = 1  # Default to option 1
        
    if choice == 1:
        # Train a model for each industry
        print(f"\nTraining separate models for {len(industries)} industries...")
        
        for i, industry in enumerate(industries):
            print(f"\nIndustry {i+1}/{len(industries)}: {industry}")
            
            try:
                train_industry_model(industry)
            except Exception as e:
                print(f"Error training model for industry {industry}: {str(e)}")
            
    elif choice == 2:
        # Train a model for a specific industry
        print("\nAvailable industries:")
        for i, industry in enumerate(industries):
            print(f"{i+1}. {industry}")
            
        try:
            industry_idx = int(input(f"Enter industry number (1-{len(industries)}): ")) - 1
            if 0 <= industry_idx < len(industries):
                train_industry_model(industries[industry_idx])
            else:
                print("Invalid selection. Training for all industries.")
                train_industry_model()
        except ValueError:
            print("Invalid input. Training for all industries.")
            train_industry_model()
    else:
        # Train a single model with all data
        train_industry_model()
    
    end_time = time.time()
    print(f"\nTotal training time: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main() 