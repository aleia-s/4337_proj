# Unemployment Rate Prediction with LSTNet

## Setup

1. Install the required packages:
   pip install -r requirements.txt

## Training the Model

1. Run the training script:
   python train.py

   Running this will:
   - Load the unemployment data
   - Train the LSTNet model and save to models/

Evaluating the Model

1. To evaluate on the same data:
   python evaluate.py

2. To evaluate on different data:
   python evaluate.py --data_file your_data.csv