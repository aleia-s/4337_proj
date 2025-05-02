"""
Main script to run the entire workflow: training models and evaluating them.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import config
import os

def ensure_directories():
    for dir_path in [config.DATA_CONFIG['models_dir'], config.DATA_CONFIG['visualizations_dir']]:
        Path(dir_path).mkdir(exist_ok=True, parents=True)

def train_models(industry=None, all_industries=False):
    print("\n=== Training Models ===")
    # Add current directory to PYTHONPATH so scripts can find the config module
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [sys.executable, "src/engine/train.py"]
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        return False
    
    return True

def evaluate_models(industry=None, all_industries=False):
    print("\n=== Evaluating Models ===")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [sys.executable, "src/engine/evaluate.py"]
    
    if all_industries:
        cmd.append("--all")
    elif industry:
        cmd.extend(["--industry", industry])
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
        return False
    
    return True

def evaluate_baselines(industry=None, all_industries=False):
    print("\n=== Evaluating Baseline Models ===")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [sys.executable, "src/baseline/evaluate_baselines.py"]
    
    if all_industries:
        cmd.append("--all")
    elif industry:
        cmd.extend(["--industry", industry])
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error during baseline evaluation: {e}")
        return False
    
    return True

def predict_models(industry=None, all_industries=False, top_n=5):
    print("\n=== Making Predictions ===")
    # Add current directory to PYTHONPATH so scripts can find the config module
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [sys.executable, "src/engine/predict.py"]
    
    if all_industries:
        cmd.extend(["--all", "--top-n", str(top_n)])
    elif industry:
        cmd.extend(["--industry", industry])
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error during prediction: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run the full workflow: train models and evaluate them')
    parser.add_argument('--industry', type=str, help='Industry to process', default=None)
    parser.add_argument('--all', action='store_true', help='Process all industries')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate existing models')
    parser.add_argument('--predict', action='store_true', help='Run predictions on existing models')
    parser.add_argument('--baseline', action='store_true', help='Evaluate baseline models')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top industries to predict in comparison mode')
    
    args = parser.parse_args()
    
    # Create necessary directories
    ensure_directories()
    
    # Check if at least one action is specified
    if not any([args.train, args.evaluate, args.predict, args.baseline]):
        parser.error("At least one of --train, --evaluate, --predict, or --baseline must be specified")
    
    # Run the requested steps
    success = True
    
    if args.train:
        success = train_models(args.industry, args.all)
        if not success:
            print("Training failed. Aborting workflow.")
            return
    
    if args.evaluate and success:
        success = evaluate_models(args.industry, args.all)
        if not success:
            print("Evaluation failed.")
            return
    
    if args.baseline and success:
        success = evaluate_baselines(args.industry, args.all)
        if not success:
            print("Baseline evaluation failed.")
            return
    
    if args.predict and success:
        success = predict_models(args.industry, args.all, args.top_n)
        if not success:
            print("Prediction failed.")
            return
    
    if success:
        print("\n=== Workflow completed successfully ===")
        print(f"Results can be found in:")
        print(f"  - Models: {config.DATA_CONFIG['models_dir']}")
        print(f"  - Visualizations: {config.DATA_CONFIG['visualizations_dir']}")
        if args.baseline:
            print(f"  - Baseline Results: src/baseline/baseline_results")

if __name__ == "__main__":
    main() 