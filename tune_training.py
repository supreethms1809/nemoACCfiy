#!/usr/bin/env python3
"""
Lightning Tuner for ModularModel Training
Optimizes hyperparameters for maximum training performance
"""

import os
import sys
import yaml
import optuna
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingTuner:
    def __init__(self, base_config_path: str = "configs/config.yaml"):
        self.base_config_path = base_config_path
        self.best_config = None
        self.best_score = float('inf')
        self.trial_results = []
        
    def load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration"""
        with open(self.base_config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def save_trial_config(self, config: Dict[str, Any], trial_number: int) -> str:
        """Save configuration for a trial"""
        trial_config_path = f"configs/trial_{trial_number}_config.yaml"
        with open(trial_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return trial_config_path
    
    def run_training_trial(self, config_path: str) -> float:
        """Run a single training trial and return validation loss"""
        try:
            # Run training command
            cmd = [
                "python", "train.py",
                "--mode", "production",
                "--model_config", "model_config_1B",
                "--stage", "stage1",
                "--use_processed_datasets",
                "--config", config_path
            ]
            
            logger.info(f"Running trial with config: {config_path}")
            start_time = time.time()
            
            # Run training and capture output
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            training_time = time.time() - start_time
            
            if result.returncode != 0:
                logger.error(f"Training failed: {result.stderr}")
                return float('inf')
            
            # Parse validation loss from output
            val_loss = self.parse_validation_loss(result.stdout)
            
            # Calculate score (lower is better)
            # Combine validation loss and training speed
            it_per_sec = self.parse_training_speed(result.stdout)
            score = val_loss + (1.0 / max(it_per_sec, 0.1))  # Penalize slow training
            
            logger.info(f"Trial completed - Val Loss: {val_loss:.4f}, Speed: {it_per_sec:.2f} it/s, Score: {score:.4f}")
            
            return score
            
        except subprocess.TimeoutExpired:
            logger.error("Training trial timed out")
            return float('inf')
        except Exception as e:
            logger.error(f"Trial failed with error: {e}")
            return float('inf')
    
    def parse_validation_loss(self, output: str) -> float:
        """Parse validation loss from training output"""
        lines = output.split('\n')
        for line in lines:
            if 'val_loss=' in line:
                try:
                    # Extract val_loss value
                    val_loss_str = line.split('val_loss=')[1].split(',')[0]
                    return float(val_loss_str)
                except:
                    continue
        return 10.0  # Default high loss if not found
    
    def parse_training_speed(self, output: str) -> float:
        """Parse training speed from output"""
        lines = output.split('\n')
        for line in lines:
            if 'it/s' in line:
                try:
                    # Extract it/s value
                    speed_str = line.split('it/s')[0].split()[-1]
                    return float(speed_str)
                except:
                    continue
        return 0.1  # Default slow speed if not found
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function"""
        # Load base config
        config = self.load_base_config()
        
        # Define search space
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
        sequence_length = trial.suggest_categorical("sequence_length", [512, 1024, 2048])
        num_workers = trial.suggest_categorical("num_workers", [8, 16, 32])
        prefetch_factor = trial.suggest_categorical("prefetch_factor", [2, 4, 8])
        cpu_offload = trial.suggest_categorical("cpu_offload", [True, False])
        
        # Update config
        stage1_config = config['training_stages']['stage1']
        stage1_config['learning_rate'] = learning_rate
        stage1_config['batch_size'] = batch_size
        stage1_config['sequence_length'] = sequence_length
        stage1_config['num_workers'] = num_workers
        stage1_config['prefetch_factor'] = prefetch_factor
        stage1_config['fsdp']['cpu_offload'] = cpu_offload
        
        # Adjust gradient accumulation to maintain effective batch size
        if batch_size == 2:
            stage1_config['gradient_accumulation_steps'] = 4
        elif batch_size == 4:
            stage1_config['gradient_accumulation_steps'] = 2
        else:  # batch_size == 8
            stage1_config['gradient_accumulation_steps'] = 1
        
        # Save trial config
        trial_number = trial.number
        config_path = self.save_trial_config(config, trial_number)
        
        # Run training trial
        score = self.run_training_trial(config_path)
        
        # Store results
        self.trial_results.append({
            'trial': trial_number,
            'params': trial.params,
            'score': score,
            'config_path': config_path
        })
        
        # Update best config
        if score < self.best_score:
            self.best_score = score
            self.best_config = config.copy()
            logger.info(f"New best score: {score:.4f}")
        
        # Clean up trial config
        try:
            os.remove(config_path)
        except:
            pass
        
        return score
    
    def run_optimization(self, n_trials: int = 10):
        """Run the optimization process"""
        logger.info(f"Starting optimization with {n_trials} trials")
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            study_name="modular_model_tuning",
            storage="sqlite:///tuning_results.db",
            load_if_exists=True
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        # Save results
        self.save_results(study)
        
        return study
    
    def save_results(self, study: optuna.Study):
        """Save optimization results"""
        # Save best config
        if self.best_config:
            best_config_path = "configs/best_tuned_config.yaml"
            with open(best_config_path, 'w') as f:
                yaml.dump(self.best_config, f, default_flow_style=False)
            logger.info(f"Best configuration saved to: {best_config_path}")
        
        # Save trial results
        results_path = "tuning_results.yaml"
        with open(results_path, 'w') as f:
            yaml.dump(self.trial_results, f, default_flow_style=False)
        logger.info(f"Trial results saved to: {results_path}")
        
        # Print summary
        logger.info("=" * 50)
        logger.info("OPTIMIZATION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Number of trials: {len(study.trials)}")
        
        # Print top 3 trials
        logger.info("\nTop 3 trials:")
        sorted_trials = sorted(study.trials, key=lambda t: t.value)
        for i, trial in enumerate(sorted_trials[:3]):
            logger.info(f"{i+1}. Score: {trial.value:.4f}, Params: {trial.params}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tune ModularModel training parameters")
    parser.add_argument("--trials", type=int, default=10, help="Number of optimization trials")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Base config file")
    
    args = parser.parse_args()
    
    # Create tuner
    tuner = TrainingTuner(args.config)
    
    # Run optimization
    study = tuner.run_optimization(args.trials)
    
    logger.info("Optimization completed!")

if __name__ == "__main__":
    main()
