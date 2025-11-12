import torch
import os
import json
import itertools
from NNSAE_IDA2026.Implementations.DecoderComparison import run_sae_experiment, run_kl_sae_experiment

def run_hyperparameter_sweep():
    """
    Runs a hyperparameter sweep for SAE and KL-SAE models.
    """
    base_hyperparameters = {
        'input_size': 784,
        'dataset': 'mnist',
        'batch_size': 1024,
        'dead_steps_threshold': 100,
        'warmup_epochs': 10,
        'use_weight_clamping': False, # Default, will be overridden for some experiments
    }

    # Define the grid of hyperparameters to search
    param_grid = {
        'learning_rate': [0.001, 0.0005],
        'hidden_size': [256, 512],
        'k_top': [10, 20],  # For AuxiliarySAE
        'aux_k': [100],
        'auxk_coef': [1/32],
        'rho': [0.05, 0.1],  # For KLSAE
        'beta': [3.0, 5.0],   # For KLSAE
        'num_epochs': [25,50],  # Longer run
    }

    # Store all experiment configurations
    all_experiments = []

    # --- Top-k SAE Experiments ---
    sae_param_keys = ['learning_rate', 'hidden_size', 'k_top', 'aux_k', 'auxk_coef', 'num_epochs']
    sae_grid_values = [param_grid[key] for key in sae_param_keys]

    for values in itertools.product(*sae_grid_values):
        hyperparameters = base_hyperparameters.copy()
        hyperparameters.update(dict(zip(sae_param_keys, values)))

        for use_clamping in [True, False]:
            hyperparameters['use_weight_clamping'] = use_clamping

            # Create a descriptive folder name
            clamping_str = "clamping_on" if use_clamping else "clamping_off"
            folder_name = (
                f"topk_sae_lr{hyperparameters['learning_rate']}_"
                f"h{hyperparameters['hidden_size']}_k{hyperparameters['k_top']}_{clamping_str}"
            )
            experiment_folder = os.path.join("results_sweep", folder_name)
            hyperparameters['experiment_folder'] = experiment_folder

            all_experiments.append({
                'model_type': 'AuxiliarySAE',
                'hyperparameters': hyperparameters.copy()
            })

            print(f"\n--- Scheduling Top-k SAE Experiment ---")
            print(json.dumps(hyperparameters, indent=2))
            run_sae_experiment(hyperparameters, num_runs=1)

    # --- KL-Divergence SAE Experiments ---
    kl_param_keys = ['learning_rate', 'hidden_size', 'rho', 'beta', 'num_epochs']
    kl_grid_values = [param_grid[key] for key in kl_param_keys]

    for values in itertools.product(*kl_grid_values):
        hyperparameters = base_hyperparameters.copy()
        hyperparameters.update(dict(zip(kl_param_keys, values)))

        for use_clamping in [True, False]:
            hyperparameters['use_weight_clamping'] = use_clamping

            # Create a descriptive folder name
            clamping_str = "clamping_on" if use_clamping else "clamping_off"
            folder_name = (
                f"kl_sae_lr{hyperparameters['learning_rate']}_"
                f"h{hyperparameters['hidden_size']}_rho{hyperparameters['rho']}_"
                f"beta{hyperparameters['beta']}_{clamping_str}"
            )
            experiment_folder = os.path.join("results_sweep", folder_name)
            hyperparameters['experiment_folder'] = experiment_folder

            all_experiments.append({
                'model_type': 'KLSAE',
                'hyperparameters': hyperparameters.copy()
            })

            print(f"\n--- Scheduling KL-SAE Experiment ---")
            print(json.dumps(hyperparameters, indent=2))
            run_kl_sae_experiment(hyperparameters, num_runs=1)


    # Save the master list of all experiments
    os.makedirs("results_sweep", exist_ok=True)
    with open(os.path.join("results_sweep", "all_experiments.json"), 'w') as f:
        json.dump(all_experiments, f, indent=4)

    print("\n--- All hyperparameter sweep experiments complete. ---")
    print("Results are in the 'results_sweep' directory.")

if __name__ == '__main__':
    run_hyperparameter_sweep()

