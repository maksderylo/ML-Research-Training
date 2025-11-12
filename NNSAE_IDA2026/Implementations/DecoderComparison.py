import torch
from torch import nn
import os
import json
import numpy as np
from sklearn.decomposition import NMF

from NNSAE_IDA2026.AuxiliarySAE import AuxiliarySparseAutoencoder
from NNSAE_IDA2026.DataLoader import load_data
from NNSAE_IDA2026.SAEAuxTraining import train_sparse_autoencoder
from NNSAE_IDA2026.ModelVisualizations import SAEVisualizer, visualize_nmf_results
from NNSAE_IDA2026.KLSAE import KLSAE
from NNSAE_IDA2026.KLSAETraining import train_kl_sae


def run_sae_experiment(hyperparameters: dict, num_runs: int = 3):
    """
    Runs a single training and visualization experiment for a given set of hyperparameters.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clamping_str = "clamping_on" if hyperparameters['use_weight_clamping'] else "clamping_off"
    folder = hyperparameters['experiment_folder']
    os.makedirs(folder, exist_ok=True)

    print(f"\n--- Running SAE Experiment: {clamping_str.replace('_', ' ').title()} ({num_runs} runs) ---")
    print(f"Results will be saved in: `{folder}`")

    # --- Save Hyperparameters ---
    with open(os.path.join(folder, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    all_training_data = []
    last_trained_model = None

    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        # --- Model & Data ---
        model = AuxiliarySparseAutoencoder(
            input_size=hyperparameters['input_size'],
            hidden_size=hyperparameters['hidden_size'],
            k_top=hyperparameters['k_top'],
            aux_k=hyperparameters['aux_k'],
            dead_steps_threshold=hyperparameters['dead_steps_threshold'],
            activation=nn.ReLU()
        ).to(device)

        # For SAE, we use normalized data
        train_loader, test_loader = load_data(
            hyperparameters['dataset'],
            batch_size=hyperparameters['batch_size'],
            normalize=True
        )

        # --- Training ---
        trained_model, training_data = train_sparse_autoencoder(
            train_loader=train_loader,
            dataset_type=hyperparameters['dataset'],
            model=model,
            num_epochs=hyperparameters['num_epochs'],
            learning_rate=hyperparameters['learning_rate'],
            use_weight_clamping=hyperparameters['use_weight_clamping'],
            auxk_coef=hyperparameters['auxk_coef']
        )
        all_training_data.append(training_data)
        last_trained_model = trained_model

    # --- Visualization (using the last model for illustrative plots) ---
    print("\n--- Generating Visualizations ---")
    visualizer = SAEVisualizer(last_trained_model, device=device)

    # 1. Aggregated Training History
    history_path = os.path.join(folder, "1_aggregated_training_history.png")
    visualizer.plot_aggregated_training_history(
        all_training_data,
        save_path=history_path,
        hyperparameters=hyperparameters
    )

    # 2. Reconstructions (from last run)
    recons_path = os.path.join(folder, "2_reconstructions.png")
    visualizer.visualize_reconstructions(
        test_loader,
        hyperparameters['dataset'],
        num_samples=10,
        save_path=recons_path
    )

    # 3. Decoder Weights (from last run, with clustering)
    weights_path = os.path.join(folder, "3_decoder_weights.png")
    visualizer.visualize_decoder_weights(
        num_features=hyperparameters['hidden_size'],
        save_path=weights_path,
        cluster_by_similarity=True,
        dendrogram_path=os.path.join(folder, "3a_decoder_dendrogram.png")
    )

    # 4. Activation Statistics (from last run)
    activations_path = os.path.join(folder, "4_activation_histogram.png")
    visualizer.plot_activation_histogram(
        test_loader,
        hyperparameters['dataset'],
        save_path=activations_path
    )

    # 5. Top Activating Examples (from last run)
    stats = visualizer.get_activation_statistics(test_loader, hyperparameters['dataset'])
    if stats['active_features'] > 0:
        top_5_feature_indices = np.argsort(stats['activation_counts'])[-5:][::-1].tolist()
        visualizer.visualize_top_activating_examples(
            data_loader=test_loader,
            dataset_type=hyperparameters['dataset'],
            feature_indices=top_5_feature_indices,
            top_k=8,
            save_path=os.path.join(folder, "5_top_activating_examples.png")
        )
    else:
        print("Skipping top activating examples plot as there were no active features.")

    # 6. Create Summary Dashboard
    summary_path = os.path.join(folder, "summary_dashboard.png")
    visualizer.create_summary_dashboard(
        output_path=summary_path,
        history_path=history_path,
        recons_path=recons_path,
        weights_path=weights_path,
        activations_path=activations_path,
        hyperparameters=hyperparameters
    )

    print(f"✓ SAE Experiment for {clamping_str} complete.")


def run_nmf_experiment(hyperparameters: dict):
    """
    Runs an NMF experiment as a baseline.
    """
    folder = experiment_folder+"nmf_baseline/"
    os.makedirs(folder, exist_ok=True)

    print(f"\n--- Running NMF Baseline Experiment ---")
    print(f"Results will be saved in: `{folder}`")

    # --- Data (NMF requires non-negative, non-normalized data) ---
    train_loader_nmf, test_loader_nmf = load_data(
        hyperparameters['dataset'],
        batch_size=10000,  # Load a large chunk for fitting
        normalize=False
    )

    # Extract training data into a single matrix
    train_data_nmf = next(iter(train_loader_nmf))[0].numpy()

    # --- NMF Model ---
    n_components = hyperparameters['hidden_size']
    nmf = NMF(
        n_components=n_components,
        init='random',
        random_state=42,
        max_iter=500,
        tol=1e-3,
        verbose=True
    )
    print(f"Fitting NMF with {n_components} components...")
    nmf.fit(train_data_nmf)

    # --- Visualization ---
    visualize_nmf_results(
        nmf_model=nmf,
        data_loader=test_loader_nmf,
        folder=folder,
        n_samples=10
    )
    print(f"✓ NMF Experiment complete.")


def run_kl_sae_experiment(hyperparameters: dict, num_runs: int = 1):
    """
    Runs a single training and visualization experiment for a KL-divergence SAE.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder = hyperparameters['experiment_folder']
    os.makedirs(folder, exist_ok=True)

    print(f"\n--- Running KL-SAE Experiment ({num_runs} runs) ---")
    if hyperparameters.get('use_weight_clamping', False):
        print("Using Decoder Weight Clamping.")
    else:
        print("Not using Decoder Weight Clamping.")
    print(f"Results will be saved in: `{folder}`")

    with open(os.path.join(folder, 'hyperparameters.json'), 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    all_training_data = []
    last_trained_model = None

    for i in range(num_runs):
        print(f"\n--- Run {i+1}/{num_runs} ---")
        model = KLSAE(
            input_size=hyperparameters['input_size'],
            hidden_size=hyperparameters['hidden_size'],
            rho=hyperparameters['rho'],
            beta=hyperparameters['beta'],
            activation = nn.ReLU()
        ).to(device)

        train_loader, test_loader = load_data(
            hyperparameters['dataset'],
            batch_size=hyperparameters['batch_size'],
            normalize=False
        )

        trained_model, training_data = train_kl_sae(
            train_loader=train_loader,
            dataset_type=hyperparameters['dataset'],
            model=model,
            num_epochs=hyperparameters['num_epochs'],
            learning_rate=hyperparameters['learning_rate'],
            warmup_epochs=hyperparameters.get('warmup_epochs', 0),
            use_weight_clamping=hyperparameters.get('use_weight_clamping', False)
        )
        all_training_data.append(training_data)
        last_trained_model = trained_model

    # --- Visualization (using the last model for illustrative plots) ---
    print("\n--- Generating Visualizations for KL-SAE ---")
    visualizer = SAEVisualizer(last_trained_model, device=device)

    # 1. Aggregated Training History
    history_path = os.path.join(folder, "1_aggregated_training_history.png")
    visualizer.plot_aggregated_training_history(
        all_training_data,
        save_path=history_path,
        hyperparameters=hyperparameters
    )

    # 2. Reconstructions
    recons_path = os.path.join(folder, "2_reconstructions.png")
    # The KL SAE was trained on non-normalized data, but for a fair visual comparison
    # of reconstructions, we should show it trying to reconstruct normalized data.
    _, test_loader_normalized = load_data(
        hyperparameters['dataset'],
        batch_size=hyperparameters['batch_size'],
        normalize=True
    )
    visualizer.visualize_reconstructions(
        test_loader_normalized,
        hyperparameters['dataset'],
        num_samples=10,
        save_path=recons_path
    )

    # 3. Decoder Weights (with clustering)
    weights_path = os.path.join(folder, "3_decoder_weights.png")
    visualizer.visualize_decoder_weights(
        num_features=hyperparameters['hidden_size'],
        save_path=weights_path,
        cluster_by_similarity=True,
        dendrogram_path=os.path.join(folder, "3a_decoder_dendrogram.png")
    )

    # 4. Activation Statistics
    activations_path = os.path.join(folder, "4_activation_histogram.png")
    # Use the original test loader (non-normalized) that the model was trained on
    visualizer.plot_activation_histogram(
        test_loader,
        hyperparameters['dataset'],
        save_path=activations_path
    )

    # 5. Top Activating Examples
    # Use the original test loader (non-normalized)
    stats = visualizer.get_activation_statistics(test_loader, hyperparameters['dataset'])
    if stats['active_features'] > 0:
        top_5_feature_indices = np.argsort(stats['activation_counts'])[-5:][::-1].tolist()
        visualizer.visualize_top_activating_examples(
            data_loader=test_loader,
            dataset_type=hyperparameters['dataset'],
            feature_indices=top_5_feature_indices,
            top_k=8,
            save_path=os.path.join(folder, "5_top_activating_examples.png")
        )
    else:
        print("Skipping top activating examples plot as there were no active features.")

    # 6. Create Summary Dashboard
    summary_path = os.path.join(folder, "summary_dashboard.png")
    visualizer.create_summary_dashboard(
        output_path=summary_path,
        history_path=history_path,
        recons_path=recons_path,
        weights_path=weights_path,
        activations_path=activations_path,
        hyperparameters=hyperparameters
    )

    clamping_status = "clamping_on" if hyperparameters.get('use_weight_clamping', False) else "clamping_off"
    print(f"✓ KL-SAE Experiment for {clamping_status} complete.")


if __name__ == '__main__':
    # --- Base Hyperparameters ---
    experiment_folder = "resultskl/mnist_comparison/klfix-200hiddensize-10ktop-rho005-beta3/"

    base_hyperparameters = {
        'input_size': 784,
        'hidden_size': 200,
        'k_top': 10,
        'aux_k': 100,
        'auxk_coef': 1/32,
        'dead_steps_threshold': 100,
        'dataset': 'mnist',
        'batch_size': 1024,
        'num_epochs': 1,
        'learning_rate': 0.001,
        'experiment_folder': experiment_folder,
        'rho': 0.05,
        'beta': 3.0,
        'warmup_epochs': 10
    }
    #
    # --- Experiment 1: SAE with Decoder Weight Clamping OFF ---
    clamping_off_params = base_hyperparameters.copy()
    clamping_off_params['use_weight_clamping'] = False
    run_sae_experiment(clamping_off_params, num_runs=1)
    #
    # --- Experiment 2: SAE with Decoder Weight Clamping ON ---
    clamping_on_params = base_hyperparameters.copy()
    clamping_on_params['use_weight_clamping'] = True
    run_sae_experiment(clamping_on_params, num_runs=1)

    # --- Experiment 3: KL-SAE with clamping ---
    kl_sae_clamping_on_params = base_hyperparameters.copy()
    kl_sae_clamping_on_params['use_weight_clamping'] = True
    run_kl_sae_experiment(kl_sae_clamping_on_params, num_runs=1)

    # --- Experiment 4: KL-SAE without clamping ---
    kl_sae_clamping_off_params = base_hyperparameters.copy()
    kl_sae_clamping_off_params['use_weight_clamping'] = False
    run_kl_sae_experiment(kl_sae_clamping_off_params, num_runs=1)

    # # --- Experiment 5: NMF Baseline ---
    # run_nmf_experiment(base_hyperparameters)

    print("\n--- All experiments complete. ---")
    print(f"Compare results in `{experiment_folder}`")
