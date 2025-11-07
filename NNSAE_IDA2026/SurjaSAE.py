# Surja Chaudhuri (TU/e)
# The code is written in collaboration and based on the work of Frits Schalij (NXP)


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JumpReLU(torch.nn.Module):
    def __init__(self, jump):
        super(JumpReLU, self).__init__()
        self.jump = jump

    def forward(self, x):
        return F.relu(x - self.jump)


class Bias(nn.Module):
    def __init__(self, input_shape):
        super(Bias, self).__init__()

        # Learnable bias initialized to zeros
        self.bias = nn.Parameter(torch.zeros(input_shape[-1]))

    def forward(self, x):
        return x + self.bias


class BiasTranspose(nn.Module):
    def __init__(self, bias):
        super(BiasTranspose, self).__init__()

        # Non-Learnable bias initialized to zeros
        self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return x - self.bias


class SparseAutoencoder(nn.Module):
    def __init__(self, input_shape, *args, **kwargs):
        super(SparseAutoencoder, self).__init__()

        self.remove_bias = Bias(input_shape)

        self.encoder = nn.Linear(input_dim, 8 * input_dim, bias=True)
        init.kaiming_normal_(self.encoder.weight, mode='fan_in',
                             nonlinearity='relu')  # in ternsorflow it is kernel_initializer='he_normal',
        self.activation = JumpReLU(jump=d_relu)

        self.decoder = nn.Linear(8 * input_dim, input_dim, bias=True)
        init.kaiming_normal_(self.decoder.weight, mode='fan_in',
                             nonlinearity='relu')  # in ternsorflow it is kernel_initializer='he_normal',

        self.add_bias = BiasTranspose(self.remove_bias.bias.detach())

    def forward(self, x):
        input = self.remove_bias(x)
        concepts = self.activation(self.encoder(input))
        x_prime = self.decoder(concepts)
        output = self.add_bias(x_prime)

        return output, concepts


def initialise_sae(sae, data_median):
    dummy_input = (torch.zeros((1, input_dim))).to(device)
    sae(dummy_input)
    print(sae)

    with torch.no_grad():
        sae.remove_bias.bias.copy_(torch.from_numpy(-data_median))

    encoder_weights = sae.encoder.weight.data

    with torch.no_grad():
        sae.decoder.weight.copy_(encoder_weights.T)
        sae.decoder.bias.copy_(torch.from_numpy(data_median))

    return sae


def train_one_epoch(model, labda, total_L0_loss):
    encoder_losses, layer_losses, image_concepts = [], [], []

    total_bins = np.zeros(Nbins)

    pbar = tqdm(total=Nbatches, position=0, leave=True)
    for step in range(Nbatches):
        ''' *** Training loop begins for batch *** '''
        optimizer.zero_grad()
        start, end = step * batch_size, min((step + 1) * batch_size, Nactivations)
        x_batch = torch.from_numpy(activations[indices[start:end]]).to(device)

        batch_count = x_batch.shape[0]
        y, layer = model(x_batch)

        encoder_loss_L2 = torch.sqrt(torch.sum(
            (x_batch - y) ** 2)) / batch_count  ## similar to L2 but not exactly, euclidean distance between vectors
        layer_loss_L1 = torch.sum(torch.abs(layer))  # L1 norm of tensor
        loss_value = encoder_loss_L2 + (labda * layer_loss_L1)
        active_concepts = (layer > 0)

        loss_value.backward()
        optimizer.step()
        ''' *** Training loop ends for batch *** '''

        ''' mathematical computations after training '''

        layer_np = layer.cpu().detach()
        active_neurons_bool = np.where(layer_np > 0)[1]
        active_concepts_np = active_concepts.cpu().detach().numpy()
        Nconcepts_per_image = np.sum(active_concepts_np, axis=1)

        concept_bins, _ = np.histogram(Nconcepts_per_image, bins=Nbins, range=[0, Nbins], density=False)

        encoder_losses.append(encoder_loss_L2.cpu().detach().numpy())
        layer_losses.append(layer_loss_L1.cpu().detach().numpy())

        total_L0_loss = total_L0_loss.union(set(active_neurons_bool))
        total_bins += concept_bins

        pbar.update()

    return encoder_losses, layer_losses, total_L0_loss, total_bins


def main():
    global labda, d_relu, Nbins, num_epochs, batch_size, encoder_epochs, epochs_encoder_losses, epochs_layer_losses, epochs_total_losses, epochs_L0_losses, epochs_bins

    global Nactivations, input_dim, indices, Nbatches, activations, optimizer, activations_median, all_activations, total_L0_loss

    labda = 3e-4  # 3e-7
    d_relu = 0.4  # 0.37
    num_epochs = 40
    Nbins = 100  # 20 is too small
    batch_size = 64  # 16, 32, 64 etc...
    encoder_epochs = 10
    # threshold = 0.3

    # Load dataset: activations derived from CNN last conv layer after pooling
    data = np.load("input/imagenette_train_predictions.npy")

    all_activations = []
    for item in data:
        all_activations.append(item)

    print((np.array(all_activations)).shape)  # shape --> (10000, 1280)

    total_L0_loss = set()
    print("initialised")
    activations = np.array(all_activations)
    Nactivations, size = activations.shape
    input_dim = (size)
    indices = np.arange(Nactivations)

    activations_median = np.median(activations, axis=0)
    activations_min = np.min(activations, axis=0)

    ### Plotting a boxplot to see activations ###

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=activations_median)
    plt.title("Boxplot of activations_median - Pytorch")
    plt.xlabel("Data")
    plt.ylabel("Values")

    plt.savefig("output/Pytorch activations imagenette")

    ### -------------------------------------- ###

    print(activations_median)

    Nbatches = ((Nactivations - 1) // batch_size) + 1

    epochs_encoder_losses, epochs_layer_losses, epochs_total_losses, epochs_L0_losses, epochs_bins = [], [], [], [], []

    sae = SparseAutoencoder(activations_median.shape).to(device)
    sae_initialised = initialise_sae(sae, activations_median)

    optimizer = torch.optim.Adam(
        sae_initialised.parameters(),
        lr=0.001,  # Default learning rate
        eps=1e-07,  # Matching tensorflow behaviour
        weight_decay=0  # No weight decay by default
    )

    for epoch in range(num_epochs):
        print('Start of epoch %d' % (epoch,))

        epoch_labda = 0 if epoch < encoder_epochs else labda

        encoder_losses, layer_losses, l0_loss, epoch_bins = train_one_epoch(sae_initialised, epoch_labda, total_L0_loss)

        encoder_losses_mean = np.mean(encoder_losses)
        layer_losses_mean = np.mean(layer_losses)
        total_losses_mean = encoder_losses_mean + epoch_labda * layer_losses_mean

        epochs_encoder_losses.append(encoder_losses_mean)
        epochs_layer_losses.append(layer_losses_mean)
        epochs_total_losses.append(total_losses_mean)

        epochs_L0_losses.append(len(l0_loss))
        epochs_bins.append(epoch_bins)

        print(
            f'\n Epoch {epoch}: encoder loss {encoder_losses_mean:.3f}, layer loss: {layer_losses_mean:.4f}, total loss: {total_losses_mean:.5f}, # active neurons: {len(l0_loss)}')

    return sae_initialised  # returning the model for later use


if __name__ == '__main__':
    main()