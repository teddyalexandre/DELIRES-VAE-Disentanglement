from src.FactorVAE import FactorVAE, Discriminator
from src.dSpritesDataset import get_dataloaders
import argparse

import torch
from torch.optim import AdamW
import numpy as np



def permute_dims(z) : 
    '''
    Implementation of the algorithm 1 'permute_dims'
    Input: z_i for i = 1, ..., B. z_i of dimension d (here z of dimension (B, d))
    '''
    B, d = z.shape()

    for j in range(d) : 
        pi = np.random.permutation(B)
        for i in range(B) : 
            z[i,j] = z[pi[i],j] #TODO : vectoriser la boucle sur i

    return z




def main(dataset_path, batch_size) : 
    
    # Get dataloaders
    train_dataloader, test_dataloader = get_dataloaders(dataset_path, 2*batch_size)

    # Define FactorVAE 
    input_dim = 64
    h_dim1 = 32
    h_dim2 = 64
    kernel_size = (4, 4)
    stride = 2
    fc_dim = 128
    output_dim = 10
    
    factorvae = FactorVAE(input_dim, h_dim1, h_dim2, kernel_size, stride, fc_dim, output_dim)

    # Define discriminator
    hidden_dim = 1000
    output_discr = 2

    discriminator = Discriminator(output_dim, hidden_dim, output_discr)

    # Training parameters
    gamma = 0.2

    vae_lr = 1e-4
    vae_beta1 = 0.9
    vae_beta2 = 0.999
    vae_opti = AdamW(factorvae.parameters(), lr = vae_lr, betas = (vae_beta1, vae_beta2))

    discr_lr = 1e-4
    discr_beta1 = 0.5
    discr_beta2 = 0.9
    discr_opti = AdamW(discriminator.parameters(), lr = discr_lr, betas = (discr_beta1, discr_beta2))


    #Training loop
    nb_epochs = 100

    for epoch in nb_epochs : 

        print(f'========== EPOCH {epoch} ============ ')
        epoch_vae_loss = 0
        epoch_discr_loss = 0

        for double_batch in train_dataloader : 
            # Split the double batch into two batches
            batch1, batch2 = torch.split(double_batch, batch_size, 0)

            # Sample z on the first batch
            y, z_mu, z_log_var = factorvae(batch1)
            z_sample = factorvae.sampling(z_mu, z_log_var)

            # Get VAE loss for the first batch
            discr_z1 = discriminator(z_sample)
            gamma_term = torch.log(discr_z1 / (1 - discr_z1)).mean()
            vae_loss = factorvae.loss_function(batch1, y, z_mu, z_log_var) + gamma * gamma_term
            epoch_vae_loss += vae_loss.item()

            # Optimization of VAE loss
            vae_opti.zero_grad()
            vae_loss.backward()
            vae_opti.step()

            # Sample z on the second batch
            y, z_mu, z_log_var = factorvae(batch2)
            z_sample = factorvae.sampling(z_mu, z_log_var)

            # Permute z
            z_permuted = permute_dims(z_sample)

            # Loss of the discriminator
            discr_z2 = discriminator(z_permuted)
            discr_loss = discriminator.discr_loss(discr_z1, discr_z2)
            epoch_discr_loss += discr_loss.item()

            # Optimization of Discriminator loss
            discr_opti.zero_grad()
            discr_loss.backward()
            discr_opti.step()

        epoch_vae_loss /= len(train_dataloader)
        epoch_discr_loss /= len(train_dataloader) 
        print('VAE loss: {epoch_vae_loss:.2f}; Discriminator loss: {epoch_discr_loss:.2f}')



if __name__ == '__main__' : 
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', default = 64)
    parser.add_argument('-dataset_path', default = 'C:\Users\Utilisateur\Documents\MVA\DELIRES\Projet\DELIRES-VAE-Disentanglement\dsprites-dataset\dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    args = parser.parse_args()
    main(args.dataset_path, args.batch_size)