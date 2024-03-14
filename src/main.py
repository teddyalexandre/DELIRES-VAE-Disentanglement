from src.FactorVAE import FactorVAE, Discriminator
from src.dSpritesDataset import get_dataloaders
import argparse

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
    train_dataloader, test_dataloader = get_dataloaders(dataset_path, batch_size)

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
        for batch in train_dataloader : 
            mu, log_var = factorvae.encode(batch)
            pass # TODO



















if __name__ == '__main__' : 
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', default = 64)
    parser.add_argument('-dataset_path', default = 'C:\Users\Utilisateur\Documents\MVA\DELIRES\Projet\DELIRES-VAE-Disentanglement\dsprites-dataset\dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    args = parser.parse_args()
    main(args.dataset_path, args.batch_size)