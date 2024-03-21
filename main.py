from src.FactorVAE import FactorVAE, Discriminator
from src.dSpritesDataset import get_dataloaders

import argparse
import numpy as np
import os

#import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch import autograd




def permute_dims(z) : 
    '''
    Implementation of the algorithm 1 'permute_dims'
    Input: z_i for i = 1, ..., B. z_i of dimension d (here z of dimension (B, d))
    '''
    B = z.size()[0]
    d = z.size()[1]

    for j in range(d) : 
        pi = np.random.permutation(B)
        for i in range(B) : 
            z[i,j] = z[pi[i],j] #TODO : vectoriser la boucle sur i

    return z




def main(dataset_path, batch_size, save_model_path) : 
    
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

    for epoch in range(nb_epochs) : 

        print(f'========== EPOCH {epoch} ============ ')

        # Training mode
        factorvae.train()
        discriminator.train()

        epoch_vae_loss = 0
        epoch_discr_loss = 0

        for i, double_batch in enumerate(train_dataloader) : 
            if i % 100 == 0 : 
                print(i)
                print(f'Current epoch VAE loss: {epoch_vae_loss}')
                print(f'Current epoch Discr loss: {epoch_discr_loss}')

            # Split the double batch into two batches
            batch1, batch2 = torch.split(double_batch, batch_size, 0)

            # Sample z on the first batch
            y, z_mu, z_log_var = factorvae(batch1)
            z_sample = factorvae.sampling(z_mu, z_log_var)

            # Get VAE loss for the first batch
            discr_z1 = discriminator(z_sample)
            #gamma_term = torch.log(discr_z1[:,0] / discr_z1[:,1]).mean()
            gamma_term = (discr_z1[:,0] - discr_z1[:,1]).mean()
            vae_loss = factorvae.loss_function(batch1, y, z_mu, z_log_var) + gamma * gamma_term
            epoch_vae_loss += vae_loss.item()

            # # Optimization of VAE loss
            # vae_opti.zero_grad()
            # vae_loss.backward(retain_graph = True)
            # vae_opti.step()

            # Sample z on the second batch
            y2, z_mu2, z_log_var2 = factorvae(batch2)
            z_sample2 = factorvae.sampling(z_mu2, z_log_var2)

            # Permute z
            z_permuted = permute_dims(z_sample2).detach()

            # Loss of the discriminator
            discr_z2 = discriminator(z_permuted)
            #discr_z1_copy = discr_z1.clone().detach()
            discr_z1_copy = discr_z1.clone()
            discr_loss = discriminator.discr_loss(discr_z1_copy, discr_z2)
            epoch_discr_loss += discr_loss.item()

            # # Optimization of Discriminator loss
            # discr_opti.zero_grad()
            # discr_loss.backward()
            # discr_opti.step()

            # Optimization of boss losses
            vae_opti.zero_grad()
            discr_opti.zero_grad()
            total_loss = vae_loss + discr_loss
            total_loss.backward()
            vae_opti.step()
            discr_opti.step()


        epoch_vae_loss /= len(train_dataloader)
        epoch_discr_loss /= len(train_dataloader) 
        print(f'Training : VAE loss: {epoch_vae_loss:.2f}; Discriminator loss: {epoch_discr_loss:.2f}')

        # Evaluation time
        print('Testing...')
        factorvae.eval()
        discriminator.eval()
        test_epoch_vae_loss = 0
        test_epoch_discr_loss = 0

        for i, double_batch in enumerate(test_dataloader) : 
            
            if i % 100 == 0 : 
                print(i)
                print(f'Current test epoch VAE loss: {test_epoch_vae_loss}')
                print(f'Current test epoch Discr loss: {test_epoch_discr_loss}')
        
            with torch.no_grad() : 
                # Split the double batch into two batches
                test_batch1, test_batch2 = torch.split(double_batch, batch_size, 0)

                # Sample z on the first batch
                test_y, test_z_mu, test_z_log_var = factorvae(test_batch1)
                test_z_sample = factorvae.sampling(test_z_mu, test_z_log_var)

                # Get VAE loss for the first batch
                test_discr_z1 = discriminator(test_z_sample)
                test_gamma_term = (test_discr_z1[:,0] - test_discr_z1[:,1]).mean()
                test_vae_loss = factorvae.loss_function(test_batch1, test_y, test_z_mu, test_z_log_var) + gamma * test_gamma_term
                test_epoch_vae_loss += test_vae_loss.item()

                # Sample z on the second batch
                test_y2, test_z_mu2, test_z_log_var2 = factorvae(test_batch2)
                test_z_sample2 = factorvae.sampling(test_z_mu2, test_z_log_var2)

                # Permute z
                test_z_permuted = permute_dims(test_z_sample2).detach()

                # Loss of the discriminator
                test_discr_z2 = discriminator(test_z_permuted)
                test_discr_z1_copy = test_discr_z1.clone()
                test_discr_loss = discriminator.discr_loss(test_discr_z1_copy, test_discr_z2)
                test_epoch_discr_loss += test_discr_loss.item()
            
        test_epoch_vae_loss /= len(test_dataloader)
        test_epoch_discr_loss /= len(test_dataloader) 
        print(f'Test : VAE loss: {test_epoch_vae_loss:.2f}; Discriminator loss: {test_epoch_discr_loss:.2f}')

        # Save model checkpoint
        print("SAVE MODEL")
        torch.save({
            'epoch': epoch,
            'vae_state_dict': factorvae.state_dict(),
            'discr_state_dict' : discriminator.state_dict(),
            'opti_vae_state_dict': vae_opti.state_dict(),
            'opti_discr_state_dict' : discr_opti.state_dict(),
            'epoch_vae_loss': epoch_vae_loss,
            'epoch_discr_loss' : epoch_discr_loss,
            'test_epoch_vae_loss' : test_epoch_vae_loss,
            'test_epoch_discr_loss' : test_epoch_discr_loss,
        }, os.path.join(save_model_path, f"checkpoint_epoch_{epoch}.pth"))




if __name__ == '__main__' : 
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', default = 64)
    parser.add_argument('-dataset_path', default = './dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    parser.add_argument('-save_model_path', default = './models_checkpoints/factorvae')
    args = parser.parse_args()
    main(args.dataset_path, args.batch_size, args.save_model_path)