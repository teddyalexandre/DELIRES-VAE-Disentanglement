from src.FactorVAE import FactorVAE, Discriminator
from src.dSpritesDataset import get_dataloaders
from src.train_factorvae import train
from src.test_factorvae import test
from src.utils import load_parameters

import argparse
import numpy as np
import os

#import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch import autograd


def main(params, device) : 


    # Get dataloaders
    train_dataloader, test_dataloader = get_dataloaders(params['dataset_path'], 2*params['batch_size'])

    #Define FactorVAE
    factorvae = FactorVAE(params['factorvae']['input_dim'],
                          params['factorvae']['h_dim1'],
                          params['factorvae']['h_dim2'], 
                          (params['factorvae']['kernel_size'], params['factorvae']['kernel_size']),
                          params['factorvae']['stride'],
                          params['factorvae']['fc_dim'],
                          params['factorvae']['output_dim'],
                          device).to(device)

    # Define discriminator
    discriminator = Discriminator(device, 
                                  params['discr']['input_dim'],
                                  params['discr']['hidden_dim'],
                                  params['discr']['output_dim']
                                  ).to(device)
    
    # Define optimizers
    vae_opti = AdamW(factorvae.parameters(), 
                     lr = params['factorvae']['vae_lr'],
                     betas = (params['factorvae']['vae_beta1'], params['factorvae']['vae_beta2']))
    
    discr_opti = AdamW(discriminator.parameters(), 
                       lr = params['discr']['discr_lr'],
                       betas = (params['discr']['discr_beta1'], params['discr']['discr_beta2']))
    

    for epoch in range(params['nb_epochs']) : 
        print(f'========== EPOCH {epoch} ============ ')

        # Training
        train_vae_loss, train_discr_loss = train(factorvae, 
                                                discriminator,
                                                vae_opti,
                                                discr_opti,
                                                train_dataloader,
                                                params['gamma'],
                                                params['batch_size'], 
                                                device)
        print(f'TRAINING : VAE loss: {train_vae_loss:.2f}; Discriminator loss: {train_discr_loss:.2f}')

        # Testing 
        test_vae_loss, test_discr_loss = test(factorvae, 
                                              discriminator, 
                                              test_dataloader, 
                                              params['gamma'],
                                              params['batch_size'], 
                                              device)
        
        print(f'TESTING : VAE loss: {test_vae_loss:.2f}; Discriminator loss: {test_discr_loss:.2f}')

        # Save model checkpoint

        print("SAVE MODEL")
        torch.save({
            'epoch': epoch,
            'vae_state_dict': factorvae.state_dict(),
            'discr_state_dict' : discriminator.state_dict(),
            'opti_vae_state_dict': vae_opti.state_dict(),
            'opti_discr_state_dict' : discr_opti.state_dict(),
            'epoch_vae_loss': train_vae_loss,
            'epoch_discr_loss' : train_discr_loss,
            'test_epoch_vae_loss' : test_vae_loss,
            'test_epoch_discr_loss' : test_discr_loss,
        }, os.path.join(params['save_model_path'], f"checkpoint_epoch_{epoch}.pth"))



# def main(params) : 
    
#     # Get dataloaders
#     train_dataloader, test_dataloader = get_dataloaders(dataset_path, 2*batch_size)
    
#     # Define FactorVAE
#     factorvae = FactorVAE(input_dim, h_dim1, h_dim2, kernel_size, stride, fc_dim, output_dim, device).to(device)

#     # Define discriminator
#     discriminator = Discriminator(device, output_dim, hidden_dim, output_discr).to(device)

#     # Define optimizers
#     vae_opti = AdamW(factorvae.parameters(), lr = vae_lr, betas = (vae_beta1, vae_beta2))
#     discr_opti = AdamW(discriminator.parameters(), lr = discr_lr, betas = (discr_beta1, discr_beta2))


#     #Training loop

#     for epoch in range(nb_epochs) : 

#         print(f'========== EPOCH {epoch} ============ ')

#         # Training mode
#         factorvae.train()
#         discriminator.train()

#         epoch_vae_loss = 0
#         epoch_discr_loss = 0

#         for i, double_batch in enumerate(train_dataloader) : 
#             if i % 100 == 0 : 
#                 print(i)
#                 print(f'Current epoch VAE loss: {epoch_vae_loss}')
#                 print(f'Current epoch Discr loss: {epoch_discr_loss}')

#             # Split the double batch into two batches
#             double_batch = double_batch.to(device)
#             batch1, batch2 = torch.split(double_batch, batch_size, 0)

#             # Sample z on the first batch
#             y, z_mu, z_log_var = factorvae(batch1)
#             z_sample = factorvae.sampling(z_mu, z_log_var)

#             # Get VAE loss for the first batch
#             discr_z1 = discriminator(z_sample)
#             #gamma_term = torch.log(discr_z1[:,0] / discr_z1[:,1]).mean()
#             gamma_term = (discr_z1[:,0] - discr_z1[:,1]).mean()
#             vae_loss = factorvae.loss_function(batch1, y, z_mu, z_log_var) + gamma * gamma_term
#             epoch_vae_loss += vae_loss.item()

#             # # Optimization of VAE loss
#             # vae_opti.zero_grad()
#             # vae_loss.backward(retain_graph = True)
#             # vae_opti.step()

#             # Sample z on the second batch
#             y2, z_mu2, z_log_var2 = factorvae(batch2)
#             z_sample2 = factorvae.sampling(z_mu2, z_log_var2)

#             # Permute z
#             z_permuted = permute_dims(z_sample2).detach()

#             # Loss of the discriminator
#             discr_z2 = discriminator(z_permuted)
#             #discr_z1_copy = discr_z1.clone().detach()
#             discr_z1_copy = discr_z1.clone()
#             discr_loss = discriminator.discr_loss(discr_z1_copy, discr_z2)
#             epoch_discr_loss += discr_loss.item()

#             # # Optimization of Discriminator loss
#             # discr_opti.zero_grad()
#             # discr_loss.backward()
#             # discr_opti.step()

#             # Optimization of boss losses
#             vae_opti.zero_grad()
#             discr_opti.zero_grad()
#             total_loss = vae_loss + discr_loss
#             total_loss.backward()
#             vae_opti.step()
#             discr_opti.step()


#         epoch_vae_loss /= len(train_dataloader)
#         epoch_discr_loss /= len(train_dataloader) 
#         print(f'Training : VAE loss: {epoch_vae_loss:.2f}; Discriminator loss: {epoch_discr_loss:.2f}')

#         # Evaluation time
#         print('Testing...')
#         factorvae.eval()
#         discriminator.eval()
#         test_epoch_vae_loss = 0
#         test_epoch_discr_loss = 0

#         for i, test_double_batch in enumerate(test_dataloader) : 
            
#             if i % 100 == 0 : 
#                 print(i)
#                 print(f'Current test epoch VAE loss: {test_epoch_vae_loss}')
#                 print(f'Current test epoch Discr loss: {test_epoch_discr_loss}')
        
#             with torch.no_grad() : 
#                 # Split the double batch into two batches
#                 test_double_batch = test_double_batch.to(device)
#                 test_batch1, test_batch2 = torch.split(test_double_batch, batch_size, 0)

#                 # Sample z on the first batch
#                 test_y, test_z_mu, test_z_log_var = factorvae(test_batch1)
#                 test_z_sample = factorvae.sampling(test_z_mu, test_z_log_var)

#                 # Get VAE loss for the first batch
#                 test_discr_z1 = discriminator(test_z_sample)
#                 test_gamma_term = (test_discr_z1[:,0] - test_discr_z1[:,1]).mean()
#                 test_vae_loss = factorvae.loss_function(test_batch1, test_y, test_z_mu, test_z_log_var) + gamma * test_gamma_term
#                 test_epoch_vae_loss += test_vae_loss.item()

#                 # Sample z on the second batch
#                 test_y2, test_z_mu2, test_z_log_var2 = factorvae(test_batch2)
#                 test_z_sample2 = factorvae.sampling(test_z_mu2, test_z_log_var2)

#                 # Permute z
#                 test_z_permuted = permute_dims(test_z_sample2).detach()

#                 # Loss of the discriminator
#                 test_discr_z2 = discriminator(test_z_permuted)
#                 test_discr_z1_copy = test_discr_z1.clone()
#                 test_discr_loss = discriminator.discr_loss(test_discr_z1_copy, test_discr_z2)
#                 test_epoch_discr_loss += test_discr_loss.item()
            
#         test_epoch_vae_loss /= len(test_dataloader)
#         test_epoch_discr_loss /= len(test_dataloader) 
#         print(f'Test : VAE loss: {test_epoch_vae_loss:.2f}; Discriminator loss: {test_epoch_discr_loss:.2f}')

#         # Save model checkpoint
#         print("SAVE MODEL")
#         torch.save({
#             'epoch': epoch,
#             'vae_state_dict': factorvae.state_dict(),
#             'discr_state_dict' : discriminator.state_dict(),
#             'opti_vae_state_dict': vae_opti.state_dict(),
#             'opti_discr_state_dict' : discr_opti.state_dict(),
#             'epoch_vae_loss': epoch_vae_loss,
#             'epoch_discr_loss' : epoch_discr_loss,
#             'test_epoch_vae_loss' : test_epoch_vae_loss,
#             'test_epoch_discr_loss' : test_epoch_discr_loss,
#         }, os.path.join(save_model_path, f"checkpoint_epoch_{epoch}.pth"))




if __name__ == '__main__' : 

    parser = argparse.ArgumentParser()
    parser.add_argument('-root_path', default = 'C:/Users/Utilisateur/Documents/MVA/DELIRES/Projet/DELIRES-VAE-Disentanglement')
    parser.add_argument('-config_path', default = '/src/config_factorvae.yaml')
    parser.add_argument('-device')
    args = parser.parse_args()

    params = load_parameters(args.root_path + args.config_path)
    if args.device is None : 
        device = params['device']
    else : 
        device = args.device

    params['save_model_path'] = args.root_path + params['save_model_path']
    params['dataset_path'] = args.root_path + params['dataset_path']

    main(params, device)