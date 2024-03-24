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


def main(params, device, subset) : 


    # Get dataloaders
    train_dataloader, test_dataloader = get_dataloaders(params['dataset_path'], 2*params['batch_size'], subset)

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



if __name__ == '__main__' : 

    parser = argparse.ArgumentParser()
    parser.add_argument('-root_path', default = 'C:/Users/Utilisateur/Documents/MVA/DELIRES/Projet/DELIRES-VAE-Disentanglement')
    parser.add_argument('-config_path', default = '/src/config_factorvae.yaml')
    parser.add_argument('-device')
    parser.add_argument('-subset', default = True)
    args = parser.parse_args()

    params = load_parameters(args.root_path + args.config_path)
    if args.device is None : 
        device = params['device']
    else : 
        device = args.device

    params['save_model_path'] = args.root_path + params['save_model_path']
    params['dataset_path'] = args.root_path + params['dataset_path']

    main(params, device, args.subset)