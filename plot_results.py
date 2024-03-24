import matplotlib.pyplot as plt
import argparse
import os
import torch
import json

from src.utils import load_parameters
from src.FactorVAE import FactorVAE, Discriminator
from src.dSpritesDataset import get_dataloaders


def main(checkpoint_dir, params, save_fig_path, subset = True, plot_losses = True, reconstitutions = True) : 
    
    if plot_losses : 
        # Plot losses

        epochs = [i for i in range(100)]
        # train_vae_losses = []
        # train_discr_losses = []
        # test_vae_losses = []
        # test_discr_losses = []


        # for epoch in epochs : 
        #     cp_file = f'checkpoint_epoch_{epoch}.pth'
        #     checkpoint = torch.load(os.path.join(checkpoint_dir, cp_file))
        #     train_vae_losses.append(checkpoint['epoch_vae_loss'])
        #     train_discr_losses.append(checkpoint['epoch_discr_loss'])
        #     test_vae_losses.append(checkpoint['test_epoch_vae_loss'])
        #     test_discr_losses.append(checkpoint['test_epoch_discr_loss'])

        with open(os.path.join(checkpoint_dir, 'losses'), 'r') as f:
            data = json.load(f)
    
        train_vae_losses = data['train_vae_losses']
        train_discr_losses = data['train_discr_losses']
        test_vae_losses = data['test_vae_losses']
        test_discr_losses = data['test_discr_losses']


        # Create subplots with 1 row and 2 columns
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # Plot curves 1 and 2 on the left plot
        axs[0].plot(epochs, train_vae_losses, label='Train')
        axs[0].plot(epochs, test_vae_losses, label='Test')
        axs[0].set_title('VAE Loss')
        axs[0].legend()

        # Plot curves 3 and 4 on the right plot
        axs[1].plot(epochs, train_discr_losses, label='Train')
        axs[1].plot(epochs, test_discr_losses, label='Test')
        axs[1].set_title('Discriminator L')
        axs[1].legend()

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Show the plot
        loss_path = os.path.join(save_fig_path, "losses")
        plt.savefig(loss_path)
        plt.show()

    if reconstitutions : 
        # Plot reconstitution
            
        last_cp_file = os.path.join(checkpoint_dir, f'checkpoint_epoch_99.pth')
        last_checkpoint = torch.load(last_cp_file)

        factorvae = FactorVAE(params['factorvae']['input_dim'],
                            params['factorvae']['h_dim1'],
                            params['factorvae']['h_dim2'], 
                            (params['factorvae']['kernel_size'], params['factorvae']['kernel_size']),
                            params['factorvae']['stride'],
                            params['factorvae']['fc_dim'],
                            params['factorvae']['output_dim'],
                            device).to(device)
        
        factorvae.load_state_dict(last_checkpoint['vae_state_dict'])

        discriminator = Discriminator(device, 
                                    params['discr']['input_dim'],
                                    params['discr']['hidden_dim'],
                                    params['discr']['output_dim']
                                    ).to(device)
        
        discriminator.load_state_dict(last_checkpoint['discr_state_dict'])

        # Load test dataset
        params['dataset_path'] = args.root_path + params['dataset_path']
        _, test_dataloader = get_dataloaders(params['dataset_path'], 2*params['batch_size'], subset = subset)

        for i, test_double_batch in enumerate(test_dataloader) : 
            with torch.no_grad() : 
                # Split the double batch into two batches
                test_double_batch = test_double_batch.to(device)
                half_length = test_double_batch.shape[0] // 2
                test_batch1, _ = torch.split(test_double_batch, half_length, 0)
                test_y, _, _ = factorvae(test_batch1)

                n = 10
                for j in range(n):
                    plt.subplot(2, n, j + 1)
                    plt.imshow(test_batch1[j].permute(1,2,0), cmap='gray')
                    plt.axis('off')
                    plt.subplot(2, n, j + 1 + n)
                    plt.imshow(test_y[j].permute(1,2,0), cmap='gray')
                    plt.axis('off')
                reconstitution_path = os.path.join(save_fig_path, f'reconstitution_{i}')
                plt.savefig(reconstitution_path)
                plt.show()



if __name__ == '__main__' : 

    parser = argparse.ArgumentParser()
    parser.add_argument('-root_path', default = 'C:/Users/Utilisateur/Documents/MVA/DELIRES/Projet/DELIRES-VAE-Disentanglement')
    parser.add_argument('-checkpoint_dir', default = 'C:/Users/Utilisateur/Documents/MVA/DELIRES/Projet/DELIRES-VAE-Disentanglement/models_checkpoints/factorvae/with_discr')
    parser.add_argument('-config_path', default = '/src/config_factorvae.yaml')
    parser.add_argument('-plot_losses', default = True)
    parser.add_argument('-reconstitutions', default = True)
    parser.add_argument('-device')
    parser.add_argument('-subset', default = True)
    parser.add_argument('-save_fig_path', default = "C:\Users\Utilisateur\Documents\MVA\DELIRES\Projet\DELIRES-VAE-Disentanglement\images")
    
    args = parser.parse_args()

    params = load_parameters(args.root_path + args.config_path)
    if args.device is None : 
        device = params['device']
    else : 
        device = args.device

    main(args.checkpoint_dir, params, args.save_fig_path, args.subset, args.plot_losses, args.reconstitutions)