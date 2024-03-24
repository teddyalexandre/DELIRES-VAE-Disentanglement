import torch
import matplotlib.pyplot as plt


from src.utils import permute_dims

def test(factorvae, discriminator, test_dataloader, gamma, batch_size, device) :

    # Evaluation time
    print('Testing...')
    factorvae.eval()
    discriminator.eval()
    test_epoch_vae_loss = 0
    test_epoch_discr_loss = 0

    for i, test_double_batch in enumerate(test_dataloader) : 
        
        print(i)
        if i % 10 == 0 : 
            print(f'Current test epoch VAE loss: {test_epoch_vae_loss}')
            print(f'Current test epoch Discr loss: {test_epoch_discr_loss}')
    
        with torch.no_grad() : 
            # Split the double batch into two batches
            test_double_batch = test_double_batch.to(device)
            half_length = test_double_batch.shape[0] // 2
            test_batch1, test_batch2 = torch.split(test_double_batch, half_length, 0)

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

            # Plot reconstruction
            # if i % 10 == 0 : 
            #     n = 10
            #     for j in range(n):
            #         print("original unique")
            #         print(test_batch1[j].unique())
            #         print("reconstruction unique")
            #         print(test_y[j].unique())
            #         plt.subplot(2, n, j + 1)
            #         plt.imshow(test_batch1[j].permute(1,2,0), cmap='gray')
            #         plt.axis('off')
            #         plt.subplot(2, n, j + 1 + n)
            #         plt.imshow(test_y[j].permute(1,2,0), cmap='gray')
            #         plt.axis('off')
            #     plt.show()
        
    test_epoch_vae_loss /= len(test_dataloader)
    test_epoch_discr_loss /= len(test_dataloader) 
    print(f'Test : VAE loss: {test_epoch_vae_loss:.2f}; Discriminator loss: {test_epoch_discr_loss:.2f}')

    return test_epoch_vae_loss, test_epoch_discr_loss