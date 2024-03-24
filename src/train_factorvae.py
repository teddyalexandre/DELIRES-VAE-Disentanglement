import torch

from src.utils import permute_dims

def train(factorvae, discriminator, vae_opti, discr_opti, train_dataloader, gamma, batch_size, device) :

    factorvae.train()
    discriminator.train()

    epoch_vae_loss = 0
    epoch_discr_loss = 0

    for i, double_batch in enumerate(train_dataloader) : 
        if i % 10 == 0 : 
            print(f'Current epoch VAE loss: {epoch_vae_loss}')
            print(f'Current epoch Discr loss: {epoch_discr_loss}')

        # Split the double batch into two batches
        double_batch = double_batch.to(device)
        batch1, batch2 = torch.split(double_batch, batch_size, 0)

        # Sample z on the first batch
        y, z_mu, z_log_var = factorvae(batch1)
        z_sample = factorvae.sampling(z_mu, z_log_var)

        # Get VAE loss for the first batch
        discr_z1 = discriminator(z_sample)
        gamma_term = (discr_z1[:,0] - discr_z1[:,1]).mean()
        vae_loss = factorvae.loss_function(batch1, y, z_mu, z_log_var) + gamma * gamma_term
        epoch_vae_loss += vae_loss.item()

        # # Optimization of VAE loss
        vae_opti.zero_grad()
        vae_loss.backward(retain_graph = True)

        # Sample z on the second batch
        y2, z_mu2, z_log_var2 = factorvae(batch2)
        z_sample2 = factorvae.sampling(z_mu2, z_log_var2)

        # Permute z
        z_permuted = permute_dims(z_sample2).detach()

        # Loss of the discriminator
        discr_z2 = discriminator(z_permuted)
        discr_loss = discriminator.discr_loss(discr_z1, discr_z2)
        epoch_discr_loss += discr_loss.item()

        # Optimization of Discriminator loss
        discr_opti.zero_grad()
        discr_loss.backward()

        # Optimization of both losses
        vae_opti.step()
        discr_opti.step()


    epoch_vae_loss /= len(train_dataloader)
    epoch_discr_loss /= len(train_dataloader) 

    return epoch_vae_loss, epoch_discr_loss
