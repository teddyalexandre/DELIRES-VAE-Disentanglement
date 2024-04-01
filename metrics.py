import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from torchvision import transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt


from src.utils import load_parameters
from src.FactorVAE import FactorVAE, Discriminator
from src.dSpritesDataset import get_dataloaders, get_data_with_factors, dSpritesDataset, dSpritesDataset_classes, RescaleBinaryImage


def find_image(latents_classes, v):
    idx = np.where((latents_classes == v).all(axis=1))[0]
    return idx

def main(root_path, checkpoint_dir, params) : 

    last_cp_file = os.path.join(checkpoint_dir, f'checkpoint_epoch_99.pth')
    last_checkpoint = torch.load(last_cp_file, map_location=torch.device('cpu'))

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


    imgs, latents_classes, latents_values = get_data_with_factors(params['dataset_path'], 2*params['batch_size'])

    # Compute s
    transform = transforms.Compose([
        transforms.ToTensor(),
        RescaleBinaryImage()
        ])
    
    empirical_std_dev = torch.zeros(params['discr']['input_dim']) # latent shape (10)
    dsprites = dSpritesDataset(imgs, transform = transform)
    subset_size = 15000
    dsprites_small = random_split(dsprites, [subset_size, len(imgs)-subset_size])[0]
    dataloader = DataLoader(dsprites_small, batch_size = 512) 

    nb_samples = 0
    for i, batch in enumerate(dataloader) : 
        nb_samples += len(batch)
        _, z_log_var = factorvae.encode(batch)
        empirical_std_dev = empirical_std_dev + torch.sqrt(torch.exp(z_log_var)).sum(axis=0)
    
    empirical_std_dev =  empirical_std_dev / nb_samples
    print("empirical std dev")
    print(empirical_std_dev)



    # Build classifier dataset

    nb_data_points = 10000 # we build a dataset of 10 000 data points
    L = 50 # number of images generated for each data point
    
    data_points = torch.zeros((nb_data_points, 2))

    for point in range(nb_data_points) : 
        fixed_factor = np.random.randint(0, 6)
        fixed_factor_value = np.random.choice(np.unique(latents_classes[:, fixed_factor]))

        imgs_indices = np.where(latents_classes[:,fixed_factor] == fixed_factor_value)[0][:L] # take only  L images

        batch = imgs[imgs_indices]
        classes = latents_classes[imgs_indices]

        dataset = dSpritesDataset_classes(batch, classes, transform = transform)
        dataloader = DataLoader(dataset, batch_size = L) # take only one batch 

        batch = next(iter(dataloader))
        batch_imgs, batch_latents_classes = batch

        batch_latents, _ = factorvae.encode(batch_imgs) # take only the mean
        normalized_latents = torch.div(batch_latents, empirical_std_dev)
        variances = torch.var(normalized_latents, dim = 0)
        argmin_var = torch.argmin(variances)

        data_points[point][0] = argmin_var
        data_points[point][1] = fixed_factor
    
    file_path = os.path.join(root_path, 'classifier_data', 'datapoints_checkpoint_99.pt')
    torch.save(data_points, file_path)


def get_classifier(data_points) : 
    print('unique fixed factors')
    print(data_points[:,1].unique())
    print('unique d')
    print(data_points[:,0].unique())
    nb_j = int(data_points[:, 0].max().item() + 1)
    nb_k = int(data_points[:, 1].max().item() + 1)

    print(nb_j)
    print(nb_k)

    V = torch.zeros((nb_j, nb_k), dtype=torch.int)

    for i in range(data_points.size(0)):
        j, k = data_points[i]
        k = int(k.item())
        j = int(j.item())
        V[j][k] += 1

    classifier = torch.zeros(nb_j)
    for j in range(nb_j) : 
        classifier[j] = torch.argmax(V[j])
    
    print('Classifier')
    print(classifier)
    
    return classifier

def classifier_metric(root_path) :
    file_path = os.path.join(root_path, 'classifier_data', 'datapoints_checkpoint_99.pt')
    data_points = torch.load(file_path)

    classifier = get_classifier(data_points)

    error_rate = 0

    for i in range(data_points.size(0)):
        j, k = data_points[i]
        k = int(k.item())
        j = int(j.item())
        k_pred = classifier[j] 
        if k != k_pred : 
            error_rate += 1

    error_rate = error_rate / data_points.size(0)
    print(error_rate)

    return error_rate

def beta_metrics(root_path, checkpoint_dir, params) :
    last_cp_file = os.path.join(checkpoint_dir, f'checkpoint_epoch_99.pth')
    last_checkpoint = torch.load(last_cp_file, map_location=torch.device('cpu'))

    factorvae = FactorVAE(params['factorvae']['input_dim'],
                        params['factorvae']['h_dim1'],
                        params['factorvae']['h_dim2'], 
                        (params['factorvae']['kernel_size'], params['factorvae']['kernel_size']),
                        params['factorvae']['stride'],
                        params['factorvae']['fc_dim'],
                        params['factorvae']['output_dim'],
                        device).to(device)
    
    factorvae.load_state_dict(last_checkpoint['vae_state_dict']) 
    factorvae.eval()

    imgs, latents_classes, latents_values = get_data_with_factors(params['dataset_path'], 2*params['batch_size'])

    # Build classifier dataset

    nb_data_points = 200 # we build a dataset of 10 000 data points
    L = 50 # number of images generated for each data point
    K = 6
    data_diffs = torch.zeros((nb_data_points, 10))
    data_classes = torch.zeros((nb_data_points))

    for point in range(nb_data_points) : 
        z_l = []
        y = np.random.randint(0, K) # fixed factor
        for l in range(L):
            # Sample a pair v1,l , v2,l such that they agree on their yth value
            v_1l = np.array([np.random.choice(np.unique(latents_classes[:,i])) for i in range(K)])
            v_2l = np.array([np.random.choice(np.unique(latents_classes[:,i])) for i in range(K)])
            v_1l[y] = v_2l[y]

            # simulate the images corresponding to the pair v1,l , v2,l
            idx_1 = np.where((latents_classes == v_1l).all(axis=1))[0]
            idx_2 = np.where((latents_classes == v_2l).all(axis=1))[0]
            
            # get the images
            imgs_1 = imgs[idx_1]
            imgs_2 = imgs[idx_2]

            z_1, _ = factorvae.encoder(torch.tensor(imgs_1, dtype=torch.float).view(1, 1, 64, 64).to(device))
            z_2, _ = factorvae.encoder(torch.tensor(imgs_2, dtype=torch.float).view(1, 1, 64, 64).to(device))
            z_1 = z_1.cpu().detach()
            z_2 = z_2.cpu().detach()
            z_diff = torch.abs(z_1 - z_2)
            z_l.append(z_diff)
        # compute the element-wise mean of the latent representation
        z_l_tensor = torch.tensor(np.array(z_l), dtype=torch.float)
        z_b_mean = torch.mean(z_l_tensor, dim=0)
        data_diffs[point] = z_b_mean
        data_classes[point] = y
    
    file_path_diffs = os.path.join(root_path, 'classifier_data', 'data_diffs_beta_metrics_checkpoint_99.pt')
    torch.save(data_diffs, file_path_diffs)    
    file_path_classes = os.path.join(root_path, 'classifier_data', 'data_classes_beta_metrics_checkpoint_99.pt')
    torch.save(data_classes, file_path_classes)


    classifier_dataset = TensorDataset(data_diffs, data_classes)
    train_size = int(0.8 * len(classifier_dataset))
    test_size = len(classifier_dataset) - train_size

    classifier_train_dataset, classifier_test_dataset = random_split(classifier_dataset, [train_size, test_size])

    classifier_traindataloader = DataLoader(classifier_train_dataset, batch_size=4, shuffle=False)
    classifier_testdataloader = DataLoader(classifier_test_dataset, batch_size=4, shuffle=False)

    classifier = nn.Sequential(
        nn.Linear(10, 6),
        nn.LogSoftmax(dim=1)
    )

    optimizer = optim.Adagrad(classifier.parameters(), lr=1e-2)
    criterion = nn.NLLLoss()

    epochs = 1000
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        classifier.train()
        for inputs, labels in classifier_traindataloader:
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    classifier.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in classifier_testdataloader:
            inputs = inputs
            outputs = classifier(inputs)
            _, predicted_indices = torch.max(outputs, 1)
            correct += (predicted_indices == labels).sum().item()
            total += inputs.size(0)
        
        score = 100 * correct / total
        print('Disentanglement metric score: %d %%' % (score))











def plot_latent_traversals_each_dim(root_path, checkpoint_dir, params, device, n=15, z_dim=10, traversal_range=3):

    # Load trained factorVAE
    last_cp_file = os.path.join(checkpoint_dir, f'checkpoint_epoch_99.pth')
    last_checkpoint = torch.load(last_cp_file, map_location=torch.device('cpu'))

    factorvae = FactorVAE(params['factorvae']['input_dim'],
                        params['factorvae']['h_dim1'],
                        params['factorvae']['h_dim2'], 
                        (params['factorvae']['kernel_size'], params['factorvae']['kernel_size']),
                        params['factorvae']['stride'],
                        params['factorvae']['fc_dim'],
                        params['factorvae']['output_dim'],
                        device).to(device)
    
    factorvae.load_state_dict(last_checkpoint['vae_state_dict'])

    # Load data
    train_dataloader, test_dataloader = get_dataloaders(params['dataset_path'], 2*params['batch_size'], subset = True)

    factorvae.eval()
    with torch.no_grad():
        # Get mean and var of each dimension of the latent space

        means = torch.zeros(z_dim)
        log_vars = torch.zeros(z_dim)
        nb_samples = 0

        for i, batch in enumerate(test_dataloader) : 
            nb_samples += len(batch)
            z_mu, z_log_var = factorvae.encode(batch)
            means = means + z_mu.sum(axis=0)
            log_vars = log_vars + z_log_var.sum(axis=0)
        
        means = means.div(nb_samples)
        log_vars = log_vars.div(nb_samples)

        print(means)
        print(log_vars)


        # Generate fixed latent codes
        #fixed_z = torch.randn(n, z_dim).to(device)
        fixed_z = factorvae.sampling(means, log_vars)
        fixed_z_repeated = fixed_z.repeat(n,1)
        
        print(fixed_z_repeated)
        # Create traversal values
        #traversal_values = torch.linspace(-traversal_range, traversal_range, n).to(device)
        
        # Loop over each dimension and create traversed latent codes
        for fixed_dim in range(z_dim):
            latent_codes = fixed_z_repeated.clone()
            std_dev = torch.sqrt(torch.exp(log_vars[fixed_dim]))
            traversal_values = torch.linspace(means[fixed_dim] - 100 * std_dev, means[fixed_dim] + 100 * std_dev, n).to(device)
            print(traversal_values)
            for i in range(n):
                latent_codes[i, fixed_dim] = traversal_values[i]
            
            # Generate images from traversed latent codes
            reconstructions = factorvae.decoder(latent_codes)
            
            # Display
            plt.figure(figsize=(n, 2))
            for i in range(n):
                plt.subplot(2, n, i + 1)
                plt.imshow(reconstructions[i].cpu().permute(1, 2, 0).squeeze(), cmap='gray')
                plt.axis('off')
            plt.suptitle(f'Latent Dimension {fixed_dim+1}')
            plt.show()

        






if __name__ == '__main__' : 

    parser = argparse.ArgumentParser()
    parser.add_argument('-root_path', default = 'C:/Users/Utilisateur/Documents/MVA/DELIRES/Projet/DELIRES-VAE-Disentanglement')
    parser.add_argument('-checkpoint_dir', default = 'C:/Users/Utilisateur/Documents/MVA/DELIRES/Projet/DELIRES-VAE-Disentanglement/models_checkpoints/factorvae/all_with_discr_lr_1e-4_1e-4')
    parser.add_argument('-config_path', default = '/src/config_factorvae.yaml')
    parser.add_argument('-device')
    
    args = parser.parse_args()

    params = load_parameters(args.root_path + args.config_path)
    params['dataset_path'] = args.root_path + params['dataset_path']
    if args.device is None : 
        device = params['device']
    else : 
        device = args.device

    main(args.root_path, args.checkpoint_dir, params)
    classifier_metric(args.root_path)
    
    plot_latent_traversals_each_dim(args.root_path, args.checkpoint_dir, params, device)
        
    beta_metrics(args.root_path, args.checkpoint_dir, params)