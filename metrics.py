import argparse
import os
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import transforms
from torch.utils.data import random_split



from src.utils import load_parameters
from src.FactorVAE import FactorVAE, Discriminator
from src.dSpritesDataset import get_data_with_factors, dSpritesDataset, dSpritesDataset_classes, RescaleBinaryImage


def find_image(latents_classes, v):
    idx = np.where((latents_classes == v).all(axis=1))[0]
    return idx

def main(root_path, checkpoint_dir, params) : 

    file_path = os.path.join(root_path, 'classifier_data', 'datapoints.pt')
    print(file_path)

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

    params['dataset_path'] = root_path + params['dataset_path']
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
    for i, batch in enumerate(dataloader) : 
        batch_latents, _ = factorvae.encode(batch)
        empirical_std_dev = torch.add(empirical_std_dev, torch.std(batch_latents, dim = 0) / len(batch))
    
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


        






if __name__ == '__main__' : 

    parser = argparse.ArgumentParser()
    parser.add_argument('-root_path', default = 'C:/Users/Utilisateur/Documents/MVA/DELIRES/Projet/DELIRES-VAE-Disentanglement')
    parser.add_argument('-checkpoint_dir', default = 'C:/Users/Utilisateur/Documents/MVA/DELIRES/Projet/DELIRES-VAE-Disentanglement/models_checkpoints/factorvae/all_with_discr_lr_1e-4_1e-4')
    parser.add_argument('-config_path', default = '/src/config_factorvae.yaml')
    parser.add_argument('-device')
    
    args = parser.parse_args()

    params = load_parameters(args.root_path + args.config_path)
    if args.device is None : 
        device = params['device']
    else : 
        device = args.device

    main(args.root_path, args.checkpoint_dir, params)
    classifier_metric(args.root_path)