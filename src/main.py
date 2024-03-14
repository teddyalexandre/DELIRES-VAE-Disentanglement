from src.FactorVAE import FactorVAE
from src.dSpritesDataset import get_dataloaders
import argparse

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

    # Training parameters
    

    #Training loop

















if __name__ == '__main__' : 
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', default = 64)
    parser.add_argument('-dataset_path', default = 'C:\Users\Utilisateur\Documents\MVA\DELIRES\Projet\DELIRES-VAE-Disentanglement\dsprites-dataset\dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

    args = parser.parse_args()
    main(args.dataset_path, args.batch_size)