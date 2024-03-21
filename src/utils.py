import os
import yaml
import numpy as np



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

def load_parameters(file):
    if os.path.isfile(file):
        with open(file, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise IOError("Parameter file not found [%s]" % file)