import torch.nn as nn
import torch.nn.functional as F
import torch 

class FactorVAE_Encoder(nn.Module) : 
    def __init__(self, 
                input_dim,
                h_dim1,
                h_dim2,
                kernel_size,
                stride,
                fc_dim,
                output_dim
                ):
        super(FactorVAE_Encoder, self).__init__()
        
        self.input_dim = input_dim # 64
        self.h_dim1 = h_dim1
        self.h_dim2 = h_dim2
        self.kernel_size = kernel_size
        self.stride = stride
        self.fc_dim = fc_dim
        self.output_dim = output_dim # 10

        self.conv1 = nn.Conv2d(1, self.h_dim1, kernel_size = self.kernel_size, stride = self.stride)
        self.conv2 = nn.Conv2d(self.h_dim1, self.h_dim1, kernel_size = self.kernel_size, stride = self.stride) #input channels to define
        self.conv3 = nn.Conv2d(self.h_dim1, self.h_dim2, kernel_size = self.kernel_size, stride = self.stride) #input channels to define
        self.conv4 = nn.Conv2d(self.h_dim2, self.h_dim2, kernel_size = self.kernel_size, stride = self.stride) #input channels to define
        self.fc1 = nn.Linear(int(self.h_dim2 * self.input_dim / (self.stride)**4) , self.fc_dim) 
        self.fc21 = nn.Linear(self.fc_dim, self.output_dim) # return mean 
        self.fc22 = nn.Linear(self.fc_dim, self.output_dim) # return log_var (diagonal) 


    def forward(self, x) : 
        '''
        Shape of x: 
        - 2D shapes data: 64*64*1 (binary images)
        - 3D shapes, CelebA, chairs data: 64*64*3 (RGB images)
        - 3D faces data: 64*64*1 (greyscale images)
        '''
        #x = torch.permute(x, (0, 3, 1, 2)) # put channels dimension to first dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.fc1(x.reshape(x.shape[0], -1))
        mu = self.fc21(x)
        log_var = self.fc22(x)
        return mu, log_var

class FactorVAE_Decoder(nn.Module) : 
    def __init__(self, 
                input_dim,
                h_dim1,
                h_dim2,
                kernel_size,
                stride,
                fc_dim,
                output_dim
                ):
        super(FactorVAE_Decoder, self).__init__()

        self.input_dim = input_dim # 10
        self.h_dim1 = h_dim1
        self.h_dim2 = h_dim2
        self.kernel_size = kernel_size
        self.stride = stride
        self.fc_dim = fc_dim 
        self.output_dim = output_dim # 64

        self.fc1 = nn.Linear(self.input_dim, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.h_dim1 * self.kernel_size[0] * self.kernel_size[1])
        self.upconv1 = nn.ConvTranspose2d(self.h_dim1, self.h_dim1, kernel_size = self.kernel_size, stride = self.stride, padding = 1)
        self.upconv2 = nn.ConvTranspose2d(self.h_dim1, self.h_dim2, kernel_size = self.kernel_size, stride = self.stride, padding = 1)
        self.upconv3 = nn.ConvTranspose2d(self.h_dim2, self.h_dim2, kernel_size = self.kernel_size, stride = self.stride, padding = 1)
        self.upconv4 = nn.ConvTranspose2d(self.h_dim2, 1, kernel_size = self.kernel_size, stride = self.stride, padding = 1)

    def forward(self, x) : 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.reshape(x, (x.shape[0], self.h_dim1, self.kernel_size[0], self.kernel_size[1])) # reshape into (batch_size, 64, 4, 4)
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.upconv4(x))
        return F.sigmoid(x) # return logits
                 

class FactorVAE(nn.Module):
    '''
    Factor VAE for 2D shapes data (DSPRITES dataset)
    
    '''
    def __init__(self, 
                 input_dim,
                h_dim1,
                h_dim2,
                kernel_size,
                stride,
                fc_dim,
                output_dim,
                device
                ):
        super(FactorVAE, self).__init__()

        
        self.input_dim = input_dim # 64
        self.h_dim1 = h_dim1 # 32
        self.h_dim2 = h_dim2 # 64
        self.kernel_size = kernel_size # (4,4)
        self.stride = stride # 2
        self.fc_dim = fc_dim # 128
        self.output_dim = output_dim # 10
        self.device = device

        self.encoder = FactorVAE_Encoder(input_dim, h_dim1, h_dim2, kernel_size, stride, fc_dim, output_dim)
        self.decoder = FactorVAE_Decoder(output_dim, h_dim2, h_dim1, kernel_size, stride, fc_dim, input_dim)
   
    def sampling(self, mu, log_var) : 
        std = torch.sqrt(torch.exp(log_var)) 
        eps = torch.randn(std.shape).to(self.device)
        return eps.mul(std).add_(mu) 
    
    def forward(self, x) : 
        z_mu, z_log_var = self.encoder(x)
        z = self.sampling(z_mu, z_log_var)
        return self.decoder(z), z_mu, z_log_var


    def loss_function(self, x, y, mu, log_var) : 
        reconstruction_error = torch.nn.BCELoss(reduction = 'mean')(y, x).to(self.device) # mean ou sum ?
        KLD = 0.5 * torch.mean(torch.exp(log_var) + mu.pow(2) - log_var - 1) # mean ou sum ?
        return reconstruction_error + KLD
    

class Discriminator(nn.Module) : 
    def __init__(self, 
                 input_size, # 10
                 hidden_dim, # 1000
                 output_size, # 2
                 batch_size = 64,
                 device
                ):
        super(Discriminator, self).__init__()
    
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.batch_size = batch_size
        self.device = device

        self.fc1 = nn.Linear(self.input_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc5 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc6 = nn.Linear(self.hidden_dim, self.output_size)

    def forward(self, z) : 
        x = F.leaky_relu(self.fc1(z))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))
        x = F.leaky_relu(self.fc6(x))
        return F.sigmoid(x) # return logits 
    
    def discr_loss(self, Dz, Dz_perm) :
        zeros = torch.zeros(self.batch_size, dtype=torch.long).to(self.device)
        ones = torch.ones(self.batch_size, dtype=torch.long).to(self.device)
        return 0.5*(F.cross_entropy(Dz, zeros) + F.cross_entropy(Dz_perm, ones))