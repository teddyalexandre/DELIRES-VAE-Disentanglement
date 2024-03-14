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
        self.fc1 = nn.Linear(self.h_dim2 * self.input_dim / (self.stride)**4 , self.fc_dim) 
        self.fc2 = nn.Linear(self.fc_dim, 2 * self.output_dim) # return mean and var (diagonal vector)


    def forward(self, x) : 
        '''
        Shape of x: 
        - 2D shapes data: 64*64*1 (binary images)
        - 3D shapes, CelebA, chairs data: 64*64*3 (RGB images)
        - 3D faces data: 64*64*1 (greyscale images)
        '''
        x = torch.permute(x, (2, 0, 1)) # put channels dimension to first dimension
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

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
        self.upconv1 = nn.ConvTranspose2d(self.h_dim1, self.h_dim1, kernel_size = self.kernel_size, stride = self.stride)
        self.upconv2 = nn.ConvTranspose2d(self.h_dim1, self.h_dim2, kernel_size = self.kernel_size, stride = self.stride)
        self.upconv3 = nn.ConvTranspose2d(self.h_dim2, self.h_dim2, kernel_size = self.kernel_size, stride = self.stride)
        self.upconv4 = nn.ConvTranspose2d(self.h_dim2, 1, kernel_size = self.kernel_size, stride = self.stride)

    def forward(self, x) : 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.reshape(x, (x.shape[0], self.h_dim1, self.kernel_size[0], self.kernel_size[1])) # reshape into (batch_size, 64, 4, 4)
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.upconv4(x))
        return x
                 

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
                output_dim
                ):
        super(FactorVAE, self).__init__()

        
        self.input_dim = input_dim # 64
        self.h_dim1 = h_dim1 # 32
        self.h_dim2 = h_dim2 # 64
        self.kernel_size = kernel_size # (4,4)
        self.stride = stride # 2
        self.fc_dim = fc_dim # 128
        self.output_dim = output_dim # 10

        self.encoder = FactorVAE_Encoder(input_dim, h_dim1, h_dim2, kernel_size, stride, fc_dim, output_dim)
        self.decoder = FactorVAE_Decoder(output_dim, h_dim2, h_dim1, kernel_size, stride, fc_dim, input_dim)
    
    def forward(self, x) : 
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

class Discriminator(nn.Module) : 
    def __init__(self, 
                ):
        super(Discriminator, self).__init__()
    
    def forward(self, x) : 
        pass