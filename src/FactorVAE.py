import torch.nn as nn

class FactorVAE_Encoder(nn.Module) : 
    def __init__(self, 
                 input_dim,
                h_dim1,
                h_dim2,
                kernel_size,
                stride,
                fc_dim,
                ):
        super(FactorVAE_Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.h_dim1 = h_dim1
        self.h_dim2 = h_dim2
        self.kernel_size = kernel_size
        self.stride = stride
        self.fc_dim = fc_dim

        self.conv1 = nn.Conv2d(self.input_dim, self.h_dim1, stride = self.stride)
        self.conv2 = nn.Conv2d(self.h_dim1, self.h_dim1, stride = self.stride) #input channels to define
        self.conv3 = nn.Conv2d(self.h_dim1, self.h_dim2, stride = self.stride) #input channels to define
        self.conv4 = nn.Conv2d(self.h_dim1, self.h_dim2, stride = self.stride) #input channels to define
        self.fc1 = nn.Linear()


    def forward(x) : 
        '''
        Shape of x: 
        - 2D shapes data: 64*64*1 (binary images)
        - 3D shapes, CelebA, chairs data: 64*64*3 (RGB images)
        - 3D faces data: 64*64*1 (greyscale images)
        '''



class FactorVAE(nn.Module):
    def __init__(self, 
                 input_dim,
                h_dim1,
                h_dim2,
                kernel_size,
                stride,
                fc_dim,
                ):
        super(FactorVAE, self).__init__()