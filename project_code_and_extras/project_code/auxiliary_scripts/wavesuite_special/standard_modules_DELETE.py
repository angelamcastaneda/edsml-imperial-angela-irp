import torch
import torch.nn as nn #
import torch.nn.functional as F 

#from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
#from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class Encoder(nn.Module):
    def __init__(self, ae_input_size , latent_space_dim):
        super(Encoder, self).__init__()
        
        self.encode1 = nn.Linear( ae_input_size , 2*latent_space_dim )
        self.encode2 = nn.Linear( 2*latent_space_dim , latent_space_dim )
        
        #self.act = nn.ReLU() #self.act = F.tanh #self.act = F.relu
        self.act = nn.ReLU() 

    def forward(self, x):
        
        #x = torch.flatten(x, start_dim=1) # I do not think this is necessary     
        x = self.act( self.encode1(x) )
        x = self.encode2(x) #or should I use self.encode2(x) only ?
        
        return x
    

class Decoder(nn.Module): 
    def __init__(self, latent_space_dim, ae_output_size):
        super(Decoder, self).__init__() 
        
        self.decode1 = nn.Linear( latent_space_dim , 2*latent_space_dim )
        self.decode2 = nn.Linear( 2*latent_space_dim , ae_output_size )
        
        self.act = nn.ReLU() #self.act = F.tanh #self.act = F.relu

    def forward(self, x):
        
        x = self.act( self.decode1(x) )
        x = self.decode2(x)
        # x = self.act(x) # or should I use x = torch.sigmoid(x) ?
        
        return x
    
    
#class taken from https://discuss.pytorch.org/t/how-to-build-a-view-layer-in-pytorch-for-sequential-models/53958/12
class View(nn.Module):
    def __init__(self, columns):
        super().__init__()
        self.columns = columns

    def __repr__(self):
        return f'View{self.columns}'

    def forward(self, input):
        '''
        Reshapes the input according to the number of columns required saved in the view data structure.
        '''
        out = input.view(-1, self.columns)
        return out