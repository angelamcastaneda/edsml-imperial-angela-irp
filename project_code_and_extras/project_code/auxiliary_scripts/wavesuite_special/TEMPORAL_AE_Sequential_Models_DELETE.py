#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# For these models, I took the code from the Machine Learning module of the 
# MSc Environmental Data Science and Machine Learning as basis for these models,
# and then changed it accordingly, particularly I added all the parts related to GCN layers.

# The model GCN_pooling_AE was taken from https://colab.research.google.com/drive/1EMgPuFaD-xpboG_ZwZcytnlOlr39rakd#scrollTo=45raJzPsjhU7
# and modified accordingly


# In[ ]:

import torch
import torch.nn as nn #
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, TopKPooling, global_mean_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from AE_Models import Encoder, Decoder
 

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
    
class GCN_Encoder(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(GCN_Encoder, self).__init__()
        
        torch.manual_seed(42)
        
        #Class variables
        self.num_nodes = num_nodes
        self.embedding_sequence = embedding_sequence
        
        # GCN layers
        self.conv1 = GCNConv(num_features, embedding_sequence[0] )
        self.conv2 = GCNConv(embedding_sequence[0] , embedding_sequence[1] )
        self.conv3 = GCNConv(embedding_sequence[1] , embedding_sequence[2] )
        
        #AutoEncoder
        self.encoder = Encoder(num_nodes*embedding_sequence[2] , latent_space_dim)
        self.decoder = Decoder(latent_space_dim , num_nodes*1)
        
        #Activation function. Both ReLU and tanh seem to be equally good
        self.act = nn.ReLU() #self.act = F.tanh
        
    def forward(self, x, edge_index):
        
        #Convolutional layers
        x = self.conv1(x, edge_index)
        x = self.act(x)   
        x = self.conv2(x, edge_index)
        x = self.act(x)   
        x = self.conv3(x, edge_index) 
        
        #Flatten the arrays to input into the AE
        x = x.view(-1, self.num_nodes*self.embedding_sequence[2])   
        
        #AE
        x = self.encoder(x)  
        
        return x 

    
class GCN_Decoder(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(GCN_Decoder, self).__init__()
        
        torch.manual_seed(42)
        
        #Class variables
        self.num_nodes = num_nodes
        self.embedding_sequence = embedding_sequence
        
        self.decoder = Decoder(latent_space_dim , num_nodes*embedding_sequence[3] )
        
        # More GCN layers after the AE
        self.conv4 = GCNConv(embedding_sequence[3] , embedding_sequence[4] )
        self.conv5 = GCNConv(embedding_sequence[4] , embedding_sequence[5] )
        self.conv6 = GCNConv(embedding_sequence[5] , 1 )
        
        #Activation function. Both ReLU and tanh seem to be equally good
        self.act = nn.ReLU() #self.act = F.tanh
        
        #The activation function for the output is special
        self.output_act = nn.Sigmoid() #torch.sigmoid 
        
    def forward(self, x, edge_index):        
        
        x = self.decoder(x)
        
        #Reshape the arrays to input into more GCN layers 
        x = x.view(-1, self.embedding_sequence[3] ) # (self.num_nodes*batch_size, self.embedding_sequence[3] )
        
        #More GCN layers after the AE
        x = self.conv4(x, edge_index)
        x = self.act(x)   
        x = self.conv5(x, edge_index)
        x = self.act(x)   
        x = self.conv6(x, edge_index)
        #x = self.act(x)    #VARIANT NUMBER 3
        
        #Resize the output
        x = x.view(-1, self.num_nodes)
        
        x = self.output_act(x)  #VARIANT NUMBER 3
        
        return x    

    
# These next models are the same as in AE_Models but here they are written so that their parameters can later be easily separated into Encoder and Decoder parameters


# In[ ]:  VARIANT 3 WAS SELECTED FOR THE classic AE MODEL   Model1-variant3


class classic_AE_Sequential(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, latent_space_dim):
        
        # Init parent
        super(classic_AE_Sequential, self).__init__()
        
        torch.manual_seed(42)

        #AutoEncoder
        self.encoder = nn.Sequential(  View(num_nodes), Encoder(num_nodes*1 , latent_space_dim) )
        self.decoder = nn.Sequential( Decoder(latent_space_dim , num_nodes*1) , nn.Sigmoid() ) 
        
    def forward(self, x, edge_index, batch_size , batch_index):  
        
        #AE
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
    
    
    
# In[ ]:  VARIANT 3 WAS SELECTED FOR THE GCN_AE MODEL   Model2-variant3


class GCN_AE_Sequential(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(GCN_AE_Sequential, self).__init__()
        
        torch.manual_seed(42)
        
        
        self.encoder = GCN_Encoder(num_nodes, num_features, embedding_sequence, latent_space_dim)
        self.decoder = nn.Sequential( Decoder(latent_space_dim , num_nodes*1) , nn.Sigmoid() ) 
        
    def forward(self, x, edge_index, batch_size , batch_index):
        
        #AE
        x = self.encoder(x, edge_index)
        x = self.decoder(x)  
        
        return x 

       
# In[ ]:  VARIANT 3 FROM TEST1 WAS SELECTED FOR THE GCN_AE_GCN MODEL   Model3-test1-variant3

class GCN_AE_GCN_Sequential(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(GCN_AE_GCN_Sequential, self).__init__()
        
        torch.manual_seed(42)

        #AE
        self.encoder = GCN_Encoder(num_nodes, num_features, embedding_sequence, latent_space_dim)
        self.decoder = GCN_Decoder(num_nodes, num_features, embedding_sequence, latent_space_dim)
        
    def forward(self, x, edge_index, batch_size , batch_index):
        
        x = self.encoder(x, edge_index)
        x = self.decoder(x, edge_index)
        
        return x    

    
# In[ ]:  NO VARIANT 3was selected for Model 4 due to its bad preliminary results  

