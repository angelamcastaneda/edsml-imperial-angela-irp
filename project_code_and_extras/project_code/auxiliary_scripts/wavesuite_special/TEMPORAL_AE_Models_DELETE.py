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


# In[ ]:


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
    
    
    

# BEST VARIANTS GO HERE


# In[ ]: VARIANT 3 WAS SELECTED FOR THE AE MODEL   Model1-variant3


class classic_AE(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, latent_space_dim):
        
        # Init parent
        super(classic_AE, self).__init__()
        
        torch.manual_seed(42)
        
        #Class variables
        self.num_nodes = num_nodes
        
        #AutoEncoder
        self.encoder = Encoder(num_nodes*1 , latent_space_dim)
        self.decoder = Decoder(latent_space_dim , num_nodes*1)
        
        #Activation function. Both ReLU and tanh seem to be equally good
        self.act = nn.ReLU() #self.act = F.tanh
        
        #The activation function for the output is special
        self.output_act = nn.Sigmoid() #torch.sigmoid 
        
    def forward(self, x, edge_index, batch_size , batch_index):
         
        #Flatten the arrays to input into the AE
        x = x.view(-1, self.num_nodes)  # *self.embedding_sequence[2]   
        
        #AE
        x = self.encoder(x)
        x = self.decoder(x)
        
        x = self.output_act(x)   
        
        return x
    
    
    
# In[ ]:  VARIANT 3 WAS SELECTED FOR THE GCN_AE MODEL   Model2-variant3


class GCN_AE(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(GCN_AE, self).__init__()
        
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
        
        #The activation function for the output is special
        self.output_act = nn.Sigmoid() #torch.sigmoid 
        
    def forward(self, x, edge_index, batch_size , batch_index):
        
        #Convolutional layers
        x = self.conv1(x, edge_index)
        x = self.act(x)   
        x = self.conv2(x, edge_index)
        x = self.act(x)   
        x = self.conv3(x, edge_index)
        # x = self.act(x)  
        
        #Flatten the arrays to input into the AE
        x = x.view(-1, self.num_nodes*self.embedding_sequence[2])   
        
        #AE
        x = self.encoder(x)
        x = self.decoder(x)
        
        x = self.output_act(x)   
        
        return x 

    

    
# In[ ]:


class GCN_AE_GCN(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(GCN_AE_GCN, self).__init__()
        
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
        self.decoder = Decoder(latent_space_dim , num_nodes*embedding_sequence[3] )
        
        # More GCN layers after the AE
        self.conv4 = GCNConv(embedding_sequence[3] , embedding_sequence[4] )
        self.conv5 = GCNConv(embedding_sequence[4] , embedding_sequence[5] )
        self.conv6 = GCNConv(embedding_sequence[5] , 1 )
        
        #Activation function. Both ReLU and tanh seem to be equally good
        self.act = nn.ReLU() #self.act = F.tanh
        
        #The activation function for the output is special
        self.output_act = nn.Sigmoid() #torch.sigmoid 
        
    def forward(self, x, edge_index, batch_size , batch_index):
        
        #Convolutional layers
        x = self.conv1(x, edge_index)
        x = self.act(x)   
        x = self.conv2(x, edge_index)
        x = self.act(x)   
        x = self.conv3(x, edge_index)
        # x = self.act(x)  
        
        #Flatten the arrays to input into the AE
        x = x.view(-1, self.num_nodes*self.embedding_sequence[2])   
        
        #AE
        x = self.encoder(x)
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

    

    
class GCN_pooling_AE(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(GCN_pooling_AE, self).__init__() 
        
        torch.manual_seed(42)
        
        #Class variables
        self.num_nodes = num_nodes
        self.embedding_sequence = embedding_sequence
        
        # GCN layers
        self.conv1 = GCNConv(num_features, embedding_sequence[0] )
        self.conv2 = GCNConv(embedding_sequence[0] , embedding_sequence[1] )
        self.conv3 = GCNConv(embedding_sequence[1] , embedding_sequence[2] )
        
        self.conv4 = GCNConv(embedding_sequence[2] , embedding_sequence[3] )
        self.conv5 = GCNConv(embedding_sequence[3] , embedding_sequence[4] )
        self.conv6 = GCNConv(embedding_sequence[4] , embedding_sequence[5] )
        
        #No parameters here required for pooling layers
        
        #AutoEncoder
        self.encoder = Encoder(2*embedding_sequence[5] , latent_space_dim)
        self.decoder = Decoder(latent_space_dim , num_nodes*1 )
        
        #Activation function. Both ReLU and tanh seem to be equally good
        self.act = nn.ReLU() #self.act = F.tanh
        
        #The activation function for the output is special
        self.output_act = nn.Sigmoid() #torch.sigmoid 
        
    def forward(self, x, edge_index, batch_size , batch_index):  
        
        #Convolutional layers
        x = self.conv1(x, edge_index)
        x = self.act(x)   
        x = self.conv2(x, edge_index)
        x = self.act(x)   
        x = self.conv3(x, edge_index)
        x = self.act(x)  
        
        x = self.conv4(x, edge_index)
        x = self.act(x)   
        x = self.conv5(x, edge_index)
        x = self.act(x)   
        x = self.conv6(x, edge_index)
        x = self.act(x)  
        
        # Global Pooling (stack different aggregations)
        x = torch.cat([gmp(x, batch_index), 
                            gap(x, batch_index)], dim=1)  
        
        # x = self.act(x) 
        
        #AE
        x = self.encoder(x)
        x = self.decoder(x)
        
        # x = self.act(x) 
        x = self.output_act(x)   
        
        return x


    
# In[ ]:   NORMAL FUNCTIONS BEFORE TESTS 



