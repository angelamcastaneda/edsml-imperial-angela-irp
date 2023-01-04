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


# In[ ]:


class model1_variant1(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(model1_variant1, self).__init__()
        
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
        
        #x = self.output_act(x)   
        
        return x
    
    


# In[ ]:


class model1_variant2(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(model1_variant2, self).__init__()
        
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
        
        x = self.act(x)   
        
        return x

    
# In[ ]:


class model1_variant3(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(model1_variant3, self).__init__()
        
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