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


class model3_variant1(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(model3_variant1, self).__init__()
        
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
        # x = self.act(x)   #VARIANT NUMBER 2
        
        #Resize the output
        x = x.view(-1, self.num_nodes)
        
        return x
    
    


# In[ ]:


class model3_variant2(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(model3_variant2, self).__init__()
        
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
        
    def forward(self, x, edge_index, batch_size , batch_index):
        
        #Convolutional layers
        x = self.conv1(x, edge_index)
        x = self.act(x)   
        x = self.conv2(x, edge_index)
        x = self.act(x)   
        x = self.conv3(x, edge_index)
        #x = self.act(x) 
        
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
        x = self.act(x) 
        
        #Resize the output
        x = x.view(-1, self.num_nodes)
        
        return x

    
# In[ ]:


class model3_variant3(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(model3_variant3, self).__init__()
        
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

    
    
# In[ ]:


class model3_variant4(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(model3_variant4, self).__init__()
        
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
        
        x = self.act(x)  #VARIANT NUMBER 6
        
        #Reshape the arrays to input into more GCN layers 
        x = x.view(-1, self.embedding_sequence[3] ) # (self.num_nodes*batch_size, self.embedding_sequence[3] )
        
        #More GCN layers after the AE
        x = self.conv4(x, edge_index)
        x = self.act(x)   
        x = self.conv5(x, edge_index)
        x = self.act(x)   
        x = self.conv6(x, edge_index)
        # x = self.act(x)   #VARIANT NUMBER 2
        
        # x = self.output_act(x)   #VARIANT NUMBER 6
        
        #Resize the output
        x = x.view(-1, self.num_nodes)
        
        return x
    
    
    


class model3_variant5(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(model3_variant5, self).__init__()
        
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
        
        x = self.act(x)   #VARIANT NUMBER 1
        
        #More GCN layers after the AE
        x = self.conv4(x, edge_index)
        x = self.act(x)   
        x = self.conv5(x, edge_index)
        x = self.act(x)   
        x = self.conv6(x, edge_index)
        x = self.act(x) 
        
        #Resize the output
        x = x.view(-1, self.num_nodes)
        
        return x

    
    
    
# In[ ]:


class model3_variant6(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(model3_variant6, self).__init__()
        
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
        
        x = self.act(x)  
        
        #Reshape the arrays to input into more GCN layers 
        x = x.view(-1, self.embedding_sequence[3] ) # (self.num_nodes*batch_size, self.embedding_sequence[3] )
        
        #More GCN layers after the AE
        x = self.conv4(x, edge_index)
        x = self.act(x)   
        x = self.conv5(x, edge_index)
        x = self.act(x)   
        x = self.conv6(x, edge_index)
        # x = self.act(x)   
        
        x = self.output_act(x)   
        
        #Resize the output
        x = x.view(-1, self.num_nodes)
        
        return x
    
# In[ ]:


class model3_variant7(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(model3_variant7, self).__init__()
        
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
        
        x = self.output_act(x)  #VARIANT NUMBER 2
        
        #Reshape the arrays to input into more GCN layers 
        x = x.view(-1, self.embedding_sequence[3] ) # (self.num_nodes*batch_size, self.embedding_sequence[3] )
        
        #More GCN layers after the AE
        x = self.conv4(x, edge_index)
        x = self.act(x)   
        x = self.conv5(x, edge_index)
        x = self.act(x)   
        x = self.conv6(x, edge_index)
        # x = self.act(x)   #VARIANT NUMBER 2
        
        #Resize the output
        x = x.view(-1, self.num_nodes)
        
        return x
    
    
    
# In[ ]: 


class model3_variant8(torch.nn.Module ):
    def __init__(self, num_nodes, num_features, embedding_sequence, latent_space_dim):
        
        # Init parent
        super(model3_variant8, self).__init__()
        
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
        
        x = self.output_act(x)  #VARIANT NUMBER 2
        
        #Reshape the arrays to input into more GCN layers 
        x = x.view(-1, self.embedding_sequence[3] ) # (self.num_nodes*batch_size, self.embedding_sequence[3] )
        
        #More GCN layers after the AE
        x = self.conv4(x, edge_index)
        x = self.act(x)   
        x = self.conv5(x, edge_index)
        x = self.act(x)   
        x = self.conv6(x, edge_index)
        x = self.act(x)   #VARIANT NUMBER 2
        
        #Resize the output
        x = x.view(-1, self.num_nodes)
        
        return x
    
    