#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch 
import vtktools 

from torch_geometric.data import Data

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

# In[ ]:


def get_node_features(vtu_object, features_list):
    
    node_features = vtu_object.GetField(features_list[0]).T

    for i in range(1,len(features_list)):
        next_feature = vtu_object.GetField(features_list[i]).T
        node_features = np.append(node_features,next_feature,axis=0)

    node_features = node_features.T
    node_features = torch.tensor(node_features)
    
    return node_features


# In[ ]:


def get_edge_list_for_node(input, vtu_graph):
    
    node_neighbours = vtu_graph.GetPointPoints(input)
    node_edge_list = []
    
    for node in node_neighbours:
        if node > input:  # change to >= to include the same node
            node_edge_list.append([input, node])
            
    return node_edge_list

def get_complete_edge_list(vtu_object, n_points):

    complete_edge_list = []

    for point in range(n_points):  #
        complete_edge_list = complete_edge_list + get_edge_list_for_node(point, vtu_object)
        
    complete_edge_list = np.array(complete_edge_list)
    complete_edge_list = torch.tensor(complete_edge_list).t()
        
    return complete_edge_list


# In[ ]:


def create_graph_data_list_vtu_file(vtu_file_locations, features_list, are_edges_fixed=True):
    
    #First object
    first_object = vtktools.vtu(vtu_file_locations[0]) 
    
    #This two parameters are the same for every graph, so they would be calculated just once instead of in every iteration
    n_points = first_object.GetField(features_list[0]).shape[0]
    fixed_edge_index = get_complete_edge_list(first_object, n_points) #Important parameter but same in every iteration
    fixed_edge_index = fixed_edge_index.long()
    
    #Create an empty graph data list
    graph_data_list = []
    
    for file_location in vtu_file_locations:
    
        vtu_object = vtktools.vtu(file_location)

        ## x
        node_features = get_node_features(vtu_object, features_list)

        ## edge_index. 
        if are_edges_fixed:
            edge_index = fixed_edge_index #It could be calculated for each one but since it is fixed we are going to use the same tensor for all nodes
        else:
            edge_index = get_complete_edge_list(vtu_object, n_points)

        #Create the Data objects and put them in the list called graph_data_list
        graph_data_object = Data(x=node_features, edge_index=edge_index)
        graph_data_list.append(graph_data_object)
        
    return graph_data_list , n_points


# In[ ]:


def create_mnist_edges_tensor():
    #Create MNIST fixed edges: horizontal, vertical, south-east diagonals and south-west diagonals
    mnist_edges = []

    #horizontals
    for i in range(784):
        if not (i%28 == 27):
            mnist_edges.append([i, i+1])

    #verticals
    for i in range(756):
            mnist_edges.append([i, i+28])

    #south-east diagonals
    for i in range(756):
        if not (i%28 == 27):
            mnist_edges.append([i, i+29])

    #south-west diagonals
    for i in range(756):
        if not (i%28 == 0):
            mnist_edges.append([i, i+27])

    mnist_edges = np.array(mnist_edges)
    mnist_edges = mnist_edges.T
    mnist_edges = torch.tensor(mnist_edges)
    
    return mnist_edges

def create_graph_data_list_mnist(num_samples_to_use=1000, mnist_path="../Datasets/"):
    
    ##Import the MNIST dataset
    #mnist_train = MNIST("../Datasets/", download=True, train=True, transform=Compose([
    #ToTensor(),
    ### Do not do this: dataset already comes in the range 0-1    
    ### Normalize(mean=[0.1307], std=[0.3081]), 
    # ]))
    
    mnist_train =  MNIST(mnist_path , 
               transform=ToTensor(), 
               download=True)

    #Fix number of points
    n_points = 28*28
    
    #Get the fixed edges
    fixed_edge_index = create_mnist_edges_tensor()

    #Create an initially empty array
    graph_data_list = []

    for i in range(num_samples_to_use):

        ## x
        standard_i = mnist_train[i][0][0].reshape(28*28,1)   # apply_standardization( mnist_train[14][0][0].reshape(28*28,1) )
        node_features = standard_i # .long() # torch.tensor( standard_i ).long()

        ## edge_index. It could be calculated for each one but since it is fixed 
        #we are going to use the same tensor for all nodes
        edge_index = fixed_edge_index.long()

        graph_data_object = Data(x=node_features, edge_index=edge_index)
        graph_data_list.append(graph_data_object)
        
    return graph_data_list , n_points


