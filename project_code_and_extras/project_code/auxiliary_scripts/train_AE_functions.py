"""Train, validate and test functions for the models implemented in the WaveSuite project.

Most of the code in this file was modified from the Machine Learning module of the MSc Environmental Data Science and Machine Learning. 
The functions implemented use GPU by default, but adding "device = torch.device("cpu")" changes this behaviour to work with CPU"""

#!/usr/bin/env python 
# coding: utf-8 

__all__ = ['train_AE_epoch', 'validate_AE_epoch', 'train_AE', 'evaluate', 'model_MSE_error', 'set_seed' ] 


# In[ ]:

import numpy as np

import torch
import torch.nn as nn 

from AE_Models import classic_AE, GCN_AE, GCN_AE_GCN, GCN_pooling_AE
from sequential.AE_Sequential_Models import classic_AE_Sequential, GCN_AE_Sequential, GCN_AE_GCN_Sequential 

from livelossplot import PlotLosses 

import random


## Use GPU for training
## Inside the functions the default is:  
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
## However, for very large datasets it is recommended to do:  device = torch.device("cpu")  given that despite taking longer times to run it has more memory. 


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False  ##uses the inbuilt cudnn auto-tuner to find the fastest convolution algorithms. -
    torch.backends.cudnn.enabled   = False

    return True


# For training


# In[ ]:


def train_AE_epoch(ae_model, optimizer, criterion, data_loader, 
                   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") , num_nodes=None , seed=42):
    """
    Updates the parameters from the inputted ae_model, by performing training on the data_loader during one epoch 

    Parameters
    ----------

    ae_model: torch.nn.Module class
    Pytorch model. ae_model stands for AutoEncoder model 
    
    optimizer: torch.optim module
    optimizer such as torch.optim.Adam or torch.optim.SGD
    
    criterion: nn.MSELoss 
    nn.MSELoss is the default and can only be modified inside the function
    
    data_loader: torch_geometric.loader.DataLoader
    Dataloader that contains a dataset comprised of graphs

    device:torch.device
    The default is torch.device("cuda:0" if torch.cuda.is_available() else "cpu") but can be changed to device = torch.device("cpu") if required for large datasets. 
    
    num_nodes : Int 
    Number of nodes in the graph, similar to number of pixels in an image. 

    seed: Int
    Random seed number 

    Returns
    -------

    float
        averaged loss of ae_model for the data_loader inputted
    """
    
    set_seed(seed)
    
    ## Set the model to mode TRAIN
    ae_model.train()
    
    batch_size = data_loader.batch_size
    
    train_loss = 0
    
    for batch in data_loader:
        
        batch = batch.to(device)  # or is it only batch.to(device) ?
        
        ## Reset gradients
        optimizer.zero_grad()
        
        ## Compute the current output values of the model...
        batch_train = batch.x.float()
        temp_train_decoded = ae_model(batch_train, batch.edge_index, batch_size , batch.batch) #batch_index = batch.batch 
        # ... and make them the same shape
        batch_train = batch_train.view(temp_train_decoded.shape[0],temp_train_decoded.shape[1])
        
        ## Compute the loss ...
        ## (Remember that label in this case is the whole batch because the model is an AE)
        # loss = criterion(temp_train_decoded, batch_train ) 
        loss = ((temp_train_decoded - batch_train)**2).mean()
        
        # ... and the gradients of the parameters ...
        loss.backward()
        ## .. to update the parameters using the gradients
        optimizer.step()  
        
        ## ... And finally add the loss in every batch (and scale the loss according to every batch of the epoch)
        if num_nodes is None:
            train_loss += loss*batch_size
        else: 
            train_loss += loss*(batch.size(0)/num_nodes)  
            
        # print('train batch_size is '+str(batch_size)+' and batch.size(0)/num_nodes is '+str(batch.size(0)/num_nodes) ) #DELETE THIS 
    
    train_loss = train_loss/len(data_loader.dataset)
    return train_loss 


# In[ ]:


def validate_AE_epoch(ae_model, criterion, data_loader,
                      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") , num_nodes=None , seed=42):
    """
    Computes the averaged loss for the data_loader when reconstructing the data using ae_model (AutoEncoder model) 

    Parameters
    ----------

    ae_model: torch.nn.Module class
    Pytorch model. ae_model stands for AutoEncoder model
    
    criterion: nn.MSELoss 
    nn.MSELoss is the default and can only be modified inside the function
    
    data_loader: torch_geometric.loader.DataLoader
    Dataloader that contains a dataset comprised of graphs

    device:torch.device
    The default is torch.device("cuda:0" if torch.cuda.is_available() else "cpu") but can be changed to device = torch.device("cpu") if required for large datasets. 
    
    num_nodes : Int 
    Number of nodes in the graph, similar to number of pixels in an image. 

    seed: Int
    Random seed number 

    Returns
    -------

    float
        averaged loss of ae_model for the data_loader inputted
    """
    
    set_seed(seed)
    
    ## Set the model to mode EVAL
    ae_model.eval()
    
    batch_size = data_loader.batch_size
    
    validation_loss = 0 
    
    for batch in data_loader:
        
        # torch.no_grad() saves time by telling the model not to take the gradients into account, 
        # because the model has already been trained
        with torch.no_grad(): 
            batch = batch.to(device) # or is it only batch.to(device) ?
            
            ## Compute the current output values of the model...
            batch_validation = batch.x.float()
            temp_validation_decoded = ae_model(batch_validation, batch.edge_index, batch_size , batch.batch) #batch_index = batch.batch
            # ... and make them the same shape
            batch_validation = batch_validation.view(temp_validation_decoded.shape[0],temp_validation_decoded.shape[1])
            
            ## Compute the loss ...
            ## (Remember that label in this case is the whole batch because the model is an AE)
            # loss = criterion(temp_validation_decoded, batch_validation ) 
            loss = ((temp_validation_decoded - batch_validation)**2).mean()
            
            ## ... and add the loss in every batch (and scale the loss according to every batch of the epoch)
            if num_nodes is None:
                validation_loss += loss*batch_size 
            else: 
                validation_loss += loss*(batch.size(0)/num_nodes)  
            
            # print('val batch_size is '+str(batch_size)+' and batch.size(0)/num_nodes is '+str(batch.size(0)/num_nodes) ) #DELETE THIS 
    
    validation_loss = validation_loss/len(data_loader.dataset)
    return validation_loss


# In[ ]:


def train_AE(train_loader, validation_loader, 
             num_nodes, num_features, embedding_sequence, latent_space_dim,
             ae_model_type='classic_AE', num_epochs=50, plot_losses=True, 
             device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") , seed=42, 
             use_sgd_instead_of_adam=False, lr=1e-1, momentum=0.2):
    """
    Trains an AutoEncoder model of type ae_model_type, on the train_loader dataset during num_epochs epochs. 
    Also outputs the model losses on train_loader and validation_loader. 

    Parameters
    ----------

    train_loader: torch_geometric.loader.DataLoader
    Dataset to train the model on
    
    validation_loader: torch_geometric.loader.DataLoader
    Dataset used to validate the results

    num_nodes : Int 
    Number of nodes in the graph, similar to number of pixels in an image.

    num_features: Int
    Number of features in each node. It is always 1 in this project. 

    embedding_sequence: Int list
    Sequence of number of channels in each GCN-layer. 3 entries are required for GCN_AE, 
    6 entries are required for GCN_AE_GCN, 6 entries are required por GCN_pooling_AE and No entries are required for classic_AE

    latent_space_dim: Int
    Number of points in the latent space

    ae_model_type:torch.nn.Module class
    Pytorch model. ae_model_type stands for AutoEncoder model

    num_epochs: Int 
    Number of epochs to use during training

    plot_losses: Boolean
    True if you want to visualize the losses of the training and validation data_loaders

    device:torch.device
    The default is torch.device("cuda:0" if torch.cuda.is_available() else "cpu") but can be changed to device = torch.device("cpu") if required for large datasets. 

    seed: Int
    Random seed number 

    use_sgd_instead_of_adam:Boolean
    Default is False. If set to True an SGD optimizer is used instead of ADAM

    lr: Float
    Learning rate to use in case use_sgd_instead_of_adam is set to True

    momentum: Float
    momentum to use in case use_sgd_instead_of_adam is set to True

    Returns
    -------

    torch.nn.Module class
        Trained AutoEncoder model (torch model) 

    list
        list of dictionaries that contains information about the train and validation loss in every epoch
    """
    
    set_seed(seed)
    
    invalid_entry = False
    
    if ae_model_type=='classic_AE':
        model = classic_AE(num_nodes, num_features, latent_space_dim).to(device) #This one does not have embedding sequence
    elif ae_model_type=='GCN_AE':
        model = GCN_AE(num_nodes, num_features, embedding_sequence, latent_space_dim).to(device)
    elif ae_model_type=='GCN_AE_GCN':
        model = GCN_AE_GCN(num_nodes, num_features, embedding_sequence, latent_space_dim).to(device)
    elif ae_model_type=='GCN_pooling_AE':
        model = GCN_pooling_AE(num_nodes, num_features, embedding_sequence, latent_space_dim).to(device) 
        
    elif ae_model_type=='classic_AE_Sequential':
        model = classic_AE_Sequential(num_nodes, num_features, latent_space_dim).to(device) #This one does not have embedding sequence
    elif ae_model_type=='GCN_AE_Sequential':
        model = GCN_AE_Sequential(num_nodes, num_features, embedding_sequence, latent_space_dim).to(device)
    elif ae_model_type=='GCN_AE_GCN_Sequential':
        model = GCN_AE_GCN_Sequential(num_nodes, num_features, embedding_sequence, latent_space_dim).to(device)
        
    else:
        invalid_entry = True
        print('\n An invalid ae_model_type was provided, a classic AE was used as default')
        model = classic_AE(num_nodes, num_features, latent_space_dim).to(device)
    
    ## The ADAM is the default, since the ADAM optimizer performs better in this case compared against SGD  
    optimizer = torch.optim.Adam(model.parameters())
    if use_sgd_instead_of_adam:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    ## The loss here is the mean_squared_error because the model is an AE
    criterion = nn.MSELoss()
    
    if plot_losses:
        liveloss = PlotLosses()
    loss_register = []
    
    for epoch in range(num_epochs):
        #During each epoch update the parameters
        train_temp_loss = train_AE_epoch(model, optimizer, criterion, train_loader, device, num_nodes)
        
        #Optional: Check the validation loss in each epoch
        validation_temp_loss = validate_AE_epoch(model, criterion, validation_loader, device, num_nodes) 
        
        #Compute the losses in each iteration 
        logs = {}
        logs['' + 'log loss'] = train_temp_loss.item()
        logs['val_' + 'log loss'] = validation_temp_loss.item()
        loss_register.append(logs)
        
        #Optionally plot the train and validation loss   
        if plot_losses:
            liveloss.update(logs)
            liveloss.draw()
            
    if invalid_entry == True:
        print('\n An invalid ae_model_type was provided, a classic AE was used as default') 
        
    print("\n Number of parameters: ", sum(p.numel() for p in model.parameters()))
    
    #At the end just return 1. the model with the trained parameters and 2. the list of losses in every epoch
    return model, loss_register 


# In[ ]:


# For evaluating after training


# In[ ]:


def evaluate(ae_model, data_loader, 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ):
    """
    Computes the averaged loss for the data_loader when reconstructing the data using ae_model (AutoEncoder model) 

    Parameters
    ----------

    ae_model: torch.nn.Module class
    Pytorch model. ae_model stands for AutoEncoder model
    
    data_loader: torch_geometric.loader.DataLoader
    Dataloader that contains a dataset comprised of graphs

    device:torch.device
    The default is torch.device("cuda:0" if torch.cuda.is_available() else "cpu") but can be changed to device = torch.device("cpu") if required for large datasets. 

    Returns
    -------

    numpy.ndarray
        Reconstructed features using the AutoEncoder ae_model

    numpy.ndarray
        Original features inputted to the AutoEncoder ae_model
    """
    
    ae_model.eval()
    
    batch_size = data_loader.batch_size 
    
    ys, y_preds = [], []
    
    for batch in data_loader:
        with torch.no_grad():
            batch = batch.to(device)  # or is it only batch.to(device) ?
            
            ## Compute the current output values of the model
            batch_test = batch.x.float()
            temp_test_decoded = ae_model(batch_test, batch.edge_index, batch_size , batch.batch) #batch_index = batch.batch
            
            #Identify y and y_pred
            y = batch_test
            y_pred = temp_test_decoded
            
            #Reshape y to have the same shape as y_pred
            y = y.reshape(y_pred.shape)
            
            #Append the results to each list 
            ys.append(y.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())
            
    return np.concatenate(y_preds, 0),  np.concatenate(ys, 0) 




def model_MSE_error(ae_model, data_loader, error_type='mean'):
    """
    Computes the averaged loss for the data_loader when reconstructing the data using ae_model (AutoEncoder model) 

    Parameters
    ----------

    ae_model: torch.nn.Module class
    Pytorch model. ae_model stands for AutoEncoder model
    
    data_loader: torch_geometric.loader.DataLoader
    Dataloader that contains a dataset comprised of graphs

    error_type:string
    Set to 'mean' to use the MSE mean loss and 'sum' to use the MSE mean loss. The default is 'mean'. 

    Returns
    -------

    float
        MSE loss comparing the original features and the decompressed (reconstructed) results from the ae_model
    """
    
    # Get the results when you test the model
    y_pred, y_real = evaluate(ae_model, data_loader)
    
    if error_type=='mean':
        error = ((y_pred-y_real)**2).mean() 
    elif error_type=='sum':
        error = ((y_pred-y_real)**2).sum() 
    else:
        error = ((y_pred-y_real)**2).mean() 
        print('Invalid error_type was provided, mean was used as default')
    
    return error 