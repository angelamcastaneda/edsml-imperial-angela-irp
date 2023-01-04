#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Parts of this code were taken and modified from the following sources: https://github.com/DL-WG/ROMS-tutorial/blob/main/MNIST_AE.ipynb 


import numpy as np
import matplotlib.pyplot as plt 
import pyvista as pv

from torch_geometric.loader import DataLoader
from train_AE_functions import evaluate 



# In[ ]:  MNIST 



def evaluate_and_plot_mnist(ae_model, data_loader, nDisplay, 
                            random=True, index_sequence=[1, 15, 30] , 
                            save_fig=False, filename='no_name.png', cut_preds_01=False):
    """
    This function randomly selects a subset of the dataset entered and plots the original results in the first row and the decompressed results of the ae_model AutoEncoder in the second row. 

    Parameters
    ----------

    ae_model: torch.nn.Module class
    Pytorch model. ae_model stands for AutoEncoder model

    data_loader: torch_geometric.loader.DataLoader
    Dataloader that contains a dataset comprised of graphs

    nDisplay: int
    Number of samples you want to display in each row

    random:If set to True then index_sequence is ignored and nDisplay number of samples are drawn from the data_loader.
    If set to False then nDisplay is ignored and the displayed images are taken from the data_loader using the index_sequence.
    Default is set to True. 
    
    index_sequence: List of integers.
    Indicates the indexes of the samples to display (not number of samples but their indexes). Default is set to [1, 15, 30]. 
     
    save_fig:Boolean
    Indicates if the final image should be saved in the filename variable locaton. Default is set to False. 

    filename:String ending in .png
    If save_fig is set to True then the final produced image is saved in the filename location 
    """
    
    #Get the results when you test the model
    y_pred, y_real = evaluate(ae_model, data_loader)
    
    #Reshape MNIST samples
    y_pred = y_pred.reshape(y_pred.shape[0],28,28)  
    y_real = y_real.reshape(y_real.shape[0],28,28) 
    
    if cut_preds_01:
        y_pred =  np.array( [ max(min(y_pred[a][b][c],1),0) 
                                   for a in range(y_pred.shape[0]) 
                                   for b in range(y_pred.shape[1]) 
                                   for c in range(y_pred.shape[2])] ).reshape(y_pred.shape)
    
        #Get the random indices to plot
    if random:
        randomIndex = np.random.randint(0, y_pred.shape[0], nDisplay)
    else:
        randomIndex = index_sequence[:y_pred.shape[0]]
    print('Indexes plotted', randomIndex) 
    
    fig, axs = plt.subplots(2, nDisplay)
    axs = axs.reshape([2,nDisplay]) #In case nDisplay=1 this will prevent error

    for i in range(nDisplay):
        
        axs[0,i].imshow(np.reshape(y_real[randomIndex[i]], (28, 28)), cmap = 'gray_r')
        axs[0,i].axis('off')
    
        axs[1,i].imshow(np.reshape(y_pred[randomIndex[i]], (28, 28)), cmap = 'gray_r')
        axs[1,i].axis('off')
        
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('saved_figures/' + filename)
    
    
# In[ ]:
    


def compare_four_plots_models_mnist(ae_model1, ae_model2, ae_model3, ae_model4, graph_data_list_input, 
                               nDisplay=3, axis_off=True, random=True, index_sequence=[20, 71, 700] , 
                               set_pred_labels_as = ['pred_model1', 'pred_model2', 'pred_model3', 'pred_model4'],
                               batch_size=32 , figsize=(16,12), save_fig=False, filename='no_name.png',
                                   cut_preds_01_model1=False ):
    """
    This function randomly selects a subset of the dataset entered and plots the original results in the first row 
    and the decompressed results of the 4 AutoEncoders entered in the succesive rows. 

    Parameters
    ----------

    ae_model1: torch.nn.Module class
    First AutoEncoder model. Its results are displayed in the second row

    ae_model2: torch.nn.Module class
    Second AutoEncoder model. Its results are displayed in the third row

    ae_model3: torch.nn.Module class
    Third AutoEncoder model. Its results are displayed in the fourth row

    ae_model4: torch.nn.Module class
    Fourth AutoEncoder model. Its results are displayed in the fifth row 

    graph_data_list_input: torch_geometric.data.Dataset  OR  list of torch_geometric.data.Data
    This variable contains a dataset comprised of graphs.  

    nDisplay: int
    Number of samples you want to display in each row

    axis_off:Boolean
    If set to False the axis are shown. Default is True. 

    random:If set to True then index_sequence is ignored and nDisplay number of samples are drawn from the data_loader.
    If set to False then nDisplay is ignored and the displayed images are taken from the data_loader using the index_sequence.
    Default is set to True. 
    
    index_sequence: List of integers.
    Indicates the indexes of the samples to display (not number of samples but their indexes). Default is set to [20, 71, 700]. 

    set_pred_labels_as:List of strings
    List containing the titles of each displayed row 

    batch_size:Int
    Indicates the number of samples per batch when creating the Dataloader. This parameters changes the execution time but not results (displayed image). 
     
    figsize:tuple
    Size of the figure. Default is (16,12). 

    save_fig:Boolean
    Indicates if the final image should be saved in the filename variable locaton. Default is set to False. 

    filename:String ending in .png
    If save_fig is set to True then the final produced image is saved in the filename location 

    cut_preds_01_model1:Boolean
    When set to True displays the first model in the [0,1] range. Default is False. 
    This was done like this to show that results from Model1 were actually the best, but they were not geing displayed properly. 
    """
    
    #Get the random indices to plot
    if random:
        randomIndex = np.random.randint(0, len(graph_data_list_input), nDisplay)
    else:
        randomIndex = index_sequence[: min(len(graph_data_list_input),len(index_sequence)) ]
       
    print('Indexes plotted', randomIndex) 
    nDisplay = len(randomIndex)
    
    #Create data loader but this time DO NOT shuffle it
    graph_data_list_plot = [ graph_data_list_input[ randomIndex[j] ] for j in range(len(randomIndex)) ]
    data_loader = DataLoader(graph_data_list_plot, batch_size=batch_size, shuffle=False)
    
    #Get the results when you test the model
    y_pred1, y_real1 = evaluate(ae_model1, data_loader)
    y_pred2, y_real2 = evaluate(ae_model2, data_loader)
    y_pred3, y_real3 = evaluate(ae_model3, data_loader)
    y_pred4, y_real4 = evaluate(ae_model4, data_loader)
    
    #Reshape MNIST samples
    y_pred1 = y_pred1.reshape(y_pred1.shape[0],28,28) 
    y_real1 = y_real1.reshape(y_real1.shape[0],28,28) 
    
    y_pred2 = y_pred2.reshape(y_pred2.shape[0],28,28) 
    y_real2 = y_real2.reshape(y_real2.shape[0],28,28)
    
    y_pred3 = y_pred3.reshape(y_pred3.shape[0],28,28) 
    y_real3 = y_real3.reshape(y_real3.shape[0],28,28)
    
    y_pred4 = y_pred4.reshape(y_pred4.shape[0],28,28) 
    y_real4 = y_real4.reshape(y_real4.shape[0],28,28)
    
    #Cut the model1 results
    if cut_preds_01_model1:
        y_pred1 =  np.array( [ max(min(y_pred1[a][b][c],1),0) 
                                   for a in range(y_pred1.shape[0]) 
                                   for b in range(y_pred1.shape[1]) 
                                   for c in range(y_pred1.shape[2])] ).reshape(y_pred1.shape)
    
    #NOW DO THE PLOTTING 
    fig, axs = plt.subplots(5, nDisplay, figsize=figsize)
    axs = axs.reshape([5,nDisplay]) #In case nDisplay=1 this will prevent error

    for i in range(nDisplay):
        axs[0,i].imshow(np.reshape(y_real1[i], (28, 28)), cmap = 'gray_r')
        axs[0,i].title.set_text('real') 
        #axs[0,i].set_ylabel('real')
        #axs[0,i].yaxis.set_ticklabels([])
        #axs[0,i].xaxis.set_ticklabels([])
        if axis_off:
            axs[0,i].axis('off')
            
        axs[1,i].imshow(np.reshape(y_pred1[i], (28, 28)), cmap = 'gray_r')
        axs[1,i].title.set_text(set_pred_labels_as[0])  #'pred_model1'
        #axs[1,i].set_ylabel('pred_model1')
        #axs[1,i].yaxis.set_ticklabels([])
        #axs[1,i].xaxis.set_ticklabels([])
        if axis_off:
            axs[1,i].axis('off')
    
        axs[2,i].imshow(np.reshape(y_pred2[i], (28, 28)), cmap = 'gray_r')
        axs[2,i].title.set_text(set_pred_labels_as[1])  #'pred_model2'
        #axs[2,i].set_ylabel('pred_model2')
        #axs[2,i].yaxis.set_ticklabels([])
        #axs[2,i].xaxis.set_ticklabels([])
        if axis_off:
            axs[2,i].axis('off')
        
        
        axs[3,i].imshow(np.reshape(y_pred3[i], (28, 28)), cmap = 'gray_r')
        axs[3,i].title.set_text(set_pred_labels_as[2])  #'pred_model3'
        #axs[3,i].set_ylabel('pred_model3')
        #axs[3,i].yaxis.set_ticklabels([])
        #axs[3,i].xaxis.set_ticklabels([])
        if axis_off:
            axs[3,i].axis('off')
            
        axs[4,i].imshow(np.reshape(y_pred4[i], (28, 28)), cmap = 'gray_r')
        axs[4,i].title.set_text(set_pred_labels_as[3])  #'pred_model4'
        #axs[4,i].set_ylabel('pred_model4')
        #axs[4,i].yaxis.set_ticklabels([])
        #axs[4,i].xaxis.set_ticklabels([])
        if axis_off:
            axs[4,i].axis('off')
        
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('saved_figures/' + filename)
  
  

    
def compare_three_plots_models_mnist(ae_model1, ae_model2, ae_model3, graph_data_list_input, 
                                     nDisplay=3, axis_off=True, random=True, index_sequence=[20, 71, 700] ,
                                     set_pred_labels_as = ['pred_model1', 'pred_model2', 'pred_model3'],
                                     batch_size=32 , figsize=(16,12), save_fig=False, filename='no_name.png',
                                     cut_preds_01_model1 = False ):
    """
    This function randomly selects a subset of the dataset entered and plots the original results in the first row 
    and the decompressed results of the 3 AutoEncoders entered in the succesive rows. 

    Parameters
    ----------

    ae_model1: torch.nn.Module class
    First AutoEncoder model. Its results are displayed in the second row

    ae_model2: torch.nn.Module class
    Second AutoEncoder model. Its results are displayed in the third row

    ae_model3: torch.nn.Module class
    Third AutoEncoder model. Its results are displayed in the fourth row

    graph_data_list_input: torch_geometric.data.Dataset  OR  list of torch_geometric.data.Data
    This variable contains a dataset comprised of graphs.  

    nDisplay: int
    Number of samples you want to display in each row

    axis_off:Boolean
    If set to False the axis are shown. Default is True. 

    random:If set to True then index_sequence is ignored and nDisplay number of samples are drawn from the data_loader.
    If set to False then nDisplay is ignored and the displayed images are taken from the data_loader using the index_sequence.
    Default is set to True. 
    
    index_sequence: List of integers.
    Indicates the indexes of the samples to display (not number of samples but their indexes). Default is set to [20, 71, 700]. 

    set_pred_labels_as:List of strings
    List containing the titles of each displayed row 

    batch_size:Int
    Indicates the number of samples per batch when creating the Dataloader. This parameters changes the execution time but not results (displayed image). 
     
    figsize:tuple
    Size of the figure. Default is (16,12). 

    save_fig:Boolean
    Indicates if the final image should be saved in the filename variable locaton. Default is set to False. 

    filename:String ending in .png
    If save_fig is set to True then the final produced image is saved in the filename location 

    cut_preds_01_model1:Boolean
    When set to True displays the first model in the [0,1] range. Default is False. 
    This was done like this to show that results from Model1 were actually the best, but they were not geing displayed properly. 
    """
    
    #Get the random indices to plot
    if random:
        randomIndex = np.random.randint(0, len(graph_data_list_input), nDisplay)
    else:
        randomIndex = index_sequence[: min(len(graph_data_list_input),len(index_sequence)) ]
       
    print('Indexes plotted', randomIndex) 
    nDisplay = len(randomIndex)
    
    #Create data loader but this time DO NOT shuffle it
    graph_data_list_plot = [ graph_data_list_input[ randomIndex[j] ] for j in range(len(randomIndex)) ]
    data_loader = DataLoader(graph_data_list_plot, batch_size=batch_size, shuffle=False)
    
    #Get the results when you test the model
    y_pred1, y_real1 = evaluate(ae_model1, data_loader)
    y_pred2, y_real2 = evaluate(ae_model2, data_loader)
    y_pred3, y_real3 = evaluate(ae_model3, data_loader)
    
    #Reshape MNIST samples
    y_pred1 = y_pred1.reshape(y_pred1.shape[0],28,28) 
    y_real1 = y_real1.reshape(y_real1.shape[0],28,28) 
    
    y_pred2 = y_pred2.reshape(y_pred2.shape[0],28,28) 
    y_real2 = y_real2.reshape(y_real2.shape[0],28,28)
    
    y_pred3 = y_pred3.reshape(y_pred3.shape[0],28,28) 
    y_real3 = y_real3.reshape(y_real3.shape[0],28,28)
    
    #Cut the model1 results
    if cut_preds_01_model1:
        y_pred1 =  np.array( [ max(min(y_pred1[a][b][c],1),0) 
                                   for a in range(y_pred1.shape[0]) 
                                   for b in range(y_pred1.shape[1]) 
                                   for c in range(y_pred1.shape[2])] ).reshape(y_pred1.shape)
    
    #NOW DO THE PLOTTING 
    fig, axs = plt.subplots(4, nDisplay, figsize=figsize)
    axs = axs.reshape([4,nDisplay]) #In case nDisplay=1 this will prevent error

    for i in range(nDisplay):
        axs[0,i].imshow(np.reshape(y_real1[i], (28, 28)), cmap = 'gray_r')
        axs[0,i].title.set_text('real') 
        #axs[0,i].set_ylabel('real')
        #axs[0,i].yaxis.set_ticklabels([])
        #axs[0,i].xaxis.set_ticklabels([])
        if axis_off:
            axs[0,i].axis('off')
            
        axs[1,i].imshow(np.reshape(y_pred1[i], (28, 28)), cmap = 'gray_r')
        axs[1,i].title.set_text(set_pred_labels_as[0])  #'pred_model1'
        #axs[1,i].set_ylabel('pred_model1')
        #axs[1,i].yaxis.set_ticklabels([])
        #axs[1,i].xaxis.set_ticklabels([])
        if axis_off:
            axs[1,i].axis('off')
    
        axs[2,i].imshow(np.reshape(y_pred2[i], (28, 28)), cmap = 'gray_r')
        axs[2,i].title.set_text(set_pred_labels_as[1])  #'pred_model2'
        #axs[2,i].set_ylabel('pred_model2')
        #axs[2,i].yaxis.set_ticklabels([])
        #axs[2,i].xaxis.set_ticklabels([])
        if axis_off:
            axs[2,i].axis('off')
        
        
        axs[3,i].imshow(np.reshape(y_pred3[i], (28, 28)), cmap = 'gray_r')
        axs[3,i].title.set_text(set_pred_labels_as[2])  #'pred_model3'
        #axs[3,i].set_ylabel('pred_model3')
        #axs[3,i].yaxis.set_ticklabels([])
        #axs[3,i].xaxis.set_ticklabels([])
        if axis_off:
            axs[3,i].axis('off')
            
        
    plt.tight_layout()
    
    if save_fig:
        plt.savefig('saved_figures/' + filename)
  
  