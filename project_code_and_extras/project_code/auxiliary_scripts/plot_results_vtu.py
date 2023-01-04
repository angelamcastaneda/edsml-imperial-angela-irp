#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Parts of this code were taken and modified from the following sources: https://github.com/DL-WG/ROMS-tutorial/blob/main/MNIST_AE.ipynb 


import numpy as np
import matplotlib.pyplot as plt 
import pyvista as pv

from torch_geometric.loader import DataLoader
from train_AE_functions import evaluate 



# In[ ]:  VTU 2D 


def plot_model_2d_vtu(ae_model, graph_data_list_input, nDisplay, filename_to_copy_structure, 
                      show_edges=False, random=True, index_sequence=[1, 15, 30] , is_data_in_loader=False, 
                      batch_size=32, save_fig=False, filename='no_name.eps'): 
    """
    This function randomly selects a subset of the dataset entered and plots the original results in the first row and the decompressed results of the ae_model AutoEncoder in the second row. 
    Since all the graphs in the dataset share the same adjacency matrix (and the positions of each point change slightly),
     it is possible to enter a filename to indicate where to take from the structure to plot. 

    Parameters
    ----------

    ae_model: torch.nn.Module class
    Pytorch model. ae_model stands for AutoEncoder model
    
    graph_data_list_input: torch_geometric.data.Dataset  OR  list of torch_geometric.data.Data  OR  torch_geometric.loader.DataLoader
    This variable contains a dataset comprised of graphs. 
    If is_data_in_loader=False then this variable can be either  torch_geometric.data.Dataset  OR  list of torch_geometric.data.Data.
    If is_data_in_loader=True then this variable must be of type torch_geometric.loader.DataLoader 

    nDisplay: int
    Number of samples you want to display in each row

    filename_to_copy_structure: string
    Indicates the location of the vtu file where the structure (adjacency matrix and positions of each node) should be read from. 

    show_edges:Boolean
    If set to True the edges of the graphs are plotted over the image that produce the node features. Default is False

    random:If set to True then index_sequence is ignored and nDisplay number of samples are drawn from the dataset graph_data_list_input.
    If set to False then nDisplay is ignored and the displayed images are taken from the dataset graph_data_list_input using the index_sequence.
    Default is set to True. 
    
    index_sequence: List of integers.
    Indicates the indexes of the samples to display (not number of samples but their indexes). Default is set to [1, 15, 30]. 
     
    is_data_in_loader:Boolean:
    Indicates if the dataset already comes in torch_geometric.loader.DataLoader format or should be converted to it. Default is set to False. 

    batch_size:Int
    Indicates the number of samples per batch when creating the Dataloader. This parameters changes the execution time but not results (displayed image). 

    save_fig:Boolean
    Indicates if the final image should be saved in the filename variable locaton. Default is set to False. 

    filename:String ending in .eps
    If save_fig is set to True then the final produced image is saved in the filename location
    """
    
    if is_data_in_loader:
        data_loader = graph_data_list_input
        
        # Get the results when you test the model
        y_pred, y_real = evaluate(ae_model, data_loader)
    
        #Get the random indices to plot
        if random:
            randomIndex = np.random.randint(0, y_pred.shape[0], nDisplay)
        else:
            randomIndex = index_sequence[: min(y_pred.shape[0],len(index_sequence)) ]
            
        print('Indexes plotted', randomIndex) 
        nDisplay = len(randomIndex)
        indices = randomIndex
        
    else:
        
        #Get the random indices to plot
        if random:
            randomIndex = np.random.randint(0, len(graph_data_list_input), nDisplay)
        else:
            randomIndex = index_sequence[:len(graph_data_list_input)]
            
        print('Indexes plotted', randomIndex) 
        nDisplay = len(randomIndex)
        
        #Create data loader but this time DO NOT shuffle it
        graph_data_list_plot = [ graph_data_list_input[ randomIndex[j] ] for j in range(len(randomIndex)) ]
        data_loader = DataLoader(graph_data_list_plot, batch_size=batch_size, shuffle=False)
    
        # Get the results when you test the model
        y_pred, y_real = evaluate(ae_model, data_loader)
        
        indices = range(nDisplay)
        
    # NOW DO THE PLOTTING 
    
    # Read one file with the fixed structure 
    mesh0 = pv.read(filename_to_copy_structure)
    mesh = mesh0.copy() 
    
    feature_select = 'TracerBackground'
    mesh.set_active_scalars(feature_select)
    
    #Create the plot mesh
    pl = pv.Plotter(shape=(nDisplay, 2))
    
    for i, j in enumerate(indices):
        
        pl.subplot(i, 0)
        mesh = mesh0.copy()
        mesh[feature_select] = y_real[j]
        mesh.set_active_scalars(feature_select)
        pl.camera_position = [(0.0, 0.0, 1.0),(0.0, 0.0, 0.0),(0.0, 1.0, 0.0)] 
        actor = pl.add_mesh(mesh, show_edges=show_edges, reset_camera=True)
        pl.add_text("Real", font_size=15)
        
        pl.subplot(i, 1)
        mesh = mesh0.copy()
        mesh[feature_select] = y_pred[j]
        mesh.set_active_scalars(feature_select)
        pl.camera_position = [(0.0, 0.0, 1.0),(0.0, 0.0, 0.0),(0.0, 1.0, 0.0)] 
        actor = pl.add_mesh(mesh, show_edges=show_edges, reset_camera=True)
        pl.add_text("Model Predictions", font_size=15)
    
    if save_fig:  #https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.save_graphic.html 
        pl.save_graphic('saved_figures/' + filename)  
    
    pl.show()


    
# In[ ]:  

    
def compare_four_plots_models_2d_vtu(ae_model1, ae_model2, ae_model3, ae_model4, 
                                graph_data_list_input, nDisplay, filename_to_copy_structure, 
                                show_edges=False, random=True, index_sequence=[1, 15, 30] , 
                 set_pred_labels_as = ["Model1 Predictions", "Model2 Predictions", "Model3 Predictions", "Model4 Predictions"], 
                                batch_size=32, save_fig=False, filename='no_name.eps'):
    """
    This function randomly selects a subset of the dataset entered and plots the original results in the first row 
    and the decompressed results of the 4 AutoEncoders entered in the succesive rows. 
    Since all the graphs in the dataset share the same adjacency matrix (and the positions of each point change slightly), 
    it is possible to enter a filename to indicate where to take from the structure to plot. 

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

    filename_to_copy_structure: string
    Indicates the location of the vtu file where the structure (adjacency matrix and positions of each node) should be read from. 

    show_edges:Boolean
    If set to True the edges of the graphs are plotted over the image that produce the node features. Default is False

    random:If set to True then index_sequence is ignored and nDisplay number of samples are drawn from the dataset graph_data_list_input.
    If set to False then nDisplay is ignored and the displayed images are taken from the dataset graph_data_list_input using the index_sequence.
    Default is set to True. 
    
    index_sequence: List of integers.
    Indicates the indexes of the samples to display (not number of samples but their indexes). Default is set to [1, 15, 30]. 
    
    set_pred_labels_as:List of strings
    List containing the titles of each displayed row 

    batch_size:Int
    Indicates the number of samples per batch when creating the Dataloader. This parameters changes the execution time but not results (displayed image). 

    save_fig:Boolean
    Indicates if the final image should be saved in the filename variable locaton. Default is set to False. 

    filename:String ending in .eps
    If save_fig is set to True then the final produced image is saved in the filename location
    """
    
    #Get the random indices to plot
    if random:
        randomIndex = np.random.randint(0, len(graph_data_list_input), nDisplay)
    else:
        randomIndex = index_sequence[: min(len(graph_data_list_input),len(index_sequence)) ]
        
    print('Indexes plotted', randomIndex) 
    nDisplay = len(randomIndex)
    
    #Create data loader but this time do notshuffle it
    graph_data_list_plot = [ graph_data_list_input[ randomIndex[j] ] for j in range(len(randomIndex)) ]
    data_loader = DataLoader(graph_data_list_plot, batch_size=batch_size, shuffle=False)
    
    # Get the results when you test the model
    y_pred1, y_real1 = evaluate(ae_model1, data_loader)
    y_pred2, y_real2 = evaluate(ae_model2, data_loader)
    y_pred3, y_real3 = evaluate(ae_model3, data_loader)
    y_pred4, y_real4 = evaluate(ae_model4, data_loader)
    
    # NOW DO THE PLOTTING 
    
    # Read one file with the fixed structure 
    mesh0 = pv.read(filename_to_copy_structure)
    mesh = mesh0.copy() 
    
    feature_select = 'TracerBackground'
    mesh.set_active_scalars(feature_select)
    
    #Create the plot mesh
    pl = pv.Plotter(shape=(5, nDisplay))
    
    for i in range(nDisplay):
        
        pl.subplot(0, i)
        mesh = mesh0.copy()
        mesh[feature_select] = y_real1[i]
        mesh.set_active_scalars(feature_select)
        #pl.camera_position = 'xy'
        pl.camera_position = [(0.0, 0.0, 1.0),(0.0, 0.0, 0.0),(0.0, 1.0, 0.0)] 
        actor = pl.add_mesh(mesh, show_edges=show_edges, reset_camera=True)
        pl.camera.zoom(1.4)
        pl.add_text("Real Image", font_size=10)  
        
        pl.subplot(1, i)
        mesh = mesh0.copy()
        mesh[feature_select] = y_pred1[i]
        mesh.set_active_scalars(feature_select)
        pl.camera_position = [(0.0, 0.0, 1.0),(0.0, 0.0, 0.0),(0.0, 1.0, 0.0)] 
        actor = pl.add_mesh(mesh, show_edges=show_edges, reset_camera=True)
        pl.camera.zoom(1.4)
        pl.add_text(set_pred_labels_as[0], font_size=10) #"Model1 Predictions"
        
        pl.subplot(2, i)
        mesh = mesh0.copy()
        mesh[feature_select] = y_pred2[i]
        mesh.set_active_scalars(feature_select)
        pl.camera_position = [(0.0, 0.0, 1.0),(0.0, 0.0, 0.0),(0.0, 1.0, 0.0)] 
        actor = pl.add_mesh(mesh, show_edges=show_edges, reset_camera=True)
        pl.camera.zoom(1.4)
        pl.add_text(set_pred_labels_as[1], font_size=10) #"Model2 Predictions"
        
        pl.subplot(3, i)
        mesh = mesh0.copy()
        mesh[feature_select] = y_pred3[i]
        mesh.set_active_scalars(feature_select)
        pl.camera_position = [(0.0, 0.0, 1.0),(0.0, 0.0, 0.0),(0.0, 1.0, 0.0)] 
        actor = pl.add_mesh(mesh, show_edges=show_edges, reset_camera=True)
        pl.camera.zoom(1.4)
        pl.add_text(set_pred_labels_as[2], font_size=10) #"Model3 Predictions"
        
        pl.subplot(4, i)
        mesh = mesh0.copy()
        mesh[feature_select] = y_pred4[i]
        mesh.set_active_scalars(feature_select)
        pl.camera_position = [(0.0, 0.0, 1.0),(0.0, 0.0, 0.0),(0.0, 1.0, 0.0)] 
        actor = pl.add_mesh(mesh, show_edges=show_edges, reset_camera=True)
        pl.camera.zoom(1.4) 
        pl.add_text(set_pred_labels_as[3], font_size=10) #"Model4 Predictions" 
        
    if save_fig:
        pl.save_graphic('saved_figures/' + filename)  
    
    pl.show()
        

        
def compare_three_plots_models_2d_vtu(ae_model1, ae_model2, ae_model3, 
                                graph_data_list_input, nDisplay, filename_to_copy_structure, 
                                show_edges=False, random=True, index_sequence=[1, 15, 30] , 
                                set_pred_labels_as = ["Model1 Predictions", "Model2 Predictions", "Model3 Predictions"], 
                                batch_size=32, save_fig=False, filename='no_name.eps'):
    """
    This function randomly selects a subset of the dataset entered and plots the original results in the first row 
    and the decompressed results of the 3 AutoEncoders entered in the succesive rows. 
    Since all the graphs in the dataset share the same adjacency matrix (and the positions of each point change slightly), 
    it is possible to enter a filename to indicate where to take from the structure to plot. 

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

    filename_to_copy_structure: string
    Indicates the location of the vtu file where the structure (adjacency matrix and positions of each node) should be read from. 

    show_edges:Boolean
    If set to True the edges of the graphs are plotted over the image that produce the node features. Default is False

    random:If set to True then index_sequence is ignored and nDisplay number of samples are drawn from the dataset graph_data_list_input.
    If set to False then nDisplay is ignored and the displayed images are taken from the dataset graph_data_list_input using the index_sequence.
    Default is set to True. 
    
    index_sequence: List of integers.
    Indicates the indexes of the samples to display (not number of samples but their indexes). Default is set to [1, 15, 30]. 
    
    set_pred_labels_as:List of strings
    List containing the titles of each displayed row 

    batch_size:Int
    Indicates the number of samples per batch when creating the Dataloader. This parameters changes the execution time but not results (displayed image). 

    save_fig:Boolean
    Indicates if the final image should be saved in the filename variable locaton. Default is set to False. 

    filename:String ending in .eps
    If save_fig is set to True then the final produced image is saved in the filename location
    """
    
    #Get the random indices to plot
    if random:
        randomIndex = np.random.randint(0, len(graph_data_list_input), nDisplay)
    else:
        randomIndex = index_sequence[: min(len(graph_data_list_input),len(index_sequence)) ]
        
    print('Indexes plotted', randomIndex) 
    nDisplay = len(randomIndex)
    
    #Create data loader but this time do notshuffle it
    graph_data_list_plot = [ graph_data_list_input[ randomIndex[j] ] for j in range(len(randomIndex)) ]
    data_loader = DataLoader(graph_data_list_plot, batch_size=batch_size, shuffle=False)
    
    # Get the results when you test the model
    y_pred1, y_real1 = evaluate(ae_model1, data_loader)
    y_pred2, y_real2 = evaluate(ae_model2, data_loader)
    y_pred3, y_real3 = evaluate(ae_model3, data_loader)
    
    # NOW DO THE PLOTTING 
    
    # Read one file with the fixed structure 
    mesh0 = pv.read(filename_to_copy_structure)
    mesh = mesh0.copy() 
    
    feature_select = 'TracerBackground'
    mesh.set_active_scalars(feature_select)
    
    #Create the plot mesh
    pl = pv.Plotter(shape=(4, nDisplay))
    
    for i in range(nDisplay):
        
        pl.subplot(0, i)
        mesh = mesh0.copy()
        mesh[feature_select] = y_real1[i]
        mesh.set_active_scalars(feature_select)
        #pl.camera_position = 'xy'
        pl.camera_position = [(0.0, 0.0, 1.0),(0.0, 0.0, 0.0),(0.0, 1.0, 0.0)] 
        actor = pl.add_mesh(mesh, show_edges=show_edges, reset_camera=True)
        pl.add_text("Real", font_size=15)
        
        pl.subplot(1, i)
        mesh = mesh0.copy()
        mesh[feature_select] = y_pred1[i]
        mesh.set_active_scalars(feature_select)
        pl.camera_position = [(0.0, 0.0, 1.0),(0.0, 0.0, 0.0),(0.0, 1.0, 0.0)] 
        actor = pl.add_mesh(mesh, show_edges=show_edges, reset_camera=True)
        pl.add_text(set_pred_labels_as[0], font_size=15) #"Model1 Predictions"
        
        pl.subplot(2, i)
        mesh = mesh0.copy()
        mesh[feature_select] = y_pred2[i]
        mesh.set_active_scalars(feature_select)
        pl.camera_position = [(0.0, 0.0, 1.0),(0.0, 0.0, 0.0),(0.0, 1.0, 0.0)] 
        actor = pl.add_mesh(mesh, show_edges=show_edges, reset_camera=True)
        pl.add_text(set_pred_labels_as[1], font_size=15) #"Model2 Predictions"
        
        pl.subplot(3, i)
        mesh = mesh0.copy()
        mesh[feature_select] = y_pred3[i]
        mesh.set_active_scalars(feature_select)
        pl.camera_position = [(0.0, 0.0, 1.0),(0.0, 0.0, 0.0),(0.0, 1.0, 0.0)] 
        actor = pl.add_mesh(mesh, show_edges=show_edges, reset_camera=True)
        pl.add_text(set_pred_labels_as[2], font_size=15) #"Model3 Predictions"
        
        
    if save_fig:
        pl.save_graphic('saved_figures/' + filename)  
    
    pl.show()
        
