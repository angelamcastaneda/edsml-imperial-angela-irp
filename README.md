This project aims to produce the best possible AutoEncoder for a given dataset by evaluating the two following methods: a first approach on how to combine Graph Convolutional layers with traditional AutoEncoders on graphs that share the same adjacency matrix; and testing and analysing different combinations of activation functions commonly used in Neural Networks.

For this matter, various functions were produced, and they are located in the subfolder "project_code_and_extras\project_code\auxiliary_scripts". All other subfolders from the folder "project_code_and_extras\project_code\" contain different jupyter notebooks that use the functions to test different models of AutoEncoders. 

Since this project is divided in subfolders, each one of them has its own README explaining their purpose, instead of the whole project having only one big README. The analysis are summarized in the report but can also be checked across the notebooks that ran the models. 

One important note is that before running any code the environment.yml file from the subfolder "project_code_and_extras.installations_and_environments" must be installed first.  


References
- [1] https://avandekleut.github.io/vae/  
- [2] https://github.com/DL-WG/ROMS-tutorial/blob/main/MNIST_AE.ipynb   
- [3] https://medium.com/@rtsrumi07/understanding-graph-neural-network-with-hands-on-example-part-2-139a691ebeac 
- [4] https://github.com/deepfindr/gvae/blob/master/dataset.py 

- [5] https://github.com/FluidityProject/fluidity/blob/main/python/vtktools.py 
- [6] https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-larger-datasets 

- [7] https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798 
- [8] https://stackoverflow.com/questions/65307833/why-is-the-decoder-in-an-autoencoder-uses-a-sigmoid-on-the-last-layer   
- [9] https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py
