
This folder is the STEP 3 of the project.

### Some things to take into account before describing STEP 3:

Initially, I thought that if the code ran for the air-pollution dataset then it would also run for the WaveSuite dataset. I WAS WRONG. 

The following reasons made me change the code significantly and they also took a significant amount of time from my project:

- Given the large size of the graphs in the WaveSuite dataset, it was not possible to read the adjacency matrix for every file (which was done in the air-pollution dataset and given its small size, the execution time did not matter). Instead, it necessary to get the adjacency matrix of just one file and storing it to use it later. This was done in the jupyter notebook "saving_complete_edge_list.ipynb" and as can be seen in that notebook,the execution time of that single step took 20 hours. The adjacency matrix is stored in the subfolder "edges_lists". 

- Given the large size of the graphs, it was necessary to implement a special class that allowed the torch-geometric package to access one file at a time, thus saving RAM memory. This special class is described in this link: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-larger-datasets ; and I also copied code from this link https://github.com/deepfindr/gvae/blob/master/dataset.py ,  and then modified it to be able to implement the required class.

P.S. There is a subfolder called "understanding wavesuite files". This folder is not important for the project. However, I decided to leave it here given that in case someone will expand this project in the future, that person will find that folder incredibly helpful to understand the Datasets of the project, something that has to be done before expanding the code.  

### STEP 3 :

Lets remember the conclusions from the previous step:

- ADAM produces better results
- The number of points in the latent space dimension is an important variable that should be tuned.
- The activation function used at the output of the AutEncoder is of great importance, and Sigmoid is not always the best option (In MNIST, no activation function turned out better than MNIST). 
- Combining GCN-layers does improve the MSE loss. However, using it requires more computational resources and execution time; and the tradeoff is not worth it. 

It could be said that STEP 3 is comprised of the following substeps:

- Step 3.1.  subfolder "choosing_latent_space_dimension"
- Step 3.2.  subfolder "checking_GCN_models"
- Step 3.3.  subfolder "extra_layer_model"
- Step 3.4.  subfolder "final_model"

### Results : 

The first thing to do is to determine an adequate number of points for the latent space dimension. This was done in the subfolder "choosing_latent_space_dimension" and it was determined that this value should be 256. In that folder it was also determined that the best output activation function for the WaveSuite dataset was no output activation at all. 

After that the GCN-AE and GCN-AE-GCN that had no output activation at all were implemented in the subfolder "checking_GCN_models".
Since they took really long to run, two more approaches were considered (in separate subfolders).

In the first approach, a classic AutoEncoder model with 256 points in the latent space dimension was implemented, but this one had an extra hidden layer. The results can be seen in the subfolder "extra_layer_model" and better results than the classic AutoEncoder were actually achieved.

Tn the second approach, the number of epochs used to run the classic AutoEncoder was increased. The results were very good and the tradeoff on execution time was not as bad as it was compared to the models that incorporated GCN-layers. 

### Future steps to consider : 

Now that the AutoEncoder was designed. It is possible to implement prediction and classification methods in the latent space dimension. It is also possible to expand this model into a Variational AutoEncoder, and that would help us to create our own simulations of the WaveSuite dataset. 
