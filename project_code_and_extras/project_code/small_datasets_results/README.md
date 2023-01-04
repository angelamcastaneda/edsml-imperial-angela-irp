
This folder is the STEP 2 of the project.

### Some things to take into account before describing STEP 2:

Coming from STEP 1 you must know that the following models were selected for plotting:

- Model 1: Variant 3
- Model 2: Variant 3
- Model 3: Test1 Variant 3
- Model 4: Test1 Variant 3

These variants were compared and plotted in the notebooks "AE models applied to air data.ipynb" and "AE models applied to regular MNIST.ipynb" located in this folder. The decompressed results of the AutoEncoder, for the air pollution dataset and MNIST dataset, are located in the "saved_figures" folder. Those figures were the ones used in the report.

### STEP 2:

This step will perform hyperparameter optimization (it is not really a hyperparameter optimization but more of a similar analysis). As mentioned in the step before, Model 4 will not be taken into account in this step given its bad results from the previous step. The actual tuning of the hyperparameters  was carried out in the subfolder "tests_hyperparameters".  

For each variable to tune a notebook was generated in which all other variables were fixed except for the one variable being tuned and the MSE results were recorded in the subfolder "tests_hyperparameters/saved_csv". In each CSV it is possible to check how much varying each variable influenced the MSE results. The variables chosen to tune were: latent space dimension, embedding sequence of the GCN layers and batch size.  

### Results : 

From the MSE results we can draw the following conclusions:

- Latent space dimension: This parameter is really important no matter the model used. In both datasets, it was checked that there is a limit were it is not possible to reduce the latent space dimension without compromising the quality of the decompressed results. For this reason, and also given the large difference in size among the datasets, this hyperparameter must be tuned for the WaveSuite dataset on its own and it is not possible to deduc it by looking at the smaller datasets.

- Embedding sequence of the GCN layers: This parameter refers to the number of channels in each GCN layer of the model. A bigger number of channels in each layer takes more computational resources since it represents a bigger number of parameters to tune. Furthermore, the number of channels in the layer that connect to the classical AutoEncoder are the ones that largely determine the real size of the number of parameters of the model. 

- Batch size: smaller batch sizes produce a lower MSE error and hence better results. However,  given enough iterations (or epochs), the MSE results tend to the same number despite of the batch size used. Therefore, this parameter is not that important to tune in order to get a low MSE error. This parameter is important however, when the time and computational resorces are taken into account since smaller batches take less computational resources but also take longer times to run. To be able to run the models on the WaveSuite dataset, a batch size of 8 will be selected for this project. 

### Next step : 

In the next step, I will finally work on the WaveSuite dataset. In general "you can not extrapolate the conclusions of a random dataset into another". However, so far in this code I have checked that:

- ADAM produces better results
- The number of points in the latent space dimension is an important variable that should be tuned.
- The activation function used at the output of the AutEncoder is of great importance, and Sigmoid is not always the best option (In MNIST, no activation function turned out better than MNIST). 
- Combining GCN-layers does improve the MSE loss. However, using it requires more computational resources and execution time; and the tradeoff is not worth it. 