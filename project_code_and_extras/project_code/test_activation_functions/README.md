
This folder is the STEP 1 of the project.

### Some things to take into account before describing STEP 1:

The principal objective of this project is to propose different AutoEncoder models that incorporate GCN layers, and to test these models on the WaveSuite dataset. 

However, given the large size of the WaveSuite dataset and the limited time and computational resources for the project, the AutoEncoder models were first tuned and tested in smaller datasets: MNIST and air_pollution. In general it is not possible to design and test Machine Learning models in certain datasets and then apply those Machine Learning models to other datasets that are not related to the ones used initially to test the models. Yet, that is exactly what I am trying to do in this project. For this, 4 models were initially proposed and then the models were tuned by testing 37 variants of these models in the smaller datasets. After that, some conclusions were drawn and the final AutoEncoder was designed taking into account the findings of the smaller datasets. 

In order to extract meaningful conclusions on the procedure previously described, it is necessary to consider the distributions of the Datasets. The notebook "checking_normalization.ipynb" located in this folder, generates the file "Datasets distribution.png" which shows the distributions of the 3 datasets. The smaller datasets have a distribution between 0 and 1. The WaveSuite dataset is distributed among 0 and 7,2. Therefore, the WaveSuite will be divided by 7,2 in order to standardize it and to make it more similar to the smaller datasets used previously. Also, standardizing a dataset between 0 and 1 before passing it to an AutoEncoder complies with the frequent used conventions for AutoEncoders.  

It is also worth mentioning that the SGD optimizer was used at the beginning of this project, but I later realized that switching to an ADAM optimizer produced much better results. This also seems to be a frequent used convention for AutoEncoders. 

To support the previous claims that standardizing the data between 0 and 1, using a sigmoid activation function at the output of the AutoEncoder and using ADAM as the default optimizer are common conventions used for AutoEncoders I present the following links: 

- https://avandekleut.github.io/vae/ 
- https://github.com/DL-WG/ROMS-tutorial/blob/main/MNIST_AE.ipynb 
- https://github.com/ese-msc-2021/ML_module/blob/master/implementation/7_VAE_GAN/7_VAE_GAN_morning/Morning_Session_6_VAEs_Solution.ipynb 


### STEP 1: 

In this STEP, 4 AutoEncoder models were proposed. 

The first model is a classic AutoEncoder, and the other models are three alternative models that incorporated CNN layers to the basis classic AutoEncoder used. 

- Model 1: Classic AE: The decoder is composed of two Fully connected layers.

- Model 2 (Alternative Model 1): Same as Model 1 but adding a 3-CNN layers module before the classic AE.

- Model 3 (Alternative Model 2): Same as Model 2 but adding 3-CNN layers module at the end of the model.

- Model 4 (Alternative Model 3): 6-CNN layers module + [mean, max] naive pooling layers of the whole graph module + classical AE module

Note that each model is described as a composition of modules. In STEP1 each model was tuned by varying the different activation functions that connected the modules within each model. Each different combination of activations functions within a model was called a variant. The different variants are summarized inside each folder. 

### Results : 

Results were obtained for all variants of the models, in both datasets: MNIST and air_pollution. 

Looking at the air pollution dataset, it is easy to see that the variants that had a Sigmoid activation function in its output were the ones that performed the best for each model. For MNIST, the variants that used no activation function in its output were the ones that performed the best for each model. However, Sigmoid also produced very good results. For this reason, the models will be compared using the following variants: 

- Model 1: Variant 3
- Model 2: Variant 3
- Model 3: Test1 Variant 3
- Model 4: Test1 Variant 3

However, all Model 4 variants performed poorly when compared to the three first models variants. For this reason, apart from plotting, Model 4 will not be used in the next step.  