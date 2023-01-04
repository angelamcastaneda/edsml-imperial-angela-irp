This is the famous MNIST dataset and it is automatically downloaded by any test notebook that works with MNIST.

Alternatively, it can be downloaded by just specifying the following commands in python:

    mnist_path="../Datasets/"

    #Import the MNIST dataset
    mnist_train =  MNIST(mnist_path , transform=ToTensor(), download=True)