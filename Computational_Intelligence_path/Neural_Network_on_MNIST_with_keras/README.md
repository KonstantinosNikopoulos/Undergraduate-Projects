# Neural Network on MNIST with keras

### Author: Konstantinos Nikopoulos


##### Test and evaluate different activation functions.



1. Preprocess dataset   
Apply PCA for dimension reduction and divide even and odd numbers.   
new shape of data: (60000, 100) (10000, 100)   
new shape of labels: (60000, 2) (10000, 2)    

2. Construction of neural network   

2.1 Model architecture   
Neural network includes 2 hidden layers with 128 neurons each and an ouput layer with 10 neurons for classification.

2.2 Compile and train   
Error function: mean squared error    
Optimizer: adam   
Batch size: 128    
Epochs: 50   

3. Evaluation of activation functions    

3.1 Softmax   
Used for output layer.   

3.2 Linear   
Applied to hidden layers.   
Train accuracy: 0.9321   
Test accuracy: 0.9265   

3.3 Sigmoid   
Applied to hidden layers.   
Train accuracy: 0.9451   
Test accuracy: 0.9408   

3.4 Tanh   
Applied to hidden layers.   
Train accuracy: 0.9865   
Test accuracy: 0.9735   

3.5 ELU   
Applied to hidden layers.   
Train accuracy: 0.993   
Test accuracy: 0.9763   

3.6 ReLU   
Applied to hidden layers.   
Train accuracy: 0.9944   
Test accuracy: 0.9757   

3.7 LeakyReLU   
Applied to hidden layers.   
Train accuracy: 0.9944   
Test accuracy: 0.9789   

3.8 Recap   
Best accuracy with LeakyReLU   


