# Radial Basis Function Network on MNIST with sklearn   
   
### Author: Konstantinos Nikopoulos   

##### Test parameters of RBF for regognition of even and odd numbers.   

1.Preprocess dataset   
Apply PCA for dimension reduction and divide even and odd numbers.   
new shape of data: (60000, 100) (10000, 100)   
new shape of labels: (60000, 2) (10000, 2)   

2. Clustering   

2.1 Clustering algorithm   
K-means with scikit learn.   

2.2 Κ clusters   
For Κ=20 clusters. Two for each digit (0,1,..,9).   

3. RBF   

3.1 Kernel   
Gaussian    
Kernels: linear and polynomial degree=3.   

3.2 Gamma   
Gamma= 0.1, 1 and 10.3   

4. MLP   

4.1 Neurons and hidden layers   
Test 1: Neural network includes 1 hidden layes with 100 neurons and an ouput layer with 2 neurons for classification.   
Test 2: Neural network includes 2 hidden layes with 50 neurons each and an ouput layer with 2 neurons for classification.   

4.2 Other parameters   
Activation function:  relu for hidden layers and softmax for output layer.   
Loss function: categorical crossentropy    
Optimizer: adam     
Epochs: 30   

5. Results   

5.1 With 1 hidden layer, 100 neurons   
kernel: linear, gamma=0.1 : acc=0.9524, loss=0.1286   
kernel: linear, gamma=1 : acc=0.9559, loss=0.1215   
kernel: linear, gamma=10 : acc=0.9522, loss=0.1310   
kernel: poly, gamma=0.1 : acc=0.9123, loss=0.2137   
kernel: poly, gamma=1 : acc=0.9099, loss=0.2208   
kernel: poly, gamma=10 : acc=0.9111, loss=0.2182   
Training time for all models: 3min   

5.2 With 2 hidden layers, 50 neurons each   
kernel: linear, gamma=0.1 : acc=0.9557, loss=0.1215   
kernel: linear, gamma=1 : acc=0.9573, loss=0.1182   
kernel: linear, gamma=10 : acc=0.9581, loss=0.1136   
kernel: poly, gamma=0.1 : acc=0.9171, loss=0.2010   
kernel: poly, gamma=1 : acc=0.9322, loss=0.1783   
kernel: poly, gamma=10 : acc=0.9214, loss=0.1983   
Training time for all models: 3min   

5.3 Final   
Best accuracy with 2 hidden layers and 50 neurons each,  kernel=linear, gamma=10   
acc=0.9581, loss=0.1136   

