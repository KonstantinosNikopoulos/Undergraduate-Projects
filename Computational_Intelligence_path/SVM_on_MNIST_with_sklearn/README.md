# Support Vector Machine on MNIST with sklearn   

### Author: Konstantinos Nikopoulos   
### Course: Neural Networks & Deep Learning

##### Test parameters of SVM for regognition of even and odd numbers.   


1.0 Preprocess dataset   
Apply PCA for dimension reduction and divide even and odd numbers.   
new shape of data: (60000, 100) (10000, 100)   
new shape of labels: (60000, 2) (10000, 2)    

2.0 Parameters   

2.1 Kernels   
Kernels: linear and polynomial (2 and 3 degree)   

2.2 C   
C: 1, 10 and 100.   

3. SVMs    
Grid search to parameters.   

3.1 Results   
Accuracy Score of all models:   
kernel: linear, C=1   : 0.874   
kernel: linear, C=10   : 0.873   
kernel: linear, C=100   : 0.873   
kernel: polynomial, degree=2, C=1   : 0.9625   
kernel: polynomial, degree=2, C=10   : 0.9595   
kernel: polynomial, degree=2, C=100   : 0.959   
kernel: polynomial, degree=3, C=1   : 0.972   
kernel: polynomial, degree=3, C=10   : 0.971   
kernel: polynomial, degree=3, C=100   : 0.971   
Best svm with kernel polynomial 3 and C=1.    
Accuracy Score on test data: 0.972.    
Training time: 6.202604s.   



