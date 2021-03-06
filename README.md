# Support-Vector-Machines

# 
# 
### 1. Support Vector Machines with Synthetic Data

For this problem, we will generate synthetic data for a nonlinear binary classification problem and partition it into training, validation and test sets. Our goal is to understand the behavior of SVMs with Radial-Basis Function (RBF) kernels with different values of  πΆ  and  πΎ .

### a. The effect of the regularization parameter,  πΆ 

Complete the Python code snippet below that takes the generated synthetic 2-d data as input and learns non-linear SVMs. Use scikit-learn's SVC function to learn SVM models with radial-basis kernels for fixed  πΎ  and various choices of  πΆβ{10β3,10β2β―,1,β―105} . The value of  πΎ  is fixed to  πΎ=1πβππ , where  π  is the data dimension and  ππ  is the standard deviation of the data set  π . SVC can automatically use these setting for  πΎ  if you pass the argument gamma = 'scale' (see documentation for more details).

Plot: For each classifier, compute both the training error and the validation error. Plot them together, making sure to label the axes and each curve clearly.

Discussion: How do the training error and the validation error change with  πΆ ? Based on the visualization of the models and their resulting classifiers, how does changing  πΆ  change the models? Explain in terms of minimizing the SVM's objective function  12π°β²π°+πΆΞ£ππ=1β(π°β£π±π,π¦π) , where  β  is the hinge loss for each training example  (π±π,π¦π) .

Final Model Selection: Use the validation set to select the best the classifier corresponding to the best value,  πΆπππ π‘ . Report the accuracy on the test set for this selected best SVM model. Note: You should report a single number, your final test set accuracy on the model corresponding to  πΆπππ π‘ .

### b. The effect of the RBF kernel parameter,  πΎ 

Complete the Python code snippet below that takes the generated synthetic 2-d data as input and learns various non-linear SVMs. Use scikit-learn's SVC function to learn SVM models with radial-basis kernels for fixed  πΆ  and various choices of  πΎβ{10β2,10β11,10,102103} . The value of  πΆ  is fixed to  πΆ=10 .

Plot: For each classifier, compute both the training error and the validation error. Plot them together, making sure to label the axes and each curve clearly.

Discussion: How do the training error and the validation error change with  πΎ ? Based on the visualization of the models and their resulting classifiers, how does changing  πΎ  change the models? Explain in terms of the functional form of the RBF kernel,  π(π±,π³)=exp(βπΎββπ±βπ³β2) 
Final Model Selection: Use the validation set to select the best the classifier corresponding to the best value,  πΎπππ π‘ . Report the accuracy on the test set for this selected best SVM model. Note: You should report a single number, your final test set accuracy on the model corresponding to  πΎπππ π‘ .

### 2. Breast Cancer Diagnosis with Support Vector Machines

For this problem, we will use the Wisconsin Breast Cancer data set, which has already been pre-processed and partitioned into training, validation and test sets. Numpy's loadtxt command can be used to load CSV files.

### 3. Breast Cancer Diagnosis with  π -Nearest Neighbors

Use scikit-learn's k-nearest neighbor classifier to learn models for Breast Cancer Diagnosis with  πβ{1,5,11,15,21} , with the kd-tree algorithm.

Plot: For each classifier, compute both the training error and the validation error. Plot them together, making sure to label the axes and each curve clearly.

Final Model Selection: Use the validation set to select the best the classifier corresponding to the best parameter value,  ππππ π‘ . Report the accuracy on the test set for this selected best kNN model. Note: You should report a single number, your final test set accuracy on the model corresponding to  ππππ π‘ .
