# Support-Vector-Machines

# 
# 
### 1. Support Vector Machines with Synthetic Data

For this problem, we will generate synthetic data for a nonlinear binary classification problem and partition it into training, validation and test sets. Our goal is to understand the behavior of SVMs with Radial-Basis Function (RBF) kernels with different values of  𝐶  and  𝛾 .

### a. The effect of the regularization parameter,  𝐶 

Complete the Python code snippet below that takes the generated synthetic 2-d data as input and learns non-linear SVMs. Use scikit-learn's SVC function to learn SVM models with radial-basis kernels for fixed  𝛾  and various choices of  𝐶∈{10−3,10−2⋯,1,⋯105} . The value of  𝛾  is fixed to  𝛾=1𝑑⋅𝜎𝑋 , where  𝑑  is the data dimension and  𝜎𝑋  is the standard deviation of the data set  𝑋 . SVC can automatically use these setting for  𝛾  if you pass the argument gamma = 'scale' (see documentation for more details).

Plot: For each classifier, compute both the training error and the validation error. Plot them together, making sure to label the axes and each curve clearly.

Discussion: How do the training error and the validation error change with  𝐶 ? Based on the visualization of the models and their resulting classifiers, how does changing  𝐶  change the models? Explain in terms of minimizing the SVM's objective function  12𝐰′𝐰+𝐶Σ𝑛𝑖=1ℓ(𝐰∣𝐱𝑖,𝑦𝑖) , where  ℓ  is the hinge loss for each training example  (𝐱𝑖,𝑦𝑖) .

Final Model Selection: Use the validation set to select the best the classifier corresponding to the best value,  𝐶𝑏𝑒𝑠𝑡 . Report the accuracy on the test set for this selected best SVM model. Note: You should report a single number, your final test set accuracy on the model corresponding to  𝐶𝑏𝑒𝑠𝑡 .

### b. The effect of the RBF kernel parameter,  𝛾 

Complete the Python code snippet below that takes the generated synthetic 2-d data as input and learns various non-linear SVMs. Use scikit-learn's SVC function to learn SVM models with radial-basis kernels for fixed  𝐶  and various choices of  𝛾∈{10−2,10−11,10,102103} . The value of  𝐶  is fixed to  𝐶=10 .

Plot: For each classifier, compute both the training error and the validation error. Plot them together, making sure to label the axes and each curve clearly.

Discussion: How do the training error and the validation error change with  𝛾 ? Based on the visualization of the models and their resulting classifiers, how does changing  𝛾  change the models? Explain in terms of the functional form of the RBF kernel,  𝜅(𝐱,𝐳)=exp(−𝛾⋅‖𝐱−𝐳‖2) 
Final Model Selection: Use the validation set to select the best the classifier corresponding to the best value,  𝛾𝑏𝑒𝑠𝑡 . Report the accuracy on the test set for this selected best SVM model. Note: You should report a single number, your final test set accuracy on the model corresponding to  𝛾𝑏𝑒𝑠𝑡 .

### 2. Breast Cancer Diagnosis with Support Vector Machines

For this problem, we will use the Wisconsin Breast Cancer data set, which has already been pre-processed and partitioned into training, validation and test sets. Numpy's loadtxt command can be used to load CSV files.

### 3. Breast Cancer Diagnosis with  𝑘 -Nearest Neighbors

Use scikit-learn's k-nearest neighbor classifier to learn models for Breast Cancer Diagnosis with  𝑘∈{1,5,11,15,21} , with the kd-tree algorithm.

Plot: For each classifier, compute both the training error and the validation error. Plot them together, making sure to label the axes and each curve clearly.

Final Model Selection: Use the validation set to select the best the classifier corresponding to the best parameter value,  𝑘𝑏𝑒𝑠𝑡 . Report the accuracy on the test set for this selected best kNN model. Note: You should report a single number, your final test set accuracy on the model corresponding to  𝑘𝑏𝑒𝑠𝑡 .
