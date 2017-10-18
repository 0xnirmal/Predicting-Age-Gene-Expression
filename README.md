
# Advanced Topics in Genomic Data Analysis Mini Project 1 #

## Running Instructions ##
```
git clone https://github.com/nkrishn9/mini_project_1.git
chmod -x run.sh
source run.sh
```
This code is intended to run on the JHU CS ugrad cluster. 

## Abstract ##
In this project, we attempted to use GTEx data to predict the age of a patient given their cis-eQTL data. This was done using the cis-eQTL gene expression data of each of four tissue types-adipose subcutaneous, muscle skeletal, thyroid, whole blood. We used a simple model, ridge regression, trained on the top 1000 genes with the highest univariate correlation with our label, age. Cross-validation on the training data was used to tune the alpha hyperparameter (regularization constant), and was then used along with the beta coefficients to predict age in our testing set. Root mean squared error was reported across different training sizes and thyroid appears to be the optimal tissue for predicting age, across training size setups. 

## Process Description ##
I recommend taking a peek at the age_prediction.ipynb to gain an understanding of the process, not the age_prediction.py because the notebook contains significant markdown comments. 

### Data Processing ###
This step was a fairly standard approach to a machine learning problem. First, we generated our design matrices for each of our tissues (separately), orienting them such that rows represented patients and columns represented genes. We then constructed our labels for each patient. This was slightly problematic because the reported GTEx ages are 10-year ranges, such at 60-69. Since we cannot infer anything else, we chose to represent this just as the first number in the range. The decision to represent age as a continuous value rather than a classification label is important because there is inherently a hierarchical structure to age. The age range 70-79 not only communicates belonging to class, but it also communicates that it is greater than the class 60-69 and less than the class 80-89. For this reason, it only makes sense to model the problem as a regression. 

We did not transform the data other than the labels, because the GTEx data is already mean centered and normalized. In terms of splitting the data, we performed a standard 70% training 30% testing split for each tissue dataset, in order to evaluate our model. On the training data, we then performed basic feature selection using an f-regression between gene and label. We chose to use an f-regression because the model we are using the predict age is a linear model (ridge regression). Therefore, we want to select features that are univariately correlated with our label in order for our model to get a good signal. We selected the top 1000 genes.

### Cross-Validation Hyperparameter Tuning ###
The training data for each tissue was split using leave-one-out cross-validation in order to evaluate regularization parameters (alphas). We evaluated alphas across different scales (0.1, 1.0, 10.0, 100), and used the alpha for each tissue that resulted in the highest root mean-squared error. 

### Training/ Prediction ###
Once the alpha parameters had been selected using CV, we trained 4 different 
