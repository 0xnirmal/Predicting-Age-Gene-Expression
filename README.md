
# Advanced Topics in Genomic Data Analysis Mini Project 1 #

## Running Instructions ##
```
git clone https://github.com/nkrishn9/mini_project_1.git
chmod -x run.sh
source run.sh
```

## Abstract ##
In this project, we attempted to use GTEx data to predict the age of a patient given their cis-eQTL data. This was done using the cis-eQTL gene expression data of each of four tissue types-adipose subcutaneous, muscle skeletal, thyroid, whole blood. We used a simple model, ridge regression, trained on the top 1000 genes with the highest univariate correlation with our label, age. Cross-validation on the training data was used to tune the alpha hyperparameter (regularization constant), and was then used along with the beta coefficients to predict age in our testing set. Root mean squared error was reported across different training sizes and thyroid appears to be the optimal tissue for predicting age, across training size setups. 

## Process Description ##
### Data Processing ###
### Cross-Validation Hyperparameter Tuning ###
### Prediction ###
