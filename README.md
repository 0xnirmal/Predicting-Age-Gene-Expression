
# Advanced Topics in Genomic Data Analysis- Mini Project 1 #

## Running Instructions ##
```
git clone https://github.com/nkrishn9/Predicting-Age-Gene-Expression.git
chmod -x run.sh
source run.sh
```
This code is intended to run on the JHU CS ugrad cluster. The figures 1, 1A, and 2 will be generated in the results directory.

## Abstract ##
In this project, we attempted to use GTEx data to predict the age of a patient given their cis-eQTL gene expression data. This was done on each of four tissue types-adipose subcutaneous, muscle skeletal, thyroid, whole blood. We used a simple model, ridge regression, trained on the top 1000 genes with the highest univariate correlation with our label, age. Cross-validation on the training data was used to tune the alpha hyperparameter (regularization constant), and was then used along with the beta coefficients to predict age in our testing set. Root mean squared error was reported across different training sizes and thyroid appears to be the optimal tissue for predicting age, across training size setups. 

## Process Description ##
I recommend taking a peek at the age_prediction.ipynb to gain an understanding of the process, not the age_prediction.py because the notebook contains significant markdown comments. 

### Data Processing ###
This step was a fairly standard approach to a machine learning problem. First, we generated our design matrices for each of our tissues (separately), orienting them such that rows represented patients and columns represented genes. We then constructed our labels for each patient. This was slightly problematic because the reported GTEx ages are 10-year ranges, such at 60-69. Since we cannot infer anything else, we chose to represent this just as the first number in the range. The decision to represent age as a continuous value rather than a classification label is important because there is inherently a hierarchical structure to age. The age range 70-79 not only communicates belonging to class, but it also communicates that it is greater than the class 60-69 and less than the class 80-89. For this reason, it only makes sense to model the problem as a regression. 

We did not transform the data other than the labels, because the GTEx data is already mean centered and normalized. In terms of splitting the data, we performed a standard 70% training 30% testing split for each tissue dataset, in order to evaluate our model. On the training data, we then performed basic feature selection using an f-regression between gene and label. We chose to use an f-regression because the model we are using the predict age is a linear model (ridge regression). Therefore, we want to select features that are univariately correlated with our label in order for our model to get a good signal. We selected the top 1000 genes.

### Cross-Validation Hyperparameter Tuning ###
The training data for each tissue was split using leave-one-out cross-validation in order to evaluate regularization parameters (alphas). We evaluated alphas across different scales (0.1, 1.0, 10.0, 100), and used the alpha for each tissue that resulted in the highest root mean-squared error. 

### Training/ Prediction ###
Once the alpha parameters had been selected using CV, we trained 4 different ridge regression model using the training data for each tissue and its corresponding alpha. We trained using increments of samples from our training data, specifically increments of 10 from 50 to 300. Its important to note that the training-testing splits occurred before incrementational model learning, so the testing data stays the same across training configurations. That is, the testing size is constant--always 30% of the original data, while the training size varies. This can in some cases be problematic if the testing size is too small, because then the testing size does not represent an empirical sampling of your actual distribution and your results compared across tissues will be inaccurate (may favor the larger testing sets over the smaller). However, we felt our testing size (~100) were large enough and the differences across the number of samples in testing set for each tissues small enough that this would not bias our results. 

Once the models were trained across our different tranings data sized increments, we predicting on our testing splits and reported the root mean squared errors for each, seen in figure 1 (for muscle-skeletal) and figure 1A (all tissues) below: 
![figure1]
![figure1a]

We trained again using all of the samples in each training split for each tissue and also using the largest common number of training examples across tissues. The results can be seen in figure 2 below: 

![figure2]


## Discussion ##
Based on the results in figures 1A and 2, the ridge regression model using thyroid cis-eQTL data is the best at predicting the age of the patient. In figure 1A, we see that across training-increment configurations, thyroid is the lowest line in the graph, indicating the lowest error on prediction in the testing set. As expected, across tissues the RMSE is minimal when the number of examples used in training is maximial, indicating that after 200 examples are seen, a good approximation of the true distribution is reached. In figure 2, we can see that in both configurations, using all samples and using the smallest common number of training examples across tissues, thyroid performs optimally. 

In terms of intepreting the results, we cannot take the RMSE to indicate the actual average error of prediction in terms of real-valued age (i.e just because our RMSE for thyroid is ~7, we cannot say that on average our prediction is off by 7 years). This is because of how the labels were encoded. Consider a case where a patient is aged 69. We place them in the 60 bucket, and then upon prediction, we predict their age using thyroid data to be 53. The error recorded is only 7, however, in actual biological space, the error is 16. Therefore, RMSE can be only used to compare the predictive accuracies of tissues, not to evaluate thyroid as a predictor of age itself. We could obviously improve our analysis here if the actual ages were provided to us upon sequencing, as there would no longer be noise associated with our labels. 

Given a longer timeframe, there are significant improvements that could be made. The GTEx consortium reports covariates for each of the patients, so we could "regress" these out of our gene expression matrices by obtaining the residuals of a linear model trained using the covariates on each gene. This could help remove batch effects and other covariates that typically overpower the signal in genomic data sets. Additionally, instead of using the top 1000 genes, we could instead cross-validate the number of genes we should use, which would likely improve our model in testing. Another idea I had was to do a polynomial feature mapping:

![feature_map]

This feature mapping would allow you to capture squared effects of features, while also allowing you to capture feature interaction. It is possible that two genes have little signal by themselves, but multiplied, produce a better predictor of age. 

In terms of the biological interpretation of thyroid being the best predictor, what this indicates to us is that the genes expressed in the thyroid vary the most (linearly) across age. That is, the genes expressed in the thyroid either change in response to or drive the aging process. In the other tissues, our model does a poorer job of picking up this age-varying process of gene expression. This could be a result of thyroid in fact being the best predictor, or that our model is not sufficient to pick up the signal in the other tissues (i.e. non-linear signal). Consequently, a natural extension of this project is to use other non-linear models and see how this affects the RMSE across tissues. 

[figure1]: https://github.com/nkrishn9/mini_project_1/blob/master/results/figure_1.png
[figure1a]: https://github.com/nkrishn9/mini_project_1/blob/master/results/figure_1A.png
[figure2]: https://github.com/nkrishn9/mini_project_1/blob/master/results/figure_2.png
[feature_map]: http://latex.codecogs.com/gif.latex?(x_1,x_2)-%3E(x_1,%20x_2,%20x_1^2,%20x_2^2,%20x_1x_2)
