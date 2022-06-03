# MIS 49Y Applied Machine Learning Final Project PyClass
## Project: House Prices - Advance Regression Techniques
We predict sales prices and practice feature engineering, RFs, and gradient boosting in this project.

### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [seaborn](https://seaborn.pydata.org/)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html).

### Code

Main code is provided in the `Group_project.ipynb` notebook file. You will also be required to use the `train.csv`, `test.csv` datasets. Also ´House Prices Visualization.ipynb´ notebook is visualization of pure data.

### Run

Run one of the following command:

```bash
jupyter notebook Group_project.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

### Data

Kaggle competition dataset which has 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa. This competition challenges us to predict the final price of each home.

[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).


---

## Project Final Report

### Problem Statement and Goal

House values and prices is important for all groups of the society. Individual buyers, renters and investment corporations are all affected by the house prices around them. So, it is important to accurately evaluate a houses value. This problem existed in our society since the introduction of private property to the human society thousands of years ago. There has been many different approaches and solutions to this problem.

Our approach is using machine learning technology to solve this problem. Machine learning algorithms proved to be highly successful in these kind of deterministic regression problems. Machine learning algorithms usually require high amounts of data to achieve success. Collecting house data is not an easy job so there usually isn’t such large amounts of data available, but the current state of this problem shows that predicting house values is one of the simpler tasks in machine learning and our algorithms can achieve very high success. This is likely due to the high dimensionality of the data ie. although we have a limited amount of data (our training dataset contains only 1500 samples) the data we have is usually detailed with a lot of important features and since it is collected manually the data quality is also generally higher.

### Literature Search

There have been numerous approaches to the problem of house price prediction. 

The selected features are self-explanatory and trivial to understand. Although reasons behind the selection of those features can be complicated, we will not discuss them due to length constraints. Our focus on this section will be explaining the frequently used machine learning models.

#### Hedonic Price Model:

According to [2] “The hedonic pricing theory suggests that house is a differentiated commodity, whose value depends on its heterogeneous characteristics.”  The hedonic price model is a machine learning model which is created by combining the hedonic price theory with existing machine learning tools like PCR, SVR and K-NN in [2]

#### Forest Regression

Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression. This model is one of the older and widely used ones. [3] claims that random forest regression can be used to predict the flat prices very efficiently. 
 
#### Multiple Linear Regression

Regression analysis is a model used to determine the relationship between variables. To evaluate the correlation of the variables, the correlation coefficient or regression equation can be used. In [4], the power of the multiple regression model can be seen when the value of the relationship between dependent and independent variables is measured. This approach also seems to be the most preferred way in this problem.

#### Support Vector Regression

SVMs or Support Vector Machines are one of the most popular and widely used algorithm for dealing with classification problems in machine learning. This algorithm acknowledges the presence of non-linearity in the data and provides a proficient prediction model. In [5] authors conduct experiments in using SVM algorithm for house price prediction and they achieve RMSE of as low as 0.2 which shows that SVM is suitable for this task.

#### Artificial Neural Networks
 
The literature around neural networks is vast, even more then all the other methods combined so we will not go into detail about them.

Other methods presented in [1] are not as frequently used in house price prediction problem. We will not discuss them either.


### Dataset Description

Our dataset comes from the Kaggle competition “House Prices - Advanced Regression Techniques” which is a “getting started competition” suitable for starters in the field. It consists of 79 features about 3000 houses, which is split in half for training and testing. The data is very well recorded, and it almost have no missing values. The most curious aspect of the data is that many of the categorical features used the “NaN” keyword to represent absence of the thing they describe (like a garage). This causes pandas to automatically assume those fields have null values where in reality those values are not null, so we had to correct those confusions by changing the “NaN” fields with appropriate alternatives.

### Methods

In the preprocessing step we had to convert classification features to numeric ones since most machine learning models demand that. We used ordinal encoding in fields where the classification data had an ordinal relationship and one hot encoding where no such relationship exists. 

After converting all fields to numerical values, we had to fill some missing values, we used mode for categorical features and mean for the numerical ones. 

Before applying our models, we splited the labeled data into train and test data using sklearn’s traintestsplit function. After that we normalized our data with a min max scaler we implemented with python.

After the preprocessing steps we used XGboost, Support Vector Machine, Gradient Boosting and Random Forest algorithms to make our predictions. Each algorithm achieved a decent RMSE but there was room for improvement.

We tried doing PCA to reduce the dimensionality of our data which included 251 features after the preprocessing steps. Although PCA managed to capture %100 of variety with 201 features we could not improve our RMSE with those and we had worse results.

After failing to improve the RMSE of our models with PCA, we turned from feature extraction to feature selection and eliminated some features. We first eliminated the features with zero variance and then removed the features with more than %90 correlation with other features. This method improved to accuracy of all our models.

Between the models we used, Random Forest was the most successful one, but we wanted to try adding an ensemble system to improve our overall accuracy by combining all our models. We implemented a voting system and experimented with different weights but to our disappointment we could not beat Random Forest’s success this way.

Then we added lime to understand the details of our algorithm’s inner workings and we found out that all four were doing very similar things. This was likely the reason why using an ensemble method did not improve our results.

Lastly we tried to change the hyper parameters of models with ones we found online and then made some experiments with them. This method proved to be beneficial, and we managed to improve our best score. RMSE of the Gradient boosting algorithm highly improved with this change and performed better than Random Forest.

### Experiment Results

In the previous section we talked about the processes we followed and their results, so in this section we will only provide the RMSE values for each different stage.

#### Stage 1: After preprocessing

```bash
XGboost: 0.047111 RMSE
SVM: 0.057991 RMSE
Random Forest: 0.036052 RMSE
Gradient Boosting: 0.061730 RMSE
```
#### Stage 2: Working with PCA

```bash
XGboost: 0.090426 RMSE
SVM: 0.094988 RMSE
Random Forest: 0.088877 RMSE
Gradient Boosting: 0.09841 RMSE
```
#### Stage 3: Feature Selection

```bash
XGboost: 0.046627 RMSE
SVM: 0.057038 RMSE
Random Forest: 0.036035 RMSE
Gradient Boosting: 0.061730 RMSE
```
#### Stage 4: Ensemble Voting
```bash
RMSE :  0.040434
```
As you can see, we achieved our lowest RMSE score with using Random Forest after feature selection.

#### Stage 5: Hyper Parameter Tunning
```bash
XGboost: 0.046627 RMSE
SVM: 0.057038 RMSE
Random Forest: 0.036035 RMSE
Gradient Boosting: 0.061730 RMSE
Ensemble: 0.040434 RMSE
```


Our submission RMSE is lower than what we achieved in our experiments; this is likely due to the differences in the normalization algorithm Kaggle uses.

### Discussion

Carefully reading and understanding features then using that understanding to preprocess them appropriately was the main contributor of our result. We tried implementing different techniques to improve it, but we were not very successful in our efforts. After reviewing other submissions and the literature we found out that we were indeed successful among peers in feature encoding and preprocessing.

Our weaknesses were that we did not do any advanced hyperparameter tunning since that was beyond our knowledge and experience level at the time. Also, some approaches in Kaggle added interesting new features using the existing ones by both combining features and doing mathematical transformations on them. In the future we plan to study those techniques and improve our ranking with them.

### Conclusion

In this work we utilized all the techniques we learned during the semester. It was a rewarding experience to combine them and make our first Kaggle submission. We learned that not all techniques are suitable for all problems, and we must try and test different approaches to a problem. Even though we can’t benefit from every single technique we learned in every ML problem, we learned how to learn different approaches and some of the techniques we learned were universal so that they were useful in almost every ML problem like data visualization.

To our knowledge none of the discussions in Kaggle does inspect every feature like we do, so this is our greatest contribution to the problem. There are also many different approaches we did not use in our work, but we plan to combine those with ours to achieve top 100 ranking in the near future.

### References

[1] House Price Prediction using a Machine Learning Model: A Survey of Literature
December 2020International Journal of Modern Education and Computer Science 12(6):46-54
DOI: 10.5815/ijmecs.2020.06.04
[2] Hedonic Housing Theory – A Machine Learning Investigation. December 2016. DOI:   10.1109/ICMLA.2016.0092. Conference: 2016 15th IEEE International Conference on Machine Learning and Applications (ICMLA)
[3] Flat Price Prediction Using Linear and Random Forest Regression Based on Machine Learning Techniques. January 2020. DOI: 10.1007/978-981-15-6025-5_19. In book: Embracing Industry 4.0
[4] S. P. Ellis and S. Morgenthaler, “Leverage and breakdown in L1regression,” J. Am. Stat. Assoc., vol. 87, no. 417, pp. 143–148, 1992, doi: 10.1080/01621459.1992.10475185.
[5] Wu, Jiao Yang, "Housing Price prediction Using Support Vector Regression" (2017). Master's Projects. 540. DOI: https://doi.org/10.31979/etd.vpub-6bgs
https://scholarworks.sjsu.edu/etd_projects/540


