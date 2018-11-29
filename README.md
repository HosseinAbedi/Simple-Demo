# Demo:

A simple demo of how to create a minimalistic ML model.


## Usage 

Install the requirements using `sudo pip3 install -r requirements.txt`

## Steps:
 
* Firstly we have a look a the data, it is a multi class classification task with 7 classes. The raw features are either numerical or categorical features with high dimentionalty so we treat them as numerical values.

* The model we use here is LighGBM, a famous implementation of the boosting famiy. We use it becasue of it is very fast and also has the merit of not needing too much data preprocessing (compared to something like NNs) to work fine. 

* In this problem, the classes are uniformy distributed so no special treatment required for handling unbalanced data, As for the object funnction, it depends mainly on the bussiness side and needs so without no knowledge of that we stick to a famous objective function (i.e. multi class logloss).

* With balanced data and absense of the time element we can easily start with a nice 5-fold validation set-up; the validation is done in two steps where 20 percent of the data is shuffled and is held out for final testing and the rest if used to create 5 models with out of fold scheme. So our final model is the blend of 5 models and the two validation errors is reported.

* We firstly start with just the raw features and an initial set of parameters with lgbm to get a 5 fold validation errors of around 0.76 and 0.69. 

* After setting up the validation we move to more advanced toppics such as feature engineering, parameter optimzation, and feature selection. After some feature engineering, feature selection and parametr optimzation we can see that the initial validation errors drops about 0.02. 

* At the end model and the list of features needed is saved for further use.

* For more info on detail please have a look at the LightGBM-v0.1.0.ipynb dashboard.

Notice: A code snippet is show for performing Bayesean optimzation for parametr tuning but because of time limit could not be used here (running it would take hours!).
        



    
