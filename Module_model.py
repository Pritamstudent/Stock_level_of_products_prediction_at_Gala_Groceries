#!/usr/bin/env python
# coding: utf-8

# # Import essential dependencies

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


# # Declare global variables

# In[3]:


#We will use 75% training samples and 25% testing samples
SPLIT = 0.75

#We will train the model K-fold times
K = 10   


# ## Import data 

# In[4]:


def import_data(path: str = '/path/csv/'):
    """
    This function will take the relative path of the csv
    file as the string and then we will read the file using
    the read_csv() function of the pandas library.
    At end it will return the data in the form of pandas
    dataframe.
    :param 1     path (optional): relative path to csv file in the form of string
    :return     df: pandas dataframe
    """
    
    df = pd.read_csv(f"{path}")
    return df

    
    


# ## Create target and predictor variables

# In[5]:


def create_target_and_predictor_variables(
    data: pd.DataFrame = None,
    target: str = "estimated_stock_pct"
):
    """
    This function will create the target variable (the one which we want 
    to predict) and the predictor variables using which we will train the model.
    The two sets of data will be X and y. These will be used to train a supervised learning model.
    :param 1 data: The pandas dataframe for the data.
    :param 2 target(optional): The target columns or the target variable.
    :return X: The predictor variables
            y: The target column or target variable
    """
    
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    X = data.drop(column = [target])
    y = data[target]
    return X,y



# ## Train the model

# In[6]:


def train_model_with_validation(
    X: pd.DataFrame = None,
    y: pd.DataFrame = None
):
    """
    This function will first split the data-set, then 
    it will scale the data using Standardization. Then, we will use 
    mean absolute error to evaluate the model.
    :param1 X: predictor variables
    :param2 y: target variables
    :return
    """
    #list to store the accuracy of models at various folds
    accuracy = []
    
    #train the model in K folds
    for iter in range(0,K):
        
        #Create the instance of model and scaler
        model = RandomForestRegressor()
        scaler = StandardScaler()
        
        #Create the training and testing samples
        X_train, X_test, y_train, y_test  = train_test_split(X,y,train_size = SPLIT,random_state = 42)
        
        #Use standard scaler to scale the data to better perform in the model
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        #Train the model
        trained_model = model.fit(X_train,y_train)
        
        #Make prediction using the trained model
        y_pred = trained_model.predict(X_test)
        
        #Let us check the accuracy of the model using MAE
        mae = mean_absolute_error(y_true = y_test , y_pred=y_pred)
        accuracy.append(mae)
        print(f"Fold {iter + 1}: MAE = {mae:.3f}")
    
    #Compute average mean absolute error
    print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.3f}")


# ## Main Function

# In[ ]:


def run():
    """
    This function executes the training pipeline of loading the prepared
    dataset from a CSV file and training the machine learning model

    :param

    :return
    """

    # Load the data first
    df = load_data()  #Add the path of the data_file

    # Now split the data into predictors and target variables
    X, y = create_target_and_predictors(data=df)

    # Finally, train the machine learning model
    train_algorithm_with_cross_validation(X=X, y=y)


# In[ ]:




