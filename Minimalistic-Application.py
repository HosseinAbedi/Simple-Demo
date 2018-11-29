
# coding: utf-8

# In[ ]:


import time 
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import pymysql
from sqlalchemy import create_engine  


def load_model(file_path='all_I_need.pickle'):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

    
if __name__ == "__main__":
    
    interval = 10 # ten second intervals
    input_table = 'data'
    output_table = 'result'
    database_address = 'mysql+pymysql://test:test1234@localhost:3306/survey?charset=utf8'
    
    # Readig the raw data    
    data = pd.read_csv('../data/train-missings.csv', index_col=0)

    # Creating a conection to MYSQL data base (created in advance)
    # Putting our raw data into the table `data`
    engine = create_engine(database_address) 
    data.to_sql(input_table, con=engine, if_exists='replace')
    
    # Loading the models and features sets
    model_dict = load_model()
    features = model_dict['features']
    selected_features = model_dict['selected_features']
    models = model_dict['models']
    
    while True:
        # Reading a whole table
        data  = pd.read_sql_table('data', con=engine, index_col='Id')

        # Creating the features needed
        for f, g, mi in np.array(features):
            if (f + '/' + g) in selected_features:
                data[f + '/' + g] = data[f] / data[g]


        # Prediction
        test_predictions = np.zeros([data.shape[0], 7])
        for clf in models:
            test_predictions += clf.predict_proba(data[selected_features])
        test_predictions = test_predictions / len(models)

        # Wrting the results into the result table
        pd.DataFrame(test_predictions, index=data.index).to_sql(output_table, con=engine, if_exists='replace')
        
        time.sleep(interval)

