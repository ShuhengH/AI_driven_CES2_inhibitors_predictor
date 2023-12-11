import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from rdkit.Chem import MACCSkeys
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error


def fit_model(data, model_clf, test_size, random_state):
    x = data.iloc[:,2:]
    y = data["pIC50"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                        random_state=random_state)
    model_clf.fit(x_train, y_train)

    y_pred_train = model_clf.predict(x_train)
    y_pred_test = model_clf.predict(x_test)
    
    # predictor evaluation
    print('Training set R2：{:.4f}'.format(r2_score(y_train, y_pred_train)), "|",
          'Test set R2：{:.4f}'.format(r2_score(y_test, y_pred_test)), "|",
          'Training set MSE：{:.4f}'.format(mean_squared_error(y_train, y_pred_train)), "|",
          'Test set MSE：{:.4f}'.format(mean_squared_error(y_test, y_pred_test)))

    

def save_model(model_clf, path):
    joblib.dump(model_clf, path)


def model_predict(data, path):
    x = data.iloc[:,2:]
    #y = data["pIC50"]

    model_clf = joblib.load(filename=path)
    y_pred_validation = model_clf.predict(x)
    
    # predictor evaluation
    #print('Validation set R2：{:.4f}'.format(r2_score(y, y_pred_validation)), "|",
    #      'Validation set MSE：{:.4f}'.format(mean_squared_error(y, y_pred_validation)))
    
    pre_result = pd.DataFrame(y_pred_validation)
    pre_result.columns=list(["Pred_pIC50"])
    
     
    #pre_result.to_csv("pIC50_for_validation_set.csv", index = True)
    
    return pre_result
    




