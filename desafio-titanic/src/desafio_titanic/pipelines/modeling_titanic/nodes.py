"""
This is a boilerplate pipeline 'modeling_titanic'
generated using Kedro 0.18.4
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb


def fill_null(train, test):

    train['Age'] = train['Age'].fillna(train.groupby(['Sex', 'Pclass', 'Embarked'])['Age'].transform('mean'))
    train = train.dropna(subset=['Embarked'])

    test['Age'] = test['Age'].fillna(test.groupby(['Sex', 'Pclass', 'Embarked'])['Age'].transform('mean'))
    test = test.dropna(subset= ['Embarked'])

    train_1 = train
    test_1 = test

    return train_1, test_1


def feature_engineering(train_1, test_1):
    #creation of features in train
    train_1['Title'] = train_1['Name'].apply(lambda x: x.split(',')[1].split('. ')[0])  

    train_1['Title'] = train_1['Title'].replace([' Don', ' Rev', ' Dr', ' Mme',
       ' Ms', ' Major', ' Lady', ' Sir', ' Mlle', ' Col', ' Capt',
       ' the Countess', ' Jonkheer'], 'Other')

    def age_classification(age):
        if age < 12:
            return 'child'
        elif age < 18:
            return 'young'
        elif age < 45:
            return 'adult'
        elif age < 90:
            return 'elderly'
        else:
            return age
    
    train_1['Age_Class'] = train_1['Age'].apply(age_classification)

    train_1["Fsize"] = train_1["SibSp"] + train_1["Parch"] + 1

    train_1 = train_1.drop(columns=['PassengerId','Ticket','Cabin', 'Name', 'SibSp', 'Parch'])

    #creation of features to test

    test_PassengerId = test_1['PassengerId']
    test_1['Title'] = test_1['Name'].apply(lambda x: x.split(',')[1].split('. ')[0])  

    test_1['Title'] = test_1['Title'].replace([' Don', ' Rev', ' Dr', ' Mme',
        ' Ms', ' Major', ' Lady', ' Sir', ' Mlle', ' Col', ' Capt',
        ' the Countess', ' Jonkheer', ' Dona' ], 'Other')

    test_1['Age_Class'] = test_1['Age'].apply(age_classification)

    test_1["Fsize"] = test_1["SibSp"] + test_1["Parch"] + 1

    test_1 = test_1.drop(columns=['PassengerId','Ticket','Cabin', 'Name', 'SibSp', 'Parch'])

    train_2 = train_1
    test_2 = test_1

    return train_2, test_2, test_PassengerId

def data_preprocessing(train_2, test_2):
    #preprocess for train
    train_2 = train_2.reset_index(drop=True)

    train_2['Pclass'] = train_2['Pclass'].astype('category')
    train_2['Sex'] = train_2['Sex'].astype('category')
    train_2['Embarked'] = train_2['Embarked'].astype('category')
    train_2['Title'] = train_2['Title'].astype('category')
    train_2['Age_Class'] = train_2['Age_Class'].astype('category')

    enc = OneHotEncoder(handle_unknown = 'ignore')  
    enc_data =  pd.DataFrame(enc.fit_transform(train_2[['Pclass', 'Sex', 'Embarked', 'Title', 'Age_Class']]).toarray())
    train_2 = train_2.join(enc_data).drop(columns=['Pclass', 'Sex', 'Embarked', 'Title', 'Age_Class'])


    #preprocess for test

    test_2 = test_2.reset_index(drop=True)

    test_2['Pclass'] = test_2['Pclass'].astype('category')
    test_2['Sex'] = test_2['Sex'].astype('category')
    test_2['Embarked'] = test_2['Embarked'].astype('category')
    test_2['Title'] = test_2['Title'].astype('category')
    test_2['Age_Class'] = test_2['Age_Class'].astype('category')

    enc = OneHotEncoder(handle_unknown = 'ignore')  
    enc_data =  pd.DataFrame(enc.fit_transform(test_2[['Pclass', 'Sex', 'Embarked', 'Title', 'Age_Class']]).toarray())
    test_2 = test_2.join(enc_data).drop(columns=['Pclass', 'Sex', 'Embarked', 'Title', 'Age_Class'])

    train_fixed = train_2
    test_fixed = test_2

    return train_fixed, test_fixed


def predict(train_fixed, test_fixed, test_PassengerId):

    X = train_fixed.iloc[:,1:]
    y = train_fixed.iloc[:,0]

    model_xgb = xgb.XGBClassifier()
    model_xgb.fit(X,y)

    y_pred = model_xgb.predict(test_fixed)

    test_survived = pd.Series(y_pred, name = "Survived").astype(int)
    results = pd.concat([test_PassengerId, test_survived],axis = 1)
    
    return results