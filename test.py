import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as sklearn_rfc
from RandomForestClassifier import RandomForestClassifier as custom_rfc

# Compare sklearn and custom RFCs on Kaggle credit card fraud dataset

def test():
    data = pd.read_csv('data/creditcard.csv')
    X = data.drop('Class', axis=1).values
    y = data['Class'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42, stratify=y)

    params = {
        'n_estimators': 10,
        'max_depth': 10,
        'max_features': 'sqrt',
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': 'balanced',
        'n_jobs': 2
    }

    rfc = sklearn_rfc(**params).fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    print(classification_report(y_test, y_pred))

    rfc = custom_rfc(max_split_values=10, **params).fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    test()
