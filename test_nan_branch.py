import numpy as np 
import pandas as pd
import read  
from sklearn.model_selection import train_test_split
import decisionTree as dt

data_path = 'Dados/breast-cancer-wisconsin-prognosis.csv'
class_name = 'outcome'
class_questionnaire = None
missing_input = 'none'
dummy = False
transform = False
use_text = False

data, original_attributes, categories  = read.readData(data_path = data_path, class_name = class_name, 
    class_questionnaire = class_questionnaire, missing_input = missing_input, dummy = dummy,
    transform_numeric = transform, use_text=use_text, skip_class_questionnaire=True)


X = data[:,0:-1]
y = np.array(data[:,-1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9,stratify=y)
missing_values_row_indexes = np.random.random_integers(0,X_train.shape[0]-1,int(0.25*X_train.shape[0]))
missing_values_column_indexes = np.random.random_integers(0,X_train.shape[1]-1,int(0.5*X_train.shape[1]))
for i in missing_values_row_indexes:
    for j in missing_values_column_indexes:
        X_train[i][j] = np.nan

pd.DataFrame(X_train,columns=original_attributes[:-1]).to_csv('bla.csv', index=False)

tree = dt.DecisionTreeClassifier(missing_branch=False,max_depth=4)
tree.fit(X_train,y_train)
tree.to_pdf(original_attributes,'bcp-mb=False.pdf')
print(tree.score(X_test,y_test))
tree = dt.DecisionTreeClassifier(missing_branch=True,max_depth=4)
tree.fit(X_train,y_train)
print(tree.score(X_test,y_test))
tree.to_pdf(original_attributes,'bcp-mb=True.pdf')
