import numpy as np 
import pandas as pd
import read  
from sklearn.model_selection import train_test_split
import decisionTree as dt
from sklearn.model_selection import KFold
data_path = '../RotEOmbroCirurgiaCategNAReduzido.csv'
class_name = 'Q92510_opcForca[RotEOmbro]'#'Q92510_snDorPos'
class_questionnaire = 'Q92510'
missing_input = 'none'
dummy = False
transform = False
use_text = False

data, original_attributes, categories  = read.readData(data_path = data_path, class_name = class_name, 
    class_questionnaire = class_questionnaire, missing_input = missing_input, dummy = dummy,
    transform_numeric = transform, use_text=use_text, skip_class_questionnaire=True)

sf = []
st = []
vp, vp1, fp, fp1, fn, fn1, vn, vn1 = 0,0,0,0,0,0,0,0
X = data[:,0:-1]
y = np.array(data[:,-1])
n_splits = X.shape[0]
sss = KFold(n_splits=n_splits,random_state=9)

for train_index, test_index in sss.split(X,y):
    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]


    tree1 = dt.DecisionTreeClassifier(missing_branch=False,max_depth=2)
    tree1.fit(X_train,y_train,)
    sf.append(tree1.score(X_test,y_test))
    if(y_test == 'SUCESSO'):
        if(tree1.predict(X_test) == 'SUCESSO'):
            vp1 += 1
        else:
            fn1+=1
    else:
        if(tree1.predict(X_test) == 'SUCESSO'):
            fp1+=1
        else:
            vn1 +=1

    tree = dt.DecisionTreeClassifier(missing_branch=True,max_depth=2)
    tree.fit(X_train,y_train)

    st.append(tree.score(X_test,y_test))
    if(y_test == 'SUCESSO'):
        if(tree.predict(X_test) == 'SUCESSO'):
            vp += 1
        else:
            fn+=1
    else:
        if(tree.predict(X_test) == 'SUCESSO'):
            fp+=1
        else:
            vn+=1

p1 = vp1/(vp1+fp1)
c1 = vp1/(vp1+fn1)
if(p1 + c1 == 0):
    f1 = 0
else:
    f1 = (2*p1*c1)/(p1+c1)


p = vp/(vp+fp)
c = vp/(vp+fn)
if(p + c == 0):
    f = 0
else:
    f = (2*p*c)/(p+c)
tree1.fit(X,y)
tree1.to_pdf(original_attributes,'mb=False.pdf')
tree.fit(X,y)
tree.to_pdf(original_attributes,'mb=True.pdf')

print(np.mean(np.array(sf)))
print(np.std(np.array(sf)))
print('false --- precisão: %r cobertura: %r medida-f: %r' % (p1,c1,f1))
print(np.mean(np.array(st)))
print(np.std(np.array(st)))
print('true -- precisão: %r cobertura: %r medida-f: %r' % (p,c,f))
print(sf)
