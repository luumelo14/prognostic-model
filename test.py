import decisionTree as dt 
import randomForest as rf
import numpy as np
import read 
import utils
import math
from sklearn.metrics import accuracy_score
dummy = False
transform = False
use_text = False


print('Testing utils...')
assert(utils.isint(10))
assert(utils.isint('50'))
assert(utils.isint('-999'))
assert(not utils.isint(1.0))
assert(not utils.isint('50.0'))
assert(utils.isint(True))
assert(not utils.isint('aba'))
assert(not utils.isint('a?a'))
assert(not utils.isint('49.x'))
assert(not utils.isfloat('0.x'))
assert(utils.isfloat('0.0'))
assert(utils.isfloat('12.984'))
assert(utils.isfloat('-0.4'))
assert(not utils.isfloat('9'))


data, original_attributes, categories  = read.readData(data_path = 'Dados/Test.csv', class_name='Class',
    dummy=dummy,transform_numeric=transform,use_text = use_text,missing_input='none')
X = data[:,0:-1]
y = np.array(data[:,-1])
print('Testing entropy, information gain, gain ratio...')
assert(utils.entropy([1,0,0,1,0,1]) == 1)
assert(utils.entropy([1,1,1]) == 0)
assert(utils.entropy([0]) == 0)
outlook_index = np.where(original_attributes == 'Outlook')[0][0]
Xs,ys,d = utils.split_categ(X,y,outlook_index,list(set(X[:,outlook_index])))
assert(np.isclose(utils.information_gain(y,ys),0.246,rtol=1e-2))
assert(np.isclose(utils.gain_ratio(y,ys,y),0.156,rtol=1e-2))

print('Testing gini index...')
assert(utils.gini_impurity([1,1,1,0,0,1,1,1,0,0,0,0]) == 0.5)
assert(utils.gini_impurity([0,0,0,0,0]) == 0)
print('Testing gini...')
assert(utils.gini([0,1,0,0,1,1,0,0,1,1,0,1],[[0,1,0,0],[1,1,0,0,1,1,0,1]]) == 0.0625)

print('Testing Decision Tree...') 
m = dt.DecisionTreeClassifier(missing_branch=False)
m.fit(X,y)
m.to_pdf(original_attributes,out='tree1.pdf')
assert(m.predict((['OVERCAST',80,90,'T'])) == 'Play'.upper())
assert(m.predict(['RAIN',80,50,'F']) == 'Play'.upper())
assert(m.predict(['RAIN',80,70,'T']) == "Don't Play".upper())
assert(m.predict(['SUNNY',50,50,'T']) == 'Play'.upper())
assert(m.predict(['SUNNY',50,91,'T']) == "Don't Play".upper())
assert(m.predict([np.nan,50,91,'T']) == "Don't Play".upper())

print('Testing Decision Tree with missing values (branch_nan = True)...')
m = dt.DecisionTreeClassifier(missing_branch=True)
data, attributes, categories  = read.readData(data_path = 'Dados/Test_with_nan.csv', class_name='Class',
    dummy=dummy,transform_numeric=transform,use_text = use_text,missing_input='none')
X = data[:,0:-1]
y = np.array(data[:,-1])
m.fit(X,y)
m.to_dot(original_attributes,out='testwithnan.dot')
outlook_index = np.where(original_attributes == 'Outlook')[0][0]
not_nan_rows = [a for a in range(X.shape[0]) if not utils.isnan(X[:,outlook_index][a])]
Xnotnan = (X[not_nan_rows,:])
ynotnan = y[not_nan_rows]
Xs,ys,d = utils.split_categ(Xnotnan,ynotnan,outlook_index,list(set(Xnotnan[:,outlook_index])))

assert(np.isclose((len(ynotnan)/len(y)) *utils.information_gain(ynotnan,ys),0.199,rtol=1e-2))
assert(np.isclose((len(ynotnan)/len(y)) *utils.gain_ratio(ynotnan,ys,y),0.110,rtol=1e-2))
#outlook, temperature, humidity, windy                   
assert(m.predict((['OVERCAST',80,90,'T'])) == 'Play'.upper())
assert(m.predict(['RAIN',80,50,'F']) == 'Play'.upper())
assert(m.predict(['RAIN',80,50,'F']) == 'Play'.upper())
assert(m.predict(['RAIN',80,70,'T']) == "Don't Play".upper())
assert(m.predict(['SUNNY',50,50,'T']) == 'Play'.upper())
assert(m.predict(['SUNNY',70,np.nan,'F']) == "Don't Play".upper())

print('Testing Decision Tree with missing values (branch_nan = False)...')
m = dt.DecisionTreeClassifier(missing_branch=False)

m.fit(X,y)
m.to_dot(original_attributes,out='testwithnanf.dot')
#m.to_pdf(original_attributes,out='out.pdf')
outlook_index = np.where(original_attributes == 'Outlook')[0][0]
not_nan_rows = [a for a in range(X.shape[0]) if not utils.isnan(X[:,outlook_index][a])]
Xnotnan = (X[not_nan_rows,:])
ynotnan = y[not_nan_rows]
Xs,ys,d = utils.split_categ(Xnotnan,ynotnan,outlook_index,list(set(Xnotnan[:,outlook_index])))

assert(np.isclose((len(ynotnan)/len(y)) *utils.information_gain(ynotnan,ys),0.199,rtol=1e-2))
assert(np.isclose((len(ynotnan)/len(y)) *utils.gain_ratio(ynotnan,ys,y),0.110,rtol=1e-2))
#outlook, temperature, humidity, windy                   
assert(m.predict((['OVERCAST',80,90,'T'])) == 'Play'.upper())
assert(m.predict(['RAIN',80,50,'F']) == 'Play'.upper())
assert(m.predict(['RAIN',80,50,'F']) == 'Play'.upper())
assert(m.predict(['RAIN',80,70,'T']) == "Don't Play".upper())
assert(m.predict(['SUNNY',50,50,'T']) == 'Play'.upper())
assert(m.predict(['SUNNY',70,np.nan,'F']) == "Don't Play".upper())

print('Testing Decision Tree score method...')

assert(m.score([['RAIN',63,50,'T'],['SUNNY',66,90,'F'],['SUNNY',50,50,'T'],
    ['OVERCAST',70,50,'F']], ['PLAY','PLAY','PLAY','PLAY']) == accuracy_score(m.predict(([['RAIN',63,50,'T'],['SUNNY',66,90,'F'],['SUNNY',50,50,'T'],
    ['OVERCAST',70,50,'F']])), ['PLAY','PLAY','PLAY','PLAY']))

print('Testing Random Forest...')

clf = rf.RandomForest(ntrees=8,mtry=math.sqrt,oob_error=True,random_state=9,missing_branch=False,prob_answer=False,max_depth=3,replace=False)
clf.fit(X,y)
fcs = clf.feature_contribution()
clf.forest[-1].to_dot(original_attributes,out='out.dot')

clf.forest[-1].to_pdf(original_attributes,out='out.pdf')

print("Testing Random Forest with missing data...")
data, original_attributes, categories  = read.readData(data_path = 'Dados/Test_with_nan2.csv', class_name='Class',
    dummy=dummy,transform_numeric=transform,use_text = use_text,missing_input='none')

X = data[:,0:-1]
y = np.array(data[:,-1])
m = rf.RandomForest(ntrees=3,oob_error=True,random_state=9,missing_branch=False,prob_answer=False,
    max_depth=3,replace=False,balance=True)

m.fit(X,y)
import pickle

with open('saved_model.pickle','wb') as handle:
   pickle.dump(m,handle)
print('Testing if model was saved correctly...')
if(m.oob_error_ == 0.428571429):
    with open('saved_model.pickle', 'rb') as handle:
        m2 = pickle.load(handle)
    assert(m2.oob_error_ == m.oob_error_)
    assert(m.predict(['OVERCAST',80,80,'T']) == ['Play'.upper()])
    assert(m.predict(['RAIN',74,81,'T']) == ["Don't Play".upper()])
# print(m.predict(['Feminino',23,np.nan,'Não'],prob=True))
# exit()
print('Testing Random Forest score method...')
assert(m.score([['RAIN',63,50,'T'],['SUNNY',66,90,'F'],['SUNNY',50,50,'T'],
    ['OVERCAST',70,50,'F']], ['PLAY','PLAY','PLAY','PLAY']) == accuracy_score(clf.predict(([['RAIN',63,50,'T'],['SUNNY',66,90,'F'],['SUNNY',50,50,'T'],
    ['OVERCAST',70,50,'F']])), ['PLAY','PLAY','PLAY','PLAY']))

print('Done!')

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# data, attributes, categories  = read.readData(data_path = 'Dados/breast-cancer-wisconsin-prognosis.csv', class_name='outcome',
#     dummy=dummy,transform_numeric=transform,use_text = use_text,missing_input='none')
# X = data[:,0:-1]
# y = np.array(data[:,-1])


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0,stratify=y)
# #clf = rf.RandomForest(ntrees=50,oob_error=True,random_state=9,
# #   mtry=math.sqrt,missing_branch=True,prob_answer=False,max_depth=None,replace=False,balance=False)
# clf = rf.DecisionTreeClassifier()
# clf.fit(X_train,y_train)
# #print(1-clf.oob_error_)
# print(clf.score(X_test,y_test))
# print(accuracy_score(y_test,clf.predict(X_test)))


#test_size=0.2,random_state=9)

# exercise_index = np.where(attributes == 'Exercício?')[0][0]
# feature_index = exercise_index
# not_nan_rows = [a for a in range(X.shape[0]) if not utils.isnan(X[:,feature_index][a])]
# Xs,ys,d = utils.split_categ(X[not_nan_rows],y[not_nan_rows],exercise_index,list(set(X[not_nan_rows,exercise_index])))
# print(utils.information_gain(y[not_nan_rows],ys))

# m.to_dot(attributes,out='out.dot')

exit()
data, original_attributes, categories  = read.readData(data_path = 'Dados/TestBaloonAdultAct.csv', class_name='inflated',
    dummy=dummy,transform_numeric=transform,use_text = use_text,missing_input='none')
X = data[:,0:-1]
y = np.array(data[:,-1])
#import plot
# plot.plot_randomforest_accuracy(X,y,original_attributes,ntrees=100,mtry=math.sqrt,replace=False,max_depth=None,missing_branch=False)
# exit()
seeds = [10,25,40,50,120,35,128,90,97,100]
import time
dif = []
i = 0
for seed in seeds:
    starttime = time.time()
    clf = rf.RandomForest(ntrees=3000,mtry=math.sqrt,oob_error=True,random_state=seed,missing_branch=True,prob_answer=False,max_depth=None,replace=False)
    clf.fit(X,y)
    print(clf.oob_error_)
    clf.forest[0].to_dot(original_attributes,out='out'+str(i)+'n.dot')
    dif.append((time.time()-starttime))
    i += 1
print('mean dif: %r' % np.mean(dif))
exit()
#fcs = clf.feature_contribution()
#clf.forest[-1].to_dot(original_attributes,out='out.dot')
print('Err Missing=True:')
variable_importances = sorted(clf.variable_importance(vitype='err',vimissing=True).items(), key=lambda x: x[1],reverse=True)
for element,importance in variable_importances:#in sorted(clf.variable_importance(vitype='auc'),key=lambda x: x[1],reverse=True):
    print('Variable: %r Importance: %r' % (original_attributes[element],importance))


print('Err Missing=False:')
variable_importances = sorted(clf.variable_importance(vitype='err',vimissing=False).items(), key=lambda x: x[1],reverse=True)
for element,importance in variable_importances:#in sorted(clf.variable_importance(vitype='auc'),key=lambda x: x[1],reverse=True):
    print('Variable: %r Importance: %r' % (original_attributes[element],importance))

