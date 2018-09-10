import numpy as np 
import math
import sys
import read
import sklearn
import time
import utils
import plot
from sklearn import  tree
import randomForest as rf
import decisionTree as dt
from scipy.stats import sem
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier, LinearRegression
from sklearn.neural_network import MLPClassifier
import pickle

# this method is based on the algorithm proposed in:
# Robin Genuer, Jean-Michel Poggi, and Christine Tuleau-Malot. 2010. 
# Variable selection using random forests. Pattern Recogn. Lett. 31, 14 (October 2010), 2225-2236. 
# DOI=http://dx.doi.org/10.1016/j.patrec.2010.03.014  
def feature_selection_threshold(X,y,ntrees,replace,mtry,max_depth,missing_branch,balance,
    cutoff,ntimes=25,title=None,missing_rate=False,vitype='err',vimissing=True,backwards=False,save_models=False):
    
    # get average feature importances for each feature
    vis =  average_varimp(X,y,ntrees,replace,mtry,max_depth,missing_branch,
        missing_rate=missing_rate,ntimes=ntimes,select=False,mean=False,vitype=vitype,vimissing=vimissing,printvi=False)
    # if backwards is True, then the feature selection will start the process with all features,
    # and eliminating the least important ones in each step
    if(backwards is True):
        reverse=False
        comp_threshold = lambda x,y: x <= y
        get_slice = lambda x,index: x[index:] 
        stop_index=-1
        chosen_model = -1
    # if it's False, then it starts with only the most important feature, and then adding
    # the most important ones in each step
    else:
        reverse = True
        comp_threshold = lambda x,y: x > y
        get_slice = lambda x: x[0:index]
        stop_index = 0
        chosen_model = 0

    ordered_features = [a[0] for a in sorted(vis,key=lambda x:np.mean(x[1]),reverse=reverse)]
    thresholds =  [np.mean(vis[a][1]) for a in ordered_features] 
    threshold_values = sorted([round(a,10) for a in set(thresholds)],reverse=reverse)

    stop_indexes = []
    scores = []
    i = 0
    nn = 0
    # for each threshold value (feature importance value), create a forest
    # only using: (a) features whose importance value is <= than the threshold
    # if backwards is True (starting from the least important), or
    # (b) features whose importance value is > than the threshold if backwards is False
    # (starting from the most important one)
    for threshold in threshold_values:

        s_index = stop_index+1
        while s_index < len(thresholds):
            if(comp_threshold(threshold,thresholds[s_index])):
                break
            else:
                s_index+=1
        stop_index = s_index
        
        features = get_slice(ordered_features,stop_index)
        seed = np.random.randint(0,10000)
        clf = rf.RandomForest(ntrees=ntrees,oob_error=True,random_state=seed,mtry=mtry,
            missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,balance=balance,
            cutoff=cutoff)

        
        clf.fit(X.values[:,features],y)
        clf.threshold = threshold
        scores.append(1-clf.oob_error_)

        if(save_models is True):
            with open('prognostic_model_' + title+ str(nn) + '.pickle', 'wb') as handle:
                pickle.dump(clf,handle)
            nn+=1
        stop_indexes.append(stop_index)


    stdm = sem(scores)
    indexes= np.where(np.array(scores) == scores[((np.abs(np.array([a for a in scores if a != max(scores)])-(max(scores)-stdm))).argmin())])[0]
    # the forest with the best score (closest to the max score subtracted from the standard error of scores) and 
    # the biggest threshold value (by index -1) is chosen as the suggested model to be used 
    index = indexes[chosen_model]


    clf = rf.RandomForest(ntrees=ntrees,oob_error=True,random_state=seed,mtry=mtry,
            missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,balance=balance,
            cutoff=cutoff)
    
    #clf.attributes = attributes[get_slice(ordered_features,stop_indexes[index])]
    clf.fit(X[X.columns[get_slice(ordered_features,stop_indexes[index])]],y)

    #importance_values = [[round(np.mean(aa),10) for aa in a[1]] for a in vis if round(np.mean(a[1]),10) >= threshold_values[index]]
    #features =  attributes[[a[0] for a in vis if round(np.mean(a[1]),10) >= threshold_values[index]]]

    #plot.boxplot(importance_values,features,title)
    if(save_models is True):

        plot.plot_feature_importance_vs_accuracy(threshold_values,scores,xlabel='threshold',title=title,special=index)
    
    return clf



def average_varimp(X,y,ntrees,replace,mtry,max_depth,missing_branch,vitype='err',vimissing=True,ntimes=25,
    select=True,printvi=False,plotvi=False,cutpoint=0.0,mean=False,title=None, missing_rate=False):
    vi = {a: [] for a in range(X.shape[1])}
    for i in range(X.shape[0]):
        if(i < ntimes):
            seed = np.random.randint(0,10000)
            clf = rf.RandomForest(ntrees=ntrees,oob_error=True,random_state=seed,mtry=mtry,
                missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,balance=True)
            clf.fit(X,y)
            varimps = clf.variable_importance(vitype=vimissing,vimissing=True)
            for var in varimps.keys():
                if(missing_rate):
                    vi[var].append(varimps[var] * utils.notNanProportion(X[X.columns[var]]))
                else:
                    vi[var].append(varimps[var])
        else:
            break

    vimean = {a: [] for a in range(X.shape[1])}
    for var in vi.keys():
        vimean[var] = np.mean(vi[var])

    if(printvi):
        vis = sorted(vimean.items(),key=lambda x: x[1],reverse=True)
        for v,i in vis:
            print('feature: %r importance: %r' % (X.columns[v],i))

    if(plotvi):
        print(cutpoint)
        importance_values = []
        features = []
        vis = sorted(vi.items(),key=lambda x: x[0])
        for v,i in vis:
            if(vimean[v] >= cutpoint):
                importance_values.append(i)
                features.append(X.columns[v])
        import plot
        plot.boxplot(importance_values,features,title)

    if(select):
        vis = sorted(vimean.items(),key=lambda x: x[1],reverse=True)
        return sorted([var[0] for var in vis if vimean[var[0]] >= cutpoint])
    if(mean):
        return sorted(vimean.items(),key=lambda x: x[1],reverse=True)
        #return [var[0] for var in vis]

    return sorted(vi.items(),key=lambda x: x[0])


def compare_models(X,y,class_name,transform=False,scale=False,n_splits=10,test_size=None,random_state=9,use_feature_selection=False):
   
    n_splits = X.shape[0]
    models = [SGDClassifier(random_state=9,penalty='l1',loss='squared_loss'),LogisticRegression(random_state=9),
    MLPClassifier(random_state=9,max_iter = 300,early_stopping=True),
    SVC(kernel='sigmoid',random_state=9), tree.DecisionTreeClassifier(criterion='entropy',random_state=9),
    RandomForestClassifier(n_estimators=100,criterion='entropy',max_features='sqrt', oob_score = True,random_state=9)]


    sss = StratifiedShuffleSplit(n_splits=n_splits,test_size=test_size,random_state=random_state)#test_size=0.2,random_state=9)
    for clf in models:
        print('------------'+str(clf)+'------------')
        print('\n')
        maxl = 0
        clf_scores = []

        for train_index, test_index in sss.split(X,y):
            Xtrain = X[train_index]
            Xval = X[test_index]
            ytrain = y[train_index]
            yval = y[test_index]

            if(use_feature_selection):
                model = SelectFromModel(clf, prefit=False)
                Xtrain = model.fit_transform(Xtrain,ytrain)
                Xval = model.transform(Xval)
            if transform and scale:
                scaler = StandardScaler()
                Xtrain = scaler.fit_transform(Xtrain)
                Xval = scaler.fit_transform(Xval)

            clf.fit(Xtrain,ytrain)

            slg = clf.score(Xval,[(yv) for yv in (yval)])
            clf_scores.append(slg)    

            if slg > maxl:
               maxl = slg


        print('Acurácia média:')
        print(np.mean(clf_scores))
        print('Desvio padrão:')
        print(np.std(clf_scores))
        print('Acurácia máxima:')
        print(maxl)


def select_params(X,y):
    times = 10
    final_scores = []
    parameters = []
    ntrees = 10#[700,600,500,400,300,200,100]#,50,25]
    mtry = [math.sqrt,None,math.sqrt,math.log2]
    max_depth = [2,3,4,None]
    missing_branch = True#[True,False]#,False]#,False]
    replace = False#[True,False]


    for md in max_depth:
        for mb in missing_branch:
            for mt in mtry:
                for r in replace:
                    parameters.append({'max_depth':md,'missing_branch':mb, 'mtry': mt, 'replace': r})
    cont = 0
    print(len(parameters))
    for params in parameters:
        cont+=1
        print('Choice %r of parameters' % cont)

        seed = np.random.randint(0,100000)

        #for seed in np.random.choice(range(1000),times):

        clf = rf.RandomForest(random_state=seed,ntrees=ntrees,oob_error=True,mtry=params['mtry'],
            missing_branch=params['missing_branch'],
            max_depth=params['max_depth'],replace=params['replace'],balance=True)
        clf.fit(X,y)
        final_scores.append(clf.oob_error_)
    
    
    min_score = min(final_scores)
    std = np.std(final_scores)
    print('Best set of parameters:')    
    indexes = np.where(np.array(final_scores) == min_score)[0]
    for index in indexes:
        print(parameters[index])

    print('Best 1.s.e. set of parameters:')
    index = (np.abs(np.array(final_scores)-(min_score+std))).argmin()
    print('%r: %r' % (parameters[index],final_scores[index]))

    print('10 best parameters:')
    for a,b in zip(np.array(parameters)[np.array(final_scores).argsort()[:10]], np.array(sorted(final_scores)[:10])):
        print('%r : %r ' % (a,b))


data_paths = [['../DorCirurgiaCategNAReduzido.csv','Q92510_snDorPos'],
['../AbdOmbroCirurgiaCategNAReduzido.csv','Q92510_opcForca[AbdOmbro]'],
['../FlexCotoveloCirurgiaCategNAReduzido.csv','Q92510_opcForca[FlexCotovelo]'],
['../RotEOmbroCirurgiaCategNAReduzido.csv','Q92510_opcForca[RotEOmbro]']]

class_questionnaire = 'Q92510'
missing_input= 'none' #'mean'
transform = False
scale = True
use_text = False
dummy = False
use_feature_selection = False

import random
seed = random.randint(0,10000)

for data_path,class_name in data_paths:
    data = read.readData(data_path = data_path, class_name = class_name, 
        class_questionnaire = class_questionnaire, missing_input = missing_input, dummy = dummy,
        transform_numeric = transform, use_text=use_text, skip_class_questionnaire=True)#skip_class_questionnaire=False)

    X = data[data.columns[:-1]]
    y = data[class_name]

    # import pickle
    # with open('prognostic_model_'+ class_name[7:] + '_' + data_path[:-4] + '.pickle', 'rb') as handle:
    #     clf = pickle.load(handle)

    ntimes = 2
    ntrees = 21
    mtry = math.sqrt
    max_depth = None
    missing_branch = True
    #seed =  89444   
    replace = False
    vitype = 'auc'
    cutoff=0.5
    if('Ombro' in data_path):
        balance = True
    else:
        balance = False

    print('--------------- MODEL: %r DATA PATH: %r' % (class_name, data_path))

    clf = feature_selection_threshold(X,y,ntrees,replace,mtry,max_depth,missing_branch,balance,
    cutoff,ntimes=ntimes,title=class_name,missing_rate=True,vitype=vitype,vimissing=True,backwards=True,save_models=True)
    
    # clf = rf.RandomForest(ntrees=ntrees,oob_error=True,random_state=seed,
    #     mtry=mtry,missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,balance=False,random_subspace=True)
    
    # clf.attributes = np.array(['Q44071_snDorPos', 'Q44071_opcLcSensTatil[C7]', 'Q61802_opctransferencias[SQ003]',
    #      'Q44071_opcLcSensor[C7]', 'Q44071_opcLcSensTatil[C8]', 'Q44071_formTempoAval',  'Q44071_opcLcSensTatil[C6]',
    #      'Q44071_opcForca[FlexDedos]', 'Q44071_opcLcSensor[C8]','Q44071_snFxPr',
    #      'Q44071_opcLcSensTatil[T1]','Q44071_opcForca[AdDedos]','Q44071_snFxAt']) 
    
    #clf1.attributes = np.array(['Q61802_opctransferencias[SQ003]', 'Q44071_opcForca[FlexCotovelo]','Q44071_snDesacordado','Q44071_lisTpAuxilio[Tipoia]','Q44071_lisTpAuxilio[Suporte]'])
    #clf1.attributes = np.array(['Q61802_opcLdCirurgia','Q44071_snCplexoAt', 'Q61802_opctransferencias[SQ003]', 'Q44071_opcForca[AbdOmbro]','Q44071_lisTpTrauma[moto]','Q44071_lisTpAuxilio[Tipoia]','Q44071_lisTpAuxilio[Suporte]'])
    #clf1.attributes = np.array(['Q44071_lisTpTrauma[moto]','Q44071_lisTpAuxilio[Tipoia]','Q44071_lisTpAuxilio[Suporte]','Q44071_snCplexoAt'])


    scores = 0
    s = []
    ivp,ifp,ifn,ivn,svp,sfp,sfn,svn = 0,0,0,0,0,0,0,0 
    #print(1-clf1.oob_error_)
    for i in range(X.shape[0]):
        Xtrain  = X.drop(i)#np.concatenate([X[0:i],X[i+1:]])
        ytrain = y.drop(i) #np.concatenate([y[0:i],y[i+1:]])
        clf.fit(Xtrain,ytrain)
        # clf1 = feature_selection_threshold(Xtrain,ytrain,original_attributes,ntrees,replace,mtry,max_depth,missing_branch,balance,
        #    cutoff,ntimes=ntimes,title=class_name,missing_rate=True,vitype=vitype,vimissing=True,backwards=True)
        
        if(y[i] == 'SUCESSO'):
            if(clf.predict(X.loc[i]) == 'SUCESSO'):
                svp+=1
                ivn+=1
            else:
                sfp+=1
                ifn+=1
        else:
            if(clf.predict(X.loc[i]) == 'SUCESSO'):
                sfn+=1
                ifp+=1
            else:
                svn+=1
                ivp+=1
        scores += clf.score(X.loc[i],y.loc[i]) 
        s.append(clf.score(X.loc[i],y.loc[i]))
    print('desvio: %r' % np.std(s))
    print(scores/X.shape[0])
    p = svp/(svp+sfp)
    c = svp/(svp+sfn)
    if(p + c == 0):
        f = 0
    else:
        f = (2*p*c)/(p+c)
    print('SUCESSO --- cobertura: %r precisão: %r medida-F: %r ' % (c,p,f))
    p = ivp/(ivp+ifp)
    c = ivp/(ivp+ifn)
    if(p + c == 0):
        f = 0
    else:
        f = (2*p*c)/(p+c)
    print('INSATISFATÓRIO --- cobertura: %r precisão: %r medida-F: %r ' % (c,p,f))

    #exit()

  

