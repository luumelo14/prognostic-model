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
#import MTdecisionTree as mtdt
#import MTrandomForest as mtrf
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.metrics import accuracy_score, classification_report
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
def feature_selection_threshold(X,y,attributes,ntrees,replace,mtry,max_depth,missing_branch,balance,
    cutoff,ntimes=25,title=None,missing_rate=False,vitype='err',vimissing=True,backwards=False):
    vis =  average_varimp(X,y,attributes,ntrees,replace,mtry,max_depth,missing_branch,balance,cutoff,
        missing_rate=missing_rate,ntimes=ntimes,select=False,mean=False,vitype=vitype,vimissing=vimissing,printvi=False)

    if(backwards):
        reverse=False
        comp_threshold = lambda x,y: x <= y
        get_slice = lambda x,index: x[index:] 
        stop_index=-1
    else:
        reverse=True
        comp_threshold = lambda x,y: x > y
        get_slice = lambda x: x[0:index]
        stop_index=0
    ordered_features = [a[0] for a in sorted(vis,key=lambda x:np.mean(x[1]),reverse=reverse)]
    thresholds =  [round(np.mean(vis[a][1]),10) for a in ordered_features]
    threshold_values = sorted([round(a,10) for a in set(thresholds)],reverse=reverse)
    print(sorted([(attributes[a[0]],round(np.mean(a[1]),10)) for a in vis],key=lambda x: x[1],reverse=True))
 
    stop_indexes = []
    scores = []
    i = 0
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
        if(i == X.shape[0]):
            i = 0
   
        clf.fit(np.append(X[0:i,features],X[i+1:,features],axis=0),np.append(y[0:i],y[i+1:]))
        scores.append(1-clf.oob_error_)
        stop_indexes.append(stop_index)

    print('Best oob errors:')
    for old_index in np.where(np.array(scores) == max(scores))[0]:
        #print('Features:')
        #print((attributes[ordered_features[0:len(ordered_features)-old_index]]))
        print(scores[old_index])

    print('Best 1 s.e. set of features:')
    stdm = sem(scores)
    indexes= np.where(np.array(scores) == scores[((np.abs(np.array(scores)-(max(scores)-stdm))).argmin())])[0]
    index = indexes[0]
    print(scores[index])
    
    #index = indexes[int(len(indexes)/2)]
    #index = max(indexes,key=lambda x: stop_indexes[x])
    #print('Features:')
    #print(attributes[ordered_features[0:len(ordered_features)-index]])

    seed = np.random.randint(0,10000)
    clf = rf.RandomForest(ntrees=ntrees,oob_error=True,random_state=seed,mtry=mtry,
            missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,balance=balance,
            cutoff=cutoff)
    
    clf.attributes = attributes[get_slice(ordered_features,stop_indexes[index])]
    clf.fit(X[:,get_slice(ordered_features,stop_indexes[index])],y)

    print('OOB ERROR: %r' % (1-clf.oob_error_))
    print('threshold: >= %r' % threshold_values[index])
    importance_values = [[round(np.mean(aa),10) for aa in a[1]] for a in vis if round(np.mean(a[1]),10) >= threshold_values[index]]
    features =  attributes[[a[0] for a in vis if round(np.mean(a[1]),10) >= threshold_values[index]]]

    plot.boxplot(importance_values,features,title)
    plot.plot_feature_importance_vs_accuracy(threshold_values,scores,xlabel='threshold',title=title,special=index)
    
    return clf


def boostrap_error(clf,X,y,weight=0.632):
    n_samples = X.shape[0]
    training_error = 1-clf.score(X,y)
    boostrap_samples = []
    loo_boostrap_error = 0
    for i in range(50):
        np.random.seed(i)
        boostrap_samples.append(np.random.choice(n_samples,n_samples,replace=True))
    #clf =  rf.RandomForest(ntrees=50,oob_error=True,random_state=9,mtry=math.sqrt,missing_branch=False,prob_answer=False,max_depth=3,replace=True)
    for i in range(n_samples):
        print('Sample %r: ' % i)
        s = 0
        boostrap_samples_without_i = [b for b in boostrap_samples if i not in b]
        for b in boostrap_samples_without_i:
            clf.fit(X[b],y[b]) 
            s += 1-clf.score(X[i],y[i])
    
        loo_boostrap_error += (1/len(boostrap_samples_without_i)) * s

    loo_boostrap_error = loo_boostrap_error / n_samples      

    err_point632=(1-weight)*training_error + weight*loo_boostrap_error
    
    print('Boostrap .632 error: %r'  % err_point632)
    print('training error: %r' % training_error)
        #1/len(b) 
    #bs = Bootstrap(X.shape[0],10,n_test=1,random_state=9)
    
    #loo_boostrap_error += 1/len(boostrap_samples_without_i) 


def plot_boxplot(X,y,attributes,ntrees,replace,mtry,max_depth,missing_branch,ntimes=25,ptype='fc',title=None):
    if(ptype == 'fc'):
        vis = average_fc(X,y,ntrees,replace,mtry,max_depth,missing_branch,ntimes)
    else:
        vis = average_varimp(X,y,ntrees,replace,mtry,max_depth,missing_branch,ntimes,select=False)

    importance_values = [a[1] for a in vis]
    features = attributes[[a[0] for a in vis]]

    import plot
    plot.boxplot(importance_values,features,title)



def average_varimp(X,y,attributes,ntrees,replace,mtry,max_depth,missing_branch,vitype='err',ntimes=25,
    select=True,printvi=False,plotvi=False,cutpoint=0.0,mean=False,title=None, missing_rate=False):
    vi = {a: [] for a in range(X.shape[1])}
    for i in range(X.shape[0]):
        if(i < ntimes):
            seed = np.random.randint(0,10000)
            clf = rf.RandomForest(ntrees=ntrees,oob_error=True,random_state=seed,mtry=mtry,
                missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,balance=True)
            #print(np.append(X[0:i,:],X[i+1:,:],axis=0).shape)
            #print(len(np.append(y[0:i],y[i+1:])))
            clf.fit(np.append(X[0:i,:],X[i+1:,:],axis=0),np.append(y[0:i],y[i+1:]))
            #clf.fit(X,y)
            varimps = clf.variable_importance(vitype=vimissing,vimissing=True)

            for var in varimps.keys():
                if(missing_rate):
                    vi[var].append(varimps[var] * utils.notNanProportion(X[:,var]))
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
            print('feature: %r importance: %r' % (attributes[v],i))

    if(plotvi):
        print(cutpoint)
        importance_values = []
        features = []
        vis = sorted(vi.items(),key=lambda x: x[0])
        for v,i in vis:
            if(vimean[v] >= cutpoint):
                importance_values.append(i)
                features.append(attributes[v])
        import plot
        plot.boxplot(importance_values,features,title)

    if(select):
        vis = sorted(vimean.items(),key=lambda x: x[1],reverse=True)
        return sorted([var[0] for var in vis if vimean[var[0]] >= cutpoint])
    if(mean):
        return sorted(vimean.items(),key=lambda x: x[1],reverse=True)
        #return [var[0] for var in vis]

    return sorted(vi.items(),key=lambda x: x[0])

def average_fc(X,y,ntrees,replace,mtry,max_depth,missing_branch,ntimes=25):
    feature_contributions = {a: [] for a in range(X.shape[1])}
    if ('N' in set(y)):
        class_of_interest = 'N'
    elif ('Sucesso' in set(y)):
        class_of_interest = 'Sucesso'
    else:
        class_of_interest = set(y)[0]
        print('class of interest set to be %r' % class_of_interest)

    for i in range(ntimes):
        seed = np.random.randint(0,10000)
        clf = rf.RandomForest(ntrees=ntrees,oob_error=True,random_state=seed,mtry=mtry,
            missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,balance=True)
        clf.fit(X,y)
        fcs = clf.feature_contribution()
        for i in range(len(fcs)):
            for f in fcs[i].keys():
                feature_contributions[f].append(fcs[i][f][class_of_interest])

    # for f in feature_contributions.keys():
    #     if(len(feature_contributions[f]) == 0):
    #         print(f)
    #     feature_contributions[f] = np.mean(feature_contributions[f])

    return sorted(feature_contributions.items(),key=lambda x: x[0],reverse=True)

def compare_models(X,y,class_name,transform=False,scale=False,n_splits=10,test_size=None,random_state=9,use_feature_selection=False):
    if transform and scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    n_splits = X.shape[0]
    models = [SGDClassifier(random_state=9,penalty='l1',loss='squared_loss'),LogisticRegression(random_state=9,penalty='l2'),
    MLPClassifier(random_state=9,activation='logistic',max_iter = 300,early_stopping=True),
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


            clf.fit(Xtrain,ytrain)

            slg = clf.score(Xval,[(yv) for yv in (yval)])
            clf_scores.append(slg)    

            if slg > maxl:
               maxl = slg


        print('Acurácia média:')
        print(np.mean(clf_scores))
        print('Acurácia mediana:')
        print(np.median(clf_scores))
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

def check_feature_rate(X,y,attributes,ntrees,replace,mtry,max_depth,missing_branch):

    seed = np.random.randint(0,10000)
    clf1 = rf.RandomForest(ntrees=ntrees,oob_error=True,random_state=seed,
        mtry=mtry,missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,balance=True)
    clf1.fit(X,y)
    attributes_used = {}
    for tree in clf1.forest:
        
        for attribute in tree.feature_indices:
            if(attribute not in attributes_used.keys()):
                attributes_used[attribute] = 1
            else:
                attributes_used[attribute] += 1

    if(len((attributes_used.keys())) != X.shape[1]):
        print(len(attributes_used.keys()))
        print(X.shape[1])
        print('not equal!!! %r' % (1-len(attributes_used.keys())/X.shape[1]))
    print({attributes[a]: b for a,b in attributes_used.items()})
    print(1-clf1.oob_error_)

def check_other_participants(filename):
    import pandas as pd
    p = []
    new_data = pd.read_csv('Dados/All_New_Participants.csv')
    data = pd.read_csv('EXPORT/NES_EXPORT_EntradaUnificada_11_04_2017/Questionnaire_data/Q44071_Avaliação-de-Entrada-Unificada/Responses_Q44071.csv')

    for participant in new_data[new_data.columns[0]]:
        if(not np.any(participant == data['participant_code'])):
            p.append(participant)
            print(participant)
    print(len(p))

data_paths = [['../DorCirurgiaCategNAReduzido.csv','Q92510_snDorPos'],
['AbdOmbroCirurgiaCategNA.csv','Q92510_opcForca[AbdOmbro]'],['AbdOmbroCategNAReduzido.csv','Q92510_opcForca[AbdOmbro]'],
['FlexCotoveloCategNA.csv','Q92510_opcForca[FlexCotovelo]'],['FlexCotoveloCirurgiaCategNAReduzido.csv','Q92510_opcForca[FlexCotovelo]'],
['RotEOmbroCirurgiaCategNA.csv','Q92510_opcForca[RotEOmbro]'],['RotEOmbroCirurgiaCategNAReduzido.csv','Q92510_opcForca[RotEOmbro]']]
#data_path='Dor.csv'
#data_paths = ['DorCirurgiaPrePosCateg.csv','DorCirurgiaPrePos.csv','DorPrePosCateg.csv','Dor.csv']
#data_path = 'DorCirurgiaPrePos.csv'
#data_path = 'DorWithoutPrePost.csv'
#data_path = 'Dados/Dor_reduzido.csv'
#data_path = 'DorCirurgiaCateg.csv'
#data_path = 'DorCirurgiaCategReduzido.csv'
#data_path = 'DorCategReduzido'
#data_path = 'DorCirurgiaReduzido.csv'
#data_path = 'AbdOmbroCirurgiaCategReduzido.csv'
#data_path = 'RotEOmbroReduzido.csv'
#data_path = 'FlexCotoveloCirurgia.csv'
#data_path = 'FlexCotovelo_reduzido.csv'
#data_path = 'RotEOmbroCirurgia.csv'
#data_path = 'RotEOmbro_reduzido.csv'
class_questionnaire = 'Q92510'
#class_name = 'Q92510_snDorPos' 
#class_name = 'Q92510_opcForca[AbdOmbro]' 
#class_name = 'Q92510_opcForca[FlexCotovelo]'
#class_name = 'Q92510_opcForca[RotEOmbro]'

missing_input= 'none'#'mean'
transform = False
scale = True
use_text = False
dummy = False
use_feature_selection = False


seed = 1994
for data_path,class_name in data_paths:
    data, original_attributes, categories  = read.readData(data_path = data_path, class_name = class_name, 
        class_questionnaire = class_questionnaire, missing_input = missing_input, dummy = dummy,
        transform_numeric = transform, use_text=use_text, skip_class_questionnaire=True)#skip_class_questionnaire=False)

    X = data[:,0:-1]
    print(X.shape)
    y = np.array(data[:,-1])

    # compare_models(X,y,class_name,transform=True,scale=True,n_splits=10,test_size=0.2,random_state=9,use_feature_selection=False)
    # exit()
    # import pickle
    # with open('prognostic_model_'+ class_name[7:] + '_' + data_path[:-4] + '.pickle', 'rb') as handle:
    #     clf = pickle.load(handle)


    ntimes = 5
    ntrees = 500
    replace = False
    mtry = math.sqrt
    max_depth = None
    missing_branch = True
    seed =  89444   
    # import plot
    # plot.plot_randomforest_accuracy(X,y,original_attributes,ntrees,replace,mtry,max_depth,missing_branch,ntimes,title=data_path+'mb=T e ntrees=15001')
    # exit()

    # print('--------------- MODEL: %r DATA PATH: %r' % (class_name, data_path))
    # clf = feature_selection_threshold(X,y,original_attributes,ntrees,replace,mtry,max_depth,missing_branch,ntimes=ntimes,
    #     missing_rate=True,title=None)
    # with open('prognostic_model_'+ '_' + data_path[:-4] + '_mrate=T_notapplicable=T'+'.pickle','wb') as handle:
    #     pickle.dump(clf,handle)
    # with open('prognostic_model_'+ '_' + data_path[:-4] + '.pickle', 'rb') as handle:
    #     clf1 = pickle.load(handle)

    #print(1-clf.oob_error_)
    # exit()

    # attributes = np.array(['Q44071_snDorPos', 'Q44071_opcLcSensor[C7]','Q44071_opcLcSensTatil[C6]','Q44071_opcForca[FlexDedos]','Q61802_opctransferencias[SQ003]',
    # 'Q44071_lisLcLPB[C]','Q44071_opcLcSensor[C8]','Q44071_lisMedicAt[outros1_Nome]','Q44071_snFxPr', 'Q44071_opcLcSensTatil[C8]', 
    # 'Q44071_opcLcSensTatil[T2]', 'Q44071_opcLcSensor[C6]', 'Q44071_snAuxilioAt', 'Q44071_snFxAt','Q44071_snCplexoAt'])#,'participant code'])
    t = time.time()
    clf1 = rf.RandomForest(ntrees=ntrees,oob_error=True,random_state=seed,
        mtry=mtry,missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,balance=True)
    print('Fitting random forest...')
    # attributes_indexes = np.array([np.where(original_attributes == a)[0][0] for a in attributes])
    # Xn = X[:,attributes_indexes]
    # ftc = np.where(original_attributes == 'Q61802_formTempoCirurg')[0][0]
    # f = np.where(original_attributes == 'Q44071_snCplexoAt')
    # for i in range(Xn.shape[0]):
    #     if(not utils.isnan(X[i][ftc])):
    #         if(X[i][f] == 'N'):
    #             Xn[i][-1] = 'S'

    clf1.fit(X,y)
    print(time.time() - t)
    print(1-clf1.oob_error_)
    # if 'Dor' in class_name:
    #     clf1.control_class = 'N'
    # else:
    #     clf1.control_class = 'SUCESSO'
    # import plot

    #plot.iter_plot_feature_contribution(clf1,attributes,dif_surgery=True)

    exit()


# clf1.forest[3].to_pdf(original_attributes,'out0.pdf')
# clf1.forest[11].to_pdf(original_attributes,'out1.pdf')
# clf1.forest[82].to_pdf(original_attributes,'out2.pdf')
# clf1.forest[80].to_pdf(original_attributes,'out3.pdf')
# clf1.forest[74].to_pdf(original_attributes,'out4.pdf')
# clf1.forest[40].to_pdf(original_attributes,'out5.pdf')


#compare_models(X,y,class_name,transform=transform,scale=scale,use_feature_selection=use_feature_selection) 
#select_params(X,y)
#check_feature_rate(X,y)
# if('Ombro' in data_path):
#     vitype = 'auc'
# else:
#     vitype = 'err'
#feature_selection(X,y,original_attributes,ntrees,replace,mtry,max_depth,missing_branch,ntimes=25)

print('calculating varimp...')
vis = average_varimp(X,y,original_attributes,ntrees,replace,mtry,max_depth,missing_branch,vitype='err',ntimes=ntimes,select=False,printvi=True,plotvi=True,cutpoint=0.0,mean=False,title=data_path)

clf2 = rf.RandomForest(ntrees=ntrees,oob_error=True,random_state=seed,
   mtry=mtry,missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,balance=True)

features = vis
clf2.fit(X[:,features],y)

print(1-clf2.oob_error_)
# import pickle

# with open('prognostic_model_'+ class_name[7:] + '_' + data_path[:-4] + '.pickle','wb') as handle:
#     pickle.dump(clf2,handle)


#plot_boxplot(X[:,features],y,original_attributes[features],ntrees,replace,mtry,max_depth,missing_branch,ntimes=50,ptype='vi')
# import plot
clf2.control_class='N'
plot.iter_plot_feature_contribution(clf2,original_attributes[features])


