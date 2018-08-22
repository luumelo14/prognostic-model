import decisionTree as dt 
import randomForest as rf
import pandas as pd  
import numpy as np
from collections import Counter
import matplotlib 
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import math
import utils
import matplotlib.patches as mpatches
import matplotlib.lines as mlines


def boxplot(collections,attributes,title=None):

    ax = plt.axes()
    ax.boxplot(collections)
    ax.set_xticklabels(attributes,rotation=90,size='small')
    x1,x2,y1,y2 = plt.axis()
    if(y2 <= 0.01):
        y2 = 0.01
    plt.axis((x1,x2,y1,y2))
    plt.tight_layout()
    plt.xlabel('features')
    plt.ylabel('feature importances')
    if(title):
        plt.title(title)
        plt.savefig('boxplot_'+title+'.png')
        plt.close()
        f = open('boxplot_data_'+title+'.txt', 'w')
        f.write('collections = %r \n' % collections)
        f.write('attributes = np.%r \n' % attributes)
    else:
        plt.show()
        print(collections)
    

def plot_feature_importance_vs_accuracy(xvalues,yvalues,xlabel='threshold',title=None,special=None):
    ax = plt.subplot(111)    
    #ax.scatter(xvalues,yvalues,marker='x',s=60)
    ax.plot(xvalues,yvalues,'x-')
    if(special is not None):
        ax.scatter(xvalues[special],yvalues[special],marker='x',s=60,color='red',linewidths=3)
    
    plt.axis((-0.005,0.007,0.6,0.85))
    plt.xlabel(xlabel)
    plt.ylabel('accuracy (1 - OOB error)')
    
    if(title):
        plt.title(title)
        plt.savefig('threshold_accuracy_plot_'+title+'.png')
        plt.close()
        f = open('threshold_accuracy_data_'+title+'.txt', 'w')
        f.write('xvalues= %r \nyvalues= %r \nspecial= %r' % (xvalues,yvalues,special))       
    else:
        plt.show()
        print(xvalues,yvalues)
        print('special: %r' % special)

def plot_mean_feature_contribution(clf,attributes):
    try:
        class_of_interest = clf.class_of_interest
    except(AttributeError):
        class_of_interest = list(set(clf.y))[0]
    fcs = clf.feature_contribution()
    pos_feature_contributions = {}
    neg_feature_contributions = {}
    pos_means = []
    neg_means = []
    colors = []

    for i in range((clf.X.shape[0])):
        for feature in fcs[i].keys():
            if(fcs[i][feature][class_of_interest] >= 0):
                if(feature not in pos_feature_contributions.keys()):
                    pos_feature_contributions[feature] = [fcs[i][feature][class_of_interest]]
                else:
                    pos_feature_contributions[feature].append(fcs[i][feature][class_of_interest])
            else:
                if(feature not in neg_feature_contributions.keys()):
                    neg_feature_contributions[feature] = [fcs[i][feature][class_of_interest]]
                else:
                    neg_feature_contributions[feature].append(fcs[i][feature][class_of_interest])

    for feature in sorted(pos_feature_contributions.keys()):
        pos_means.append(np.median(pos_feature_contributions[feature]))

    for feature in sorted(neg_feature_contributions.keys()):
        neg_means.append(np.median(neg_feature_contributions[feature]))


    plt.bar(range(len(pos_feature_contributions.keys())),pos_means,color='blue')
    plt.bar(range(len(neg_feature_contributions.keys())),neg_means,color='red')
    pos = range(len(attributes))
    width = 1.0     # gives histogram aspect to the bar diagram
    ax = plt.axes()
    ax.set_xticklabels(attributes[sorted(pos_feature_contributions.keys())])
    ax.set_xticks(np.array(pos) +  (width / 2))

    plt.show()

    

def iter_plot_feature_contribution(clf,attributes,dif_surgery=False):
    try:
        class_of_interest = clf.control_class
    except(AttributeError):
        class_of_interest = list(set(clf.y))[0]
    print(class_of_interest)
    if(clf.feature_contribution is False):
        print('Feature contribution not calculated for model. Please use "feature_contribution = True" when creating the model.')
        exit(-1)

    fcs = clf.feature_contribution()

    if(len(clf.X) == 0):
        print('Please fit some data to the model first.')
        exit(-1)

    while(1):
        f = 0
        feature_name = input('Enter the feature name (or enter "\q" to quit): ')
        if(feature_name == '\q'):
            break
        try:
            print(attributes[:-1])
            print( np.where(attributes[:-1] == feature_name))
            feature_index = np.where(attributes[:-1] == feature_name)[0][0]
            f = 1
        except(IndexError):
            print('Feature %r could not be found.' % (feature_name))
            print('List of possible features:')
            for attribute in attributes[:-1]:
                print(attribute)
        if(f == 1):
            if(dif_surgery):
                plot_feature_contributions_surgery_class(clf.X,clf.y,feature_index,fcs,attributes,class_of_interest)
            else:
                plot_feature_contributions(clf.X,feature_index,fcs,attributes,class_of_interest)


def plot_feature_contributions_surgery_class(X,y,feature_index,fcs,attributes,class_of_interest,title=None):
    surgery_index = np.where(attributes == 'Q44071_snCplexoAt')[0][0]

    if(not utils.isint(X[utils.firstNotNan(X[:,feature_index])][feature_index]) and not utils.isfloat(X[utils.firstNotNan(X[:,feature_index])][feature_index])):
        values = [i for i in set(X[:,feature_index]) if not utils.isnan(i)] + [np.nan]

        x_surgery  = []
        surgery_colors = []
        x_no_surgery = []
        no_surgery_colors = []
        x_nan = []
        nan_colors = []
        y_surgery = []
        y_no_surgery = []
        y_nan = []

        contributions = {}

        for i in range(X.shape[0]):
            
            if(feature_index in fcs[i].keys()):
            
                if(X[i][surgery_index] == 'S' or X[i][surgery_index] == 'Y'):
                    x_surgery.append(fcs[i][feature_index][class_of_interest])
                    y_surgery.append(values.index(X[i][feature_index]))
                    if(y[i] == class_of_interest):
                        surgery_colors.append('blue')
                    else:
                        surgery_colors.append('red')

                elif(utils.isnan(X[i][surgery_index])):
                    x_nan.append(fcs[i][feature_index][class_of_interest])
                    #this is necessary because of weird behavior when X[i][feature_index] is nan
                    #and for some reason it says that nan is not values
                    y_nan.append(len(values)-1)
                    if(y[i] == class_of_interest):
                        nan_colors.append('blue')
                    else:
                        nan_colors.append('red')
                else:
                    x_no_surgery.append(fcs[i][feature_index][class_of_interest])
                    y_no_surgery.append(values.index(X[i][feature_index]))
                    if(y[i] == class_of_interest):
                        no_surgery_colors.append('blue')
                    else:
                        no_surgery_colors.append('red')

                # if(X[i][feature_index] not in contributions.keys()):
                #     contributions[X[i][feature_index]] = [fcs[i][feature_index][class_of_interest]]
                # else:
                #     contributions[X[i][feature_index]].append(fcs[i][feature_index][class_of_interest])
        coi = str(class_of_interest)
        ax = plt.subplot(111)    
        ax.scatter(x_surgery,y_surgery,marker='o',s=60,edgecolors=surgery_colors,facecolors='none')
        ax.scatter(x_no_surgery,y_no_surgery,marker='x',s=60,edgecolors=no_surgery_colors,facecolors='none')
        ax.scatter(x_nan,y_nan,marker='d',s=60,edgecolors=nan_colors,facecolors='none')
        plt.xlabel('feature contribution')
        plt.ylabel('values of feature %r' % attributes[feature_index])
        ax.set_yticks(np.array(range(len(values)+2))-1)
        ax.set_yticklabels([str('')]+values+[str('')])
        red_patch = mpatches.Patch(color='red')
        blue_patch = mpatches.Patch(color='blue')
        xmarker = mlines.Line2D([], [], color='black', marker='x', markersize=10, linestyle='None')
        omarker = mlines.Line2D([], [], color='black', marker='o', markersize=10, linestyle='None',
            markerfacecolor='None',markeredgecolor='black')
        #plt.legend(handles=[red_patch,blue_patch])

        plt.legend([red_patch,blue_patch,xmarker,omarker],['Classe da instância ≠ '+ coi,
            'Classe da instância = '+coi,'Não passou por cirurgia','Passou por cirurgia'],numpoints=1,fontsize='small')
        plt.show()

    else:

        values = sorted([round(i,4) for i in (set(X[:,feature_index])) if not utils.isnan(i)])# + [np.nan]
        print(values)
        nan_index = values[-1]-values[-2]
        x_surgery  = []
        surgery_colors = []
        x_no_surgery = []
        no_surgery_colors = []
        x_nan = []
        nan_colors = []
        y_surgery = []
        y_no_surgery = []
        y_nan = []

        for i in range(X.shape[0]):
            if(feature_index in fcs[i].keys()):
                if(X[i][surgery_index] == 'S' or X[i][surgery_index] == 'Y'):
                    x_surgery.append(fcs[i][feature_index][class_of_interest])
                    y_surgery.append((X[i][feature_index]))
                    if(y[i] == class_of_interest):
                        surgery_colors.append('blue')
                    else:
                        surgery_colors.append('red')
                elif(utils.isnan(X[i][surgery_index])):
                    x_nan.append(fcs[i][feature_index][class_of_interest])
                    #this is necessary because of weird behavior when X[i][feature_index] is nan
                    #and for some reason it says that nan is not values
                    y_nan.append(values[-1]+nan_index)
                    if(y[i] == class_of_interest):
                        nan_colors.append('blue')
                    else:
                        nan_colors.append('red')
                else:
                    x_no_surgery.append(fcs[i][feature_index][class_of_interest])
                    y_no_surgery.append((X[i][feature_index]))
                    if(y[i] == class_of_interest):
                        no_surgery_colors.append('blue')
                    else:
                        no_surgery_colors.append('red')
        coi = str(class_of_interest)                       
        fig,ax = plt.subplots()    
        ax.scatter(x_surgery,y_surgery,marker='o',s=60,facecolors='none',edgecolors=surgery_colors)
        ax.scatter(x_no_surgery,y_no_surgery,marker='x',s=60,edgecolors=no_surgery_colors)
        ax.scatter(x_nan,y_nan,marker='d',s=60,facecolors='none',edgecolors=nan_colors)
        fig.canvas.draw()
        labels =  ['']+[item.get_text() for item in ax.get_yticklabels()]+['']  
        if(values[-1]+nan_index < ax.get_yticks()[-1]):
            plt.yticks([values[0]-nan_index]+sorted(list(ax.get_yticks())+[values[-1]+nan_index]))       
        else:
            plt.yticks([values[0]-nan_index]+sorted(list(ax.get_yticks())+[values[-1]+nan_index,values[-1]+2*nan_index]))
        labels[-2] = 'nan'

        plt.xlabel('feature contribution')
        plt.ylabel('values of feature %r' % attributes[feature_index])
        ax.set_yticklabels(labels)
        red_patch = mpatches.Patch(color='red')
        blue_patch = mpatches.Patch(color='blue')
        xmarker = mlines.Line2D([], [], color='black', marker='x', markersize=10, label='Bla', linestyle='None')
        omarker = mlines.Line2D([], [], color='black', marker='o', markersize=10,label='Bla', linestyle='None',
            markerfacecolor='None',markeredgecolor='black')
        #plt.legend(handles=[red_patch,blue_patch])
        plt.legend([red_patch,blue_patch,xmarker,omarker],['Classe da instância ≠ '+ coi,
            'Classe da instância = '+coi,'Não passou por cirurgia','Passou por cirurgia'],numpoints=1,fontsize='small')
        plt.show()
    
    if(title is not None):        
        plt.savefig(title)
        plt.close()    
        f = open(title,'w')
        f.write('X='+str(X))
        f.write('\ny='+str(y))
        f.write('\nfcs='+str(fcs))
        f.write('\nfeatures='+str(attributes))
        f.write('\nfeature_index='+str(feature_index))
        f.write('\nvalues='+str(values))
        f.write('\nx_surgery='+str(x_surgery))
        f.write('\ny_surgery='+str(y_surgery))
        f.write('\nsurgery_colors='+str(surgery_colors))
        f.write('\nx_no_surgery='+str(x_no_surgery))
        f.write('\ny_no_surgery='+str(y_no_surgery))
        f.write('\nno_surgery_colors='+str(no_surgery_colors))
        f.write('\nx_nan='+str(x_nan))
        f.write('\ny_nan='+str(y_nan))
        f.write('\nnan_colors='+str(nan_colors))

def plot_feature_contributions_instance(i,fcs,attributes,class_of_interest,title=None):
    attribute_names = {'Q44071_snDorPos':'Sente dor após a lesão?',
    'Q44071_opcLcSensor[C6]': 'Sensibilidade superficial \ndolorosa [C6]',
    'Q44071_opcLcSensor[C7]': 'Sensibilidade superficial \ndolorosa [C7]',
    'Q44071_opcLcSensor[C8]': 'Sensibilidade superficial \ndolorosa [C8]',
    'Q44071_opcLcSensTatil[C6]': 'Sensibilidade superficial \ntátil [C6]',
    'Q44071_opcLcSensTatil[C7]': 'Sensibilidade superficial \ntátil [C7]',
    'Q44071_opcLcSensTatil[C8]': 'Sensibilidade superficial \ntátil [C8]',
    'Q44071_opcLcSensTatil[T1]': 'Sensibilidade superficial \ntátil [T1]',
    'Q44071_opcForca[FlexDedos]': 'Força muscular \n[Flexão dos Dedos]',
    'Q44071_opcForca[FlexCotovelo]': 'Força muscular \n[Flexão do Cotovelo]',
    'Q44071_opcForca[AbdOmbro]': 'Força muscular \n[Abdução do Ombro]',
    'Q44071_opcForca[AdDedos]': 'Força muscular \n[Adução dos Dedos]',
    'Q44071_snFxPr':'Tem história prévia \nde fratura?', 
    'Q44071_snFxAt':'Teve fratura associada \nà lesão?', 
    'Q61802_opctransferencias[SQ003]': 'Transferência realizada \n[Oberlin]',
    'Q44071_lisMedicAtNer': 'Faz uso de medicamento com ação \nsobre o sistema nervoso?',
    'Q44071_lisTpTrauma[moto]': 'Evento que levou ao \ntrauma [moto]',
    'Q44071_lisTpAuxilio[Tipoia]': 'Se faz uso de dispositivo \nauxiliar [Tipoia]',
    'Q44071_lisTpAuxilio[Suporte]': 'Se faz uso de dispositivo \nauxiliar [Suporte de Ombro]',
    'Q44071_formTempoAval': 'Período entre a lesão \ne a primeira consulta \nno INDC',
    'Q44071_snDesacordado': 'Ficou desacordado?',
    'Q44071_snCplexoAt': 'Já fez alguma cirurgia \ndo plexo braquial?',
    'Q61802_opcLdCirurgia': 'Qual o lado operado?'}
    values = []
    pos_fcs = []
    y_fcs = []
    colors = []
    for feature_index in range(len(attributes)):
        values.append(str(attribute_names[attributes[feature_index]]) +' : ' + str(i[feature_index]))
        if(feature_index in fcs.keys()):
            pos_fcs.append(fcs[feature_index][class_of_interest])
            y_fcs.append(feature_index)
            if(fcs[feature_index][class_of_interest] > 0):
                colors.append('blue')
            elif(fcs[feature_index][class_of_interest] < 0):
                colors.append('red')
            else:
                colors.append('black')

    print(pos_fcs)
    print(y_fcs)
    ax = plt.subplot(111)    
    plt.plot(y_fcs,pos_fcs,'x')#,colors=colors)
    #plt.plot(neg_fcs,neg_values,'x',color='red')
    #plt.plot(zero_fcs,zero_values,'x',color='black')
    plt.ylabel('contribuição')
    plt.xlabel('atributos')

    ax.set_xticks(np.array(range(len(values)+2))-1)
    ax.set_xticklabels([str('')]+values+[str('')],rotation=90)

    plt.plot(np.arange(len(y_fcs)+2)-1,np.array([0 for i in (np.arange(len(y_fcs)+2)-1)]),'r--',color='black')
    plt.tight_layout()
    plt.show()

    
    if(title is not None):        
        #plt.savefig(title)
        #plt.close()        
        f = open(title,'w')
        #f.write('fcs='+str(fcs))
        f.write('X='+str(i))
        f.write('\nfeatures_values='+str(values))
        f.write('\ncontributions='+str(pos_fcs))
def plot_feature_contributions(X,feature_index,fcs,attributes,class_of_interest,title=None):
    
    if(not utils.isint(X[utils.firstNotNan(X[:,feature_index])][feature_index]) and not utils.isfloat(X[utils.firstNotNan(X[:,feature_index])][feature_index])):
        values = [i for i in set(X[:,feature_index]) if not utils.isnan(i)] + [np.nan]

        pos_fcs = []
        neg_fcs = []
        pos_values = []
        neg_values = []
        zero_fcs = []
        zero_values = []
        contributions = {}
        
        for i in range(X.shape[0]):
            
            if(feature_index in fcs[i].keys()):
                if(fcs[i][feature_index][class_of_interest] > 0):
                    pos_fcs.append(fcs[i][feature_index][class_of_interest])
                    #this is necessary because of weird behavior when X[i][feature_index] is nan
                    #and for some reason it says that nan is not values
                    if(utils.isnan(X[i][feature_index])):
                        pos_values.append(len(values)-1)
                    else:
                        pos_values.append(values.index(X[i][feature_index]))
                elif(fcs[i][feature_index][class_of_interest] == 0):
                    zero_fcs.append(0)
                    if(utils.isnan(X[i][feature_index])):
                        zero_values.append(len(values)-1)
                    else:
                        zero_values.append(values.index(X[i][feature_index]))
                else:
                    neg_fcs.append(fcs[i][feature_index][class_of_interest])
                    if(utils.isnan(X[i][feature_index])):
                        neg_values.append(len(values)-1)
                    else:
                        neg_values.append(values.index(X[i][feature_index]))
                if(X[i][feature_index] not in contributions.keys()):
                    contributions[X[i][feature_index]] = [fcs[i][feature_index][class_of_interest]]
                else:
                    contributions[X[i][feature_index]].append(fcs[i][feature_index][class_of_interest])


        print('Contributions:')
        for value in contributions.keys():
            print('Value %r' % value)
            print( '\nMean: %r Variance: %r' % (np.mean(contributions[value]),np.var(contributions[value])))

        c =(contributions.items())
        boxplot([a[1] for a in c],[a[0] for a in c],title=None)


        ax = plt.subplot(111)    
        plt.plot(pos_fcs,pos_values,'x',color='blue')
        plt.plot(neg_fcs,neg_values,'x',color='red')
        plt.plot(zero_fcs,zero_values,'x',color='black')
        plt.xlabel('feature contribution')
        plt.ylabel('values of feature %r' % attributes[feature_index])
        ax.set_yticks(np.array(range(len(values)+2))-1)
        ax.set_yticklabels([str('')]+values+[str('')])
        plt.show()

    else:

        values = sorted([round(i,4) for i in (set(X[:,feature_index])) if not utils.isnan(i)])# + [np.nan]

        nan_index = values[-1]-values[-2]
        pos_fcs = []
        neg_fcs = []
        pos_values = []
        neg_values = []
        zero_fcs = []
        zero_values = []
        contributions = {}

        for i in range(X.shape[0]):
            if(feature_index in fcs[i].keys()):

                if(fcs[i][feature_index][class_of_interest] > 0):
                    pos_fcs.append(fcs[i][feature_index][class_of_interest])
                    #this is necessary because of weird behavior when X[i][feature_index] is nan
                    #and for some reason it says that nan is not values
                    if(utils.isnan(X[i][feature_index])):
                        pos_values.append(values[-1]+nan_index)
                    else:
                        pos_values.append(X[i][feature_index])
                elif(fcs[i][feature_index][class_of_interest] == 0):
                    zero_fcs.append(0)
                    if(utils.isnan(X[i][feature_index])):
                        zero_values.append(values[-1]+nan_index)
                    else:
                        zero_values.append(X[i][feature_index])
                else:
                    neg_fcs.append(fcs[i][feature_index][class_of_interest])
                    if(utils.isnan(X[i][feature_index])):
                        neg_values.append(values[-1]+nan_index)
                    else:
                        neg_values.append((X[i][feature_index]))
                if(utils.isnan(X[i][feature_index])):
                    if('nan' in contributions.keys()):
                        contributions['nan'].append(fcs[i][feature_index][class_of_interest])
                    else:
                        contributions['nan'] = [fcs[i][feature_index][class_of_interest]]
                elif(X[i][feature_index]in contributions.keys()):
                    contributions[(X[i][feature_index])].append(fcs[i][feature_index][class_of_interest])
                else:
                    contributions[(X[i][feature_index])] = [fcs[i][feature_index][class_of_interest]]

        print('Contributions:')
        for value in contributions.keys():
            print('Value %r' % value)
            print( 'Mean: %r Variance: %r' % (np.mean(contributions[value]),np.std(contributions[value])))
        c =(contributions.items())
        boxplot([a[1] for a in c],[a[0] for a in c],title=None)
        fig,ax = plt.subplots()    
        plt.plot(pos_fcs,pos_values,'x',color='blue')
        plt.plot(neg_fcs,neg_values,'x',color='red')
        plt.plot(zero_fcs,zero_values,'x',color='black')
        fig.canvas.draw()
        labels =  ['']+[item.get_text() for item in ax.get_yticklabels()]+['']  
        if(values[-1]+nan_index < ax.get_yticks()[-1]):
            plt.yticks([values[0]-nan_index]+sorted(list(ax.get_yticks())+[values[-1]+nan_index]))       
        else:
            plt.yticks([values[0]-nan_index]+sorted(list(ax.get_yticks())+[values[-1]+nan_index,values[-1]+2*nan_index]))
        labels[-2] = 'nan'

        plt.xlabel('feature contribution')
        plt.ylabel('values of feature %r' % attributes[feature_index])
        ax.set_yticklabels(labels)

        plt.show()
    
    if(title is not None):        
        plt.savefig(title)
        plt.close()

def plot_randomforest_accuracy_nfeatures(X,y,original_attributes,features,ntrees,replace,mtry,max_depth,missing_branch):
    nf = []
    accuracy = []
    nfeatures = len(features)
    seed = np.random.randint(0,10000)
    clf = rf.RandomForest(ntrees=ntrees,mtry=mtry,missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,random_state=seed)
    clf.fit(X,y)
    nf.append(0)
    accuracy.append(1-clf.oob_error_)
    for i in range(1,nfeatures):
        print('eliminating feature %r...' % original_attributes[features[-i]])
        seed = np.random.randint(0,10000)
        clf = rf.RandomForest(ntrees=ntrees,mtry=mtry,missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,random_state=seed)
        clf.fit(X[:,features[:-i]],y)

        nf.append(i) 
        accuracy.append(1-clf.oob_error_)

    plt.plot(nf,accuracy,'bo',color='blue')
    plt.xlabel('number of features not being considered')
    plt.ylabel('accuracy')
    plt.show()


def plot_randomforest_accuracy_threshold(X,y,original_attributes,variable_importances,ntrees,replace,mtry,max_depth,missing_branch):
    thrs = []
    accuracy = []
    nfeatures = len(variable_importances)
    features = [a[0] for a in variable_importances]
    thresholds = [a[1] for a in variable_importances]
    seed = np.random.randint(0,10000)
    clf = rf.RandomForest(ntrees=ntrees,mtry=mtry,missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,random_state=seed)
    clf.fit(X,y)
    thrs.append(thresholds[-1])
    #nf.append(nfeatures)
    accuracy.append(1-clf.oob_error_)
    for i in range(1,nfeatures):
        print('eliminating feature %r...' % original_attributes[features[-i]])
        seed = np.random.randint(0,10000)
        clf = rf.RandomForest(ntrees=ntrees,mtry=mtry,missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,random_state=seed)
        clf.fit(X[:,features[:-i]],y)
        #print(features[:-i])
        #nf.append(nfeatures-i) 
        thrs.append(thresholds[-i-1])
        accuracy.append(1-clf.oob_error_)

    plt.plot(thrs,accuracy,'bo',color='blue')
    plt.xlabel('threshold')
    plt.ylabel('accuracy')
    plt.show()


def plot_randomforest_accuracy(X,y,attributes,ntrees,replace,mtry,max_depth,missing_branch,ntimes,title=None):
    missing_branch_dict = {}
    missing_c45 = []
    seeds = []
    for i in range(ntimes):
    #for seed in range(0,1000,50):
        seed = np.random.randint(100000)
        #print('seed: %r' % seed)
        clf = rf.RandomForest(ntrees=ntrees,mtry=mtry,missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,random_state=seed)
        clf.fit(X,y)

        if round(1-clf.oob_error_,2) not in missing_branch_dict.keys():
            missing_branch_dict[round(1-clf.oob_error_,2)] = 1
        else:
            missing_branch_dict[round(1-clf.oob_error_,2)] += 1
 

    k = sorted(missing_branch_dict.items(),key=lambda x: x[0])

    plt.bar(range(len([i[0] for i in k])),[i[1] for i in k])
    pos = np.arange(len(k))
    width = 1.0     # gives histogram aspect to the bar diagram
    ax = plt.axes()
    ax.set_xticks(pos +  (width / 2))
    ax.set_xticklabels([round(i[0],2) for i in k])
    ax.set_yticks(range(0,50,5))

    plt.xlabel('acurácia com missing branch = ' + str(missing_branch) )

    if(title is not None):
        plt.title(title)
    plt.ylabel('frequência')

    plt.show()


def plot_randomforest_seed(X,y,attributes):
    missing_branch = []
    missing_c45 = []
    seeds = []
    for seed in range(0,1000,10):
        print('seed:')
        print(seed)

        clf = rf.RandomForest(ntrees=300,mtry=math.sqrt,missing_branch=True,prob_answer=False,max_depth=4,replace=False,random_state=seed)
        clf.fit(X,y)

        missing_branch.append(1-clf.oob_error_)
        print(1-clf.oob_error_)

        clf2 = rf.RandomForest(ntrees=300,mtry=math.sqrt,missing_branch=False,prob_answer=False,max_depth=4,replace=False,random_state=seed)
        clf2.fit(X,y)

        missing_c45.append(1-clf2.oob_error_)
        print(1-clf2.oob_error_)

        seeds.append(seed)


    plt.plot(missing_c45,missing_branch,'x',color='blue')
    plt.show()


def plot_entropy_pmissing(X,y,attributes):
    import matplotlib.pyplot as plt
    entrops = []
    entropsc = []
    features = []
    featuresc = []
    pmissings = []
    pmissingsc = []
    n_samples = X.shape[0]


    for feature_index in range(X.shape[1]):
        best_entrpy = np.float('inf')
        #best_entrpy = -np.float('inf')

        not_nan_rows = [a for a in range(n_samples) if not isnan(X[:,feature_index][a])]
        pmissing = (n_samples-len(not_nan_rows))/n_samples
        
        Xtmp = (X[not_nan_rows,:])
        ytmp = y[not_nan_rows]


        if(isnum(Xtmp[0,feature_index])):
            values = (set(Xtmp[:, feature_index]))
            values.discard(np.nan)
            values = sorted(values)     
            for j in range(len(values) - 1):
                value = (values[j] + values[j+1])/2
                [X_true,X_false], [y_true, y_false], [t,f] = split_num(Xtmp, ytmp, feature_index, value)#threshold)

                entrpy = (entropy(y_true)+entropy(y_false)) / 2

                if entrpy < best_entrpy:

                    best_entrpy = entrpy
                    best_feature = feature_index

                    best_value = [value]

        else:       
            values = (set(Xtmp[:, feature_index]))
            values.discard(np.nan)
            values = sorted(values)

            Xs,ys,d = split_categ(Xtmp, ytmp,feature_index,values)

            entrpy = sum(list(entropy(k) for k in ys)) / len(values)

            if entrpy < best_entrpy:

                best_entrpy = entrpy
                best_feature = feature_index
                best_value = values

        if len(best_value) == 1:
            entrops.append(best_entrpy)
            features.append(attributes[best_feature] + ': ' + str(best_value[0]))
            pmissings.append(pmissing)

            
        else:
            entropsc.append(best_entrpy)
            pmissingsc.append(pmissing)
            featuresc.append(attributes[best_feature])

    plt.plot(entrops,pmissings,'x',color='blue')
    plt.plot(entropsc,pmissingsc,'x',color='red')


    plt.axis([-0.25,1.25,-0.25,1])

    plt.xlabel('entropy')
    plt.ylabel('pmissing')

    for i in range(len(pmissingsc)):
        if(pmissingsc[i] < 0.5):
            if(entropsc[i] < 0.2):
                print(attributes[i])

    plt.show()
