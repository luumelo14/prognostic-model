import decisionTree as dt 
import randomForest as rf
import pandas as pd  
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import math
import utils




def boxplot(collections,attributes,title=None):

    ax = plt.axes()
    ax.boxplot(collections)
    ax.set_xticklabels(attributes,rotation=90,size='small')
    plt.tight_layout()
    plt.xlabel('features')
    plt.ylabel('feature importances')
    if(title):
        plt.title(title)
        plt.savefig('boxplot_'+title+'.png')
        plt.close()

    plt.show()

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
    #ax.set_xticklabels([round(i[0],2) for i in k])
    #ax.set_yticks(range(0,50,5))
    plt.show()

    

def iter_plot_feature_contribution(clf,attributes):
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
            feature_index = np.where(attributes == feature_name)[0][0]
            f = 1
        except(IndexError):
            print('Feature %r could not be found.' % (feature_name))
            print('List of possible features:')
            for attribute in attributes:
                print(attribute)
        if(f == 1):
            plot_feature_contributions(clf.X,feature_index,fcs,attributes,class_of_interest)


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
        # print(X[:,feature_index])
        # print(feature_index)
        # print(attributes[feature_index])
        # print(max((values)))
        # print(max(X[:,feature_index]))
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
        # if (ax.get_yticks()[-1] > values[-1]+nan_index):
        #     labels[-2] = 'nan'    
        # else:
        #     labels[-1]= 'nan'

        #print(labels)
        
        #labels[-1] = 'nan'
        #print(labels)
        #ax.ylim=[values[0]-nan_index,values[-1]+nan_index]
        #ax.set_yticks(np.array(range(len(values)+2))-1)
        #x1,x2,y1,y2 = plt.axis()
        #plt.margins(x=0.1,y=0.1)
        #ax.set_ylim([values[0]-nan_index,values[1]+2*nan_index])
        #ax.set_yticklabels(['nan'],minor=True)
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
        #print('eliminating feature %r...' % original_attributes[features[i]])
        seed = np.random.randint(0,10000)
        clf = rf.RandomForest(ntrees=ntrees,mtry=mtry,missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,random_state=seed)
        clf.fit(X[:,features[:-i]],y)
        #print(features[:-i])
        nf.append(i) 
        #nf.append(i)
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
        print('seed: %r' % seed)
        clf = rf.RandomForest(ntrees=ntrees,mtry=mtry,missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,random_state=seed)
        clf.fit(X,y)

        #clf2 = rf.RandomForest(ntrees=ntrees,mtry=mtry,variable_importance=True,missing_branch=missing_branch,prob_answer=False,max_depth=max_depth,replace=replace,random_state=seed+1)
        #clf2.fit(X[:,[a[0] for a in clf.variable_importance_ if a[1] > 0]],y)

    #print('Acurácia oob RF Anatoli:')
        if round(1-clf.oob_error_,2) not in missing_branch_dict.keys():
            missing_branch_dict[round(1-clf.oob_error_,2)] = 1
        else:
            missing_branch_dict[round(1-clf.oob_error_,2)] += 1
        #missing_branch.append(1-clf.oob_error_)
#clf.forest[0].to_dot(attributes,out='foresttree0.dot')

        #clf = rf.RandomForest(ntrees=50,mtry=math.sqrt,missing_branch=False,prob_answer=True,max_depth=3,replace=False,random_state=seed)
        #clf.fit(X[:,:-3],y)

        #missing_c45.append(1-clf.oob_error_)

        #seeds.append(seed)
        #return clf
    #plt.plot(missing_c45,missing_branch,'x',color='blue')
    #plt.plot(seeds,missing_c45,'x',color='blue')
    #plt.bar(range(len(missing_branch)),missing_branch)

    k = sorted(missing_branch_dict.items(),key=lambda x: x[0])
    plt.bar(range(len([i[0] for i in k])),[i[1] for i in k])
    pos = np.arange(len(k))
    width = 1.0     # gives histogram aspect to the bar diagram
    ax = plt.axes()
    ax.set_xticks(pos +  (width / 2))
    ax.set_xticklabels([round(i[0],2) for i in k])
    ax.set_yticks(range(0,50,5))
    #plt.axis([0,1,0,70])
    #plt.axis([-0.25,4,-0.25,1])
    #plt.axis([-0.25,1,-0.25,1])
    #plt.xlabel('missing_c45')
   # plt.xlabel('teste b) acurácia C4.5')
    plt.xlabel('acurácia com missing branch = ' + str(missing_branch) )
    #plt.xlabel('information gain and gain ratio mixed')
    #plt.xlabel('gini')
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
    #print('Acurácia oob RF Anatoli:')
        #if 1-clf.oob_error_ not in missing_branch.keys():
        #    missing_branch[1-clf.oob_error_] = 1
        #else:
        #    missing_branch[1-clf.oob_error_] += 1
        missing_branch.append(1-clf.oob_error_)
        print(1-clf.oob_error_)
#clf.forest[0].to_dot(attributes,out='foresttree0.dot')

        clf2 = rf.RandomForest(ntrees=300,mtry=math.sqrt,missing_branch=False,prob_answer=False,max_depth=4,replace=False,random_state=seed)
        clf2.fit(X,y)

        missing_c45.append(1-clf2.oob_error_)
        print(1-clf2.oob_error_)

        seeds.append(seed)

    clf.forest[14].to_dot(attributes,'out.dot')
    clf2.forest[14].to_dot(attributes,'out2.dot')
    clf.forest[0].to_dot(attributes,'out3.dot')
    plt.plot(missing_c45,missing_branch,'x',color='blue')

    #plt.plot(seeds,missing_c45,'x',color='blue')
    #plt.bar(range(len(missing_branch)),missing_branch)
    #k = sorted(missing_branch.items(),key=lambda x: x[0])
    #plt.bar(range(len([i[0] for i in k])),[i[1] for i in k])
    #pos = np.arange(len(k))
    #width = 1.0     # gives histogram aspect to the bar diagram
    #ax = plt.axes()
    #ax.set_xticks(pos + (width / 2))
    #ax.set_xticklabels([i[0] for i in k])
    #plt.axis([-0.25,1.25,-0.25,1])
    #plt.axis([-0.25,4,-0.25,1])
    #plt.axis([-0.25,1,-0.25,1])
    #plt.xlabel('missing_c45')
    #plt.xlabel('information gain')
    #plt.xlabel('gain ratio')
    #plt.xlabel('information gain and gain ratio mixed')
    #plt.xlabel('gini')
    #plt.ylabel('missing_branch')

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

       # if(isnum(X[firstNotNan((X[:,feature_index])),feature_index])):
        if(isnum(Xtmp[0,feature_index])):
            values = (set(Xtmp[:, feature_index]))
            values.discard(np.nan)
            values = sorted(values)     
            for j in range(len(values) - 1):
                value = (values[j] + values[j+1])/2
                [X_true,X_false], [y_true, y_false], [t,f] = split_num(Xtmp, ytmp, feature_index, value)#threshold)
                #entrpy = gini(ytmp,[y_true,y_false])
                #entrpy = information_gain(ytmp, [y_true, y_false])
                #entrpy = gain_ratio(ytmp, [y_true,y_false])
                entrpy = (entropy(y_true)+entropy(y_false)) / 2
                # if(entrpy == 0):
                #     print(attributes[feature_index])
                if entrpy < best_entrpy:
                #if entrpy > best_entrpy:
                    #gr = gain_ratio(ytmp, [y_true,y_false])
                    #if(gr > best_entrpy):
                    #    best_entrpy = gr
                    best_entrpy = entrpy
                    best_feature = feature_index
                    #print(value)
                    best_value = [value]

        else:       
            values = (set(Xtmp[:, feature_index]))
            values.discard(np.nan)
            values = sorted(values)

            Xs,ys,d = split_categ(Xtmp, ytmp,feature_index,values)

            entrpy = sum(list(entropy(k) for k in ys)) / len(values)
            #entrpy = information_gain(ytmp,ys)
            #entrpy = gini(ytmp,ys)
            #entrpy = gain_ratio(ytmp,ys)
            # if(entrpy == 0):
            #     print(attributes[feature_index])
            if entrpy < best_entrpy:
            #if entrpy > best_entrpy:
                #gr = gain_ratio(ytmp, ys)
                #if(gr > best_entrpy):
                #    best_entrpy = gr
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
    #plt.axis([-0.25,4,-0.25,1])
    #plt.axis([-0.25,1,-0.25,1])
    plt.xlabel('entropy')
    #plt.xlabel('information gain')
    #plt.xlabel('gain ratio')
    #plt.xlabel('information gain and gain ratio mixed')
    #plt.xlabel('gini')
    plt.ylabel('pmissing')

    for i in range(len(pmissingsc)):
        if(pmissingsc[i] < 0.5):
            if(entropsc[i] < 0.2):
                print(attributes[i])
    #     if(np.isclose(pmissings[i],0,atol=0.1)):
    #         print(features[i] + ':')
    #         print(entrops[i])
    # for i in range(len(pmissingsc)):
    #     if(np.isclose(pmissingsc[i], 0.4,atol=0.01)):
    #         if(np.isclose(entropsc[i],0.9,atol=0.05)):
    #             print(featuresc[i]) 
    plt.show()
