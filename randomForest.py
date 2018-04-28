# This is an implementation of Random Forests as proposed by: 
# Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from scipy.stats import entropy

import random
from scipy.stats import mode
import utils
import decisionTree as dt
import time
from math import ceil

class RandomForest(object):

    def __init__(self, ntrees=300, mtry=np.sqrt, max_depth=None,
        min_samples_split=0.8, bootstrap=0.8, oob_error = True,replace = True,
        missing_branch=False, balance=True,prob_answer = False, 
        random_state=9):

        # number of trees
        self.ntrees = ntrees
        #  a function that determines how many features will used to build the tree
        self.mtry = mtry
        # defines the depth to which each tree should be grown  
        self.max_depth = max_depth
        # fraction of samples to used to grow each tree 
        self.bootstrap = bootstrap
        # list of tree objects
        self.forest = []
        # if oob_error is True, then the out-of-bag error of the forest will be calculated
        self.oob_error = oob_error
        # if the random choice of samples should be made with or without replacement
        self.replace = replace
        # if missing_branch = False, then missing values will be handled according to the C4.5 algorithm approach.
        # if missing_branch = True, then a branch will descend from each node of the tree for the missing values
        self.missing_branch = missing_branch 
        # if answer should be returned as a final class (prob_answer = False) or a class distribution
        self.prob = prob_answer
        # defines the seed that will be used to randomly select the features for each tree
        self.random_state = random_state
        # if balance = True, then keep the same class proportion for each randomly selected subsample 
        self.balance = balance
        # control class
        self.control_class = None

    # fits a forest for the data X with classes y
    def fit(self, X, y):

        self.forest = []
        self.X = X
        self.y = y
        n_samples = len(y)

        if(self.replace is True):
            n_sub_samples = n_samples
        else:
            n_sub_samples = round(n_samples*self.bootstrap)

        # list of out-of-bag sets     
        self.oob = []
        index_oob_samples = np.array([])
        #if(self.balance):

            # min_class = np.argmin([len(c) for c in classes])
            # print(classes[min_class])
            # print(classes[(min_class+1)%2])

            # if(self.replace is False):
            #     subsets = []
            #     nsubsets = ceil(len(classes[(min_class+1)%2])/len(classes[min_class]))
            #     print('nsubsets: %r' % nsubsets)
            #     subsets.append(np.random.choice(classes[(min_class+1)%2],len(classes[min_class]),replace=False))
            #     print('initial subset: %r' % subsets)
            #     for j in range(1,nsubsets):
            #         remaining_samples = np.array(classes[(min_class+1)%2])
            #         for k in range(j):
            #             print(-k-1)
            #             print(subsets[-k-1])
            #             print('remaining samples begining: %r ' % remaining_samples)
            #             print(np.where(np.array(subsets[-k-1]) == remaining_samples))
            #             remaining_samples = (np.delete(remaining_samples,np.where(subsets[-k-1] == remaining_samples)[0]))
            #             print('reamaing samples after delete: %r' % remaining_samples)
            #             if(j != nsubsets-1):
            #                 remaining_samples = np.random.choice(remaining_samples,len(classes[min_class]),replace=False)
            #             print('remaining: %r' % remaining_samples)
            #         subsets.append(remaining_samples)

            #         #subsets[-1] = sorted(np.append(subsets[-1],classes[min_class]))
            #     for j in range(nsubsets):
            #         subsets[j] = np.append(subsets[j],classes[min_class])
            # print(subsets)
            # exit()
        classes = []
    # separate samples according to their classes
        for c in set(y):
            classes.append([j for j in range(len(y)) if y[j] == c])
        #print('Creating trees...')
        # for each tree
        for i in range(self.ntrees):

            np.random.seed(self.random_state+i)
            #same proportion of instances from each class
            if(self.balance):
                # select indexes of sub samples considering class balance    
                index_sub_samples = sorted([k for l in [np.random.choice(a, round(n_sub_samples*(len(a)/n_samples)),
                    replace=self.replace) for a in classes] for k in l])
                # if(self.replace is False):
                #     index_sub_samples = sorted(np.random.choice(subsets[(i%nsubsets)],round(n_samples*self.bootstrap),replace=False))
                #     index_oob_samples = np.delete(np.array(subsets[(i%nsubsets)]),index_sub_samples)
                #     print(index_sub_samples)
                #     print(index_oob_samples)
                #     exit()
                # else:
                #     index_sub_samples = np.append(sorted(np.random.choice(classes[min_class],round(n_samples/2),replace=True)),
                #         np.random.choice(classes[(min_class+1)%2],round(n_samples/2),replace=True))
                index_oob_samples = np.delete(np.array(range(n_samples)),index_sub_samples) 

            else:
                index_sub_samples = sorted(np.random.choice(range(n_samples),n_sub_samples,replace=self.replace))
            
                index_oob_samples = np.delete(np.array(range(n_samples)),index_sub_samples)

            self.oob.append(index_oob_samples)

            X_subset = X[index_sub_samples]
            y_subset = y[index_sub_samples]
            tree = dt.DecisionTreeClassifier(max_depth=self.max_depth,mtry=self.mtry,
                missing_branch=self.missing_branch, random_state=self.random_state+i)
            #tree.index = i
            tree.fit(X_subset,y_subset)
            
            self.forest.append(tree)
        # if out-of-bag error should be calculated
        if self.oob_error is True:

            #print('Calculating oob error...')
            # set of all intances that belong to at least one out-of-bag set
            oob_set = set([j for i in self.oob for j in i])

            ypred = {}

            # for each tree 
            for t in range(self.ntrees):
                # error counting
                err = 0
                # for each instance at the tree out-of-bag set
                for i in self.oob[t]:
                    if i not in ypred:
                        ypred[i] = {}

                    # predict the class (or the class distribution) for instance X[i]
                    tmp = self.forest[t].predict(X[i].reshape(1,-1),self.prob)[0]
                    # in case of class prediction (not distribution)
                    if(self.prob is False):
                        # add a vote for class "tmp"
                        if tmp not in ypred[i]:
                            ypred[i][tmp] = 1
                        else:
                            ypred[i][tmp] += 1
                        # wrong prediction
                        if (tmp != y[i]):
                            err += 1

                    # in case of class distribution
                    else:
                        for k in tmp.keys():
                            if k not in ypred[i].keys():
                                ypred[i][k] = tmp[k]
                            else:
                                ypred[i][k] += tmp[k]
                            
                            yp = max(ypred[i].keys(), key= (lambda k: ypred[i][k]))
                            if(yp != y[i]):
                                err += 1
                    
            err = 0
            dif = 0

            # calculate the out-of-bag error
            for i in ypred.keys():

                #in case of a tie, assign the class that appears the most in the training set
                if(len(ypred[i]) > 1 and list(ypred[i].values())[0] == list(ypred[i].values())[1]):
                    yp = mode(y)[0][0]
                else:
                    yp = max(ypred[i].keys(), key= (lambda k: ypred[i][k]))
                if(yp != y[i]):
                    err += 1

                # if(len(ypred[i].keys()) > 1):
                #     dif += abs(ypred[i][list(ypred[i].keys())[0]] - ypred[i][list(ypred[i].keys())[1]])/(ypred[i][list(ypred[i].keys())[0]] + ypred[i][list(ypred[i].keys())[1]])
                # else:
                #     dif += 1

            #dif = dif / len(ypred.keys())
            #self.dif = dif
        
        self.oob_error_ = err / len(set(oob_set))

    def predict(self, X):

        if(len(np.array(X).shape)) == 1: 
            X = [X]
            n_samples = 1
        else:
            n_samples = X.shape[0]

        n_trees = len(self.forest)
        predictions = np.empty(n_samples,dtype=object)
        #y = [{}] * n_samples
        for i in range(n_samples):
            ypreds = []
            for j in range(n_trees):
                ypreds.append(self.forest[j].predict(X[i],self.prob))
            if(self.prob is False):
                predictions[i] = mode(ypreds)[0][0]
            else:
                predictions[i] = {}
                for ydict in ypreds:
                    for c in ydict.keys():
                        if c not in predictions[i].keys():
                            predictions[i][c] = ydict[c]
                        else:
                            predictions[i][c] += ydict[c]

        return predictions
        


    def score(self, X, y):

        y_predict = self.predict(X)
        if(self.prob is True):
            y_predict = max(y_predict.keys(), key=lambda k: y_predict[k])
        n_samples = len(y)
        if(isinstance(y,str)):
            y = [y]
            n_samples = 1
    
        correct = 0
        for i in range(n_samples):
            if y_predict[i] == y[i]:
                correct = correct + 1
        accuracy = correct/n_samples
        return accuracy

    # this function implements the method proposed in:
    # Palczewska, A., Palczewski, J., Robinson, R. M., & Neagu, D. (2013). 
    # Interpreting random forest models using a feature contribution method. 
    # In 2013 IEEE 14th International Conference on Information Reuse and Integration (pp. 112–119). 
    # Retrieved from http://eprints.whiterose.ac.uk/79159/1/feature_contribution_camera_ready.pdf
    def feature_contribution(self):
        print('calculating feature contribution')

        fcs = []
        C = set(self.y)

        for i in range(self.X.shape[0]):

            FC = {}
            c = 0
            for k in C:
                t_index = 0
                # if(i_index == 9):
                #     import pdb
                #     pdb.set_trace()
                
                for t in self.forest:
                    if(i in self.oob[t_index]):
                        #print(oob[t_index])
                        t_index+=1
                        continue

                    t_index +=1
                    child_list = [[1,t.root]]   
                    

                    while len(child_list) > 0:
                        w, parent = child_list.pop(0)
                        
                        while parent.is_class is False:
                            f = parent.feature_index

                            #print(i[f])
                            #print(parent.values)
                            if(f not in FC.keys()):
                                FC[f] =  {c:0 for c in C}

                            if(utils.isnan(self.X[i][f])):
                                if(parent.branch_nan is None):
                                    sp = sum(parent.distr.values())
                                    for c in parent.branches:
                                        child_list.append([round(w*(sum(c.distr.values()))/sp,2),c])
                                    w,child = child_list.pop(0)
                                else:
                                    child = parent.branch_nan
                            else:
                                if(len(parent.values) == 1):
                                    if self.X[i][f] <= parent.values[0]:
                                        child = parent.branches[0]
                                    else:
                                        child = parent.branches[1]
                                else:
                                    child = parent.branches[parent.values.index(str(self.X[i][f]))]


                            sc = sum(child.distr.values())
                            sp = sum(parent.distr.values())
                            
                            FC[f][k] = FC[f][k] + w*(child.distr[k]/sc - parent.distr[k]/sp)

                            parent = child

            for element in FC:
                for el in FC[element]:
                    FC[element][el] = FC[element][el] / self.ntrees

            fcs.append(FC)

        return fcs

    # variable importance calculation for Random Forests.
    # --- when vitype='err' and vimissing=False, then calculation is made as proposed in:
    #       Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
    # --- when vitype='err' and vimissing=True, then calculation is made as proposed in:
    #       Hapfelmeier, A., & Ulm, K. (2014). Variable selection by Random Forests using data with missing values.
    #         Computational Statistics and Data Analysis, 80, 129–139. https://doi.org/10.1016/j.csda.2014.06.017
    # --- when vitype='auc' and vimissing=False, then calculation is made as proprosed in:
    #       Janitza, S., Strobl, C., & Boulesteix, A.-L. (2012). 
    #       An AUC-based Permutation Variable Importance Measure for Random Forests. 
    #           Retrieved from http://www.stat.uni-muenchen.de
    # --- when vitype='auc' and vimissing=True, then calculation is made by joining the two methods above
    def variable_importance(self,vitype='err',vimissing=True,y=None):

        if(y is None):
            y = self.y
        if(vitype == 'auc'):
            ntreesc = 0
        else:

            ntreesc = self.ntrees

        variable_importance = {attribute: 0 for attribute in range(self.X.shape[1])}
        for t in range(self.ntrees):

                #print(self.forest[t].attributes_in_tree)
            for m in self.forest[t].feature_indices:


                X_permuted = self.X.copy() 
                if(vimissing is False):
                    np.random.shuffle(X_permuted[:,m])
                    sa = None
                else:
                    sa = m


                # import pdb
                # pdb.set_trace()
                if(vitype == 'auc'):
                    if(len(set(y[self.oob[t]])) > 1):
                        ntreesc += 1
                        auc_before = self.forest[t].auc(self.X[self.oob[t]],y[self.oob[t]],shuffle_attribute=None,control_class=self.control_class)
                        auc = self.forest[t].auc(X_permuted[self.oob[t]],y[self.oob[t]],shuffle_attribute=sa,control_class=self.control_class)
                        variable_importance[m] += auc_before - auc

                else:    
                    err = 1-self.forest[t].score(self.X[self.oob[t]],y[self.oob[t]],shuffle_attribute=None)
                    err_permuted = 1 - self.forest[t].score(X_permuted[self.oob[t]], y[self.oob[t]],shuffle_attribute=sa)
                    variable_importance[m] += (err_permuted - err)

            
        return {a:b/ntreesc for a,b in variable_importance.items()} 

