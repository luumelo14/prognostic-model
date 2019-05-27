import randomForest as rf
import pickle
import read 
from os import listdir
from os.path import isfile, join

def models_transform_to_JSON(class_name,file_name=None):
    import json
    F = {}
    #onlyfiles = [f for f in listdir('../saved_models') if isfile(join('../saved_models', f))]
    max_accuracy = 0.0
    scnd_max_accuracy = 0.0
    scnd_max_threshold = -2.0
    max_threshold = -2.0
    for i in range(200): 
        special = False
        try:
            filename = '../saved_models/prognostic_model_Q92510_'+class_name+str(i)+'.pickle'
            with open(filename,'rb') as handle:
                clf = pickle.load(handle)
                F[filename[16:]] = {'threshold': format(clf.threshold,'.6f'), 
                'accuracy':round(1-clf.oob_error_,4), 'features' : list(clf.X.columns),
                'datapath': filename, 'special': False}
                
        except(FileNotFoundError):
            break
            
    if(file_name is None):
        file_name = class_name +'_prognostic_models.json'
    f = open(file_name,'w')
    jsonfile = json.dumps(F,ensure_ascii=False)
    f.write(jsonfile)



def transform_to_JSON(clf,fcs,out='FeatureContributions.json',diffsur=True,X=None,addline=None):
    import json 
    import pandas as pd
    import utils 
    if(X is None):
        if(not isinstance(clf.X,pd.DataFrame)):
            X = pd.DataFrame(clf.X,columns=clf.attributes)
        else:
            X = clf.X
    
    #data = read.readData(data_path = data_path, class_name = class_name)
    #newcolumns = np.append(X.columns,['Q44071_snCplexoAt',class_name])
    #newX = pd.merge(data,X,how='inner',on='Q44071_participant_code')[newcolumns]
    F = {}

    for i in range(len(fcs)):
        if(diffsur):
            for feature_index in fcs[i].keys():    
                if(feature_index not in F):
                    F[feature_index] = {'name' : clf.attributes[feature_index], 
                    'ycategs' : sorted(list([a for a in set(X[X.columns[feature_index]]) if not utils.isnan(a)])) + ['nan'],
                    'redopoints' : [], 'redxpoints' : [], 'blueopoints' : [], 'bluexpoints' : [] }
                if(clf.compX['Q44071_snCplexoAt'][i] == 'S'):
                    if(clf.y[i] == 'INSATISFATORIO'):
                        if(not utils.isnan(X[X.columns[feature_index]][i])):
                            F[feature_index]['redopoints'].append([round(fcs[i][feature_index],5),F[feature_index]['ycategs'].index(X[X.columns[feature_index]][i])])
                        else:
                            F[feature_index]['redopoints'].append([round(fcs[i][feature_index],5),len(F[feature_index]['ycategs'])-1])
                    else:
                        if(not utils.isnan(X[X.columns[feature_index]][i])):
                            F[feature_index]['blueopoints'].append([round(fcs[i][feature_index],5),F[feature_index]['ycategs'].index(X[X.columns[feature_index]][i])])
                        else:
                            F[feature_index]['blueopoints'].append([round(fcs[i][feature_index],5),len(F[feature_index]['ycategs'])-1])

                else:
                    if(y[i] == 'INSATISFATORIO'):
                        if(not utils.isnan(X[X.columns[feature_index]][i])):
                            F[feature_index]['redxpoints'].append([round(fcs[i][feature_index],5),F[feature_index]['ycategs'].index(X[X.columns[feature_index]][i])])
                        else:
                            F[feature_index]['redxpoints'].append([round(fcs[i][feature_index],5),len(F[feature_index]['ycategs'])-1])

                    else:
                        if(not utils.isnan(X[X.columns[feature_index]][i])):
                            F[feature_index]['bluexpoints'].append([round(fcs[i][feature_index],5),F[feature_index]['ycategs'].index(X[X.columns[feature_index]][i])])
                        else: 
                            F[feature_index]['bluexpoints'].append([round(fcs[i][feature_index],5),len(F[feature_index]['ycategs'])-1])
        else:
            for feature_index in fcs[i].keys():
                if(feature_index not in F.keys()):
                    if(isinstance(X,pd.DataFrame)):
                        F[feature_index] = {'name': clf.attributes[feature_index], 'value': X.values[i][feature_index] if not utils.isnan(X.values[i][feature_index]) else 'nan',
                        'contribution': 0 }
                    else:
                        F[feature_index] = {'name': clf.attributes[feature_index], 'value': X[i][feature_index] if not utils.isnan(X[i][feature_index]) else 'nan',
                        'contribution': 0 }
                
                F[feature_index]['contribution'] = fcs[i][feature_index]

    file = open(out,'w')
    if(addline is not None):
        F['classification'] = addline
    jsonfile = json.dumps(F,ensure_ascii=False)
    file.write(jsonfile)


def classify(pacient_filename,model_filename,class_name):
    if(model_filename[-7:] != '.pickle' ):
        model_filename = model_filename + '.pickle'
    try:
        with open(model_filename, 'rb') as handle:
            clf = pickle.load(handle)   
    except(FileNotFoundError):
        print('Could not find file %r.\n' % model_filename)
        exit()

    data = read.readData(data_path = pacient_filename, class_name = class_name)
    X = data[data.columns[:-1]]
    classdict = (clf.predict(X,prob=True))[0]
    outcome = max(classdict,key=classdict.get)
    print(f"Outcome {outcome} with {classdict[outcome]/sum(classdict.values())*100}% of probabily.")
    transform_to_JSON(clf,clf.feature_contribution(X),out='classification_'+class_name+'.json',diffsur=False,addline=classdict)


# with open('prognostic_model_'+ class_name[7:] + '.pickle', 'rb') as handle:
#     clf = pickle.load(handle)
# transform_to_JSON(clf,clf.feature_contribution())
# continue

classify('out.csv','prognostic_model_Q92510_snDorPos0.pickle','Q92510_snDorPos')
# classify('../Patient3.csv','prognostic_model_opcForca[AbdOmbro].pickle','Q92510_opcForca[AbdOmbro]')
# classify('../Patient3.csv','prognostic_model_opcForca[FlexCotovelo].pickle','Q92510_opcForca[FlexCotovelo]')
# classify('../Patient3.csv','prognostic_model_opcForca[RotEOmbro].pickle','Q92510_opcForca[RotEOmbro]')
# models_transform_to_JSON('snDorPos')
# models_transform_to_JSON('opcForca[AbdOmbro]')
# models_transform_to_JSON('opcForca[FlexCotovelo]')
# models_transform_to_JSON('opcForca[RotEOmbro]')