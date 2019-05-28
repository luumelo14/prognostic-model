import randomForest as rf
import read 
import pickle
from os import listdir
from os.path import isfile, join
from featureContribution import transform_to_JSON

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




classify('data/out.csv','prognostic_model_Q92510_snDorPos0.pickle','Q92510_snDorPos')
# classify('../Patient3.csv','prognostic_model_opcForca[AbdOmbro].pickle','Q92510_opcForca[AbdOmbro]')
# classify('../Patient3.csv','prognostic_model_opcForca[FlexCotovelo].pickle','Q92510_opcForca[FlexCotovelo]')
# classify('../Patient3.csv','prognostic_model_opcForca[RotEOmbro].pickle','Q92510_opcForca[RotEOmbro]')
# models_transform_to_JSON('snDorPos')
# models_transform_to_JSON('opcForca[AbdOmbro]')
# models_transform_to_JSON('opcForca[FlexCotovelo]')
# models_transform_to_JSON('opcForca[RotEOmbro]')