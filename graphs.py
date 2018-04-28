# this script was used to generate the graphs for my master's dissertation, with excpetion of the pie plots (fig 4.1 and fig 4.4),
# which was generated through this website: https://nces.ed.gov/nceskids/createagraph/ 

import matplotlib.pyplot as plt
import read
import utils
import pandas as pd
import numpy as np 
from collections import Counter
import re

def plot_side_distribution():
    data_path = '~/Faculdade/Mestrado/Projeto/scripts/Working Scripts/'
    data_path = data_path + 'EXPERIMENT_DOWNLOAD/Group_patients-with-brachial-plexus-injury/Per_questionnaire_data/'
    data_path = data_path + 'Q44071_unified-admission-assessment/Responses_Q44071.csv'

    data = pd.read_csv(data_path, header=0, delimiter=",",
            na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)
    side_distribution = sorted(Counter(data['opcLdLesao']).items())
    side_labels = {'D': 'Direito', 'E': 'Esquerdo', 'DE': 'Ambos'}
    width = 0.6
    plt.bar(range(len(side_distribution)), [a[1] for a in side_distribution],width=width)
    plt.xticks(np.array(range(len(side_distribution)))+width/2,[side_labels[a[0]] for a in side_distribution])
    plt.yticks(range(0,max(side_distribution,key=lambda x: x[1])[1]+5,5))

    plt.show()

def plot_event():

    data_path = '~/Faculdade/Mestrado/Projeto/scripts/Working Scripts/'
    data_path = data_path + 'EXPERIMENT_DOWNLOAD/Group_patients-with-brachial-plexus-injury/Per_questionnaire_data/'
    data_path = data_path + 'Q44071_unified-admission-assessment/Responses_Q44071.csv'

    data = pd.read_csv(data_path, header=0, delimiter=",",
            na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)

    events_right = data.filter(like='lisTpTraumaD')
    events_left = data.filter(like='lisTpTraumaE')
    events_description = {'lisTpTrauma[arma]': 'Arma de fogo', 'lisTpTrauma[moto]':'Acidente motociclístico',
    'lisTpTrauma[auto]': 'Acidente automobilístico', 'lisTpTrauma[atropelamento]':'Atropelamento',
    'lisTpTrauma[cirurgia]':'Cirurgia','lisTpTrauma[corte]':'Objeto cortante',
    'lisTpTrauma[ocupacao]':'Acidente ocupacional', 'lisTpTrauma[other]':'Outros'}
    event_names = {}
    for c in events_right.columns:
        event_names[re.sub('D','',c)] = [re.sub('D','E',c), c]


    index = 0
    x = np.arange(len(event_names.keys()))
    
    width = 0.4
    fig = plt.figure()
    ax = fig.add_subplot(111)
    i = 0
    events_in_plot = []
    for event in sorted(event_names.keys()): 
        yleft = sum([a[1] for a in Counter(events_left[event_names[event][0]]).items() if not utils.isnan(a[0])])
        yright = sum([a[1] for a in Counter(events_right[event_names[event][1]]).items() if not utils.isnan(a[0])])
        if(yleft != 0 or yright != 0):
            l = plt.bar(i,yleft,width,color='blue')
            r = plt.bar(i+width,yright,width,color='red')
            events_in_plot.append(event)
        else:
            continue
        i+=1

    #print([[Counter(events_left[event_names[event][0]]) for event in y] for event in y])
    #exit()
    # y = sorted(event_names.keys())
    # left = plt.bar(x, [Counter(events_left[event_names[event][0]])['Y'] for event in y], width,color='blue')
    # right = plt.bar(x+width, [Counter(events_right[event_names[event][1]])['Y'] for event in y], width,color='red')
    ax.set_xticks(np.arange(i)+width)
    ax.set_xticklabels([events_description[e] for e in events_in_plot],rotation=90)

    ax.legend((l,r), ('Esquerdo','Direito'))
    plt.tight_layout()
    #plt.width = width
    plt.show()

def calculate_followup_return_period():

    data_path = '~/Faculdade/Mestrado/Projeto/scripts/Working Scripts/'
    data_path = data_path + 'EXPERIMENT_DOWNLOAD/Group_patients-with-brachial-plexus-injury/Per_questionnaire_data/'
    #data_path = data_path + 'Q61802_unified-surgical-evaluation/Responses_Q61802.csv'
    data_path = data_path + 'Q92510_unified-follow-up-assessment/Responses_Q92510.csv'

    data = pd.read_csv(data_path, header=0, delimiter=",",
            na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)

    patients_considered = {}
    for i,row in data.iterrows():
        if(row['participant code']) not in patients_considered:
            patients_considered[row['participant code']] = row['formTempoAval']
        else:
            if(row['formTempoAval'] < patients_considered[row['participant code']]):
                patients_considered[row['participant code']] = row['formTempoAval']

    for k in patients_considered.keys():
        patients_considered[k] = int(patients_considered[k]/30)
    print(sorted(Counter(patients_considered.values()).items(),key=lambda x: x[0]))

def calculate_mean_followup_return_period():
    data_path = '~/Faculdade/Mestrado/Projeto/scripts/Working Scripts/'
    data_path = data_path + 'EXPERIMENT_DOWNLOAD/Group_patients-with-brachial-plexus-injury/Per_questionnaire_data/'
    #data_path = data_path + 'Q61802_unified-surgical-evaluation/Responses_Q61802.csv'
    data_path = data_path + 'Q92510_unified-follow-up-assessment/Responses_Q92510.csv'

    data = pd.read_csv(data_path, header=0, delimiter=",",
            na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)
 
    s = []
    cont = 0
    patients_considered = {}
    for i,row in data.iterrows():
        if(row['participant code']) not in patients_considered:
            patients_considered[row['participant code']] = [row['formTempoAval']]
        else:
            patients_considered[row['participant code']] = sorted(patients_considered[row['participant code']]+[row['formTempoAval']])
            s.append(patients_considered[row['participant code']][-1] - patients_considered[row['participant code']][-2])
    cont = 0
    m = 0
    for p in patients_considered.keys():
        if(len(patients_considered[p]) > 1):
            cont+=1    
        m += len(patients_considered[p])
    print(np.mean(s))
    print(cont)
    print(m/len(patients_considered))
    #print(sorted(Counter(patients_considered.values()).items(),key=lambda x: x[0]))

def plot_followup_pain():
    data_path = '~/Faculdade/Mestrado/Projeto/scripts/Working Scripts/'
    data_path = data_path + 'EXPERIMENT_DOWNLOAD/Group_patients-with-brachial-plexus-injury/Per_questionnaire_data/'
    #data_path = data_path + 'Q61802_unified-surgical-evaluation/Responses_Q61802.csv'
    data_path = data_path + 'Q92510_unified-follow-up-assessment/Responses_Q92510.csv'
    #data_path = data_path + 'Q44071_unified-admission-assessment/Responses_Q44071.csv'

    data = pd.read_csv(data_path, header=0, delimiter=",",
            na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)

    # admission_data = pd.read_csv('~/Faculdade/Mestrado/Projeto/scripts/Working Scripts/Dor.csv', header=0, delimiter=",",
    #         na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)

    outcome = 'snDorPos'#'opcForcaD[FlexCotovelo]' #'snDorPos'
    #outcome_left = 'snDorPos'#'opcForcaE[FlexCotovelo]'#'snDorPos'
    #print(len(([int(a/30) for a in data['formTempoAval']])))
    patients_considered = {}
    patient_outcomes = {}
    #return_periods = []
    for i,row in data.iterrows():
        if(row['participant code']) not in patients_considered:
            patients_considered[row['participant code']] = row['formTempoAval']
            patient_outcomes[row['participant code']] = row[outcome]

            #return_periods.append(row['formTempoAval'])
        else:
            if(row['formTempoAval'] > patients_considered[row['participant code']]):
                if(row[outcome] != 'NINA' and not utils.isnan(row[outcome])):
                    patient_outcomes[row['participant code']] = row[outcome]
                    patients_considered[row['participant code']] = row['formTempoAval']

            else:
                if(utils.isnan(patient_outcomes[row['participant code']])):
                    if(row[outcome] != 'NINA' and not utils.isnan(row[outcome])):
                        patient_outcomes[row['participant code']] = row[outcome]
                        patients_considered[row['participant code']] = row['formTempoAval']


                #print(row['participant code'])
    #import pdb
    #pdb.set_trace()
    #labels = {'S':'Sim','N':'Não',np.nan:'Não informado'}
    for k in patients_considered.keys():
        patients_considered[k] = int(patients_considered[k]/30)

    xlabels = ['N','S',np.nan]
    labels = {'S':'Sim','N':'Não',np.nan:'Não informado'}
    y = [ (Counter(patient_outcomes.values())[x]) for x in xlabels]
    width=0.8
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar(range(len(xlabels)),y,width=width)
    ax.set_xticks(np.arange(len(xlabels))+width/2)
    ax.set_xticklabels([labels[l] for l in xlabels])
    plt.xlabel('Sente dor após a lesão?')
    plt.show()

def plot_followup_movements():

    data_path = '~/Faculdade/Mestrado/Projeto/scripts/Working Scripts/'
    data_path = data_path + 'EXPERIMENT_DOWNLOAD/Group_patients-with-brachial-plexus-injury/Per_questionnaire_data/'
    #data_path = data_path + 'Q61802_unified-surgical-evaluation/Responses_Q61802.csv'
    data_path = data_path + 'Q92510_unified-follow-up-assessment/Responses_Q92510.csv'

    data = pd.read_csv(data_path, header=0, delimiter=",",
            na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)

    admission_data = pd.read_csv('~/Faculdade/Mestrado/Projeto/scripts/Working Scripts/FlexCotoveloNew.csv', header=0, delimiter=",",
            na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)

    outcome_right = 'opcForcaD[FlexCotovelo]' #'snDorPos'
    outcome_left = 'opcForcaE[FlexCotovelo]'#'snDorPos'
    #print(len(([int(a/30) for a in data['formTempoAval']])))
    patients_considered = {}
    patient_outcomes = {}
    #return_periods = []
    for i,row in data.iterrows():
        if(row['participant code']) not in patients_considered:
            patients_considered[row['participant code']] = row['formTempoAval']

            if(np.all(admission_data['Q44071_opcLdLesao'][admission_data['participant code'] == row['participant code']] == 'D')):
                patient_outcomes[row['participant code']] = row[outcome_right]
            elif(np.all(admission_data['Q44071_opcLdLesao'][admission_data['participant code'] == row['participant code']] == 'E')):
                patient_outcomes[row['participant code']] = row[outcome_left]
            else:
                'Preprocessing of side {0} not implemented'.format(admission_data['Q44071_opcLdLesao'][admission_data['participant code']])

            #return_periods.append(row['formTempoAval'])
        else:
            if(row['formTempoAval'] > patients_considered[row['participant code']]):
                if(np.all(admission_data['Q44071_opcLdLesao'][admission_data['participant code'] == row['participant code']] == 'D')):
                    if(row[outcome_right] != 'NINA' and not utils.isnan(row[outcome_right])):
                        patient_outcomes[row['participant code']] = row[outcome_right]
                        patients_considered[row['participant code']] = row['formTempoAval']
                elif(np.all(admission_data['Q44071_opcLdLesao'][admission_data['participant code'] == row['participant code']] == 'E')):
                    if(row[outcome_left] != 'NINA' and not utils.isnan(row[outcome_left])):
                        patient_outcomes[row['participant code']] = row[outcome_left]
                        patients_considered[row['participant code']] = row['formTempoAval']

            else:
                if(utils.isnan(patient_outcomes[row['participant code']])):
                    if(np.all(admission_data['Q44071_opcLdLesao'][admission_data['participant code'] == row['participant code']] == 'D')):
                        if(row[outcome_right] != 'NINA' and not utils.isnan(row[outcome_right])):
                            patient_outcomes[row['participant code']] = row[outcome_right]
                            patients_considered[row['participant code']] = row['formTempoAval']
                    elif(np.all(admission_data['Q44071_opcLdLesao'][admission_data['participant code'] == row['participant code']] == 'E')):
                        if(row[outcome_left] != 'NINA' and not utils.isnan(row[outcome_left])):
                            patient_outcomes[row['participant code']] = row[outcome_left]
                            patients_considered[row['participant code']] = row['formTempoAval']

                #print(row['participant code'])
    #import pdb
    #pdb.set_trace()
    #labels = {'S':'Sim','N':'Não',np.nan:'Não informado'}
    for k in patients_considered.keys():
        patients_considered[k] = int(patients_considered[k]/30)

    xlabels = list(np.arange(6))+[np.nan]#['N','S',np.nan]
    label = lambda x: 'Não informado' if utils.isnan(x) else x 
    #labels = {'S':'Sim','N':'Não',np.nan:'Não informado'}
    y = [0]*7
    for value in patient_outcomes.values():
        if(utils.isnan(value)):
            y[-1] += 1
        else:
            y[int(value)] += 1
    width=0.8
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar(range(len(xlabels)),y,width=width)
    ax.set_xticks(np.arange(len(xlabels))+width/2)
    ax.set_yticks(range(0,30,5))
    ax.set_xticklabels([label(l) for l in xlabels])
    print(Counter(patient_outcomes.values()))
    plt.xlabel('Força muscular avaliada sobre flexão do cotovelo')
    plt.show()

    #print(sorted(Counter([int(a/30) for a in patients_considered.values()]).items()))

#plot_side_distribution()
#plot_event()
#calculate_followup_return_period()
#plot_followup_pain()
plot_followup_movements()
#calculate_mean_followup_return_period()