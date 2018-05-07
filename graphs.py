# this script was used to generate the graphs for my master's dissertation, with excpetion of the pie plots (fig 4.1 and fig 4.4),
# which was generated through this website: https://nces.ed.gov/nceskids/createagraph/ 

import matplotlib.pyplot as plt
import read
import utils
import pandas as pd
import numpy as np 
from collections import Counter
import re

colors = ['forestgreen','gold','blue','firebrick','darkviolet','orangered','darkturquoise']
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

def plot_followup_return_period():

    data_path = '~/Faculdade/Mestrado/Projeto/scripts/Working Scripts/'
    data_path = data_path + 'EXPERIMENT_DOWNLOAD/Group_patients-with-brachial-plexus-injury/Per_questionnaire_data/'
    #data_path = data_path + 'Q61802_unified-surgical-evaluation/Responses_Q61802.csv'
    data_path = data_path + 'Q92510_unified-follow-up-assessment/Responses_Q92510.csv'

    data = pd.read_csv(data_path, header=0, delimiter=",",
            na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)

    more_than_one_return = {}
    patients_considered = {}
    for i,row in data.iterrows():
        if(row['participant code']) not in patients_considered:
            patients_considered[row['participant code']] = row['formTempoAval']
            more_than_one_return[row['participant code']] = False
        else:
            if(row['formTempoAval'] > patients_considered[row['participant code']]):
                patients_considered[row['participant code']] = row['formTempoAval']
            more_than_one_return[row['participant code']] = True

    patients_to_plot_m1r = {}
    patients_to_plot_1r = {}
    for k in patients_considered.keys():
        if(more_than_one_return[k]):
            patients_to_plot_m1r[k] = int(patients_considered[k]/30)
        else:
            patients_to_plot_1r[k] = int(patients_considered[k]/30)
    values = (sorted(Counter(patients_to_plot_m1r.values()).items(),key=lambda x: x[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar([a[0] for a in values],[a[1] for a in values],width=0.8)
    ax.set_xticks(np.arange(0,100,10)+0.4)
    ax.set_xticklabels(np.arange(0,100,10))
    ax.set_yticks(np.arange(7))
    plt.xlabel('Período entre a lesão e o último retorno ao INDC em meses')
    plt.ylabel('Número de pacientes')
    plt.show()

    values = (sorted(Counter(patients_to_plot_1r.values()).items(),key=lambda x: x[0]))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.bar([a[0] for a in values],[a[1] for a in values],width=0.8)
    ax.set_xticks(np.arange(0,100,10)+0.4)
    ax.set_xticklabels(np.arange(0,100,10))
    plt.xlabel('Período entre a lesão e o primeiro retorno ao INDC em meses')
    plt.ylabel('Número de pacientes')
    plt.show()
    # time_groups = [0,0,0,0,0,0,0]
    # for time,frequency in values:
    #     if(time <= 6):
    #         time_groups[6] += frequency
    #     elif(time <= 12):
    #         time_groups[5] += frequency
    #     elif(time <= 18):
    #         time_groups[4] += frequency
    #     elif(time <= 24):
    #         time_groups[3] += frequency
    #     elif(time <= 30):
    #         time_groups[2] += frequency
    #     elif(time <= 36):
    #         time_groups[1] += frequency
    #     else:
    #         time_groups[0] += frequency 
    # labels = ['37 meses ou mais','31 a 36 meses','25 a 30 meses','19 a 24 meses','13 a 18 meses','7 a 12 meses','0 a 6 meses']
    # plt.pie(time_groups,colors=colors[::-1],
    # labels=labels,autopct='%1.1f%%',startangle=90) #,'gold'],)
    # plt.axis('equal')
    # plt.show()

def plot_surgery_period():

    data_path = '~/Faculdade/Mestrado/Projeto/scripts/Working Scripts/'
    data_path = data_path + 'EXPERIMENT_DOWNLOAD/Group_patients-with-brachial-plexus-injury/Per_questionnaire_data/'
    data_path = data_path + 'Q61802_unified-surgical-evaluation/Responses_Q61802.csv'
    #data_path = data_path + 'Q92510_unified-follow-up-assessment/Responses_Q92510.csv'

    data = pd.read_csv(data_path, header=0, delimiter=",",
            na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)

    patients_considered = {}
    for i,row in data.iterrows():
        if(row['participant code']) not in patients_considered:
            patients_considered[row['participant code']] = row['formTempoCirurg']
        else:
            if(row['formTempoCirurg'] < patients_considered[row['participant code']]):
                patients_considered[row['participant code']] = row['formTempoCirurg']

    for k in patients_considered.keys():
        patients_considered[k] = int(patients_considered[k]/30)
    values = (sorted(Counter(patients_considered.values()).items(),key=lambda x: x[0]))
    time_groups = [0,0,0]
    for time,frequency in values:
        if(time <= 6):
            time_groups[0] += frequency
        elif(time <= 12):
            time_groups[1] += frequency
        else:
            time_groups[2] += frequency
    labels = ['0 a 6 meses', '7 a 12 meses', '13 meses ou mais']
    plt.pie(time_groups,colors=colors,labels=labels,autopct='%1.1f%%') #,'gold'],)
    plt.axis('equal')
    plt.show()

def plot_ages():
    data_path = '~/Faculdade/Mestrado/Projeto/scripts/Working Scripts/'
    data_path = data_path + 'EXPERIMENT_DOWNLOAD/Group_patients-with-brachial-plexus-injury/Per_questionnaire_data/'
    #data_path = data_path + 'Q61802_unified-surgical-evaluation/Responses_Q61802.csv'
    #data_path = data_path + 'Q92510_unified-follow-up-assessment/Responses_Q92510.csv'
    data_path = data_path + 'Q44071_unified-admission-assessment/Responses_Q44071.csv'

    data = pd.read_csv(data_path, header=0, delimiter=",",
            na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)

    age_groups = [0,0,0,0]
    for age in data['formIdadeLesao']:
        if(age <= 30):
            age_groups[0] += 1
        elif(age <= 45):
            age_groups[1] += 1
        elif(age <= 60):
            age_groups[2] += 1
        else:
            age_groups[3] += 1
    labels = ['15 a 30 anos', '31 a 45 anos', '45 a 60 anos', '61 anos ou mais']
    fig,ax = plt.subplots()
    ax.pie(age_groups,colors=colors,labels=labels,autopct='%1.1f%%',
    startangle=90,labeldistance=1.05,pctdistance=0.5) #,'gold'],)
    centre_circle = plt.Circle((0,0),0.65,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

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

def plot_followup_improvements():

    data_path = '~/Faculdade/Mestrado/Projeto/scripts/Working Scripts/'
    data_path = data_path + 'EXPERIMENT_DOWNLOAD/Group_patients-with-brachial-plexus-injury/Per_questionnaire_data/'
    #data_path = data_path + 'Q61802_unified-surgical-evaluation/Responses_Q61802.csv'
    data_path = data_path + 'Q92510_unified-follow-up-assessment/Responses_Q92510.csv'

    followup_data = pd.read_csv(data_path, header=0, delimiter=",",
            na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)

    data_path = '~/Faculdade/Mestrado/Projeto/scripts/Working Scripts/'
    data_path = data_path + 'EXPERIMENT_DOWNLOAD/Group_patients-with-brachial-plexus-injury/Per_questionnaire_data/'
    data_path = data_path + 'Q44071_unified-admission-assessment/Responses_Q44071.csv'

    admission_data = pd.read_csv(data_path, header=0, delimiter=",",
            na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)
    print(admission_data.shape)
    print(followup_data.shape)
    return_value = {}
    return_period = {}
    surgery_patients = []
    injury_side_column = admission_data.filter(like='opcLdLesao').columns[0]
    merged_data = admission_data.merge(followup_data,how='inner',on='participant code',suffixes=('_a','_f'))
    for ix, row in merged_data.iterrows():

        if row['participant code'] in return_value.keys():
            if(not utils.isnan(row['opcForca'+row[injury_side_column]+'[AbdOmbro]_f'])):
                return_value[row['participant code']].append(row['opcForca'+row[injury_side_column]+'[AbdOmbro]_f'])
            
                if(row['formTempoAval_f'] < return_period[row['participant code']][-1]):
                    return_value[row['participant code']][-1], return_value[row['participant code']][-2] = return_value[row['participant code']][-2], return_value[row['participant code']][-1]
                    tmp = return_period[row['participant code']][-1]
                    return_period[row['participant code']][-1] = row['formTempoAval_f']
                    return_period[row['participant code']].append(tmp)
                else:
                    return_period[row['participant code']].append(row['formTempoAval_f'])
        else:
            if(not utils.isnan(row['opcForca'+row[injury_side_column]+'[AbdOmbro]_a'])):
                return_value[row['participant code']] = [row['opcForca'+row[injury_side_column]+'[AbdOmbro]_a']]
                return_period[row['participant code']] = [row['formTempoAval_a']]

                if(not utils.isnan(row['opcForca'+row[injury_side_column]+'[AbdOmbro]_f'])):
                    return_value[row['participant code']].append(row['opcForca'+row[injury_side_column]+'[AbdOmbro]_f'])
                    return_period[row['participant code']].append(row['formTempoAval_f'])
        if(row['snCplexoAt_a'] == 'S' or row['snCplexoAt_f'] == 'S'):
            surgery_patients.append(row['participant code'])

    spatients_to_plot = []    
    speriods_to_plot = []
    nspatients_to_plot = []
    nsperiods_to_plot = []
    for patient in return_value.keys():
        if(len(return_value[patient]) >= 3):
            if(patient in surgery_patients):
                spatients_to_plot.append(return_value[patient])
                speriods_to_plot.append(return_period[patient])
            else:
                nspatients_to_plot.append(return_value[patient])
                nsperiods_to_plot.append(return_period[patient])
    
    print(min([b for a in return_period.values() for b in a ]))
    print(max([b for a in return_period.values() for b in a]))
    exit()

    for j in range(0,len(spatients_to_plot),5):
        ax = plt.subplot(111) 
        plt.axis((0,3000,-1,6))
        for i in range(j,j+5):
            if(i < len(spatients_to_plot)):   
                ax.plot(speriods_to_plot[i],spatients_to_plot[i],'x-')#,color=colors[i])
            else:
                break
        plt.show()

    ax = plt.subplot(111) 
    plt.axis((0,3000,-1,6))
    for i in range(len(nspatients_to_plot)):
        ax.plot(nsperiods_to_plot[i],nspatients_to_plot[i],'x-')
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

def plot_surgery_procedures():

    data_path = '~/Faculdade/Mestrado/Projeto/scripts/Working Scripts/'
    data_path = data_path + 'EXPERIMENT_DOWNLOAD/Group_patients-with-brachial-plexus-injury/Per_questionnaire_data/'
    data_path = data_path + 'Q61802_unified-surgical-evaluation/Responses_Q61802.csv'

    data = pd.read_csv(data_path, header=0, delimiter=",",
        na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)


    procedures = data.filter(like='lisprocedimentos')

    d = {c:0 for c in procedures.columns}
    for ix,row in procedures.iterrows():
        for column in procedures.columns:
            if(row[column] == 'Y'):
                    d[column] += 1
    procedure_names = {'lisprocedimentos[SQ001]': 'Neurólise', 'lisprocedimentos[SQ002]': 'Transferência de nervo',
    'lisprocedimentos[SQ003]': 'Enxertia', 'lisprocedimentos[SQ004]': 'Dissecção de Neuroma',
     'lisprocedimentos[SQ005]': 'Não informado'}
    procedure_frequencies = sorted(d.items(),key=lambda x: x[0])
    width = 0.8
    fig,ax = plt.subplots()

    plt.bar(range(len(procedure_frequencies)),[a[1] for a in procedure_frequencies],width=width,color='blue')
    ax.set_xticks(np.arange(len(procedure_frequencies))+width/2)
    ax.set_xticklabels([procedure_names[l] for l in [a[0] for a in procedure_frequencies]])
    # plt.pie([a[1] for a in procedure_frequencies],
    #     labels=[procedure_names[l] for l in [a[0] for a in procedure_frequencies]],colors=colors[::-1],
    #     autopct='%1.1f%%', radius=1)
    
    
    plt.show()



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



    #print(sorted(Counter([int(a/30) for a in patients_considered.values()]).items()))

#plot_side_distribution()
#plot_event()
plot_followup_return_period()
#plot_ages()
#plot_followup_pain()
#plot_followup_movements()
#plot_surgery_period()
#plot_surgery_procedures()
#calculate_mean_followup_return_period()
#plot_followup_improvements()
