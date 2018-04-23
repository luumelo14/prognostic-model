import pandas as pd
import numpy as np
import re
import math
import csv
import os
import sys
import utils
from datetime import datetime
import pdb

# some assumptions made in order for this script to work properly:
#   - the answers from the same patient should be together (no other patient answer between them)
#   - metadata filenames start with "Fields_[questionnaire_code]"
#   - it currently works only for pt-BR language 

# the ideia is that I want to join columns that has the same meaning but that are separated for the sides (Direito and Esquerdo).
# so I want to use the metadata info to do that.  

def processMetadata(metadata):#
    #metadata = pd.read_csv(file_name, header=0, delimiter=",", na_values=['N/A', 'None', 'NAAI'], quoting=0, encoding='utf8', mangle_dupe_cols=False)
    import pdb
    field_names = metadata.loc[:,['question_code', 'question_description','question_scale','question_scale_label']].drop_duplicates()
    code_description_fields = {}
    descriptions = {}
    equivalent_descriptions = []
    equivalent_fields = {}
    right_side_fields = np.array([])
    left_side_fields = np.array([])
    not_explicited_side_fields = {}
    regex_sides = re.compile(r'Direito|direito|Direita|direita|DIREITO|DIREITA|Esquerdo|esquerdo|Esquerda|esquerda|ESQUERDO|ESQUERDA|,|\"|\.|:') 
    regex_coded_sides = re.compile(r'D|E')

    for code,description,opt,side in field_names.values:
        #f = 0
        try:
            parsed_description = re.sub(regex_sides,"",description) 
        except(TypeError):
            parsed_description = code

        if(not utils.isnan(side)):
            
            if(not re.match(r'\s+',side)):
                if(code not in not_explicited_side_fields.keys()):
                    #pdb.set_trace()
                    not_explicited_side_fields[code] = ['['+str(int(opt))+']',side]
                    
                # else:
                #     print('code %r in side_fields_not_explicit.keys()' % code)
                #     print(not_explicited_side_fields) 

        #     else:
        #         f = 1
        # else:
        #     f = 1

        parsed_code = re.sub(regex_coded_sides,"",code)

        if parsed_code in code_description_fields.keys():
            if(code == equivalent_fields[parsed_code]):
                continue
            if code.count('D') > equivalent_fields[parsed_code].count('D'):
                right_side_fields = np.append(right_side_fields,code)
                left_side_fields = np.append(left_side_fields,equivalent_fields[parsed_code]) 
            else:
                left_side_fields = np.append(left_side_fields,code)
                right_side_fields = np.append(right_side_fields,equivalent_fields[parsed_code])

        else:
            code_description_fields[parsed_code] =parsed_description
            equivalent_fields[parsed_code] =code
            if(parsed_description not in descriptions.keys()):
                descriptions[parsed_description] = code
            elif(parsed_description not in field_names['question_description']):
                if(descriptions[parsed_description].count('D') > code.count('D')):
                    equivalent_descriptions.append([code, descriptions[parsed_description]])
                else:
                    equivalent_descriptions.append([descriptions[parsed_description],code])

        # if(side is not None):
        #     if(re.match(rege))
        #     code = code + 

    for codel,coder in equivalent_descriptions:
        right_side_fields = np.append(right_side_fields,(coder))
        left_side_fields = np.append(left_side_fields,(codel))


    return code_description_fields, right_side_fields, left_side_fields, not_explicited_side_fields

def unifyColumsBySide(data,metadata_paths,class_questionnaire,class_name): #entrada_dados.csv   
    #data = pd.read_csv(file_name, header=0, delimiter=",", na_values=['N/A', 'None', 'NAAI'], quoting=0, encoding='utf8', mangle_dupe_cols=False)
    
    metadata = join_metadata_files(metadata_paths,'participant code')
    cdf,right_side_fields,left_side_fields,not_explicited_side_fields = processMetadata(metadata)
    matched_fields = {}
    side_code = data.filter(like='opcLdLesao').columns[0]
    regex_sides = re.compile(r'Direito|direito|Direita|direita|DIREITO|DIREITA|Esquerdo|esquerdo|Esquerda|esquerda|ESQUERDO|ESQUERDA|,|\"|\s+a\s+|\s+|\.|:') 
    regex_coded_sides = re.compile(r'D|E')
    field_names = np.array(data.columns)
    index_new_to_old_names = {}
    new_to_old_names = {}
    visited = []
    #old_to_new_names = {}
    new_field_names = []



    for field in field_names:
        
        new_field_name = field

        if re.search('(\[.*\])',field):
            field_name, option, foo = re.split('(\[.*\])',field)
            if option == '[AAxi]':
                option = '[AAXi]'
            # if('[1]' in option):
            #     if(field_name +'E' not in left_side_fields):
            #        left_side_fields = np.append(left_side_fields,field_name+'E')
            #     if(field_name + 'D' not in right_side_fields):
            #        right_side_fields = np.append(right_side_fields,field_name+'D')
            #     field_name = field_name + 'E'
                #option = re.sub(r'\[1\]',"",option)

                
            # if('[2]' in option):
            #     if(field_name +'D' not in right_side_fields):
            #        right_side_fields = np.append(right_side_fields,field_name+'D')
            #     if(field_name + 'E' not in left_side_fields):
            #        left_side_fields = np.append(left_side_fields,field_name+'E')
            #     field_name = field_name + 'D'
                #option = re.sub(r'\[2\]',"",option)

        else:
        # if there's no [], then there's no option on the field
            field_name = field
            option = ''
        if field_name in right_side_fields:
            #print(field_name)
            #print(right_side_fields)
            i, = np.where(right_side_fields == field_name)[0]
            if (str(i) + option) in visited:
                continue
            new_field_name = re.sub(r'D|E',"",field_name) + option#re.sub(r'\[2\]',"",option)
            #if new_field_name in new_to_old_names.keys():
            #    new_to_old_names[new_field_name] = np.insert(new_to_old_names[new_field_name], 0, field) 
            #else:
            #    new_to_old_names[new_field_name] = np.array([field])
            #    new_field_names.append(new_field_name)

            # if('[2]' in option):
            #     #option = re.sub(r'\[2\]',"",option)
            #     lsf = re.sub(r'D|E',"",left_side_fields[i])
            #     option = re.sub(r'\[2\]',"[1]",option)
            # else:
            lsf = left_side_fields[i]
            new_to_old_names[new_field_name] = np.array([lsf+option , field])

            visited.append(str(i)+option)
            new_field_names.append(new_field_name)

        elif field_name in left_side_fields:

            i, = np.where(left_side_fields == field_name)[0]
            if (str(i) + option) in visited:
                continue
            new_field_name = re.sub(r'D|E',"",field_name) + option#re.sub(r'\[1\]',"",option)
            #if new_field_name in new_to_old_names.keys():
            #    new_to_old_names[new_field_name] = np.append(new_to_old_names[new_field_name], field) 
            #else:
            #    new_to_old_names[new_field_name] = np.array([field])
            #    new_field_names.append(new_field_name))
            
            # if('[1]' in option):
            #    rsf = re.sub(r'D|E',"",right_side_fields[i])
            #    option = re.sub(r'\[1\]',"[2]",option)
            # else:
            rsf = right_side_fields[i]
            new_to_old_names[new_field_name] = np.array([field,rsf+option])
            #left_side_fields = np.delete(left_side_fields,i)
            #right_side_fields = np.delete(right_side_fields,i)

            visited.append(str(i)+option)
            new_field_names.append(new_field_name)

        elif field_name in not_explicited_side_fields.keys():
            #print('field name in not explicited side fields') 
            #not_explicited_side_fields[field_name][0] = [1] or [2]
            #pdb.set_trace()
            new_field_name = field_name + re.sub(r'\[1\]|\[2\]','',option)
            if(new_field_name in new_field_names):
                continue
            if(not_explicited_side_fields[field_name][0] in option):
                if(re.match(r'Direito|direito|Direita|direita|DIREITO|DIREITA',not_explicited_side_fields[field_name][1])):
                    if(not_explicited_side_fields[field_name][0] == '[1]'):
                        lsf = re.sub(r'\[1\]','[2]',field)
                    else:
                        lsf = re.sub(r'\[2\]','[1]',field)
                    new_to_old_names[new_field_name] = np.array([lsf,field])
                else:
                    if(not_explicited_side_fields[field_name][0] == '[1]'):
                        rsf = re.sub(r'\[1\]','[2]',field)
                    else:
                        rsf = re.sub(r'\[2\]','[1]',field)
                    new_to_old_names[new_field_name] = np.array([field,rsf])
            else:
                if(re.match(r'Direito|direito|Direita|direita|DIREITO|DIREITA',not_explicited_side_fields[field_name][1])):
                    if(not_explicited_side_fields[field_name][0] == '[1]'):
                        rsf = re.sub(r'\[1\]','[2]',field)
                    else:
                        rsf = re.sub(r'\[2\]','[1]',field)
                    new_to_old_names[new_field_name] = np.array([field,rsf])
                else:
                    if(not_explicited_side_fields[field_name][0] == '[1]'):
                        lsf = re.sub(r'\[1\]','[2]',field)
                    else:
                        lsf = re.sub(r'\[2\]','[1]',field)
                    new_to_old_names[new_field_name] = np.array([lsf,field])

            new_field_names.append(new_field_name)


            #if(not_explicited_side_fields[field_name])
            #new_to_old_names[new_field_name] = np.array()

        else:
            new_to_old_names[new_field_name] = np.array([field])
            new_field_names.append(new_field_name)

    final_data = pd.DataFrame(columns = new_field_names)



    #print(new_field_names)
    #print(final_data.columns[final_data.columns.str.endswith('[Cotovelo]')])
    for i in range(len(data[field_names[0]])):
        row = []
        side = data[side_code][i]   
        if side == 'D':
            for field in new_field_names:
                #if('[Subluxacao][2]' in new_to_old_names[field][-1]):
                #    print((data[new_to_old_names[field][-1]])[i])

                #print('d: %r ' % new_to_old_names[field][-1])
                string = re.sub(r',|\n|;','',str(np.array(data[new_to_old_names[field][-1]])[i]))
                #else:
                #    string = re.sub(r',|\n|;','',str(np.array(data[new_to_old_names[field]])[i]))
                row.append(string)

        elif side == 'E':

            for field in new_field_names:

                #if new_to_old_names[field] == -1:
                #    string = re.sub(r',|\n|;','',str(np.array(data[field])[i]))
                #else:
                #print(new_to_old_names[field][0])
                # if(len(new_to_old_names[field]) > 1):
                #     print('e: %r ' % (new_to_old_names[field]))
                string = re.sub(r',|\n|;','',str(np.array(data[new_to_old_names[field][0]])[i]))
                row.append(string)

    
        elif side == 'DE':
            print('ops')
            continue
        final_data.loc[i] = row

    
    final_data = (final_data.T).dropna(how='all').T
    print(final_data.shape)

    class_code = class_questionnaire + '_' + class_name#final_data.filter(like=class_questionnaire+'_'+class_name).columns[0]
    tmp = final_data[class_code]
    del final_data[class_code]
    final_data.insert(len(final_data.columns),class_code,tmp)

    #return final_data
    #final_data.to_csv(out,index=False)
    #print(final_data.shape)
    return final_data


def differentiatePreAndPostSurgery(data,class_name):
    #data =  pd.read_csv(filename,header=0,delimiter=",",
    #    quoting=0,encoding='utf8')
    questionnaires = []
    columns_to_change = ['opcInspecao','opcEscoliose','opcTinel','opcLcSensTatil','opcLcSensor',
        'opcLcArtrestesia','opcLcCinestesia','opcLcPalestesia','snFisioAt','lisTpAuxilio',
        'lisMedicAt','snDorPos', 'snMedicAt','opcForca','intAM']


    for questionnaire in data.columns[data.columns.str.contains('_snCplexoAt')]:

        m = re.match('(Q\d+)\_',questionnaire)
        if m:
            q = m.group(1)
        for column in columns_to_change:
            for column_name in data.columns[data.columns.str.startswith(q+'_'+column)]:
                if(column_name == class_name):
                    continue
                for i in data.index:
                    if(data.loc[i,column_name] != 'NAAI' and  
                        data.loc[i,column_name] != 'NINA' and not utils.isnan(data.loc[i,column_name])):
                            if(data[q+'_snCplexoAt'][i] == 'S'):
                                data.loc[i,column_name] = data.loc[i,column_name] + ' pos'
                            # here NINA's on snCplexoAt are considered "no"    
                            else:
                                data.loc[i,column_name] = data.loc[i,column_name] + ' pre'

                # data.loc[data[q+'_snCplexoAt'] == 'S', 
                # column_name].loc[data[column_name] != 'NAAI'] = data.loc[data[q+'_snCplexoAt'] == 'S',
                #     column_name].loc[data[column_name] != 'NAAI'] + ' pos'
                # data.loc[data[q+'_snCplexoAt'] != 'S',
                #     column_name] = data.loc[data[q+'_snCplexoAt'] != 'S', column_name] + ' pre'
    #data.to_csv(out,index=False)

    return data

def to_scores(filename):

    roots = ['C5', 'C6','C7','C8','T1']
    segment = ['Indicador', 'Cotovelo', 'Ombro']
    segment2 = ['Clavicula','Umero','Ulna']

    modalities = {
        'opcLcSensTatil': {'Ane':2, 'Hiper':1, 'Hipo':1, 'Sem':0, 'EvaluatedOn':roots},
        'opcLcSensor': {'Ana':2, 'Hiper':1, 'Hipo':1, 'Sem':0, 'EvaluatedOn':roots},
        'opcLcArtrestesia': {'Alter':1, 'Prese':0,'EvaluatedOn':segment},
        'opcLcCinestesia': {'Alter':1, 'Prese': 0,'EvaluatedOn': segment},
        'opcLcPalestesia': {'Apa':2, 'Hipo':1, 'P':0, 'EvaluatedOn': segment2}}

    modality_score = 0
    root_score = 0

    data =  pd.read_csv(filename,header=0,delimiter=",",
        quoting=0,encoding='utf8')
    for modality in modalities.keys():
        columns_names = data.columns[data.columns.str.contains(modality)]


#convert numeric class column values into two classes given a threshold,
#so that instances whose value <= threshold belong to class1 and whose
#value > threshold belong to class2
def numeric_to_binary(data,feature,class1,class2,threshold):#,out):
   # data =  pd.read_csv(filename,header=0,delimiter=",",
    #    quoting=0,encoding='utf8')
    for column in (data.filter(like=feature).columns):
        #data = data.drop(np.where([e == 'NAAI' or e == 'NINA' or utils.isnan(e) for e in data[data.columns[i]]])[0])
        d = {'True':class1, True:class1, 'False':class2, False:class2, 'NINA':'NINA'}
        # import pdb
        # pdb.set_trace()
        #True when class value <= threshold and False otherwise 
        comp_threshold = lambda x: np.array([float(a) < threshold if (utils.isfloat(a) or utils.isint(a)) else 'NINA' for a in x])
        #class1 when value is True and class2 when it's False

        mask = [d[l] for l in comp_threshold(data[column])]
        data[column] = mask
    #data.to_csv(out,index=False)
    return data

# read and merge two csv files given a certain condition, and write on a new file
# with name defined by "out"
def merge_files(filename_left,data_right,condition,how):
    l = pd.read_csv(filename_left,header=0,delimiter=",",
        quoting=0,encoding='utf8')
    return l.merge(data_right,how=how,on=condition) #l.merge(pd.read_csv(filename_right,header=0,delimiter=",",
        #quoting=0,encoding='utf8'),how=how,on=condition)
    #'Seguimento_dor_socio.csv'
    
    #data.to_csv(out,index=False)

# concat data files on participant_code, follow-up being the one that we need to preserve all the rows, and
# the other ones being complementary. At this point the attributes need to be preceded by an id for the questionnaires.
# Then union metadata files adding the questionnaire id to the name of the attributes. Then produce the data with
# the output file -> that no longer will be a file.
def get_data(filename,condition):
    r = pd.read_csv(filename, header=0, delimiter=",",
        quoting=0, encoding='utf8')
    questionnaire_id = re.search('.*(Q[0-9]+)',filename).group(1)
    r.columns = [questionnaire_id + '_' + column for column in r.columns]
    r = r.rename(columns={questionnaire_id+'_'+condition: condition})

    return r

def get_metadata(filename,condition):
    r = pd.read_csv(filename, header=0, delimiter=",",
        quoting=0, encoding='utf8')
    questionnaire_id = re.search('.*(Q[0-9]+)',filename).group(1)

    r['question_code'] = [questionnaire_id + '_']*len(r['question_code']) + r['question_code']
    r.loc[r['question_code'] == questionnaire_id+'_'+condition,'question_code'] = condition

    #r['question_description'] = r['question_description'].astype(object)
    for i in range(len(r['question_description'])):
        if(utils.isnan(r['question_description'][i])):
            r['question_description'].iloc[i] = str(r['question_code'][i])
        r['question_description'].loc[i] = str(questionnaire_id + '_' + r['question_description'][i])

           
    #r['question_description'] = [questionnaire_id + '_']*len(r['question_description']) + r['question_description']

    r.loc[r['question_description'] == questionnaire_id + '_' + condition, 'question_description'] = condition

    return r



#the file with the class inner join entrada if it's not entrada
#then left join the other ones
def join_data_files(list_of_files,condition,main_questionnaire='Q44071',class_questionnaire='Q92510',class_name = ''):
    

    cq = False

    for file_index in range(len(list_of_files)):
        if main_questionnaire in list_of_files[file_index]:
            tmp = list_of_files[0]
            list_of_files[0] = list_of_files[file_index]
            list_of_files[file_index] = tmp
        elif class_questionnaire in list_of_files[file_index]:
            cq = True
            if len(list_of_files) > 1:
                k = 1
            else:
                k = 0
            tmp = list_of_files[k]
            list_of_files[k] = list_of_files[file_index]
            list_of_files[file_index] = tmp 

    
    data = treat_main_questionnaire_data(list_of_files[0],condition,main_questionnaire) #r
    k = 1
    if(cq):
        r_to_merge = treat_class_questionnaire_data(list_of_files[k],condition,class_questionnaire,class_name)
        #r_to_merge = r_to_merge.filter(regex=re.escape(condition) + '|' + re.escape(class_name) + '|' + 'date')
        data = data.merge(r_to_merge, how = 'inner', on=condition)
        #data = r
        k += 1
    
    if len(list_of_files) > k:
        s = None
        
        for file_index in range(k,len(list_of_files)):
            #s = get_data(list_of_files[file_index],condition)
            if('Q61802' in list_of_files[file_index]):
                s_to_merge = treat_surgical_questionnaire_data(list_of_files[file_index],condition,'Q61802')
                if(s is None):
                    s = s_to_merge
                else:
                    s = s.merge(s_to_merge, how = 'outer', on=condition)
                #s = s_to_merge#s.merge(s_to_merge, how = 'outer', on=condition)
        #k += 1
            else:
                s_to_merge = get_data(list_of_files[file_index],condition)
                if(s is None):
                    s = s_to_merge
                else:
                    s = s.merge(s_to_merge, how = 'outer', on=condition)
        # s.to_csv('Cirurgica.csv',index=False)
        # print(os.getcwd())
        # print('done')
        # exit()
        # print('shape before: {0}'.format(r.shape))
        # print('shape of surgical: {0}'.format(s.shape))
        data = data.merge(s, how = 'left', on = condition)
        # print('shape after: {0}'.format(data.shape))
        # exit()
    
    #data.to_csv('eita.csv',index=False)
    #r = ((((r.T).drop_duplicates(keep='first'))).T)#.dropna(how='all')).T)
    #r.to_csv('out.csv')
    return data

def treat_main_questionnaire_data(filename,condition,main_questionnaire_code):
    r = get_data(filename,condition)
    columns = np.array(r.columns)
    for column_index in range(len(columns)):
        if('[' in columns[column_index]):
            m = re.match('(.+)(\[.+\])',columns[column_index])
            if m:
                variable_name = m.group(1)
            else:
                print('regex not found: {0} '.format(columns[column_index]))

            variable_columns = r.filter(like=variable_name)
            variable_columns = variable_columns.filter(like=variable_name)#(regex=r''+re.escape(variable_name) + '(?!\[NINA\])')
            for ix, row in variable_columns.iterrows():
                for ir in range(len(row)):
                    if(not utils.isnan(row[ir])):
                        for it in range(len(row)):
                            if(it == ir):
                                continue
                            else:
                                if(utils.isnan(r[variable_columns.columns[it]][ix])):
                                    if(variable_name+'[NINA]' not in r.columns or 
                                        (variable_name+'[NINA]' in r.columns and r[variable_name+'[NINA]'][ix] != 'Y')):
                                        r.ix[ix,variable_columns.columns[it]] = 'N'
                                    #r.set_value(ix,variable_columns.columns[it],'N')

    r = r.drop((r.filter(like='[NINA]').columns),axis=1)
    for ix, row in r.filter(like=main_questionnaire_code+'\_lisTpTrauma[other]'):
        if(not utils.isnan(row)):
            r.set_value(ix,main_questionnaire_code+'_lisTpTrauma[other]','Y')
    return r


def class_value_is_valid(row,tmp,class_questionnaire,class_name):
    try:
        class_index = np.where(tmp.columns == class_questionnaire+'_'+class_name)[0][0]
        if row[class_index] != 'NAAI' and row[class_index] != 'NINA' and not utils.isnan(row[class_index]) :
            return True
    except(IndexError):
        m = re.match('(.+)(\[.+\])',class_name)
        name = class_questionnaire+'_'+m.group(1)
        option = m.group(2)

        class_indexes = [np.where(tmp.columns == name+'E'+option)[0][0],np.where(tmp.columns == name+'D'+option)[0][0]]
        if((row[class_indexes[0]] != 'NAAI' and row[class_indexes[0]] != 'NINA' and not utils.isnan(row[class_indexes[0]]))
            or (row[class_indexes[1]] != 'NAAI' and row[class_indexes[1]] != 'NINA' and not utils.isnan(row[class_indexes[1]]))):
            return True
    return False

def treat_class_questionnaire_data(filename,condition,class_questionnaire,class_name):

    tmp = get_data(filename,condition)
    acquisition_time_code = tmp.filter(like=class_questionnaire+'_formTempoAval').columns[0]
    r_to_merge = pd.DataFrame(columns = tmp.columns)
    #class_index = np.where(tmp.columns == class_questionnaire+'_'+class_name)[0][0]

    i = 0
    for ix,row in tmp.iterrows():
        if i == 0 or tmp[condition][i] != r_to_merge[condition].values[-1]:
            r_to_merge.loc[i] = row
        else:
            if(tmp[acquisition_time_code][ix] > r_to_merge[acquisition_time_code].values[-1] and 
                class_value_is_valid(row,tmp,class_questionnaire,class_name)):
                r_to_merge.iloc[-1] = row
        i += 1
    # for ix,row in tmp.iterrows():
    #     # if(row[0] == 'P23391'):
    #     #    pdb.set_trace()
    #     #print(row[0])
    #     if i == 0 or tmp[condition][ix] != r_to_merge[condition].values[-1]: 
    #         r_to_merge = r_to_merge.append(row)
    #     else:
    #         # if(row[class_index] == 'NAAI'):
    #         #     print(row[0])

    #                 #r_to_merge.set_value(r_to_merge.shape[0]-1,cs,'Y')
    #         #if datetime.strptime(tmp[acquisitiondate_code][i],dateformat) > datetime.strptime(r_to_merge[acquisitiondate_code].values[-1],dateformat): 
    #         old_row = np.array(r_to_merge.iloc[-1])
    #         if tmp[acquisition_time_code][ix] > r_to_merge[acquisition_time_code].values[-1] and row[class_index] != 'NAAI' and row[class_index] != 'NINA':
    #             r_to_merge.iloc[-1] = row


    #         css = r_to_merge.filter(regex=r''+re.escape(class_questionnaire)+'\_.+At').columns
    #         for cs in css:
    #             #print(cs)
    #             if(tmp[cs][ix] == 'S' or old_row[np.where(cs == r_to_merge.columns)[0][0]] == 'S'):
    #                 #print(r_to_merge)
    #                 old = str(r_to_merge[cs].values[-1])
    #                 #r_to_merge[cs][r_to_merge.shape[0]-1] = 'S'#set_value(r_to_merge.shape[0]-1,cs,'S')
    #                 r_to_merge.set_value(r_to_merge.index[-1],cs,'S')
    #                 #r_to_merge.iloc[-1][cs] = 'S'
    #                 #print('{0} -> {1}'.format(old,r_to_merge[cs].values[-1]))
    #             elif(tmp[cs][ix] == 'Y' or old_row[np.where(cs == r_to_merge.columns)[0][0]] == 'Y'):
    #                 #print(r_to_merge[cs].values[-1])

    #                 old = str(r_to_merge[cs].values[-1])
    #                 #r_to_merge[cs][r_to_merge.shape[0]-1] = 'Y'#r_to_merge.iloc[-1][cs]
    #                 r_to_merge.set_value(r_to_merge.index[-1],cs,'Y')
    #                 #print('{0} -> {1}'.format(old,r_to_merge[cs].values[-1]))

    #    i += 1

    return r_to_merge

def treat_surgical_questionnaire_data(filename,condition,surgical_questionnaire_code):
    tmp = get_data(filename,condition)
    acquisition_time_code = tmp.filter(like=surgical_questionnaire_code+'_formTempoCirurg').columns[0]
    remaining_columns = ['participant code','formTempoCirurg', 'opcLdCirurgia',
    'lisprocedimentos', 'lisneurolise[', 'lisneuroliselraiz','lisneurolisenervo','lisneurolisetronco',
    'lisneurolisedivisao', 'lisneurolisecordao', 'opctransferencias','lisenxerto[', 'lisenxertoqualraiz', 'lisenxertoqualtronco',
    'lisenxertonervos','listdissecneuromarai']
    code_procedure = {'neurolise':'[SQ001]' , 'transferencia':'[SQ002]', 'enxerto':'[SQ003]', 'dissecneuroma':'[SQ004]', 'nan':'[SQ005]'}
    nan_codes = {'lisprocedimentos[SQ005]': '(lisprocedimentos)(?!\[SQ005\])','lisneurolise[7]': '(lisneurolise)(?!\[7\])','lisneurolisenervo[20]':'(lisneurolisenervo)(?!\[20\])',
     'lisneurolisetronco[4]':'(lisneurolisetronco)(?!\[4\])', 'lisneurolisecordao[SQ004]':'(lisneurolisecordao)(?!\[SQ004\])',
      'lisneurolisecordao[SQ004]':'(lisneurolisecordao)(?!\[SQ004\])','lisneurolisedivisao[SQ007]': '(lisneurolisedivisao)(?!\[SQ007\])',
      'opctransferencias[SQ017]':'(opctransferencias)(?!\[SQ017\])','lisenxerto[7]':'(lisenxerto)(?!\[7\])',
      'lisenxertoqualraiz[8]':'(lisenxertoqualraiz)(?!\[8\])','lisenxertoqualtronco[4]':'(lisenxertoqualtronco)(?!\[4\])',
      'lisenxertonervos[20]':'(lisenxertonervos)(?!\[20\])','listdissecneuromarai[8]':'listdissecneuromarai)(?!\[8\])'}
    df = tmp.filter(like=remaining_columns[0])
    for rc_index in range(1,len(remaining_columns)):
        df = df.join(tmp.filter(like=remaining_columns[rc_index]))
        #df = df.join(dftmp)

    r_to_merge = pd.DataFrame(columns=df.columns,dtype=str)
    #i = 0
    for i,row in df.iterrows():

        #pdb.set_trace()
        if i == 0 or row[condition] != r_to_merge[condition].values[-1]:
            r_to_merge.loc[r_to_merge.shape[0]] = row.values

            for procedure in code_procedure.keys():

                if(r_to_merge[surgical_questionnaire_code+'_lisprocedimentos'+code_procedure[procedure]].values[-1] == 'Y'):
                    css = r_to_merge.filter(regex=r'(lisprocedimentos)(?!' + re.escape(code_procedure[procedure])+ ')').columns
                    for j in range(len(css)):
                        if(r_to_merge[css[j]].values[-1] != 'Y'):
                            r_to_merge.ix[r_to_merge.shape[0]-1,css[j]] = 'N'
                
                for rc in remaining_columns[3:]:
                    if(procedure in rc):
                        css = r_to_merge.filter(like=rc).columns
                        for cs in css:
                            if(r_to_merge[cs].values[-1] != 'Y' and 
                                r_to_merge[surgical_questionnaire_code+'_lisprocedimentos'+code_procedure['nan']].values[-1] != 'Y'):
                                r_to_merge.ix[r_to_merge.shape[0]-1,cs] = 'N'
                            #else:
                            #    print('ops! something wrong is not right: %r' % cs)
                                #r_to_merge.replace(to_replace=None,r_to_merge.shape[0]-1,css[j],'N')
           

            # if it's not informed
            # if(r_to_merge[surgical_questionnaire_code+'_lisprocedimentos'+code_procedure['nan']].values[-1]) == 'Y':
            #     r_to_merge.set_value(r_to_merge.shape[0]-1,
            #     r_to_merge.filter(regex=r'(lisprocedimentos)(?!' + re.escape(code_procedure['nan']) + ')').columns,np.nan)
                #r_to_merge = r_to_merge.drop(surgical_questionnaire_code+'_lisprocedimentos[SQ005]',axis=1)
            for nan_code in nan_codes.keys():
                if(r_to_merge[surgical_questionnaire_code+'_'+nan_code].values[-1] == 'Y'):
                    r_to_merge.set_value(r_to_merge.index[-1],r_to_merge.filter(regex=r''+nan_codes[nan_code]).columns,np.nan)
            # if r_to_merge[surgical_questionnaire_code+'_lisneurolise[7]'].values[-1] == 'Y':
            #     r_to_merge.set_value(r_to_merge.shape[0]-1,r_to_merge.filter(regex=r'(lisneurolise)(?!\[7\])').columns,np.nan)
            #     #r_to_merge = r_to_merge.drop(surgical_questionnaire_code_code+'_lisneurolise[7]',axis=1)
            # if r_to_merge[surgical_questionnaire_code+'_lisneurolisenervo[20]'].values[-1] == '':
            #     r_to_merge.set_value(r_to_merge.shape[0]-1,r_to_merge.filter(regex=r'(lisneurolisenervo)(?!\[20\])').columns,np.nan)
            #     #r_to_merge = r_to_merge.drop(surgical_questionnaire_code_code+'_lisneurolisenervo[20]',axis=1)
            # if r_to_merge[surgical_questionnaire_code+'_lisneurolisetronco[4]'].values[-1] == 'Y':
            #     r_to_merge.set_value(r_to_merge.shape[0]-1,r_to_merge.filter(regex=r'(lisneurolisetronco)(?!\[4\])').columns,np.nan)
            #     #r_to_merge = r_to_merge.drop(surgical_questionnaire_code_code+'_lisneurolisetronco[4]',axis=1)
            # if r_to_merge[surgical_questionnaire_code+'_lisneurolisecordao[SQ004]'].values[-1] == 'Y':
            #     r_to_merge.set_value(r_to_merge.shape[0]-1,r_to_merge.filter(regex=r'(lisneurolisecordao)(?!\[SQ004\])').columns,np.nan)
            #     #r_to_merge = r_to_merge.drop(surgical_questionnaire_code_code+'_lisneurolisedivisao[SQ007]',axis=1)
            # if r_to_merge[surgical_questionnaire_code+'_lisneurolisedivisao[SQ007]'].values[-1] == 'Y':
            #     r_to_merge.set_value(r_to_merge.shape[0]-1,r_to_merge.filter(regex=r'(lisneurolisedivisao)(?!\[SQ007\])').columns,np.nan)
            #     #r_to_merge = r_to_merge.drop(surgical_questionnaire_code_code+'_lisneurolisecordao[SQ004]',axis=1)
            # if r_to_merge[surgical_questionnaire_code+'_opctransferencias[SQ017]'].values[-1] == 'Y':
            #     r_to_merge.set_value(r_to_merge.shape[0]-1,r_to_merge.filter(regex=r'(opctransferencias)(?!\[SQ017\])').columns,np.nan)
            #     #r_to_merge = r_to_merge.drop(surgical_questionnaire_code_code+'_opctransferencias[SQ017]',axis=1)
            # if r_to_merge[surgical_questionnaire_code+'_lisenxerto[7]'].values[-1] == 'Y':
            #     r_to_merge.set_value(r_to_merge.shape[0]-1,r_to_merge.filter(regex=r'(lisenxerto)(?!\[7\])').columns,np.nan)
            #     #r_to_merge = r_to_merge.drop(surgical_questionnaire_code_code+'_lisenxerto[7]',axis=1)
            # if r_to_merge[surgical_questionnaire_code+'_lisenxertoqualraiz[8]'].values[-1] == 'Y':
            #     r_to_merge.set_value(r_to_merge.shape[0]-1,r_to_merge.filter(regex=r'(lisenxertoqualraiz)(?!\[8\])').columns,np.nan)
            #     #r_to_merge = r_to_merge.drop(surgical_questionnaire_code_code+'_lisenxertoqualraiz[8]',axis=1)
            # if r_to_merge[surgical_questionnaire_code+'_lisenxertoqualtronco[4]'].values[-1] == 'Y':
            #     r_to_merge.set_value(r_to_merge.shape[0]-1,r_to_merge.filter(regex=r'(lisenxertoqualtronco)(?!\[4\])').columns,np.nan)
            #     #r_to_merge = r_to_merge.drop(surgical_questionnaire_code_code+'_lisenxertoqualtronco[4]',axis=1)
            # if r_to_merge[surgical_questionnaire_code+'_lisenxertonervos[20]'].values[-1] == 'Y':
            #     r_to_merge.set_value(r_to_merge.shape[0]-1,r_to_merge.filter(regex=r'(lisenxertonervos)(?!\[20\])').columns,np.nan)
            #     #r_to_merge = r_to_merge.drop(surgical_questionnaire_code_code+'_lisenxertonervos[20]',axis=1)
            # if r_to_merge[surgical_questionnaire_code+'_listdissecneuromarai[8]'].values[-1] == 'Y':
            #     r_to_merge.set_value(r_to_merge.shape[0]-1,r_to_merge.filter(regex=r'(listdissecneuromarai)(?!\[8\])').columns,np.nan)
                #r_to_merge = r_to_merge.drop(surgical_questionnaire_code_code+'_listdissecneuromarai[8]',axis=1)

        else:

            if row[acquisition_time_code] < r_to_merge[acquisition_time_code].values[-1]:
                r_to_merge.set_value(r_to_merge.index[-1],acquisition_time_code, row[acquisition_time_code])
                #css = r_to_merge.filter(regex=r'(lisprocedimentos)(?!' + re.escape(code_procedure[procedure])+ ')').columns
            if(row[surgical_questionnaire_code+'_lisprocedimentos'+code_procedure['nan']] == 'Y'):
                continue

            for procedure in code_procedure.keys():
                # if(row[condition] == 'P87791'):
                #     pdb.set_trace()
                if(row[surgical_questionnaire_code+'_lisprocedimentos'+code_procedure[procedure]] == 'Y'):
                    r_to_merge.set_value(r_to_merge.index[-1],surgical_questionnaire_code+'_lisprocedimentos'+code_procedure[procedure], 'Y')
                    css = r_to_merge.filter(regex=r'(lisprocedimentos)(?!' + re.escape(code_procedure[procedure])+ ')').columns
                    for cs in css:
                        if(r_to_merge[cs].values[-1] != 'Y'):
                            r_to_merge.ix[r_to_merge.shape[0]-1,cs] = 'N'
                for rc in remaining_columns[3:]:
                    if(procedure in rc):
                        css = r_to_merge.filter(like=rc).columns
                        for cs in css:
                            if(row[cs] == 'Y'):
                                r_to_merge.ix[r_to_merge.shape[0]-1,cs] = 'Y' 
                            else:
                                if(r_to_merge[cs].values[-1] != 'Y'):
                                    r_to_merge.ix[r_to_merge.shape[0]-1,cs] = 'N'



            #if df[surgical_questionnaire+'_lisprocedimentos[SQ001]'][i] == 'Y':
        #i+=1
    nan_codes_with_questionnaire_id = np.array(list(nan_codes.keys()),dtype=object)

    for i in range(len(nan_codes_with_questionnaire_id)):
        nan_codes_with_questionnaire_id[i] = surgical_questionnaire_code+ '_'+ nan_codes_with_questionnaire_id[i]
    r_to_merge = r_to_merge.drop(nan_codes_with_questionnaire_id,axis=1)

    return r_to_merge
    



def get_frequencies_of_return(filename,condition):

    data = pd.read_csv(filename, header=0, delimiter=",",
        quoting=0, encoding='utf8')
    dateformat = '%Y-%m-%d %H:%M:%S' 
    e_acquisitiondate_code = data.filter(like='44071_acquisitiondate').columns[0]
    s_acquisitiondate_code = data.filter(like='92510_acquisitiondate').columns[0]
    #i = 0
    periods = {}
    
    for i in range(data.shape[0]):
        d = (datetime.strptime(data[s_acquisitiondate_code][i],dateformat) - datetime.strptime(data[e_acquisitiondate_code][i],dateformat))
        d = round(d.days/30)
        if d not in periods.keys():
            periods[d] = 1
        else:
            periods[d] += 1
        #i+=1
        
    import matplotlib.pyplot as plt
    k = sorted(periods.items(),key=lambda x: x[0])
    plt.bar(range(0,2*len([i[0] for i in k]),2),[i[1] for i in k])
    pos = np.arange(0,2*len(k),2)
    width = 1.0     # gives histogram aspect to the bar diagram
    ax = plt.axes()
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels([i[0] for i in k])
    plt.xlabel('período (meses)')
    plt.ylabel('frequência')
    plt.show()


def join_metadata_files(list_of_files,condition):


    filename = list_of_files[0]         
    r = get_metadata(filename,condition)
    
    for file_index in range(1,len(list_of_files)):
        if('Q61802' not in list_of_files[file_index]):
            r_to_merge = get_metadata(list_of_files[file_index],condition) 
            r = r.append(r_to_merge)
    print(r.shape)

    return r

def reduce(data, main_questionnaire,class_questionnaire,class_name, condition):
    r_columns = [condition, main_questionnaire+'_snFxPr',  main_questionnaire+'_snCortPr',
    main_questionnaire+'_snCcerPr',main_questionnaire+'_snCnerPr', main_questionnaire+'_snTCEPr', main_questionnaire+'_snTRMPr',
    main_questionnaire+'_snDorPr', main_questionnaire+'_formIdadeLesao', main_questionnaire+'_opcLdLesao',
    main_questionnaire+'_lisTpTrauma[moto]',main_questionnaire+'_snFxAt', main_questionnaire+'_snLuxAt',
    main_questionnaire+'_snTCEAt',main_questionnaire+'_snCortAt', main_questionnaire+'_snCcerAt', main_questionnaire+'_snTRMAt',
    main_questionnaire+'_snDrenoAt', main_questionnaire+'_snVasoAt', main_questionnaire+'_snDesacordado', 
    main_questionnaire+'_snFisioAt', main_questionnaire+'_lisTpAuxilio[Tipoia]',main_questionnaire+'_lisMedicAt[Opioides_Nome]',
    main_questionnaire+'_lisMedicAt[Antidepressivos_Nome]', main_questionnaire+'_lisMedicAt[Anticonvulsivantes_Nome]',
    main_questionnaire+'_lisMedicAt[Neurolepticos_Nome]', main_questionnaire+'_snCplexoAt', main_questionnaire+'_snCdorAt',
    main_questionnaire+'_opcInspecao[Subluxacao]', main_questionnaire+'_opcInspecao[Alada]',
    main_questionnaire+'_opcInspecao[Horner]', main_questionnaire+'_opcInspecao[Edema]', 
    main_questionnaire+'_opcInspecao[Cicatriz]', main_questionnaire+'_opcInspecao[Trofismo]',
    main_questionnaire+'_opcEscoliose[SQ007]', main_questionnaire+'_opcTinel[SQ007]', main_questionnaire+'_opcLcSensTatil[C5]',
    main_questionnaire+'_opcLcSensTatil[C6]',main_questionnaire+'_opcLcSensTatil[C7]',main_questionnaire+'_opcLcSensTatil[C8]',
    main_questionnaire+'_opcLcSensTatil[T1]', main_questionnaire+'_opcLcSensor[C5]', main_questionnaire+'_opcLcSensor[C6]',
    main_questionnaire+'_opcLcSensor[C7]', main_questionnaire+'_opcLcSensor[C8]', main_questionnaire+'_opcLcSensor[T1]',
    main_questionnaire+'_opcLcArtrestesia[Indicador]',main_questionnaire+'_opcLcArtrestesia[Cotovelo]',
    main_questionnaire+'_opcLcArtrestesia[Ombro]', main_questionnaire+'_opcLcCinestesia[Indicador]', 
    main_questionnaire+'_opcLcCinestesia[Cotovelo]', main_questionnaire+'_opcLcCinestesia[Ombro]',
    main_questionnaire+'_opcLcPalestesia[Clavicula]',  main_questionnaire+'_opcLcPalestesia[Umero]',
    main_questionnaire+'_opcLcPalestesia[Ulna]', main_questionnaire+'_intAMflexombro',main_questionnaire+'_intAMabduombro',
    main_questionnaire+'_intAMrotex', main_questionnaire+'_intAMflexcotovelo', main_questionnaire+'_intAMextcotovelo',
    main_questionnaire+'_intAMsupinacao', main_questionnaire+'_intAMpronacao', main_questionnaire+'_intAMflexpunho',
    main_questionnaire+'_intAMextpunho', main_questionnaire+'_opcForca[AbdOmbro]', main_questionnaire+'_opcForca[RotEOmbro]',
    main_questionnaire+'_opcForca[RotIOmbro]', main_questionnaire+'_opcForca[ElevEscapula]', 
    main_questionnaire+'_opcForca[AbdRotSEscapula]', main_questionnaire+'_opcForca[FlexCotovelo]',
    main_questionnaire+'_opcForca[ExtCotovelo]', main_questionnaire+'_opcForca[ExtPunho]',
    main_questionnaire+'_opcForca[FlexPunho]', main_questionnaire+'_opcForca[FlexDedos]',
    main_questionnaire+'_opcForca[AbdDedos]',main_questionnaire+'_opcForca[AdDedos]',
    main_questionnaire+'_opcForca[Oponencia]', main_questionnaire+'_snDorPos'] 

    surgery_columns = data.filter(like='Q61802').columns
    #class_colum = data[class_questionnaire+'_'+class_name].columns
    remaining_columns = r_columns + list(surgery_columns) + [class_questionnaire+'_'+class_name]
    data = data[remaining_columns]
    medic_columns = data.filter(like=main_questionnaire+'_lisMedicAt').columns
    for ix, row in data[medic_columns].iterrows():
        for j in range(len(medic_columns)):
            if(row[medic_columns[j]] != 'N' and not utils.isnan(row[medic_columns[j]])):
                data.set_value(ix,medic_columns[0],'S')
    data = data.rename(columns = {medic_columns[0]:main_questionnaire+'_'+'lisMedicAtNer'})
    data = data.drop(medic_columns[1:],axis=1)


    return data


    

# if(len(sys.argv) > 1):
#     path = sys.argv[1]
# else:
#     print("Please provide the files path. \nUsage: $ python3 preprocess.py my/path/") 
#     exit(-1)    


def preprocess(path,main_questionnaire,class_questionnaire,class_name,out,surgery=True,dif_surgery=False,reduced=False,to_binary=True,language='pt'):
    data_paths = []
    metadata_paths = []

    if(language != 'pt' and language != 'pt-BR' and language != 'pt-br'):# and language != 'en'):
        print('Language not identified. Changing to language = pt-BR')
        language = 'pt'

    print('Getting data path...')
    dirname = 'Group_patients-with-brachial-plexus-injury'
    # get data_paths of per questionnaires data 
    for d,ds,filenames in os.walk(os.path.join(path,dirname,'Per_questionnaire_data')):
        for filename in filenames:
            if('.~lock.' in filename or (not surgery and 'Q61802' in filename)):
                continue
            data_paths.append(os.path.join(d,filename))
    # get data_paths of per questionnaires data
    for d,ds,filenames in os.walk(os.path.join(path,dirname,'Questionnaire_metadata')):
        for filename in filenames:
            if(language not in filename or '.~lock.' in filename or (not surgery and 'Q61802' in filename)):
                continue
            metadata_paths.append(os.path.join(d,filename))
    print('Joining datafiles...')

    data = join_data_files(data_paths,'participant code',main_questionnaire,class_questionnaire,class_name)
    print('data size: {0}'.format(data.shape))
    #metadata = join_metadata_files(metadata_paths,'participant code')
    #cdf,right_side_fields,left_side_fields,not_explicited_side_fields = processMetadata(metadata)
    print('Unifying columns by side...')
    data = unifyColumsBySide(data,metadata_paths,class_questionnaire,class_name)
    #data = merge_files('Dados_sociodemograficos.csv', data, condition = 'participant_code', how = 'right')
    if 'Dor' not in class_name:
        data = numeric_to_binary(data,class_questionnaire+'_'+class_name,'Insatisfatorio','Sucesso',3)
    if(to_binary):
        print('Transforming numeric features to binary...')
        data = numeric_to_binary(data,main_questionnaire+'_opcForca','Menor que 3','Maior ou igual a 3',3)
        data = numeric_to_binary(data,main_questionnaire+'_formIdadeLesao','Menor que 30','Maior ou igual a 30',30)
        data = numeric_to_binary(data,main_questionnaire+'_intAMflexombro','Menor que 180','Maior ou igual a 180',180)
        data = numeric_to_binary(data,main_questionnaire+'_intAMextombro','Menor que 50','Maior ou igual a 50',50)
        data = numeric_to_binary(data,main_questionnaire+'_intAMabduombr','Menor que 170','Maior ou igual a 170',170) 
        data = numeric_to_binary(data,main_questionnaire+'_intAMrotex','Menor que 60','Maior ou igual a  60',60)
        data = numeric_to_binary(data,main_questionnaire+'_intAMflexcotovelo','Menor que 40','Maior ou igual a 40',40)
        data = numeric_to_binary(data,main_questionnaire+'_intAMextcotovelo','Menor que 180','Maior ou igual a 180',180) 
        data = numeric_to_binary(data,main_questionnaire+'_intAMsupinacao','Menor que 80','Maior ou igual a 80',80)
        data = numeric_to_binary(data,main_questionnaire+'_intAMpronacao','Menor que 80','Maior ou igual a 80',80)
        data = numeric_to_binary(data,main_questionnaire+'_intAMflexpunho','Menor que 60','Maior ou igual a 60',60)
        data = numeric_to_binary(data,main_questionnaire+'_intAMextpunho','Menor que 60','Maior ou igual a 60',60)
        

    if(dif_surgery):
        print('Adding pre and post surgery info to features...')
        data = differentiatePreAndPostSurgery(data,class_questionnaire+'_'+class_name)
    data = data.dropna(subset=[class_questionnaire+'_'+class_name])
    print('Dropping some variables...')
    data = data.drop(np.where([e == 'NAAI' or e == 'NINA' for e in data[data.columns[-1]]])[0])
    data = data.drop(data.columns[data.columns.str.endswith('id')], 1)
    data = data.drop(data.columns[data.columns.str.endswith('token')], 1)
    data = (data.drop(data.columns[data.columns.str.endswith('ipaddr')],1))
    data = (data.drop(data.columns[data.columns.str.endswith('stamp')],1))
    # print(data.columns[data.columns.str.endswith('gender')])
    # print(data.columns[data.columns.str.endswith('gender')][1:])
    data = (data.drop(data.columns[data.columns.str.endswith('gender')][1:],1))
    if(reduced):
        data = reduce(data, main_questionnaire,class_questionnaire,class_name,'participant code')
    #final_data = (final_data.T).dropna(how='all').T
    print(data.shape)
    data = data.drop(data.T[np.array([np.all([data[k] == 'nan']) for k in data])].T.columns,axis=1)
    final_data = data
    print(final_data.shape)
    final_data.to_csv(out,index=False)

#preprocess('EXPORT', 'Q44071','Q92510','opcForca[RotEOmbro]', out = 'RotEOmbro.csv')
#preprocess('EXPORT', 'Q44071','Q92510','opcForca[FlexCotovelo]', out = 'FlexCotovelo.csv')
#preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[AbdOmbro]', out = 'AbdOmbroNew.csv',surgery=False,to_binary=True)
#preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[FlexCotovelo]', out = 'FlexCotoveloNew.csv',surgery=False,to_binary=True)
#preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[RotEOmbro]', out = 'RotEOmbroNew.csv',surgery=False,to_binary=True)
#preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','snDorPos', surgery=False,out = 'DorTeste.csv',to_binary=False)
#preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','snDorPos', out = 'DorCirurgiaCateg.csv',surgery=True,dif_surgery=False,to_binary=True,language='pt')
# preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[AbdOmbro]', out = 'AbdOmbroCirurgiaPrePosCateg.csv',surgery=True,dif_surgery=True,to_binary=True,language='pt')
# preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[AbdOmbro]', out = 'AbdOmbroCirurgiaPrePos.csv',surgery=True,dif_surgery=True,to_binary=False,language='pt')
# preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[AbdOmbro]', out = 'AbdOmbroPrePos.csv',surgery=False,dif_surgery=True,to_binary=False,language='pt')
# preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[AbdOmbro]', out = 'AbdOmbroPrePosCateg.csv',surgery=False,dif_surgery=True,to_binary=True,language='pt')
#preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','snDorPos', out = 'DorCirurgia.csv',surgery=True,dif_surgery=False,to_binary=False,language='pt')
preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','snDorPos', out = 'DorCirurgiaReduzidoTeste.csv',reduced=True,surgery=True,dif_surgery=False,to_binary=False,language='pt')

#preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[AbdOmbro]', out = 'AbdOmbroCirurgia.csv',surgery=True,dif_surgery=False,to_binary=False,language='pt')

#print('DONE WITH ABDOMBRO!')
#print('\n\n')
# preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[RotEOmbro]', out = 'RotEOmbroCirurgiaPrePosCateg.csv',surgery=True,dif_surgery=True,to_binary=True,language='pt')
# preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[RotEOmbro]', out = 'RotEOmbroCirurgiaPrePos.csv',surgery=True,dif_surgery=True,to_binary=False,language='pt')
# preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[RotEOmbro]', out = 'RotEOmbroPrePos.csv',surgery=False,dif_surgery=True,to_binary=False,language='pt')
# preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[RotEOmbro]', out = 'RotEOmbroPrePosCateg.csv',surgery=False,dif_surgery=True,to_binary=True,language='pt')
#preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[RotEOmbro]', out = 'RotEOmbroCirurgia.csv',surgery=True,dif_surgery=False,to_binary=False,language='pt')

#print('DONE WITH ROTEOMBRO!')
#print('\n\n')
# preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[FlexCotovelo]', out = 'FlexCotoveloCirurgiaPrePosCateg.csv',surgery=True,dif_surgery=True,to_binary=True,language='pt')
# preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[FlexCotovelo]', out = 'FlexCotoveloCirurgiaPrePos.csv',surgery=True,dif_surgery=True,to_binary=False,language='pt')
# preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[FlexCotovelo]', out = 'FlexCotoveloPrePos.csv',surgery=False,dif_surgery=True,to_binary=False,language='pt')
# preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[FlexCotovelo]', out = 'FlexCotoveloPrePosCateg.csv',surgery=False,dif_surgery=True,to_binary=True,language='pt')
#preprocess('EXPERIMENT_DOWNLOAD', 'Q44071','Q92510','opcForca[FlexCotovelo]', out = 'FlexCotoveloCirurgia.csv',surgery=True,dif_surgery=False,to_binary=False,language='pt')

#merge_files('Dados_sociodemograficos.csv','FlexCotovelo.csv',out='FlexCotovelo.csv',condition='participant_code',how='right')
#numeric_to_binary('FlexCotovelo.csv','Insatisfatorio','Sucesso',2,'FlexCotovelo.csv')
#differentiatePreAndPostSurgery('FlexCotovelo.csv','FlexCotovelo.csv')


#get_frequencies_of_return('Dor_com_seguimento_pacientes_repetidos.csv','participant_code')