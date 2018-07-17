import pandas as pd
import numpy as np
import re
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import matplotlib.pyplot as plt
import utils

#import preprocess as pp

sw = ['a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 
'até', 'com', 'como', 'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois', 
'do', 'dos', 'e', 'ela', 'elas', 'ele', 'eles', 'em', 'entre', 'era', 'eram', 'essa', 'essas', 
'esse', 'esses', 'esta', 'estamos', 'estas', 'estava', 'estavam', 'este', 'esteja', 'estejam', 
'estejamos', 'estes', 'esteve', 'estive', 'estivemos', 'estiver', 'estivera', 'estiveram',
'estiverem', 'estivermos', 'estivesse', 'estivessem', 'estivéramos', 'estivéssemos', 'estou',
'está', 'estávamos', 'estão', 'eu', 'ficar','ficou','ficaram','foi', 'fomos', 'for', 'fora', 'foram', 'forem', 'formos', 
'fosse', 'fossem', 'fui', 'fôramos', 'fôssemos', 'haja', 'hajam', 'hajamos', 'isso', 'isto', 
'lhe', 'lhes', 'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'muito', 'na', 'nas',
 'no', 'nos', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 'nós', 'o', 'os', 'ou', 
 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 'que', 'quem', 'se', 
 'seja', 'sejam', 'sejamos', 'serei', 'seremos', 'seria', 'sendo', 'seriam', 'será', 'serão', 'seríamos', 
 'seu', 'seus', 'somos', 'sou', 'sua', 'suas', 'são', 'só', 'também', 'te', 'teu', 'teus', 
 'tu', 'tua', 'tuas', 'ter','teve','tem', 'tínhamos', 'um', 'uma', 'você', 'vocês', 'vos', 'à', 'às', 'éramos']


def readData(class_name,class_questionnaire='Q92510',data_path=None,missing_input='mean',dummy=False,transform_numeric = True,use_text=True,skip_class_questionnaire=True):
	# attributes are separated by commas (',')
	# "nan" is assigned to fields with 'N/A' or 'None'


	print('Reading data...')
	data = pd.read_csv(data_path, header=0, delimiter=",",
		na_values=['N/A', 'None','nan','NAAI','NINA'], quoting=0, encoding='utf8', mangle_dupe_cols=False)

	if(not transform_numeric):
		dummy = False

	#print(data.columns[-1])
	# data = data.dropna(subset=[class_name])
	# data = data.drop(np.where([e == 'NAAI' or e == 'NINA' for e in data[data.columns[-1]]])[0])
	# print(data.shape)
	data = data.drop(data.columns[data.columns.str.endswith('id')], 1)
	data = data.drop(data.columns[data.columns.str.endswith('token')], 1)
	data = (data.drop(data.columns[data.columns.str.endswith('ipaddr')],1))
	data = (data.drop(data.columns[data.columns.str.endswith('date')],1))
	data = (data.drop(data.columns[data.columns.str.endswith('stamp')],1))
	data = (data.drop(data.columns[data.columns.str.endswith('participant_code')],1))
	data = (data.drop(data.columns[data.columns.str.endswith('datLesao')],1))
	data = (data.drop(data.columns[data.columns.str.endswith('datNasc')],1))
	# data = (data.drop(data.columns[data.columns.str.endswith('Origem')],1))
	# data = (data.drop(data.columns[data.columns.str.endswith('Cidade')],1))
	# data = (data.drop(data.columns[data.columns.str.endswith('Estado')],1))
	# data = (data.drop(data.columns[data.columns.str.endswith('País')],1))
	# data = (data.drop(data.columns[data.columns.str.endswith('participant_code')],1))
	# data = (data.drop(data.columns[data.columns.str.endswith('Natural de')],1))

	data = ((((data.T).drop_duplicates(keep='first')).dropna(how='all')).T)
	#dropping columns that are constant
	data = data.loc[:,data.apply(pd.Series.nunique) != 1]

	## data = pp.preprocess(data_path,class_name)
	n_samples = data.shape[0]
	n_features = data.shape[1]
	regex_date = re.compile('(\d{4})-(\d{2})-(\d{2})\s?((\d{2}):(\d{2}):(\d{2}))?')


	treatment = np.empty(n_features,dtype='U5')

	attributes = []
	categories = []
	transformedData = []
	index = 0
	si = 0

	print('Transforming data...')
	### representing the categories with numbers

	for attribute in data.columns:

		if skip_class_questionnaire and class_questionnaire in attribute and class_name not in attribute:
			index +=1
			continue
		# else:
		# 	if class_questionnaire in attribute and 'Ombro' not in attribute:
		# 		index+=1
		# 		continue

		t = pd.factorize(data[attribute].values,sort=True)
		#temp = t[0]
		i = utils.firstNotNan(data[attribute].values)
		
		try:
			result = regex_date.match(data[attribute].values[i])
			if(result):
				treatment[index] = 'date'

			elif(len(t[1]) > 0.9*n_samples):
				# if(attribute == 'participant_code'):
				# 	temp = t[0]
				# 	treatment[index] = 'int'
				# else:
					treatment[index] = 'text'		
			else:
				if(utils.isfloat(data[attribute].values[i])):
					# index+=1
					# continue
					#temp = [float(x) for x in t[0]]
					treatment[index] = 'float'

				elif(not dummy):

					if(transform_numeric or utils.isint(data[attribute].values[i])):  
						temp = t[0]
						# if not utils.isint(data[attribute].values[i]):
						# 	index += 1
						# 	continue

					else:
						temp = data[attribute].values
					treatment[index] = 'int'

				else:
					treatment[index] = 'bin'

		except TypeError:


			if(utils.isfloat(data[attribute].values[i])):
				# index+=1
				# continue
				temp = np.array(data[attribute].values).reshape(-1,1)
				treatment[index] = 'float'
			elif(utils.isint(data[attribute][i])):
				# index+=1
				# continue
				temp = (np.array(data[attribute].values)*1).reshape(-1,1)
				treatment[index] = 'int'
			else:
				print("could not identify type of attribute %s" % attribute)
				exit(-1)



		#treatment of class	attribute	
		if(class_name in attribute):
			temp = t[0]
			treatment[index] = 'int'
		
	
		
		if(treatment[index] == 'float'):
			if(missing_input != 'none'):
				imp = preprocessing.Imputer(strategy=missing_input,axis=0)
				temp = imp.fit_transform(X=np.array(data[attribute].values).reshape(-1,1))
			else:
				temp = data[attribute].values
			#print(np.array(list((float(x) for x in temp))).reshape(-1,1).shape)
			transformedData.append(np.array(list((float(x) for x in temp))).reshape(-1,1))

		else:
			# t[0] corresponds to the translated numeric data 
			# t[1] corresponds to a list with the possible values for each feature'
			# (different values in a column, e.g. [sim, não]).
			# the index of that value in the list corresponds to its numeric representation 
			# (e.g. [sim, não] -> sim is represented by 0 and não by 1).
			# if(missing_input != 'none' and treatment[index] != 'bin'):	
			# 	imp = preprocessing.Imputer(missing_values=-1,strategy=missing_input,axis=0)
			# 	temp = imp.fit_transform(X=temp.reshape(-1,1))
				
			if(treatment[index] == 'bin'):
				#imp = preprocessing.Imputer(missing_values=-1,strategy='mean',axis=0)
				#temp = imp.fit_transform(X=np.array(temp).reshape(-1,1))
				
				temp = pd.get_dummies(np.ravel(data[attribute].values))
				for x in temp.columns:
					attributes.append(attribute+'='+x)
					#print(temp[x].reshape(-1,1).shape)
					transformedData.append(temp[x].reshape(-1,1))



			elif(treatment[index] == 'int'):
				if (not transform_numeric):
					temp = data[attribute].values
					for temp_index in range(len(temp)):
						if(isinstance(temp[temp_index],str)):
							temp[temp_index] = temp[temp_index].upper()

					i = utils.firstNotNan(data[attribute].values)
					if (utils.isint(data[attribute].values[i]) and missing_input != 'none'):
						temp[data[attribute].values == 'NAAI'] = -1
						temp[np.isnan(np.array(data[attribute].values,dtype=float))] = -1
						imp = preprocessing.Imputer(missing_values=-1,strategy=missing_input,axis=0)
						temp = imp.fit_transform(X=np.array(list(int(x) for x in temp)).reshape(-1,1))
						

				elif(missing_input != 'none'):	
					imp = preprocessing.Imputer(missing_values=np.nan,strategy=missing_input,axis=0)
					temp = imp.fit_transform(X=np.array(temp).reshape(-1,1))

				#print(np.array(temp).reshape(-1,1).shape)
				transformedData.append(np.array(temp).reshape(-1,1))
				
				
			elif(treatment[index] == 'date'):
				temp = []
				for date in data[attribute].values:
					if(not isinstance(date,float)):
						temp.append(int(date[:4]))
					else:
						temp.append(-1)
				if(missing_input != 'none'):
					imp = preprocessing.Imputer(strategy='most_frequent',axis=0)	

					temp = imp.fit_transform(X=np.array(temp).reshape(-1,1))

				#print(np.array(temp).reshape(-1,1).shape)
				transformedData.append(np.array(temp).reshape(-1,1))

			elif(use_text and treatment[index] == 'text'):
				#try:
				bigword = ''
				#print(attribute)
				try:
					bag_of_words = CountVectorizer(min_df=0.25, stop_words=sw,ngram_range=(1,4))
					#print(data[attribute])
					words = np.array(bag_of_words.fit_transform(((data[attribute].values))).todense())
					c = 0
					for word in bag_of_words.get_feature_names():
						bigword = bigword + word + ' '
						attributes.append(attribute + ' termo: ' + word)
						transformedData.append(words[:,c].reshape(-1,1))
						c+=1 
						# wordcloud = WordCloud(stopwords=sw,background_color='white').generate(bigword,)	
						# plt.imshow(wordcloud)
						# plt.axis('off')
						# plt.show()
				except (ValueError, AttributeError):

					index += 1
					continue	
			else:
				index+=1
				continue
			# else:
			# 	print('undefined option for pre processing: (%s, %s) ' % (categ) )
			# 	exit(-1)

		categories.append(t[1])
		if(treatment[index] != 'text' and treatment[index] != 'bin'):
			attributes.append(attribute)		

		index += 1

	data = np.array(transformedData).reshape(-1,n_samples).T


	# pd.DataFrame(data,columns=attributes).to_csv('out.csv', index=False)
	# f = open('DorR.csv', 'w')
	# f.write(','.join(np.array(attributes, dtype=object)))


	# for d in data:
	# 	f.write('\n')
	# 	f.write(','.join(str(dd) for dd in d))
	# exit()

	return data, np.array(attributes, dtype=object), np.array(categories)
	###

