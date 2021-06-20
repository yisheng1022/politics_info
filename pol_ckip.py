# from ckiptagger import WS,POS,NER
import pandas as pd
import os,re
import numpy as np

def stopwordlist():
	stopwords = [line.strip() for line in open('stop_word_all.txt', 'r', encoding='utf-8').readlines()]
	return stopwords

def ckip_cut_gpu(input_data,data_col,do_NER = False): #whole csv dataframe, colname wait for cut
	from ckiptagger import WS, construct_dictionary

	User_Dict = {}
	with open("dict2.txt","r",encoding = 'utf-8') as USDic:
		for tmpwords in USDic:
			words = tmpwords.strip().split(" ")
			if len(words) > 1:
				User_Dict[words[0]] = words[1]
			else:
				User_Dict[words[0]] = 10
	dictionary = construct_dictionary(User_Dict)
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	ws = WS("./data",disable_cuda=False)

	input_data = input_data.replace(np.nan,'',regex = True)
	tmp_text = list(input_data[data_col])
	stopwordslist = stopwordlist()
	ckip_cut_result = pd.DataFrame(columns = ['CKIP_Result'])
	ckip_cut_result['CKIP_Result'] = ckip_cut_result['CKIP_Result'].astype('str')
	total = len(tmp_text)
	counter = 1
	tmp_things = []
	for things in tmp_text:
		print("Now: ",str(counter)," of ",total)
		tmp_things.append(things)
		ckip_cut = ws(tmp_things,sentence_segmentation=True,segment_delimiter_set = {",", "。", ":", "?", "!", ";", "、"}) #sentence_segmentation=True,segment_delimiter_set = {",", "。", ":", "?", "!", ";", "、"},coerce_dictionary = dictionary
		tmp_things.clear()
		if do_NER:
			print("Not yet.")
		else:
			text = ''
			for cutted in ckip_cut:
				if cutted not in stopwordslist:
					text = str(cutted) + " " + text
			text = re.sub(r'[0-9]','',text)
			text = re.sub(r'[^\w\s]','',text)
			text = re.sub(r'[a-zA-Z]','',text)
			tmp = pd.Series({'CKIP_Result' : text})
			ckip_cut_result = ckip_cut_result.append(tmp,ignore_index = True)
			counter += 1
	del ws
	return ckip_cut_result

def do_LDA(cut_Result,F_name,topic_count = 10,word_count = 10): #input cut result only
	from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
	from sklearn.decomposition import LatentDirichletAllocation
	import pyLDAvis,pyLDAvis.sklearn
	cut_list = cut_Result.values.tolist()
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(cut_list)
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(X)
	# word = vectorizer.get_feature_names()
	# do tfidf upon

	LDA = LatentDirichletAllocation(n_components = topic_count, random_state = None)
	LDA.fit(tfidf)
	f = open(F_name,'w+',encoding = 'utf-8')
	for x,topic in enumerate(LDA.components_):
		f.write("Topic #{}\n".format(x))
		f.write(str([vectorizer.get_feature_names()[index] for index in topic.argsort()[word_count*-1:]]))
		f.write("\n")
	f.close()

	# pic_data = pyLDAvis.sklearn.prepare(LDA,tfidf,vectorizer)
	# pyLDAvis.save_html(pic_data,'tryLDA.html')

	topic_fit = list()
	fit_topic = LDA.transform(tfidf)
	for aa in range(fit_topic.shape[0]):
		topic_fit_in = fit_topic[aa].argmax()
		topic_fit.append(topic_fit_in)
	tmp = {"cut_result":cut_list,"topic":topic_fit}
	LDA_result = pd.DataFrame(tmp)

	return LDA_result

def match_check(tp1,tp2):
	counter = 0
	tmp_match = list()
	for th1 in tp1:
		for th2 in tp2:
			if th1 == th2:
				counter += 1
				tmp_match.append(th1)
	return round(float(counter/len(tp2)),3),tmp_match

def LDA_matchloop(cut_Result1,cut_Result2,loop_time = 100,topic_count = 10,word_count = 10): #data wanna check = cut_Result1 ; check bennchmark = cut_Result2
	from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
	from sklearn.decomposition import LatentDirichletAllocation
	match_rate = list()
	match_list = list()
	for times in range(0,loop_time):
		cut_list = cut_Result1.values.tolist()
		vectorizer = CountVectorizer()
		X = vectorizer.fit_transform(cut_list)
		transformer = TfidfTransformer()
		tfidf = transformer.fit_transform(X)
		del cut_list
		LDA = LatentDirichletAllocation(n_components = topic_count, random_state = None)
		LDA.fit(tfidf)
		topic_word = list()
		for counter,topic in enumerate(LDA.components_):
			out_topic = [vectorizer.get_feature_names()[index] for index in topic.argsort()[word_count*-1:]]
			for things in out_topic:
				topic_word.append(things)
		del LDA
		del out_topic
		unique_topic = set(topic_word)
		del topic_word
		topic_words = list(unique_topic)
		del unique_topic

		cut_list2 = cut_Result2.values.tolist()
		vectorizer2 = CountVectorizer()
		X2 = vectorizer2.fit_transform(cut_list2)
		transformer2 = TfidfTransformer()
		tfidf2 = transformer2.fit_transform(X2)
		del cut_list2
		LDA2 = LatentDirichletAllocation(n_components = topic_count, random_state = None)
		LDA2.fit(tfidf2)
		topic_word2 = list()
		for counter2,topic2 in enumerate(LDA2.components_):
			out_topic2 = [vectorizer2.get_feature_names()[index2]for index2 in topic2.argsort()[word_count*-1:]]
			for things2 in out_topic2:
				topic_word2.append(things2)
		del LDA2
		del out_topic2
		unique_topic2 = set(topic_word2)
		del topic_word2
		topic_words2 = list(unique_topic2)
		del unique_topic2

		tmp_match_rate, tmp_match_list = match_check(topic_words,topic_words2)
		match_rate.append(tmp_match_rate)
		match_list.append(tmp_match_list)
		del tmp_match_list
		del tmp_match_rate
		print("Done loop No. ",times+1)

	return match_rate,match_list





