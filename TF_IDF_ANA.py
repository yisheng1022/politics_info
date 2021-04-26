import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba,re,threading,time
from wordcloud import WordCloud
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation



def sortdate(t_list,period = 'M'):
	t_list['Date'] = pd.to_datetime(t_list['created_time_taipei']).dt.date
	t_list['YYYY-MM'] = pd.to_datetime(t_list['Date']).dt.to_period(period)
	# print(t_list)
	return t_list
	
def sortvalue(tmp_list,sorting_method,ascend = False):
	if sorting_method == 'like':
		tmp_list = tmp_list.sort_values(by = ['like_count'],ascending = ascend)
	elif sorting_method == 'share':
		tmp_list = tmp_list.sort_values(by = ['share_count'],ascending = ascend)
	# print(tmp_list)
	return tmp_list

def cutfunc(target):
	jieba.load_userdict("dict.txt")
	punctuation = " //，：:""()\n!！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏""<->#。！⋯.➡?=&▶_%♀!❗🎉⏰💪🔥⁉❓"
	re_punctuation = "[{}] ".format(punctuation)

	target = target.replace(np.nan,'',regex = True)
	target_text = list(target['message'])
	jieba_docs = pd.DataFrame(columns=['jieba_results'])
	jieba_docs['jieba_results'] = jieba_docs['jieba_results'].astype('str')

	for i in range(0,len(target)):
		cutword = jieba.lcut(target_text[i],cut_all = False)
		text = ''
		for words in cutword:
			text = str(words) + ' ' +text

		text = re.sub(re_punctuation, "", text)
		text = re.sub(r'[0-9]','',text)
		text = re.sub(r'[a-zA-Z]','',text)
		s = pd.Series({'jieba_results': text})
		jieba_docs = jieba_docs.append(s,ignore_index = True)
		# jieba_docs = jieba_docs.append(,ignore_index = True)
	print("↓↓↓↓↓↓↓↓↓↓↓Cutting function Done !!↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
	return jieba_docs
		
def drawpic(name,target_list,sort_method):
	info_count = target_list[str(sort_method)+'_count']#str(sort_method)+
	print(info_count)
	tmp_pic = info_count.hist(grid = True, bins = 100, rwidth = 0.9 , color = '#607c8e')
	tmp_pic.figure.savefig(name+'_'+sort_method+'.png')

def count_frq(t_info):
	jieba_list = t_info['jieba_Result'].values.tolist()
	vectorizer = CountVectorizer()  
	X = vectorizer.fit_transform(jieba_list)
	word = vectorizer.get_feature_names()
	return X

def do_tf_idf(t_info):
	X = count_frq(t_info)
	transform = TfidfTransformer()
	tfidf = transform.fit_transform(X)
	return tfidf

def record_topic(LDA,t_info,name):
	f = open(str(name)+"FB_Topic.txt",'w+')
	vectorizer = CountVectorizer()
	jieba_list = t_info['jieba_Result'].values.tolist()
	vectorizer = CountVectorizer()  
	X = vectorizer.fit_transform(jieba_list)
	word = vectorizer.get_feature_names()
	for x,topic in enumerate(LDA.components_):
		f.write("TOP 10 WORDS PER TOPIC #{}\n".format(x))
		f.write(str([vectorizer.get_feature_names()[index] for index in topic.argsort()[-10:]]))
		f.write('\n')
	f.close()
	print("Topic has been record.")

def LDA_Topic_tfidf(tf_idf,topicnum,t_info,target_name):
	LDA = LatentDirichletAllocation(n_components = int(topicnum), random_state = None)
	LDA.fit(tf_idf)
	jieba_list = t_info['jieba_Result'].values.tolist()
	vectorizer = CountVectorizer()  
	X = vectorizer.fit_transform(jieba_list)
	word = vectorizer.get_feature_names()
	threading.Thread(target = record_topic, name = 'Topic record',args =(LDA.fit(tf_idf),t_info,target_name)).start()
	print('Active thread count: ', threading.active_count())
	print('Now using thread: ', threading.current_thread())
	print('Using thread name: ', threading.current_thread().name)
	print('Active thread info: ', threading.enumerate)
	time.sleep(3)

	for x,topic in enumerate(LDA.components_):
		print("TOP 10 WORDS PER TOPIC #{}".format(x))
		print([vectorizer.get_feature_names()[index] for index in topic.argsort()[-10:]])
    
def makecloud(target_message,name):
	font = 'C:\\Users\\ianle\\AppData\\Local\\Microsoft\\Windows\\Fonts\\JasonHandwriting1.ttf'
	text = ''
	for things in target_message:
		text = text + things
		text = text + ' '
	cloud = WordCloud(font_path = font).generate(text)
	cloud.to_file(str(name)+'_output.png')
	
def politics_func(politics,target_name,topic_num):
	politics_person = politics[politics.姓名 == target_name].政見.to_string()
	punctuation = " //，：:""()\n!！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏""<->#。！⋯.➡?=&▶_%♀!❗🎉⏰💪🔥⁉❓"
	re_punctuation = "[{}] ".format(punctuation)
	politics_person = re.sub(re_punctuation, "", politics_person)
	politics_person = re.sub(r'[0-9]','',politics_person)
	politics_person = re.sub(r'[a-zA-Z]','',politics_person)
	jieba.load_userdict("dict.txt")
	words = jieba.lcut(politics_person, cut_all = False)
	vectorizer = CountVectorizer()  
	X = vectorizer.fit_transform(words)  
	word = vectorizer.get_feature_names()
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(X)
	LDA = LatentDirichletAllocation(n_components=int(topic_num), random_state=None)
	LDA.fit(tfidf)

	f = open(target_name+'_promise.txt','w+')
	for i,topic in enumerate(LDA.components_):
		f.write("TOP 10 WORDS PER TOPIC #{}\n".format(i))
		f.write(str([vectorizer.get_feature_names()[index] for index in topic.argsort()[-10:]]))
		f.write('\n')
	f.close()
	print("Politics promise done !")

def main(target_name,sort_method):
	pd.set_option('display.max_colwidth', None)
	pd.options.mode.chained_assignment = None  # default='warn'
	fb = pd.read_csv('nysu_10902_2019_research_right.csv')
	politics = pd.read_csv('9th_legislator_promise.csv')
	target = fb[fb.page_name == target_name]
	n_target = sortdate(target)
	# print(n_target)
	n_target = sortvalue(n_target,sort_method)
	# print(n_target)
	n_target.reset_index(inplace = True)
	# print(n_target)
	cutting_words = cutfunc(n_target)
	n_target['jieba_Result'] = cutting_words #n_target contain all info
	print(n_target)
	n_info = n_target[['Date','message','jieba_Result',str(sort_method)+'_count']] #n_info contain only brief info
	print(n_info)
	drawpic(target_name,n_info,sort_method)
	print("Plz choose your target range !")
	upperrange = input("Max：")
	lowerrange = input("Min：")
	target_info = n_info[(n_info[str(sort_method)+'_count']<=int(upperrange)) & (n_info[str(sort_method)+'_count']>=int(lowerrange))]
	tf_idf = do_tf_idf(target_info)
	topic_num = input("Topic count :")
	threading.Thread(target = politics_func, name = 'Promise jieba',args =(politics,target_name,topic_num,)).start()
	print('Active thread count: ', threading.active_count())
	print('Now using thread: ', threading.current_thread())
	print('Using thread name: ', threading.current_thread().name)
	print('Active thread info: ', threading.enumerate)
	time.sleep(3)
	LDA_Topic_tfidf(tf_idf,topic_num,target_info,target_name)
	makecloud(target_info['jieba_Result'],target_name)

main("江啟臣",'like')