import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba,re
from tqdm import tqdm



def sortdate(t_list,period = 'M'):
	t_list['Date'] = pd.to_datetime(t_list['created_time_taipei']).dt.date
	t_list['YYYY-MM'] = pd.to_datetime(t_list['Date']).dt.to_period(period)
	print("Day sorted")
	return t_list
	
def sortvalue(tmp_list,sorting_method,ascend = False):
	if sorting_method == 'Like':
		tmp_list = tmp_list.sort_values(by = ['like_count'],ascending = ascend)
	elif sorting_method == 'Share':
		tmp_list = tmp_list.sort_values(by = ['share_count'],ascending = ascend)
	return tmp_list

def cutfunc(target):
	jieba.load_userdict("dict.txt")
	punctuation = " //，：:""()\n!！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏""-<>#。！⋯.➡?=&▶_%♀!❗🎉⏰💪🔥⁉❓"
	re_punctuation = "[{}] ".format(punctuation)

	target = target.replace(np.nan,'',regex = True)
	target_text = list(target['message'])
	jieba_docs = pd.DataFrame(columns=['jieba_results'])
	jieba_docs['jieba_results'] = jieba_docs['jieba_results'].astype('str')

	for i in range(0,len(n_target)):
		cutword = jieba.lcut(target_text[i],cut_all = False)
		text = ''
		for words in cutword:
			text = str(words) + '' +text

		text = re.sub(re_punctuation, "", text)
    	text = re.sub(r'[0-9]','',text)
   		text = re.sub(r'[a-zA-Z]','',text)
   		s = pd.Series({'jieba_results': text})
   		jieba_docs = jieba_docs.append(s,ignore_index = True)
   	print("↓↓↓↓↓↓↓↓↓↓↓Cutting function Done !!↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
   	print(jieba_docs,'\n')
   	return jieba_docs
		
def drawpic(target_list,sort_method):
	info_count = target_list[str(sort_method)]
	info_count.plot.hist(grid = True, bins = 100, rwidth = 0.9,color = 'Blue')
	plt.grid(axis='y', alpha=0.75)
	

def main(target_name,sort_method):
	pd.set_option('display.max_colwidth', None)
	fb = pd.read_csv('nysu_10902_2019_research_right.csv')
	politics = pd.read_csv('9th_legislator_promise.csv')
	target = fb[fb.page_name == target_name]
	n_target = sortdate(target)
	n_target = sortvalue(n_target,sort_method)
	n_target = n_target.reset_index(inplace = True)
	print(n_target)
	cutting_words = cutfunc(n_target)
	n_target['jieba_Result'] = cutting_words
	n_info = n_target[['new_date','message','jieba_results',str(sort_method)]]
	print(n_info)
	drawpic(n_info,sort_method)