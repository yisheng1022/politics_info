import tensorflow as tf
# import keras
import pandas as pd
import numpy as np
import gc
import math,threading,re,os,time


def read_in_fbcsv(fbname):
	fb = pd.read_csv('nysu_10902_2019_research_right.csv')
	target_fb = fb[fb.page_name == fbname]
	return target_fb

def read_in_procsv(proname):
	promise = pd.read_csv('9th_legislator_promise.csv')
	# print(promise)
	target_pro = promise[promise.姓名 == proname]
	# print(target_pro)
	return target_pro

def clean_fb_data(infb_csv):
	infb_csv["Date"] = pd.to_datetime(infb_csv['created_time_taipei']).dt.date
	infb_csv["YYYY-MM"] = pd.to_datetime(infb_csv["Date"]).dt.to_period('M')
	return infb_csv

def stopwordlist():
	stopwords = [line.strip() for line in open('stop_word_all.txt', 'r', encoding='utf-8').readlines()]
	return stopwords

def do_NER(ws_result):
	from ckiptagger import POS,NER
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	pos = POS("./data",disable_cuda=False)
	ner = NER("./data",disable_cuda=False)
	pos_result = pos(ws_result)
	ner_result = ner(ws_result,pos_result)
	class_l = []
	ner_l = []
	# print(ner_result)
	for a in range(len(ner_result[0])):
		popobj = ner_result[0].pop()
		class_l.append(popobj[2])
		ner_l.append(popobj[3])
	ner_output = pd.DataFrame({"Class":list(class_l), "NER":list(ner_l)})
	ner_output.drop_duplicates(inplace = True)
	tmp2 = ner_output.loc[ner_output['Class'].isin(['LOC','PERSON', 'ORG', 'LAW', 'EVENT','GPE','NORP'])]
	tmp3 = tmp2[tmp2['NER'].map(len) >= 2]
	del tmp2
	clean_NER = tmp3.sort_values(by = ['Class'])
	del tmp3
	if os.path.isfile("pol_NER.csv"):
		clean_NER.to_csv("pol_NER.csv",mode = 'a+',header = False, index = False)
		return
	else:
		clean_NER.to_csv("pol_NER.csv",mode = 'a+',header = True, index = False)
		return


def cut_func(input_data,data_col,name):
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	from ckiptagger import data_utils, construct_dictionary, WS
	User_Dict = {}
	with open("dict.txt","r",encoding = 'utf-8') as USDic:
		for tmpwords in USDic:
			words = tmpwords.strip().split(" ")
			if len(words) > 1:
				User_Dict[words[0]] = words[1]
			else:
				User_Dict[words[0]] = 10
	dictionary = construct_dictionary(User_Dict)
	ws = WS("./data",disable_cuda=False)
	# pos = POS("/data")
	# ner = NER("/data")
	print(input_data)

	punctuation = " 的也//，：:""()\n!！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏""<->#。！⋯.➡?=&▶_%♀!❗🎉⏰💪🔥⁉❓"
	re_punctuation = "[{}] ".format(punctuation)
	input_data = input_data.replace(np.nan,'',regex = True)
	tmp_fbtext = list(input_data[data_col])
	stopwordslist = stopwordlist()
	ckip_pd = pd.DataFrame(columns = ['CKIP_Result'])
	ckip_pd['CKIP_Result'] = ckip_pd['CKIP_Result'].astype('str')
	print("Total Data to process: ",len(tmp_fbtext),'\n','----------------')
	counter = 1
	tmp_things = []
	for things in tmp_fbtext:
		print("Now processing:", name," No.",counter)
		tmp_things.append(things)
		ckip_cut = ws(tmp_things,sentence_segmentation=True,segment_delimiter_set = {",", "。", ":", "?", "!", ";", "、"},coerce_dictionary = dictionary)
		text = ''
		tmp_things.clear()
		ner_thread = threading.Thread(target = do_NER, args = (ckip_cut,))
		ner_thread.start()
		for cutted in ckip_cut:
			if cutted not in stopwordslist:
				text = str(cutted) + " " + text
		text = re.sub(r'[0-9]','',text)
		text = re.sub(r'[a-zA-Z]','',text)
		text = re.sub(r'[^\w\s]','',text)
		text = re.sub(re_punctuation,'',text)
		tmp = pd.Series({'CKIP_Result' : text})
		ckip_pd = ckip_pd.append(tmp,ignore_index = True)
		ner_thread.join()
		counter += 1
	return ckip_pd


def main(fb_name,colname):
	# tmpfb_csv = read_in_fbcsv(fb_name)
	tmppromise_csv = read_in_procsv(fb_name)
	# clean_fb = clean_fb_data(tmppromise_csv)
	print(tmppromise_csv)
	clean_fb = tmppromise_csv.reset_index(inplace = False)
	print(clean_fb)
	cut_fb = cut_func(clean_fb,colname,fb_name)
	fb_after_ckip = clean_fb[["縣市","選區","姓名","在任狀態","政黨","政見"]] #Need to be changed if doing other data "page_name","Date","message","like_count","share_count","permalink"
	fb_after_ckip["ckip_result"] = cut_fb
	if os.path.isfile("PRO_withckip.csv"):
		fb_after_ckip.to_csv("PRO_withckip.csv",mode = 'a+',header = False, index = False)
	else:
		fb_after_ckip.to_csv("PRO_withckip.csv",mode = 'a+',header = True, index = False)


name_list = ['江啟臣','顏寬恒','沈智慧','何欣純','洪慈庸','張廖萬堅','黃國書','蔡其昌','蔡 易 餘','許 淑 華','柯呈枋','黃秀芳','徐志榮']
for names in name_list:
	main(names,'政見')
# from ckiptagger import data_utils
# data_utils.download_data_gdown("./")
# data_utils.download_data_url("./")
# FB: '江啟臣','顏寬恒','沈智慧','何欣純','洪慈庸','堅持．張廖萬堅','黃國書','蔡其昌','蔡易餘 家己人','許淑華','柯呈枋','黃秀芳','徐志榮'
# 政見: '江啟臣','顏寬恒','沈智慧','何欣純','洪慈庸','張廖萬堅','黃國書','蔡其昌','蔡 易 餘','許 淑 華','柯呈枋','黃秀芳','徐志榮'