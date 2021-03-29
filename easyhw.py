import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_colwidth', None)
import random
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy,statistics
# class politics_info(object):
# 	"""docstring for politics_info"""
# 	def __init__(self, d_file, p_file, name):
# 		super(politics_info, self).__init__()
# 		self.arg = arg
		
def normalshow(datafile,promisefile,name0,period = 'M'):
	fb = pd.read_csv(datafile)
	politics = pd.read_csv(promisefile)
	print(fb.head())
	targetperson = fb[fb.page_name == name0]
	print(targetperson.head())
	targetperson['new_date'] = pd.to_datetime(targetperson['created_time_taipei']).dt.date
	targetperson['month_year'] = pd.to_datetime(targetperson['new_date']).dt.to_period(period)
	results = targetperson.groupby('month_year').sum()
	results.index.name = 'YYYY-MM'
	results.reset_index(inplace=True)
	print(results)

	#畫出逐月Like-share散布圖
	pic0 = plt.figure(figsize = (20,10))
	myfont = FontProperties(fname='C:\\Windows\\Fonts\\msjh.ttc') #字型設定須注意
	for aa in range(0,len(results)):
	    plt.scatter(results.loc[aa,['like_count']],results.loc[aa,['share_count']])
	plt.legend(results['YYYY-MM'],loc = [1,0])
	plt.title(str(name0)+" like-share scaater plot",fontproperties=myfont)
	plt.xlabel("Like")
	plt.ylabel("Share")

	#畫出趨勢線
	z = numpy.polyfit(results['like_count'], results['share_count'] , 1)
	p = numpy.poly1d(z)
	plt.plot(results['like_count'],p(results['like_count']),'black')
	plt.savefig(str(name0) + '_all_scatter.png')

def nopicshow(datafile,promisefile,name0,period):
	print('↓↓↓↓↓↓View the specific time info.↓↓↓↓↓↓')
	fb = pd.read_csv(datafile)
	politics = pd.read_csv(promisefile)
	targetperson = fb[fb.page_name == name0]
	targetperson['new_date'] = pd.to_datetime(targetperson['created_time_taipei']).dt.date
	targetperson['month_year'] = pd.to_datetime(targetperson['new_date']).dt.to_period(period)
	results = targetperson.groupby('month_year').sum()
	results.index.name = 'YYYY-MM'
	results.reset_index(inplace=True)
	print(statistics.mean(results['like_count']))
	return targetperson

def specialissue(datafile,promisefile,name0,specialmonth,period = 'M'):
	#檢視該指定時間段的資訊概要
	target = nopicshow(datafile,promisefile,name0,period)
	nsp = target[target.month_year == specialmonth]
	nsp_sum = nsp.groupby('new_date').sum()
	nsp_sum.index.name = 'Year_Month'
	nsp_sum.reset_index(inplace = True)
	print(nsp_sum)

	#畫出該指定時間段Like-Share散布圖
	pic1 = plt.figure(figsize = (20,10))
	for bb in range(0,len(nsp_sum)):
		plt.scatter(nsp_sum.loc[bb,['like_count']],nsp_sum.loc[bb,['share_count']])
	myfont = FontProperties(fname='C:\\Windows\\Fonts\\msjh.ttc') #字型設定須注意
	plt.legend(nsp_sum['Year_Month'],loc = [1,0])
	plt.title(str(name0) + "'s like-share scaater plot",fontproperties=myfont)
	plt.xlabel("Like")
	plt.ylabel("Share")
	pic1.savefig(str(name0) + '_special_scatter.png')

	#畫出當月折線圖
	pic2 = plt.figure(figsize = (20,10))
	plt.plot(nsp_sum['Year_Month'],nsp_sum['like_count'],'o-')
	plt.plot(nsp_sum['Year_Month'],nsp_sum['share_count'],'s-')
	plt.title("Like-Share",fontproperties=myfont)
	pic2.savefig(str(name0) + '.png')

def specialday(datafile,promisefile,name0,specialdate,period = 'M'):
	pd.set_option('display.max_colwidth', None)
	target = nopicshow(datafile,promisefile,name0,period)
	target['new_date'] = target['new_date'].map(str)
	special_share = target[target['new_date'] == specialdate]
	special_share.to_clipboard(sep = ',')
	print(special_share)
	a = target.loc[13518,['permalink']]
	print(a[0])
	print("Please paste ur result to notebook !")
