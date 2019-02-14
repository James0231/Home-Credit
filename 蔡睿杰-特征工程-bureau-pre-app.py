# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 18:02:19 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

FILE_NAME = 'C:/Users/Administrator/Desktop/Home-credit-default-risk/bureau.csv'
df = pd.read_csv(FILE_NAME,encoding = 'utf-8')
df = df.drop(['SK_ID_BUREAU'],axis=1)
df_final = pd.DataFrame()
df_final_previous_application = pd.DataFrame()

#----------------统计第一个变量-----------------
df['CREDIT_ACTIVE'].where(df['CREDIT_ACTIVE'] == 'Active', other = 0, inplace = True)
df['CREDIT_ACTIVE'].where(df['CREDIT_ACTIVE'] == 0, other = 1, inplace = True)
df['CREDIT_ACTIVE'] = pd.to_numeric(df['CREDIT_ACTIVE'])
CREDIT_ACTIVE_GROUP = df['CREDIT_ACTIVE'].groupby(df['SK_ID_CURR'])
CREDIT_ACTIVE_RATE = pd.DataFrame(CREDIT_ACTIVE_GROUP.mean())
df_final = pd.concat([df_final,pd.DataFrame(CREDIT_ACTIVE_RATE.index)],axis=1)
CREDIT_ACTIVE_RATE = CREDIT_ACTIVE_RATE['CREDIT_ACTIVE'].tolist()  #重新设置索引
CREDIT_ACTIVE_RATE = pd.DataFrame(CREDIT_ACTIVE_RATE)
df_final = pd.concat([df_final,CREDIT_ACTIVE_RATE],axis=1)
df_final.rename(columns={0:'CREDIT_ACTIVE_RATE'},inplace = True)


#----------------统计第二个变量-----------------
df['CREDIT_CURRENCY'].where(df['CREDIT_CURRENCY'] == 'currency 1', other = 1, inplace = True)
df['CREDIT_CURRENCY'].where(df['CREDIT_CURRENCY'] == 1, other = 0 , inplace = True)
df['CREDIT_CURRENCY'] = pd.to_numeric(df['CREDIT_CURRENCY'])
CREDIT_CURRENCY_GROUP = df['CREDIT_CURRENCY'].groupby(df['SK_ID_CURR'])
NONE_CURRENCY_1_RATE = pd.DataFrame(CREDIT_CURRENCY_GROUP.mean())
NONE_CURRENCY_1_RATE = NONE_CURRENCY_1_RATE['CREDIT_CURRENCY'].tolist()  
NONE_CURRENCY_1_RATE = pd.DataFrame(NONE_CURRENCY_1_RATE)
df_final = pd.concat([df_final,NONE_CURRENCY_1_RATE],axis=1)
df_final.rename(columns={0:'NONE_CURRENCY_1_RATE'},inplace = True)


#----------------统计第三个变量-----------------
df['DAYS_CREDIT'] = df['DAYS_CREDIT'].fillna(0)
DAYS_CREDIT_GROUP = df['DAYS_CREDIT'].groupby(df['SK_ID_CURR'])
DAYS_CREDIT_MEAN = pd.DataFrame(DAYS_CREDIT_GROUP.mean())
DAYS_CREDIT_MEAN = DAYS_CREDIT_MEAN['DAYS_CREDIT'].tolist()  
DAYS_CREDIT_MEAN = pd.DataFrame(DAYS_CREDIT_MEAN)
MAX = np.max(DAYS_CREDIT_MEAN)
MIN = np.min(DAYS_CREDIT_MEAN)
DAYS_CREDIT_MEAN = (DAYS_CREDIT_MEAN-MIN)/(MAX-MIN)
df_final = pd.concat([df_final,DAYS_CREDIT_MEAN],axis=1)
df_final.rename(columns={0:'DAYS_CREDIT_MEAN'},inplace = True)

DAYS_CREDIT_MEDIAN = pd.DataFrame(DAYS_CREDIT_GROUP.median())
DAYS_CREDIT_MEDIAN = DAYS_CREDIT_MEDIAN['DAYS_CREDIT'].tolist()  
DAYS_CREDIT_MEDIAN = pd.DataFrame(DAYS_CREDIT_MEDIAN)
MAX = np.max(DAYS_CREDIT_MEDIAN)
MIN = np.min(DAYS_CREDIT_MEDIAN)
DAYS_CREDIT_MEDIAN = (DAYS_CREDIT_MEDIAN-MIN)/(MAX-MIN)
df_final = pd.concat([df_final,DAYS_CREDIT_MEDIAN],axis=1)
df_final.rename(columns={0:'DAYS_CREDIT_MEDIAN'},inplace = True)

DAYS_CREDIT_MAX = pd.DataFrame(DAYS_CREDIT_GROUP.max())
DAYS_CREDIT_MAX = DAYS_CREDIT_MAX['DAYS_CREDIT'].tolist()  
DAYS_CREDIT_MAX = pd.DataFrame(DAYS_CREDIT_MAX)
MAX = np.max(DAYS_CREDIT_MAX)
MIN = np.min(DAYS_CREDIT_MAX)
DAYS_CREDIT_MAX = (DAYS_CREDIT_MAX-MIN)/(MAX-MIN)
df_final = pd.concat([df_final,DAYS_CREDIT_MAX],axis=1)
df_final.rename(columns={0:'DAYS_CREDIT_MAX'},inplace = True)

DAYS_CREDIT_MIN = pd.DataFrame(DAYS_CREDIT_GROUP.min())
DAYS_CREDIT_MIN = DAYS_CREDIT_MIN['DAYS_CREDIT'].tolist()  
DAYS_CREDIT_MIN = pd.DataFrame(DAYS_CREDIT_MIN)
MAX = np.max(DAYS_CREDIT_MIN)
MIN = np.min(DAYS_CREDIT_MIN)
DAYS_CREDIT_MIN = (DAYS_CREDIT_MIN-MIN)/(MAX-MIN)
df_final = pd.concat([df_final,DAYS_CREDIT_MIN],axis=1)
df_final.rename(columns={0:'DAYS_CREDIT_MIN'},inplace = True)


#----------------统计第四个变量-----------------
df['CREDIT_DAY_OVERDUE'].where(df['CREDIT_DAY_OVERDUE'] <1 , other = 0, inplace = True)
df['CREDIT_DAY_OVERDUE'] = pd.to_numeric(df['CREDIT_DAY_OVERDUE'])
CREDIT_DAY_OVERDUE_GROUP = df['CREDIT_DAY_OVERDUE'].groupby(df['SK_ID_CURR'])
CREDIT_DAY_OVERDUE_1_RATE = pd.DataFrame(CREDIT_DAY_OVERDUE_GROUP.mean())
CREDIT_DAY_OVERDUE_1_RATE = CREDIT_DAY_OVERDUE_1_RATE['CREDIT_DAY_OVERDUE'].tolist()  
CREDIT_DAY_OVERDUE_1_RATE = pd.DataFrame(CREDIT_DAY_OVERDUE_1_RATE)
df_final = pd.concat([df_final,CREDIT_DAY_OVERDUE_1_RATE],axis=1)
df_final.rename(columns={0:'CREDIT_DAY_OVERDUE_1_RATE'},inplace = True)

#----------------统计第五个变量-----------------
df['DAYS_CREDIT_ENDDATE'] = df['DAYS_CREDIT_ENDDATE'].fillna(0)
df['DAYS_CREDIT_ENDDATE'].where(df['DAYS_CREDIT_ENDDATE'] >= 0, other = 1, inplace = True)
df['DAYS_CREDIT_ENDDATE'].where(df['DAYS_CREDIT_ENDDATE'] == 1, other = 0, inplace = True)
DAYS_CREDIT_ENDDATE_GROUP = df['DAYS_CREDIT_ENDDATE'].groupby(df['SK_ID_CURR'])
DAYS_CREDIT_ENDDATE_1_RATE = pd.DataFrame(DAYS_CREDIT_ENDDATE_GROUP.mean())
DAYS_CREDIT_ENDDATE_1_RATE = DAYS_CREDIT_ENDDATE_1_RATE['DAYS_CREDIT_ENDDATE'].tolist()  
DAYS_CREDIT_ENDDATE_1_RATE = pd.DataFrame(DAYS_CREDIT_ENDDATE_1_RATE)
df_final = pd.concat([df_final,DAYS_CREDIT_ENDDATE_1_RATE],axis=1)
df_final.rename(columns={0:'DAYS_CREDIT_ENDDATE_1_RATE'},inplace = True)

#----------------统计第六个变量-----------------
df['AMT_CREDIT_MAX_OVERDUE'] = df['AMT_CREDIT_MAX_OVERDUE'].fillna(0)
DAYS_CREDIT_GROUP = df['AMT_CREDIT_MAX_OVERDUE'].groupby(df['SK_ID_CURR'])
AMT_CREDIT_MAX_OVERDUE_MAX = pd.DataFrame(DAYS_CREDIT_GROUP.max())
AMT_CREDIT_MAX_OVERDUE_MAX = AMT_CREDIT_MAX_OVERDUE_MAX['AMT_CREDIT_MAX_OVERDUE'].tolist()  
AMT_CREDIT_MAX_OVERDUE_MAX = pd.DataFrame(AMT_CREDIT_MAX_OVERDUE_MAX)
MAX = np.max(AMT_CREDIT_MAX_OVERDUE_MAX)
MIN = np.min(AMT_CREDIT_MAX_OVERDUE_MAX)
AMT_CREDIT_MAX_OVERDUE_MAX = (AMT_CREDIT_MAX_OVERDUE_MAX-MIN)/(MAX-MIN)
df_final = pd.concat([df_final,AMT_CREDIT_MAX_OVERDUE_MAX],axis=1)
df_final.rename(columns={0:'AMT_CREDIT_MAX_OVERDUE'},inplace = True)

#----------------统计第七个变量-----------------
df['CNT_CREDIT_PROLONG'] = df['CNT_CREDIT_PROLONG'].fillna(0)
CNT_CREDIT_PROLONG_GROUP = df['CNT_CREDIT_PROLONG'].groupby(df['SK_ID_CURR'])
CNT_CREDIT_PROLONG_TIMES = pd.DataFrame(CNT_CREDIT_PROLONG_GROUP.max())
CNT_CREDIT_PROLONG_TIMES = CNT_CREDIT_PROLONG_TIMES['CNT_CREDIT_PROLONG'].tolist()  
CNT_CREDIT_PROLONG_TIMES = pd.DataFrame(CNT_CREDIT_PROLONG_TIMES)
CNT_CREDIT_PROLONG_TIMES.where(CNT_CREDIT_PROLONG_TIMES <2, other = 2,inplace = True)
encoder = LabelBinarizer()
CNT_CREDIT_PROLONG_TIMES = encoder.fit_transform(CNT_CREDIT_PROLONG_TIMES)
CNT_CREDIT_PROLONG_TIMES = pd.DataFrame(CNT_CREDIT_PROLONG_TIMES)
df_final = pd.concat([df_final,CNT_CREDIT_PROLONG_TIMES],axis=1)
df_final.rename(columns={0:'CNT_CREDIT_PROLONG_0',
                         1:'CNT_CREDIT_PROLONG_1',
                         2:'CNT_CREDIT_PROLONG_2ORMORE'},inplace = True)

df_final.to_csv('C:/Users/Administrator/Desktop/蔡睿杰-bureau-特征工程.csv',index = False)


#----------------统计第八个变量-----------------previous_application中的CHANNEL_TYPE
FILE_NAME = 'C:/Users/Administrator/Desktop/Home-credit-default-risk/previous_application.csv'
df1 = pd.read_csv(FILE_NAME)
df_COPY = df1.copy()
df_COPY['CHANNEL_TYPE'].where(df_COPY['CHANNEL_TYPE'] == 'Country-wide',other = 0,inplace = True)
df_COPY['CHANNEL_TYPE'].where(df_COPY['CHANNEL_TYPE'] == 0, other = 1,inplace = True)
df_COPY['CHANNEL_TYPE'] = pd.to_numeric(df_COPY['CHANNEL_TYPE'])
CHANNEL_TYPE_GROUP = df_COPY['CHANNEL_TYPE'].groupby(df_COPY['SK_ID_CURR'])
CHANNEL_TYPE_COUNTRY_WIDE = pd.DataFrame(CHANNEL_TYPE_GROUP.mean())
df_final_previous_application = pd.concat([df_final_previous_application,pd.DataFrame(CHANNEL_TYPE_COUNTRY_WIDE.index)],
                                          axis =1)
CHANNEL_TYPE_COUNTRY_WIDE = CHANNEL_TYPE_COUNTRY_WIDE['CHANNEL_TYPE'].tolist()  
CHANNEL_TYPE_COUNTRY_WIDE = pd.DataFrame(CHANNEL_TYPE_COUNTRY_WIDE)
df_final_previous_application = pd.concat([df_final_previous_application,CHANNEL_TYPE_COUNTRY_WIDE],axis=1)
df_final_previous_application.rename(columns={0:'CHANNEL_TYPE_COUNTRY_WIDE'},inplace = True)

df_COPY = df1.copy()
df_COPY['CHANNEL_TYPE'].where(df_COPY['CHANNEL_TYPE'] == 'Credit and cash offices',other = 0,inplace = True)
df_COPY['CHANNEL_TYPE'].where(df_COPY['CHANNEL_TYPE'] == 0, other = 1,inplace = True)
df_COPY['CHANNEL_TYPE'] = pd.to_numeric(df_COPY['CHANNEL_TYPE'])
CHANNEL_TYPE_GROUP = df_COPY['CHANNEL_TYPE'].groupby(df_COPY['SK_ID_CURR'])
CHANNEL_TYPE_CREDIT = pd.DataFrame(CHANNEL_TYPE_GROUP.mean())
CHANNEL_TYPE_CREDIT = CHANNEL_TYPE_CREDIT['CHANNEL_TYPE'].tolist()  
CHANNEL_TYPE_CREDIT = pd.DataFrame(CHANNEL_TYPE_CREDIT)
df_final_previous_application = pd.concat([df_final_previous_application,CHANNEL_TYPE_CREDIT],axis=1)
df_final_previous_application.rename(columns={0:'CHANNEL_TYPE_CREDIT'},inplace = True)

df_final_previous_application.to_csv('C:/Users/Administrator/Desktop/蔡睿杰-previous-application-特征工程.csv',index = False)
