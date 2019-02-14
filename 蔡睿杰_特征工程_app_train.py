# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:32:47 2018

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer


df = pd.read_csv('C:/Users/Administrator/Desktop/application_train_feature_engineering.csv')
SK_ID_Current = df['SK_ID_CURR']
SK_ID_Current = pd.DataFrame(SK_ID_Current)
df = df.drop(['SK_ID_CURR'],axis=1)
df = df.drop(['TARGET'],axis =1) #标签
df_first_31 = df.ix[:,0:31]

df_final = pd.DataFrame() #用于储存最后的数据集
df_final = pd.concat([df_final,SK_ID_Current],axis=1)

#-------------标记第一个变量------------1：贷款类型为非现金；0：贷款类型为现金
encoder = LabelBinarizer()
Name_Contact = df_first_31['NAME_CONTRACT_TYPE']
Name_Contact_1hot = encoder.fit_transform(Name_Contact)
Name_Contact_1hot = pd.DataFrame(Name_Contact_1hot)
df_final = pd.concat([df_final,Name_Contact_1hot],axis=1)
df_final.rename(columns={0:'NAME_CONTRACT_TYPE'}, inplace = True)


#-------------标记第二个变量------------1：男性；0：女性
encoder = LabelEncoder()
CODE_GENDER = df_first_31['CODE_GENDER']
CODE_GENDER_1hot = encoder.fit_transform(CODE_GENDER)
CODE_GENDER_1hot = pd.DataFrame(CODE_GENDER_1hot)
df_final = pd.concat([df_final,CODE_GENDER_1hot],axis=1)
df_final.rename(columns={0:'CODE_GENDER'}, inplace = True)


#-------------标记第三个变量------------1：客户有车；0：客户没车
encoder = LabelEncoder()
FLAG_OWN_CAR = df_first_31['FLAG_OWN_CAR']
FLAG_OWN_CAR_1hot = encoder.fit_transform(FLAG_OWN_CAR)
FLAG_OWN_CAR_1hot = pd.DataFrame(FLAG_OWN_CAR_1hot)
df_final = pd.concat([df_final,FLAG_OWN_CAR_1hot],axis=1)
df_final.rename(columns={0:'FLAG_OWN_CAR'}, inplace = True)

#-------------标记第四个变量------------1：客户拥有不动产；0：客户拥有动产
encoder = LabelEncoder()
FLAG_OWN_REALTY = df_first_31['FLAG_OWN_REALTY']
FLAG_OWN_REALTY_1hot = encoder.fit_transform(FLAG_OWN_REALTY)
FLAG_OWN_REALTY_1hot = pd.DataFrame(FLAG_OWN_REALTY_1hot)
df_final = pd.concat([df_final,FLAG_OWN_REALTY_1hot],axis=1)
df_final.rename(columns={0:'FLAG_OWN_REALTY'}, inplace = True)


#-------------标记第五个变量------------拥有孩子数量，分为0,1,2或者以上
encoder = LabelBinarizer()
CNT_CHILDREN = df_first_31['CNT_CHILDREN']
CNT_CHILDREN = pd.DataFrame(CNT_CHILDREN)
CNT_CHILDREN.where(CNT_CHILDREN<2, other = 2.0,inplace = True)
CNT_CHILDREN_1hot = encoder.fit_transform(CNT_CHILDREN)
CNT_CHILDREN_1hot = pd.DataFrame(CNT_CHILDREN_1hot)
df_final = pd.concat([df_final,CNT_CHILDREN_1hot],axis=1)
df_final.rename(columns={0:'CNT_CHILDREN_0'}, inplace = True) #表示没有孩子
df_final.rename(columns={1:'CNT_CHILDREN_1'}, inplace = True) #表示只有一个孩子
df_final.rename(columns={2:'CNT_CHILDREN_2_or_more'}, inplace = True) #表示有两个或两个以上孩子

#-------------标记第六个变量------------客户的月收入
AMT_INCOME_TOTAL = (df_first_31['AMT_INCOME_TOTAL'])/12
MAX = np.max(AMT_INCOME_TOTAL)
MIN = np.min(AMT_INCOME_TOTAL)
AMT_INCOME_TOTAL = (AMT_INCOME_TOTAL-MIN)/(MAX-MIN)
AMT_INCOME_TOTAL = pd.DataFrame(AMT_INCOME_TOTAL)
df_final = pd.concat([df_final,AMT_INCOME_TOTAL],axis=1)
df_final.rename(columns={0:'AMT_INCOME_TOTAL'}, inplace = True) 


#-------------标记第七个变量------------客户贷款金额
AMT_CREDIT = df_first_31['AMT_CREDIT']
MAX = np.max(AMT_CREDIT)
MIN = np.min(AMT_CREDIT)
AMT_CREDIT = (AMT_CREDIT-MIN)/(MAX-MIN)
AMT_CREDIT = pd.DataFrame(AMT_CREDIT)
df_final = pd.concat([df_final,AMT_CREDIT],axis=1)
df_final.rename(columns={0:'AMT_CREDIT'}, inplace = True)

#-------------标记第八个变量------------贷款年金
median = df_first_31['AMT_ANNUITY'].median()
df_first_31['AMT_ANNUITY'] = df_first_31['AMT_ANNUITY'].fillna(median)
AMT_ANNUITY = df_first_31['AMT_ANNUITY']
MAX = np.max(AMT_ANNUITY)
MIN = np.min(AMT_ANNUITY)
AMT_ANNUITY = (AMT_ANNUITY-MIN)/(MAX-MIN)
AMT_ANNUITY = pd.DataFrame(AMT_ANNUITY)
df_final = pd.concat([df_final,AMT_ANNUITY],axis=1)
df_final.rename(columns={0:'AMT_ANNUITY'}, inplace = True)

#-------------标记第九个变量------------消费金额
median = df_first_31['AMT_GOODS_PRICE'].median()
df_first_31['AMT_GOODS_PRICE'] = df_first_31['AMT_GOODS_PRICE'].fillna(median)
AMT_GOODS_PRICE = df_first_31['AMT_GOODS_PRICE']
MAX = np.max(AMT_GOODS_PRICE)
MIN = np.min(AMT_GOODS_PRICE)
AMT_GOODS_PRICE = (AMT_GOODS_PRICE-MIN)/(MAX-MIN)
AMT_GOODS_PRICE = pd.DataFrame(AMT_GOODS_PRICE)
df_final = pd.concat([df_final,AMT_GOODS_PRICE],axis=1)
df_final.rename(columns={0:'AMT_GOODS_PRICE'}, inplace = True)

#-------------标记第十个变量------------客户在申请贷款是陪同人员情况
encoder = LabelBinarizer()
df_first_31['NAME_TYPE_SUITE'] = df_first_31['NAME_TYPE_SUITE'].fillna('None')
NAME_TYPE_SUITE = df_first_31['NAME_TYPE_SUITE']
NAME_TYPE_SUITE_1hot = encoder.fit_transform(NAME_TYPE_SUITE)
NAME_TYPE_SUITE_1hot = pd.DataFrame(NAME_TYPE_SUITE_1hot)
df_final = pd.concat([df_final,NAME_TYPE_SUITE_1hot],axis=1)
df_final.rename(columns={0:'NAME_TYPE_SUITE_CHILDREN',1:'NAME_TYPE_SUITE_FAMILY',
                         2:'NAME_TYPE_SUITE_GROUP_OF_PEOPLE',
                         3:'NAME_TYPE_SUITE_NONE',
                         4:'NAME_TYPE_SUITE_OTHER_A',
                         5:'NAME_TYPE_SUITE_OTHER_B',
                         6:'NAME_TYPE_SUITE_SPOUSE_OR_PARTNER',
                         7:'NAME_TYPE_SUITE_UNCOMPANIED'}, inplace = True)

#-------------标记第十一个变量------------客户收入类型
encoder = LabelBinarizer()
NAME_INCOME_TYPE = df_first_31['NAME_INCOME_TYPE']
NAME_INCOME_TYPE_1hot = encoder.fit_transform(NAME_INCOME_TYPE)
NAME_INCOME_TYPE_1hot = pd.DataFrame(NAME_INCOME_TYPE_1hot)
df_final = pd.concat([df_final,NAME_INCOME_TYPE_1hot],axis=1)
df_final.rename(columns={0:'NAME_INCOME_TYPE_BUSINESSMAN',1:'NAME_INCOME_TYPE_COMMERCIAL_ASSOCIATE',
                         2:'NAME_INCOME_TYPE_MATERNITY_LEAVE',
                         3:'NAME_INCOME_TYPE_PENSIONER',
                         4:'NAME_INCOME_TYPE_STATE_SERVANT',
                         5:'NAME_INCOME_TYPE_STUDENT',
                         6:'NAME_INCOME_TYPE_UNEMPLOYED',
                         7:'NAME_INCOME_TYPE_WORKING'}, inplace = True)

#-------------标记第十二个变量------------客户受教育的程度
encoder = LabelBinarizer()
NAME_EDUCATION_TYPE = df_first_31['NAME_EDUCATION_TYPE']
NAME_EDUCATION_TYPE_1hot = encoder.fit_transform(NAME_EDUCATION_TYPE)
NAME_EDUCATION_TYPE_1hot = pd.DataFrame(NAME_EDUCATION_TYPE_1hot)
df_final = pd.concat([df_final,NAME_EDUCATION_TYPE_1hot],axis=1)
df_final.rename(columns={0:'NAME_EDUCATION_TYPE_ACADEMIC_DEGREE',
                         1:'NAME_EDUCATION_TYPE_HIGHER_EDUCATION',
                         2:'NAME_EDUCATION_TYPE_INCOMPLETE_HIGHER',
                         3:'NAME_EDUCATION_TYPE_LOWER',
                         4:'NAME_EDUCATION_TYPE_SECONDARY'})

#-------------标记第十三个变量------------客户的家庭状况
encoder = LabelBinarizer()
NAME_FAMILY_STATUS = df_first_31['NAME_FAMILY_STATUS']
NAME_FAMILY_STATUS_1hot = encoder.fit_transform(NAME_FAMILY_STATUS)
NAME_FAMILY_STATUS_1hot = pd.DataFrame(NAME_FAMILY_STATUS_1hot)
df_final = pd.concat([df_final,NAME_FAMILY_STATUS_1hot],axis=1)
df_final.rename(columns={0:'NAME_FAMILY_STATUS_CIVIL_MARRIAGE',
                         1:'NAME_FAMILY_STATUS_MARRIED',
                         2:'NAME_FAMILY_STATUS_SEPARATED',
                         3:'NAME_FAMILY_STATUS_SINGLE',
                         4:'NAME_FAMILY_STATUS_UNKNOWN',
                         5:'NAME_FAMILY_STATUS_WIDOW'})

#-------------标记第十四个变量------------客户的住房情况（租房or与父母同住)
encoder = LabelBinarizer()
NAME_HOUSING_TYPE = df_first_31['NAME_HOUSING_TYPE']
NAME_HOUSING_TYPE_1hot = encoder.fit_transform(NAME_HOUSING_TYPE)
NAME_HOUSING_TYPE_1hot = pd.DataFrame(NAME_HOUSING_TYPE_1hot)
df_final = pd.concat([df_final,NAME_HOUSING_TYPE_1hot],axis=1)
df_final.rename(columns={0:'NAME_HOUSING_TYPE_CO_APARTMENT',
                         1:'NAME_HOUSING_TYPE_HOUSE',
                         2:'NAME_HOUSING_TYPE_MUNICIPAL',
                         3:'NAME_HOUSING_TYPE_OFFICE',
                         4:'NAME_HOUSING_TYPE_RENTED',
                         5:'NAME_HOUSING_TYPE_WITH_PARENTS'})

#-------------标记第十五个变量------------客户居住区域人口情况(数字越大代表人口越多)
REGION_POPULATION_RELATIVE = df_first_31['REGION_POPULATION_RELATIVE']
MAX = np.max(REGION_POPULATION_RELATIVE)
MIN = np.min(REGION_POPULATION_RELATIVE)
REGION_POPULATION_RELATIVE = (REGION_POPULATION_RELATIVE-MIN)/(MAX-MIN)
REGION_POPULATION_RELATIVE = pd.DataFrame(REGION_POPULATION_RELATIVE)
df_final = pd.concat([df_final,REGION_POPULATION_RELATIVE],axis=1)
df_final.rename(columns={0:'REGION_POPULATION_RELATIVE'}, inplace = True)


#-------------标记第十六个变量------------客户申请时年龄（天数)
DAYS_BIRTH = df_first_31['DAYS_BIRTH']
MAX = np.max(DAYS_BIRTH)
MIN = np.min(DAYS_BIRTH)
DAYS_BIRTH = (DAYS_BIRTH-MIN)/(MAX-MIN)
DAYS_BIRTH = pd.DataFrame(DAYS_BIRTH)
df_final = pd.concat([df_final,DAYS_BIRTH],axis=1)
df_final.rename(columns={0:'DAYS_BIRTH'}, inplace = True)


#-------------标记第十七个变量------------申请人就业的天数
DAYS_EMPLOYED = df_first_31['DAYS_EMPLOYED']
MAX = np.max(DAYS_EMPLOYED)
MIN = np.min(DAYS_EMPLOYED)
DAYS_EMPLOYED = (DAYS_EMPLOYED-MIN)/(MAX-MIN)
DAYS_EMPLOYED = pd.DataFrame(DAYS_EMPLOYED)
df_final = pd.concat([df_final,DAYS_EMPLOYED],axis=1)
df_final.rename(columns={0:'DAYS_EMPLOYED'}, inplace = True)


#-------------标记第十八个变量------------客户在申请前多少天更改了注册时间，仅相对于申请时间
DAYS_REGISTRATION = df_first_31['DAYS_REGISTRATION']
MAX = np.max(DAYS_REGISTRATION)
MIN = np.min(DAYS_REGISTRATION)
DAYS_REGISTRATION = (DAYS_REGISTRATION-MIN)/(MAX-MIN)
DAYS_REGISTRATION = pd.DataFrame(DAYS_REGISTRATION)
df_final = pd.concat([df_final,DAYS_REGISTRATION],axis=1)
df_final.rename(columns={0:'DAYS_REGISTRATION'}, inplace = True)


#-------------标记第十九个变量------------客户在申请前多少天更改了申请贷款的身份证明文件，相对于限申请时间
DAYS_ID_PUBLISH = df_first_31['DAYS_ID_PUBLISH']
MAX = np.max(DAYS_ID_PUBLISH)
MIN = np.min(DAYS_ID_PUBLISH)
DAYS_ID_PUBLISH = (DAYS_ID_PUBLISH-MIN)/(MAX-MIN)
DAYS_ID_PUBLISH = pd.DataFrame(DAYS_ID_PUBLISH)
df_final = pd.concat([df_final,DAYS_ID_PUBLISH],axis=1)
df_final.rename(columns={0:'DAYS_ID_PUBLISH'}, inplace = True)


#-------------标记第二十一个变量------------客户是否有车
df_first_31['OWN_CAR_AGE'] = df_first_31['OWN_CAR_AGE'].fillna(0)
OWN_CAR_AGE = df_first_31['OWN_CAR_AGE']
OWN_CAR_AGE = pd.DataFrame(OWN_CAR_AGE)
OWN_CAR_AGE.where(OWN_CAR_AGE<=0.0, other = 1.0,inplace = True)
df_final = pd.concat([df_final,OWN_CAR_AGE],axis=1)
df_final.rename(columns={0:'OWN_CAR'}, inplace = True) #1代表有车，0代表没车。

#-------------标记第二十一到二十六个变量------------
#FLAG_MOBIL:客户是否提供移动电话，１：有，０：没有
#FLAG_EMP_PHONE:客户是否提供工作电话，１/０
#FLAG_WORK_PHONE:客户是否提供家庭电话，１/０
#FLAG_CONT_MOBILE:移动电话是否管用，１/０
#FLAG_PHONE:家庭是否有电话1/0
#FLAG_EMAIL:家庭是否有电子邮箱1/0
       
df_first_NO_MODIFY = df_first_31.ix[:,('FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE',
                             'FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL')]
df_final = pd.concat([df_final,df_first_NO_MODIFY],axis=1)
    

#-------------标记第二十七个变量------------职业类型
encoder = LabelBinarizer()
df_first_31['OCCUPATION_TYPE'] = df_first_31['OCCUPATION_TYPE'].fillna('None')
OCCUPATION_TYPE = df_first_31['OCCUPATION_TYPE']
OCCUPATION_TYPE_1hot = encoder.fit_transform(OCCUPATION_TYPE)
OCCUPATION_TYPE_1hot = pd.DataFrame(OCCUPATION_TYPE_1hot)
df_final = pd.concat([df_final,OCCUPATION_TYPE_1hot],axis=1)
df_final.rename(columns={0:'ACCOUNTANTS',1:'CLEANING_STAFF',
                         2:'COOK_STAFF',
                         3:'CORE_STAFF',
                         4:'DRIVERS',
                         5:'HIGH_SKILL_TECH_STAFF',
                         6:'HR_STAFF',
                         7:'IT_STAFF',
                         8:'LABOURS',
                         9:'LOW_SKILL_STAFF',
                         10:'MANAGERS',
                         11:'MEDICINE_STAFF',
                         12:'NONE',
                         13:'PRIVATE_SERVICE_STAFF',
                         14:'REALITY_AGENT',
                         15:'SALES_STAFF',
                         16:'SECRETARIES',
                         17:'SECURITY_STAFFS',
                         18:'WAITERS'}, inplace = True)


#-------------标记第二十八个变量------------客户家庭成员数量，分为0,1,2或者以上
encoder = LabelBinarizer()
df_first_31['CNT_FAM_MEMBERS'] = df_first_31['CNT_FAM_MEMBERS'].fillna(0)
CNT_FAM_MEMBERS = df_first_31['CNT_FAM_MEMBERS']
CNT_FAM_MEMBERS = pd.DataFrame(CNT_FAM_MEMBERS)
CNT_FAM_MEMBERS.where(CNT_FAM_MEMBERS<2,other =2, inplace = True)
CNT_FAM_MEMBERS_1hot = encoder.fit_transform(CNT_FAM_MEMBERS)
CNT_FAM_MEMBERS_1hot = pd.DataFrame(CNT_FAM_MEMBERS_1hot)
df_final = pd.concat([df_final,CNT_FAM_MEMBERS_1hot],axis=1)
df_final.rename(columns={0:'CNT_FAM_MEMBERS_0',
                 1:'CNT_FAM_MEMBERS_1',
                 2:'CNT_FAM_MEMBERS_2'},inplace = True)

#-------------标记第二十九个变量------------对客户所在地区评级（１，２，３）
encoder = LabelBinarizer()
REGION_RATING_CLIENT = df_first_31['REGION_RATING_CLIENT']
REGION_RATING_CLIENT = pd.DataFrame(REGION_RATING_CLIENT)
REGION_RATING_CLIENT_1hot = encoder.fit_transform(REGION_RATING_CLIENT)
REGION_RATING_CLIENT_1hot = pd.DataFrame(REGION_RATING_CLIENT_1hot)
df_final = pd.concat([df_final,REGION_RATING_CLIENT_1hot],axis=1)
df_final.rename(columns={0:'REGION_RATING_CLIENT_1',
                 1:'REGION_RATING_CLIENT_2',
                 2:'REGION_RATING_CLIENT_3'},inplace = True)

#-------------标记第三十个变量------------对客户居住地区所在的城市评级(1,2,３）
encoder = LabelBinarizer()
REGION_RATING_CLIENT_W_CITY = df_first_31['REGION_RATING_CLIENT_W_CITY']
REGION_RATING_CLIENT_W_CITY = pd.DataFrame(REGION_RATING_CLIENT_W_CITY)
REGION_RATING_CLIENT_W_CITY_1hot = encoder.fit_transform(REGION_RATING_CLIENT_W_CITY)
REGION_RATING_CLIENT_W_CITY_1hot = pd.DataFrame(REGION_RATING_CLIENT_W_CITY_1hot)
df_final = pd.concat([df_final,REGION_RATING_CLIENT_W_CITY_1hot],axis=1)
df_final.rename(columns={0:'REGION_RATING_CLIENT_W_CITY_1',
                 1:'REGION_RATING_CLIENT_W_CITY_2',
                 2:'REGION_RATING_CLIENT_W_CITY_3'},inplace = True)

#-------------标记第三十一个变量------------客户在周几申请的贷款
encoder = LabelBinarizer()
WEEKDAY_APPR_PROCESS_START = df_first_31['WEEKDAY_APPR_PROCESS_START']
WEEKDAY_APPR_PROCESS_START = pd.DataFrame(WEEKDAY_APPR_PROCESS_START)
WEEKDAY_APPR_PROCESS_START_1hot = encoder.fit_transform(WEEKDAY_APPR_PROCESS_START)
WEEKDAY_APPR_PROCESS_START_1hot = pd.DataFrame(WEEKDAY_APPR_PROCESS_START_1hot)
df_final = pd.concat([df_final,WEEKDAY_APPR_PROCESS_START_1hot],axis=1)
df_final.rename(columns={0:'FRIDAY',
                 1:'MONDAY',
                 2:'SATUARDAY',
                 3:'SUNDAY',
                 4:'THURSDAY',
                 5:'TUESDAY',
                 6:'WEDNESDAY'},inplace = True)

#-------------添加四个特征-----------------
CREDIT_INCOME_PERCENT = df_first_31['AMT_CREDIT']/df_first_31['AMT_INCOME_TOTAL']
MAX = np.max(CREDIT_INCOME_PERCENT)
MIN = np.min(CREDIT_INCOME_PERCENT)
CREDIT_INCOME_PERCENT = (CREDIT_INCOME_PERCENT-MIN)/(MAX-MIN)
CREDIT_INCOME_PERCENT = pd.DataFrame(CREDIT_INCOME_PERCENT)

ANNUITY_INCOME_PERCENT = df_first_31['AMT_ANNUITY']/df_first_31['AMT_INCOME_TOTAL']
MAX = np.max(ANNUITY_INCOME_PERCENT)
MIN = np.min(ANNUITY_INCOME_PERCENT)
ANNUITY_INCOME_PERCENT = (ANNUITY_INCOME_PERCENT-MIN)/(MAX-MIN)
ANNUITY_INCOME_PERCENT = pd.DataFrame(ANNUITY_INCOME_PERCENT)

ANNUITY_CREDIT_PERCENT = df_first_31['AMT_ANNUITY']/df_first_31['AMT_CREDIT']
MAX = np.max(ANNUITY_CREDIT_PERCENT)
MIN = np.min(ANNUITY_CREDIT_PERCENT)
ANNUITY_CREDIT_PERCENT = (ANNUITY_CREDIT_PERCENT-MIN)/(MAX-MIN)
ANNUITY_CREDIT_PERCENT = pd.DataFrame(ANNUITY_CREDIT_PERCENT)

DAYS_EMPLOYED_PERCENT = df_first_31['DAYS_EMPLOYED']/df_first_31['DAYS_BIRTH']
MAX = np.max(DAYS_EMPLOYED_PERCENT)
MIN = np.min(DAYS_EMPLOYED_PERCENT)
DAYS_EMPLOYED_PERCENT = (DAYS_EMPLOYED_PERCENT-MIN)/(MAX-MIN)
DAYS_EMPLOYED_PERCENT = pd.DataFrame(DAYS_EMPLOYED_PERCENT)

df_final = pd.concat([df_final,CREDIT_INCOME_PERCENT],axis=1)
df_final.rename(columns={0:'CREDIT_INCOME_PERCENT'},inplace = True)
df_final = pd.concat([df_final,ANNUITY_INCOME_PERCENT],axis=1)
df_final.rename(columns={0:'ANNUITY_INCOME_PERCENT'},inplace = True)
df_final = pd.concat([df_final,ANNUITY_CREDIT_PERCENT],axis=1)
df_final.rename(columns={0:'ANNUITY_CREDIT_PERCENT'},inplace = True)
df_final = pd.concat([df_final,DAYS_EMPLOYED_PERCENT],axis=1)
df_final.rename(columns={0:'DAYS_EMPLOYED_PERCENT'},inplace = True)

df_final.to_csv('C:/Users/Administrator/Desktop/蔡睿杰-application_train_feature_engineering.csv',index = False)    
    