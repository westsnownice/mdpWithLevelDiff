import Files.functions as fc
import json
import pandas as pd
#import datetime
import progressbar
from tqdm import tqdm
import re
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans as km
from sklearn import tree


result1 = pd.read_csv(filepath_or_buffer="C://Users/zhang/Desktop/anaconda/pomdp_eole/extraction_results_23133.csv")
result1.index = pd.Series(range(1, len(result1)+1))# 0123456 as index
data1 = pd.read_csv(filepath_or_buffer="C://Users/zhang/Desktop/anaconda/pomdp_eole/extraction_events_23133.csv")
# data1.drop(["courseId"])
# courseId.drop(["courseId"])
print("Reading csv, JSON files and normalise them in DataFrame as input")
allEventList, finalExamList, resultExamList = fc.readJsonToList()

print("Add event time to results from original JSON file")
result1 = fc.addTimeToResult(result1, resultExamList)
UserId = list(set(data1["userId"]))

# 这个循环是带表头的，而表头中没有数字的存在，所以我们要从1开始循环。
for i in (range(1, len(result1))):
    result1.loc[i, 'examName'] = int(re.findall(r"\d+\.?\d*", result1.loc[i]['examName'])[0])

d1 = data1.copy(deep=True)
r1 = result1.copy(deep=True)

# I want to use this rename, but the interpreter don't know this function :/
# r1.rename(columns={r1.columns{'examName': "resourceId"}), inplace=True)
r1["resourceId"] = r1['examName']
r1 = r1.drop(columns=['examName'])

d1 = d1.drop(columns=['resourceType', 'resourceName', 'resourceDesc'])

r1 = r1.drop(index=0)
r1['eventTime'].loc[-1] = '2018-10-02 21:35:05.755000'
d1['score'] = 0
d1['scoreMax'] = 0
r1['action'] = "Submitted"

# Drop the 'submitted' in study record because each visit of quiz has a unique resourceId, even the same resource
d1 = (d1[d1['action'].isin(['Viewed'])])
d1 = d1[['userId', 'courseId', 'score', 'scoreMax', 'eventTime', 'resourceId']]

#Here, scoreMax is 18
FinalExam = fc.treatFinalExam(finalExamList)
FinalExam['courseId'] = d1.loc[0, 'courseId']
FinalExam['scoreMax'] = 18
FinalExam['eventTime'] = "2019-10-02 21:35:05.755000"
FinalExam['resourceId'] = 666666 #标号我自己取的
FinalExam = FinalExam[['userId', 'courseId', 'score', 'scoreMax', 'eventTime', 'resourceId']]

"""
        合成整体的计算数据,并按照学生的成绩类别进行分类
"""
frames = [d1, r1, FinalExam]
df_total = pd.concat(frames, sort=True)
label = []
q1TB = []
q1PASS = []
q1FAIL = []
q2TB = []
q2PASS = []
q2FAIL = []

"""Here is for the quiz part"""
for i in range(len(r1)):
    if r1.iloc[i]['score']>=8:
       label.append('TB')
    elif r1.iloc[i]['score']<8 and r1.iloc[i]['score']>=6:
        label.append('PASS')
    else:
        label.append('FAIL')
r1['label'] = label

"""
Here is for the Final Exam Part
"""

FinalExam_label = []
FinalTB = []
FinalPASS = []
FinalFAIL = []
for i in (FinalExam.score.astype(float)):
    if i >= 14:
       FinalExam_label.append('TB')
    elif i<14 and i>=8:
        FinalExam_label.append('PASS')
    else :
        FinalExam_label.append('FAIL')
FinalExam['label'] = FinalExam_label

Q1 = r1[r1['resourceId'] == 83669]
Q2 = r1[r1['resourceId'] == 84682]
q1 = []
q2 = []
final = []
for i in UserId:
    if i in list(Q1['userId']) and i in list(Q2['userId']) and i in list(FinalExam['userId']):

        a = float(Q1[Q1['userId'] == i].get('score'))
        q1.append(a)
        q2.append(float(Q2[Q2['userId'] == i].get('score')))
        final.append(float(FinalExam[FinalExam['userId'] == i].get('score')))


d = {'userId':"" , 'score_Q1': "", 'scoreQ2': "", 'score_final': ""}



x = np.array([q1, q2, final])
x = x.reshape(len(x[0]), 3)
kmeans = km(n_clusters=3, random_state=0).fit(x)

# kmeans.labels_ 这个是分类好了的表格
#quizAndExam KMEANS 效果贼差，决定不用了。
quizAndExam = pd.DataFrame(data=x, columns={"q1Score", "q2Score", "finalScore"})

quizAndExam['label'] = kmeans.labels_

quizAndExam.to_csv(path_or_buf="成绩和分类.csv")
quizAndExam.to_excel('output1.xlsx', engine='xlsxwriter')
"""
三个学生的成绩等级。TB20个，B36个，F47个。咱大学的学生水平不行啊。
"""
TB = []
B = []
F = []
for i in range(len(FinalExam)):
    if FinalExam.iloc[i]["label"] == 'TB':
        TB.append(FinalExam.loc[i]['userId'])
    if FinalExam.iloc[i]["label"] == 'PASS':
        B.append(FinalExam.loc[i]['userId'])
    if FinalExam.iloc[i]['label'] == 'FAIL':
        F.append(FinalExam.loc[i]['userId'])

#######################################################################################################################
# 计算一下用所有数据的转移矩阵
# 肯定是按照学生分的，先按照每个学生的情况将转移的次数加到矩阵里，然后在做那个矩阵除法就好了
# 我们一共有117个学生，按照每个学生的名字做最外层循环。
# 里面是transition matrix一样的计算过程

toCalculateMatrix = df_total[['eventTime', 'resourceId']].copy(deep=True)
toCalculateMatrix = toCalculateMatrix.drop(index=156)
toCalculateMatrix = toCalculateMatrix.sort_values(by='eventTime')
numIndex = 0
resourceIndex = {}


###########################################################################
#存储resource的index编号,每个resource我都按照
# 这一步能够简化计算action转移概率的 复杂度，非常重要。
for i in toCalculateMatrix.resourceId:
    if i not in resourceIndex.keys():
        resourceIndex[i] = numIndex
        numIndex += 1
"""
我们的数据将所有的学习记录都放在了一起，但是实际中的例子不是这样的。
我以学生为单位，把每个学生的学习轨迹单独的计算出来，能够去掉影响resource转移的因素
"""
df_tb = pd.DataFrame(columns=df_total.columns)
df_b = pd.DataFrame(columns=df_total.columns)
df_f = pd.DataFrame(columns=df_total.columns)

for i in TB:
    df_tb = pd.concat([df_tb, df_total[df_total['userId']==i]])

for i in B:
    df_b = pd.concat([df_b, df_total[df_total['userId']==i]])

for i in F:
    df_f = pd.concat([df_f, df_total[df_total['userId']==i]])
# 这块矩阵可以改成按照学生的成绩等级分类,一共三个等级，TB，B，F
M_tb = np.array(fc.transition_diff(resourceIndex, df_tb))
M_b = fc.transition_diff(resourceIndex, df_b)
M_f = fc.transition_diff(resourceIndex, df_f)
S = []

for i in range(len(M_tb)):
    S.append(i)


M1 = np.zeros([38,38])
for x in range(len(M_tb[1:])):
     for y in range(len(M_tb[x][1:])):
        M1[x][y]=(M_tb[x][y])