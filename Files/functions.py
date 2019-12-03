import json
import pandas as pd
import datetime
#from datetime import date
import time
import progressbar
from tqdm import tqdm
import numpy as np


p = progressbar.ProgressBar()

# 读取三个json文件并存储在list中方便处理
def readJsonToList():
    """read the json file and put them in 3 lists"""

    eventFile = "C://Users/zhang/Desktop/anaconda/pomdp_eole/events_23133.json"
    finalExamFile = "C://Users/zhang/Desktop/anaconda/pomdp_eole/results_final_exam_23133.json"
    resultExamFile = "C://Users/zhang/Desktop/anaconda/pomdp_eole/results_23133.json"
    allEventList = []
    finalExamList = []
    resultExamList = []
    # Reading data back
    for line in open(eventFile, 'r'):
        allEventList.append(json.loads(line))

    for line in open(finalExamFile, 'r'):
        finalExamList.append(json.loads(line))

    for line in open(resultExamFile, 'r'):
        resultExamList.append(json.loads(line))
    return allEventList, finalExamList, resultExamList

'''
def timestamp2string(timeStamp):
    """the timestamp is not the same, so I should chang them in the same format
    and find the relation between the event and result"""
    try:
        timeStamp = timeStamp / 1000
        d = date.fromtimestamp(timeStamp)
        str1 = d.strftime("%Y-%m-%d %H:%M:%S.%f")
        # 2015-08-28 16:43:37.283000'
        return str1
    except Exception as e:
        print (e)
        return ''
'''


def timestamp2string(timeStamp):
    """the timestamp is not the same, so I should chang them in the same format
    and find the relation between the event and result"""
    try:
        timeStamp = timeStamp / 1000
        d = datetime.datetime.fromtimestamp(timeStamp)
        str1 = d.strftime("%Y-%m-%d %H:%M:%S.%f")
        # 2015-08-28 16:43:37.283000'
        print(str1)
        return str1
    except Exception as e:
        print (e)
        return ''




def addTimeToResult(result1, resultExamList):
    """ Here I add the time to the quiz. Return a new result1 with time stamp
    I need these three : userId,examName,eventTime"""
    result1['eventTime'] = '0'

    for i in resultExamList:
        for j in range(len(result1)):
            examName = i['lineitemSourcedId']
            userId = i['userSourcedId']
            eventTime = int(i['result']['date']['$date']['$numberLong'])
            eventTime = timestamp2string(eventTime)
            # indexResult = (result1[(result1.userId==userId)&(result1.examName==examName)].index)
            # too hard to save a 'for' loop :/
            if (result1.iloc[j]['userId'] == userId) and (result1.iloc[j]['examName'] == examName):
                result1.loc[j, 'eventTime'] = eventTime
    return result1


# calculate probability of all resources in the action
# 计算一个resource在所有学习记录中出现的概率
"""
def CalResourceNumber(data1):
    # input is a event dataframe,
    # out put is a dictionary with key(resourceId),value(probability)
    dic_resource = {}
    totalNumberAction = len(data1)
    p.start(len(data1))
    for dl in data1["resourceId"]:
        dic_resource[dl] = dic_resource.get(dl, 0) + 1
        p.update(dl + 1)
    p.finish()
    for k, v in dic_resource.items():
        dic_resource[k] = v / totalNumberAction

    return dic_resource
"""
# calculate probability of all resources in the action
def CalResourceNumber(data1):
    """input is a event dataframe,
    out put is a dictionary with key(resourceId),value(probability)"""
    dic_resource = {}
    totalNumberAction = len(data1)
    for dl in data1["resourceId"] :
        if dl != '23133' and dl!= 23133:
            # delete welcome page
            dic_resource[dl] = dic_resource.get(dl, 0) + 1
    for k,v in dic_resource.items():
        dic_resource[k] = v / totalNumberAction
    return dic_resource

def addProbaForAllResource(data1,probaForAllResource):
    print('en train de calcul')
    listProba = [x for x in range(len(data1)) ]
    # print(probaForAllResource)

    for k, v in probaForAllResource.items():
        for i in range(len(data1)):
            if k == data1.iloc[i]["resourceId"]:
                listProba[i] = v
    data1["probaForAllResource"] = listProba
    return data1

# Transition Matrix
# 如何输出转移矩阵for row in m: print(' '.join('{0:.2f}'.format(x) for x in row))
def transition_matrix(transitions,len_n):
    n = 1+ len_n #number of states
    print(n)
    M = [[0]*n for _ in range(n)]

    for (i, j) in tqdm(zip(transitions,transitions[1:])):
        M[i][j] += 1

    #now convert to probabilities:
    for row in (M):
        s = sum(row)
        if s > 0:
            row[:] = [f/s for f in row]
    return M

#这是一个可能没用的函数
def stateGet(resourceOrder):

    dic_t = {}
    count = 0
    list_t = []
    for i in resourceOrder:
        ''''''
        if i not in dic_t.keys():
            dic_t[i] = count
            count += 1

        list_t.append(dic_t[i])

    return dic_t, list_t


# here the function will use the
def tm(list_t):
    n = 1 + max(list_t)  # number of states
    M = np.zeros((n, n))
    for (i, j) in zip(list_t,list_t[1:]):
            M[i][j] += 1
    print("和在这里。")
    #print(sum(M))
    # now convert to probabilities:
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]
    return M

#计算不同成绩分类的同学的 transition matrxi
#calculate the transition matrix for ddifferent level students: TB, B,FAIL

def transition_diff(resourceIndex, df_total):
    print("calculating the transition matrix from all the records")
    n = len(resourceIndex)  # number of states
    M = np.zeros((n, n))  # Matrix for all the
    UserID = []  # 存储处理的用户的顺序。
    actionpace = []
    for i in set(df_total['userId']):
        print("treating user :" + i)
        i_user = df_total[df_total['userId'].isin([i])]
        i_user = i_user.sort_values(by='eventTime')
        UserID.append(i_user)
        listResource = []
        for j in i_user.resourceId:
            listResource.append(resourceIndex[j])
        actionpace.append(listResource)  # 将进行过的每个用户的学习过程全部记录下来
        for (i, j) in zip(listResource, listResource[1:]):
            M[i][j] += 1

    # print(sum(M))
    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]
    return M

"""
def R1PlusD1(result1, data1):
    d1 = data1.copy(deep=True)
    r1 = result1.copy(deep=True)
    r1.rename(columns={r1.columns['examName']: "resourceId"}, inplace=True)
    d1.drop(columns=['resourceType', 'resourceName', 'resourceDesc', 'probaForAllResource'])
    # r1.drop(columns=['examName'])
    r1.drop(index=0)
    d1['score'] = 0
    d1['scoreMax'] = 0
"""



# 预测状态的马尔可夫模型的函数
def study_forecast(resource, T, states):
    # 选择初始状态
    resourceNow = 0
    print("Start state: " + str(resourceNow))
    # 应该记录选择的状态序列。这里现在只有初始状态。
    activityList = [resourceNow]
    i = 0
    # 计算 activityList 的概率
    prob = 1
    while i != resource:
        change = np.random.choice(list(states), replace=True, p=T[resourceNow])
        activityList.append(change)
        prob = prob*T[resourceNow][change]
        resourceNow = change
        i += 1
    print("Possible states: " + str(activityList))
    print("End state after "+ str(resource) + " resource: " + str(resourceNow))
    print("Probability of the possible sequence of states: " + str(prob))


# Markov decision model
"""
def mdp():
    # have 27 actions and 27 states
    #从一个点出发，所有proba的加和是1
    return

# reward function
def reward(type,score):
    if type == "Viewed":
        reward = -1
    else:
        if 1:
            pass
"""

#Output the final exam as DF.
def treatFinalExam(finalExamList):
    userId = []
    score = []
    for i in range(len(finalExamList)):
        userId.append(finalExamList[i]['userSourcedId'])
        try:

            score.append(finalExamList[i]['result']['score']['$numberInt'])
        except KeyError:
            score.append(finalExamList[i]['result']['score']['$numberDouble'])

        # metadata = finalExamList[i]['result']['metadata']
    f = {'userId': userId, 'score': score}
    finalExam = pd.DataFrame(data=f)
    return finalExam

# 根据值返回key的函数
def get_key(dic, value):
    return [k for k, v in dic.items() if v == value]

#get the scores before


def getQuizScore(resourceIndex, s, userId, r1):

    resourceId = get_key(resourceIndex, s)[0]
    for i in r1:
        if i['resourceId'] == resourceId and i['userId'] == userId:
            return i['score'], i['scoreMax']


"""Reward function"""
"""My reward function is based on the student's records, so the userId is needed"""


def prob_next_state(s_next, s_current, action,M):
    if str(s_current) + ' + ' + str(s_next) == action:
        return M[s_current][s_next]
    else:
        return 0



