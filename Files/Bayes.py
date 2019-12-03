import Files.treat_data as ts
import pandas as pd
import tqdm
import numpy as np
import datetime
import heapq
"""
Here we calculate the priori probability of Bayes
"""

UserId = list(set(ts.df_tb["userId"]))
df_tb = ts.df_f.copy(deep=True)
df_b = ts.df_b.copy(deep=True)
df_f = ts.df_f.copy(deep=True)
df_tb.index = range(len(df_tb))


def treat_input(N):
    if N == 'TB':
        return df_tb
    if N == 'B':
        return df_b
    if N == 'F':
        return df_f
dfItemProba = pd.DataFrame(columns=["userId",'resourceId','ProbaResourceForEachUser'])
# data1["ProbaResourceForEachUser"] = 0
# print(data1[(data1.userId=="adam125u")]) caculat by userId

"""
计算每个学生的学习过的素材的先验概率,没有使用
(Pas encore utililser) Calculer la probabilité prior des resource appris de chaque étudiant

for i in tqdm(UserId):
    temp_dict = {}
    total_resource = 0
    for j in range(len(df_tb)):
        if df_tb.iloc[j]["userId"] == i:
            total_resource = total_resource + 1
            temp_dict[df_tb.iloc[j]["resourceId"]] = temp_dict.get(df_tb.iloc[j]["resourceId"],0) +1
    for k,v in temp_dict.items():
        temp_dict[k] = v / total_resource
    for k,v in temp_dict.items():
        dfItemProba.loc[len(dfItemProba)+1] = [i, k, v]
"""
# 每个学生的所有学习资源中，这个resource被学习次数占的比重
dfItemProba[(dfItemProba["resourceId"] == 8935) & (dfItemProba["userId"]=="adam125u")]

# calculate probability of all resources in the action
def addProbaForAllResource(data1,probaForAllResource):
    # print("addProbaForAllResource fonction time:")
    # a = datetime.datetime.now()
    listProba = [x for x in range(len(data1))]
    # print(probaForAllResource)

    for k, v in probaForAllResource.items():
        for i in range(len(data1)):
            if k == data1.iloc[i]["resourceId"]:
                listProba[i] = v
    data1["probaForAllResource"] = listProba
    # b = datetime.datetime.now()
    # print(b-a)
    return data1


"""
    # input is a event dataframe,
    # out put is a dictionary with key(resourceId),value(probability)
    输出是字典形式的所有resource的出现先验概率。其中删除了welcome page 22133的概率
"""
def CalResourceNumber(data1):
    print("CalResourceNumber fonction time:")
    a = datetime.datetime.now()
    """input is a event dataframe,
    out put is a dictionary with key(resourceId),value(probability)"""
    dic_resource = {}
    totalNumberAction = len(data1[data1.resourceId != 23133])
    for dl in data1["resourceId"]:
        if dl != '23133' and dl != 23133:
            # delete welcome page
            dic_resource[dl] = dic_resource.get(dl, 0) + 1
    for k,v in dic_resource.items():
        dic_resource[k] = v / totalNumberAction
    b = datetime.datetime.now()
    # print(b-a)
    return dic_resource

probaForAllResource = CalResourceNumber(df_tb)


"""
Estimation the proba sum should be 1.0
"""
print("Total probability is:")
#print(sum(probaForAllResource.values()))


""" reward """
#av_tb = np.mean(df_tb[df_tb['score']!=0]['score'].astype(float))

def get_key(dic, value):
    return [k for k, v in dic.items() if v == value]

def reward_function(s_next, s_current, action,layer,N): # here action is equal to s_next.
    # print("reward_function fonction time:")
    a = datetime.datetime.now()
    inputDF = treat_input(N)
    # probaForAllResource = CalResourceNumber(inputDF)

    resourceId = get_key(ts.resourceIndex, s_next)[0]

    try:
        # Welcom page
        if s_next == 23133:
            if action == "avance" or action =="stay":
            # b = datetime.datetime.now()
            # print(b-a)
                return -1
            elif action == "back":
                return -100

        # Final
        elif s_next == 666666:
            if N == 'TB':
                b = datetime.datetime.now()
                print("terminal")
                return 15
            elif N == 'B':

                return 10
            elif N == 'F':

                return 6
        # Q1
        elif s_next == 83669:
            if action =="avance":
                return np.mean(inputDF[(inputDF['resourceId'] == 83669) & (inputDF['score'] != -1)].astype({'score': 'float'})['score'])/3.0-1
            elif action =="stay":
                return -100
            elif action =="back":
                return np.mean(inputDF[(inputDF['resourceId'] == 83669) & (inputDF['score'] != -1)].astype({'score': 'float'})['score'])/3.0-1.5

        # Q2
        elif s_next == 84682:

            return np.mean(inputDF[(inputDF['resourceId'] == 84682) & (inputDF['score'] != -1)].astype({'score': 'float'})['score'])/3.0-1

        else:
            proba = [v for k, v in probaForAllResource.items() if k == resourceId][0]
            if layer == 0:

                return np.mean(inputDF[(inputDF['resourceId'] == 83669) & (inputDF['score'] != -1)].astype({'score': 'float'})['score']) *proba/3.0-1
            elif layer == 1:

                return np.mean(inputDF[(inputDF['resourceId'] == 84682) & (inputDF['score'] != -1)].astype({'score': 'float'})['score'])*proba/3.0-1
            elif layer == 3:

                return np.mean(inputDF[(inputDF['resourceId'] == 666666) & (inputDF['score'] != -1)].astype({'score': 'float'})['score'])*proba/3.0-1
            else:

                return -1
    except IndexError:

        return -1


# define a best traverse option of horizon
def input_for_randamChoice( s_current, S, M, action):
    r1 = []
    r2 = []
    # should delete the 0 proba to fit the randomchoice
    for i in range(len(M[s_current])):
        # print("M.shape:",M.shape)
        # print("M[s_current][i]",M[s_current][i])
        # print("Some thing of M", M[s_current][i].ndim)
        if M[s_current][i] != 0:
            r1.append(S[i])
            r2.append(M[s_current][i])

    # Normalisation de r2
    # print("R1", r1)
    # print("R2", r2)
    r2 = [x/sum(r2) for x in r2]
    if action == 'avance':
        return [x > s_current for x in r1], r2
    elif action == 'back':

        return [x < s_current for x in r1], r2


def nextState(s_current, S, M):
    r1 = []
    r2 = []
    # should delete the 0 proba to fit the randomchoice
    for i in range(len(M[s_current])):
        # print("M.shape:",M.shape)
        # print("M[s_current][i]",M[s_current][i])
        # print("Some thing of M", M[s_current][i].ndim)
        if M[s_current][i] != 0 and i != 0: # can't be the welcom page.
            r1.append(S[i])
            r2.append(M[s_current][i])

    index_noneZero = [i for i, e in enumerate(r2) if e != 0]#map(r2.index, heapq.nlargest(15, r2))
    result = []
    for i in index_noneZero:
        result.append(r1[i])
    return result
