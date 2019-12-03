import numpy as np
from Files.treat_data import S
import Files.treat_data as ts
from Files.Bayes import reward_function
# from goto import goto, label

from datetime import datetime
gamma = 0.95
error = 0.00001
max_iter = 10
A = ["avance", "back", "stay"]

"""
Here we should note that each quiz or exam can just submit one time! 
How to do this constrain ? 
"""
# Create a table to store the value for every s at each time step k

PassedQuizz = []


def value(S, A, error, max_iter, M, Neaveau):

    v_array = np.zeros([len(S), 1])
    layer = 0

    for i in range(max_iter):
        # calculer le temps
        a = datetime.now()
        temp_array = np.zeros([len(S), 1])
        # creat a dict to store a = pi(s)
        pi_optimal = {}
        for s_current in S:

            # create a Q_table to store the Q value at each step
            q_table = np.zeros([len(A), 1])
            for action in A:
                    for s_next in S:
                        if s_next in PassedQuizz: # if the quiz is passed, we don't go into it
                            continue

                        else:
                            if s_current == 12: # 83669
                                layer = 1
                                PassedQuizz.append(12)
                            if s_current == 20: #84682:
                                layer = 2
                                PassedQuizz.append(20)
                            reward = reward_function(s_next, layer, Neaveau)
                            value_next = v_array[S.index(s_next)]
                            q_table[A.index(action)] += M[s_current,s_next]*(reward + gamma*value_next)
                            if s_next == 38:# 666666
                                temp_array[S.index(s_current)] = max(q_table)
                                a_optimal = A[q_table.argmax()]
                                pi_optimal[s_current] = a_optimal
                                break
                            """"
                            here I  want to jump out two loops but continue the third loop :/
                            """
            temp_array[S.index(s_current)] = max(q_table)
            a_optimal = A[q_table.argmax()]
            # print("The optimal value is ", v_array[S.index(s_current),k],"the optimal action is ",a_optimal,"at state",s_current.pos,s_current.stat, "at time step ",k)
            pi_optimal[s_current] = a_optimal
        b = datetime.now()
        print("Time for cycle", i)

        print(b-a)

        #print("iteration times ", i)
        if (abs(v_array - temp_array) < error).all():
            print("*********************** Coverged ************************")
            for s in S:
                print("State :", s, "value: ", v_array[S[s]],", optimal action :", pi_optimal[s])
            return v_array

        v_array = temp_array



a = datetime.now()
"""
TB group
"""
TB_updated_v_array = value(S, A, error, max_iter, ts.M_tb, 'TB')
b = datetime.now()
print("Time for TB is:")
print(b-a)
"""
B group

B_updated_v_array = value(S, A, error, max_iter, ts.M_b, 'B')
c = datetime.now()
print("Time for TB is:")
print((c-b))
"""
"""
F group

B_updated_v_array = value(S, A, error, max_iter, ts.M_f, 'F')
d = datetime.now()
print("Time for F is :")
print((d-c))
print("Time total is")
print(d-a)
"""
