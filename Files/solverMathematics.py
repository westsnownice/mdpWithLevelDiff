import numpy as np
from Files.treat_data import S
import Files.treat_data as ts
from Files.Bayes import reward_function, nextState
from Files.treat_data import resourceIndex
from datetime import datetime
gamma = 0.95
error = 0.00001
max_iter = 10000
A = ["avance", "back", "stay"]
"""
Here we should note that each quiz or exam can just submit one time! 
How to do this constrain ? 
"""
# Create a table to store the value for every s at each time step k
Quiz = [83669, 84682, 666666]


def value(S, A, error, max_iter, M, Neaveau):
    v_array = np.zeros([len(S), 1])

    for i in range(max_iter):
        PassedQuizz = []
        a = datetime.now()
        # calculer le temps
        layer = 0
        temp_array = np.zeros([len(S), 1])
        # creat a dict to store a = pi(s)
        pi_optimal = {}
        for s_current in S:

            if s_current == 12:  # 83669
                layer = 1
                pi_optimal[s_current] = "avance"
                PassedQuizz.append(12)
            if s_current == 20:  # 84682:
                layer = 2
                pi_optimal[s_current] = "avance"
                PassedQuizz.append(20)
            q_table = np.zeros([len(A), 1])
            next_list = nextState(s_current, S, M)
            for s_next in S:
                s_next = int(s_next)
                for action in A:
                    #if s_current == 0:
                    #   s_next = np.random.choice(S)

                    """which action?
                    if action == "avance" and (s_next) <= s_current:
                        break
                    elif action == "back" and s_next >= s_current:
                            break
                    elif action == "stay":
                        if s_next != s_current:
                            break
                    """
                    # think about the state Quiz?Terminal exam?
                    if s_next in PassedQuizz:  # if the quiz is passed, we don't go into it
                        continue
                    else:
                        if s_current == 12:  # 83669
                            layer = 1
                            PassedQuizz.append(12)
                        if s_current == 20:  # 84682:
                            layer = 2
                            PassedQuizz.append(20)
                    reward = reward_function(s_next, layer, Neaveau)


                    # Here we can write v_array[S[s_next]] because
                    # S.index(s_next) == s_next
                    value_next = v_array[S.index(s_next)]
                    #print(s_current,s_next)
                    #print("M[s_next][s_current]",M[s_current][s_next])
                    #print("(reward + gamma*value_next)",(reward + gamma*value_next))
                    #print("M[s_next, s_current]*(reward + gamma*value_next)",M[s_next][s_current]*(reward + gamma*value_next))
                    q_table[A.index(action)] += M[s_current][s_next] * (reward + gamma * value_next)

                    if s_next == 38:  # 666666 Terminal state
                        temp_array[S.index(s_current)] = np.amax(q_table)
                        a_optimal = A[np.argmax(q_table)] # first axis is the horizon, so argmax
                        pi_optimal[s_current] = a_optimal
                        break
                #print("循环之后的输出:")
                #print(np.amax(q_table[horizon]))
                temp_array[S.index(s_current)] = np.amax(q_table)
                a_optimal = A[np.argmax(q_table)]
                pi_optimal[s_current] = a_optimal

        b = datetime.now()
        print("Time for cycle", i)
        print(b - a)
        if (abs(v_array - temp_array) < error).all():
            print("*********************** Coverged ************************")
            for s in S:
                try:
                    print("State :",  s, "value: ", v_array[S[s]], ", optimal action :", pi_optimal[s])

                except KeyError:
                    print("State :", s, "value: ", v_array[S[s]], ", optimal action : none")

            return v_array
        v_array = temp_array

    return v_array


a = datetime.now()
TB_updated_v_array = value(S, A, error, max_iter, ts.M_tb, 'TB')

b = datetime.now()
print("Time for TB is:")

print(b-a)
a = datetime.now()
TB_updated_v_array = value(S, A, error, max_iter, ts.M_b, 'B')
b = datetime.now()
print("Time for B is:")

print(b-a)

a = datetime.now()
TB_updated_v_array = value(S, A, error, max_iter, ts.M_f, 'B')
b = datetime.now()
print("Time for F is:")

print(b-a)