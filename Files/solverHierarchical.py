import numpy as np
from Files.treat_data import S
import Files.treat_data as ts
from Files.Bayes import reward_function, nextState
from Files.treat_data import resourceIndex
from datetime import datetime
gamma = 0.95
error = 0.00001
max_iter = 10000

def solver_hierachical(S, A, error, max_iter, M, Neaveau):
    """
    Here we have 3 MDPs: 0-12, 0+13-20, 0+21-37(Guessed)
    We will do the optimisation with these three
    """