## Gottesman-Kill

import numpy as np

## Initialization process
## a four qubit state
N = 4

## Start with identity matrix
table_rep = np.zeros((2 * N + 1, 2 * N))
table_rep[0: 2 * N, :] = np.identity(2 * N)
r_vec = np.zeros(2 * N + 1)

## g (x_1, z_1, x_2, z_2) takes in representation of 2 stablizer states
## compute the sign after multiplying them
def helper_g(x_1, z_1, x_2, z_2):
    match (x_1, z_1):
        case (0, 0):
            return 0
        
        case (1, 1):
            return z_2 - x_2
        
        case (1, 0):
            return z_2 * (2 * x_2 - 1)
        
        case(0, 1):
            return x_2 * (1 - 2 * z_2)

## rowsum computes the sign of multiplying two stablizer string
## That is two tensor products of pauli strings
## Notice that, the resulting summation means how much is i raised to power
## In order for stablizer states to be commutative, it has to be either 0, 2
## each representing + and -
## This function multiplies the state S_h and S_i and store the result in h
def rowsum(h, i):
    r_h = r_vec[h]
    r_i = r_vec[i]
    g = 0
    for j in range(N):
        x_1 = table_rep[i, j]
        z_1 = table_rep[i, j + N]
        x_2 = table_rep[h, j]
        z_2 = table_rep[h, j + N]
        g = g + helper_g(x_1, z_1, x_2, z_2)
    res = r_h * 2 + r_i * 2 + g
    if np.mod(res, 4) == 0:
        r_vec[h] = 0
    else:
        r_vec[h] = 1
    x_h = np.asarray(table_rep[h, 0 : N])
    x_i = np.asarray(table_rep[i, 0 : N])
    z_h = np.asarray(table_rep[h, N : 2 * N])
    z_i = np.asarray(table_rep[i, N : 2 * N])
    table_rep[h, 0 : N] = np.mod(x_h + x_i, 2)
    table_rep[h, N : 2 * N] = np.mod(z_h + z_i, 2)

## CNOT update, controlled over ath qubit to bth qubit
## a < N, b < N, because a, and b corresponds to two seperate bits
## a = X b = I => a' = X, b' = X
## a = I b = X => a' = I, b' = X
## a = Z b = I => a' = Z, b' = I
## a = I b = Z => a' = Z, b' = Z
def cnot_gate (a, b):
    for i in range(2 * N):
        r_i = r_vec[i]
        ## condition on when to flip for CNOT gate, when we have YY or XZ
        flip_flag = table_rep[i, a] * table_rep[i, b + N] * np.mod(table_rep[i, b] + table_rep[i, a + N] + 1, 2)
        r_vec[i] = np.mod(r_i + flip_flag, 2)
        table_rep[i, b] = np.mod(table_rep[i, b] + table_rep[i, a], 2)
        table_rep[i, a + N] = np.mod(table_rep[i, b + N] + table_rep[i, a + N], 2)

## hadamard gate sign is changed when we are working on Y
def hadamard_gate(a):
    for i in range(2 * N):
        r_i = r_vec[i]
        conjugate_sign = table_rep[i, a] * table_rep[i, a + N]
        r_vec[i] = np.mod(r_i + conjugate_sign, 2)
        temp_x = table_rep[i, a]
        table_rep[i, a] = table_rep[i, a + N]
        table_rep[i, a + N] = temp_x

def phase_gate(a):
    for i in range(2 * N):
        r_i = r_vec[i]
        conjugate_sign = table_rep[i, a] * table_rep[i, a + N]
        r_vec[i] = np.mod(r_i + conjugate_sign, 2)
        table_rep[i, a + N] = np.mod(table_rep[i, a + N] + table_rep[i, a], 2)

## To be implemented: X, Y, Z

## We assume we measure over standard basis; as a result, we only know the
## measurement deterministically if our state is just |0> or |1>. Which means,
## at the ath qubit, we can only have Z gate or I gate that stablizes that
## qubit.
## when we have X or Y, there are equal probability for us to measure either
## |0> or |1>. When it is random, we need to decide whether we want to measure
## |0> or |1> (which corresponds to setting the state in qubit a as either Z
## or - Z).
def measurements(a):
    # In the case when it is random
    list_one = np.flatnonzero(table_rep[N : 2 * N, a])
    print(list_one)
    if list_one.size == 0:
        # when there are no X, we enter determined case
        list_one = np.flatnonzero(table_rep[0: N, a])
        return determined_measure(list_one)
    else:
        return random_measure(list_one, a)



def determined_measure(list_one):
    table_rep[2 * N, :] = np.zeros(2 * N)
    for i in list_one:
        rowsum(2 * N, i + N)
    return r_vec[2 * N]


def random_measure(list_one, a):
    p = list_one[0]
    for i in list_one[1 :]:
        rowsum(i + N, p + N)
        table_rep[p, :] = table_rep[p + N, :]
        r_vec[p] = r_vec[p + N]
    table_rep[p + N, :] = np.zeros(2 * N)
    table_rep[p + N, a + N] = 1
    r_vec[p + N] = np.random.randint(2)
    return r_vec[p + N]

hadamard_gate(0)
print(measurements(0))
