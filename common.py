import time
import numpy as np
import scipy.linalg as sl

import hashlib as hl
import fractions
from math import log
import sys

if sys.version_info[0] != 3:
    raise ValueError()

def pretty_print(datum):
    s = "{"
    for key,val in datum.items():
        if isinstance(val, float):
            s+= ("'%s': %.3f, "%(key, val))
        elif isinstance(val, str):
            s+= ("'%s': '%s', "%(key, val))
        else:
            s+= ("'%s': %s, "%(key, str(val)))
    s+= "}"
    print(s)


def isprime(x):
    for i in range(2, int(x**(1./2) + 1)):
        if x % i == 0:
            return False
    else:
        return True


def ispow2(x):
    e = int(log(x)/log(2))
    return x == 2**e


def decomp_prime_pow(x):

    for p in range(2, int(x**(1./2) + 1)):
        if x % p == 0:
            assert isprime(p)
            e = int(round(log(x)/log(p)))
            if not p**e == x:
                print(x, p, e)
                raise ValueError("This is not a prime power")
            return (p, e)
    else:
        return (x, 1)


def proj_orth(u, v):
    '''
    project v orthogonally to span(u)
    '''
    return v - ((np.dot(v,u) / np.dot(u,u)) * u)


def extract_linindep(W):
    '''
    extract a maximal subset of linearly independant vectors
    '''
    B = np.zeros((W.shape[1]+1,W.shape[1]))
    i = 0
    for x in W :
        B_temp = B.copy()
        B_temp[i,:] = x[:]
        if np.linalg.matrix_rank(B_temp) == i+1 :
            B[i,:] = x[:]
            i += 1
    return B[:-1,:]


def gram_schmidt_orthogonalization(B):
    '''
    Compute the Gram Schmidt Orthogonalization of a basis B (a list of vector)
    '''
    d = len(B)
    Q, R = np.linalg.qr(np.array(B, dtype=float).transpose())
    Q = Q.transpose()
    for i in range(d):
       Q[i] *= R[i,i]

    return Q

def nearest_plane_reduction(B, Bstar, x):
    ''' 
    Apply the Nearest Plane algorithm to reduce input x modulo the lattice L(B)
    '''
    
    d = len(B)
    s = x

    for i in range(d)[::-1]:
        c = np.dot(s, Bstar[i]) / np.dot(Bstar[i], Bstar[i])
        s = s - int(round(c)) * B[i]

    return s


def round_off_reduction(B, Binv, x):
    ''' 
    Apply the Round-off algorithm to reduce input x modulo the lattice L(B)
    '''
    y = np.dot(x, Binv)
    y -= np.round(y)

    return np.dot(y, B)