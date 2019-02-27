#!/usr/bin/python3 -u

from common import *
from stickelberger import *
from logunit import *
import numpy as np
from copy import copy
from math import log, pi
from scipy.special import loggamma
from functools import lru_cache
import sys
import begin
from hminus import h

sys.setrecursionlimit(9999)

@lru_cache(maxsize=2**30)
def N(d, b):
    if b< 0:
        return 0
    if b==0:
        return 1
    if d==1:
        return 2*b+1
    else:
        return N(d-1, b) + 2*sum([N(d-1, b-k) for k in range(1, b+1)])

def stickelberger_covering_bound(d, logvol):
    M = int(d * d * int(exp(logvol / d)) + 20)

    for b in range(M):
        if log(N(d, b)) > logvol:
            return b-1

    raise ValueError("M is not large enough")

@begin.start
@begin.convert(_automatic=True)
def run(max_conductor=400):
    
    for m in range(5, max_conductor):
        prime = isprime(m)
        pow2 =ispow2(m)
        if not (pow2 or prime):
            continue

        H = CyclotomicLog(m, zs_only=True)
        n = H.n

        print("m=%d, n=%d"%(m, n))

        d = n/2
        B = np.array(H.Basis)
        BB = B.dot(B.transpose())
        _, logvol = np.linalg.slogdet(BB)
        rootvol = exp(logvol / (2.*(d-1)))
        pred_rootvol = sqrt(n)/2

        print("Logunit       : root-vol / prediction    =", rootvol / pred_rootvol)
        
        if n > 100 and prime:
            assert abs(rootvol / pred_rootvol - 1) < 1e-2


        R = CyclotomicGroupRing(m, vs_only=True)
        B = np.array(R.Basis)
        BB = B.dot(B.transpose())
        _, logvol = np.linalg.slogdet(BB)
        logvol *= .5
        logvol_predicted =  (d-1)*log(2) + log(h[m])

        if prime:
            print("Stickelberger : volume / hminus*2^(d-1)  =", exp(logvol - logvol_predicted))
            assert abs(logvol - logvol_predicted ) < 1e-6
            log_hminus_prediction = log(2*m) + (d/2) * log(m / (4*pi**2))

        else:
            logvol_predicted += log(2)
            print("Stickelberger : volume / hminus*2^d      =", exp(logvol - logvol_predicted))
            assert abs(logvol - logvol_predicted ) < 1e-6
            log_hminus_prediction = log(2*m) + (d/2) * log(m / (8*pi**2))


        ratio = exp((log(h[m]) - log_hminus_prediction)/d)
        print("Stickelberger : root-hminus / prediction =",  ratio)

        if n > 100 and prime:
           assert abs(ratio-1) < 1e-2


