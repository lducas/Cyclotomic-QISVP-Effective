#!/usr/bin/python3 -u

from common import *
from stickelberger import *
from logunit import *
import numpy as np
from copy import copy
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
def run(step="st", conductor="prime", max_conductor=50):

    datum_base = {'step':step, 'cond': conductor}

    for m in range(5, max_conductor):
        n = None
        if conductor=="prime":
            if not isprime(m):
                continue
            n = m-1

        elif conductor=="pow2":
            if not ispow2(m):
                continue
            n = m/2

        else:
            raise ValueError("conductor must be 'prime' or 'pow2'")

        if step== "lu":
            H = CyclotomicLog(m, zs_only=True)
            d = n/2
            B = np.array(H.Basis)
            BB = B.dot(B.transpose())
            _, logvol = np.linalg.slogdet(BB)
            logvol *= .5
            logballvol = (d-.5) * log(d) - loggamma(d)
            lower_bound = exp((logvol - logballvol)/(d-1)) + .5*log(2)


        elif step=="st":
            d = n/2
            logvol_predicted =  (d-1)*log(2) + log(h[m])
           
            R = CyclotomicGroupRing(m, vs_only=True)
            B = np.array(R.Basis)
            BB = B.dot(B.transpose())
            _, logvol = np.linalg.slogdet(BB)
            logvol *= .5

            assert abs(logvol - logvol_predicted ) < 1e-3

            lower_bound = stickelberger_covering_bound(d, logvol)

        else:
            raise ValueError("step must be 'lu' or 'st'")

        datum = copy(datum_base)

        datum["m"]  =  m
        datum["n"]  =  n
        datum["lower_bound"]  = lower_bound

        pretty_print(datum)



