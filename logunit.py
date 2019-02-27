import time
import numpy as np
import scipy.linalg as sl
from fractions import gcd
import fractions
from math import log, sqrt, exp
import sys
from random import getrandbits
from common import *



def pseudo_norm(x) :
    '''
    input : x a vector in the Log Embedding
    ouput : log(||Exp(x)||_2)
    '''
    sup = np.max(x)
    return sup + log(2 * sum(np.exp(2*(x-sup))) )/2


class CyclotomicLog:
    def __init__(self, m, zs_only=False):
        
        self.m = m
        p, e = decomp_prime_pow(m)
        self.G = [i for i in range(self.m) if fractions.gcd(i,self.m) == 1]
        self.n = len(self.G)
        self.Gplus = self.G[:self.n//2]
        self.zs, self.zs_hash = self.compute_zs()
        self.Basis = [self.zs[i] - self.zs[0] for i in range(1, self.n//2)]

        if not zs_only:
            self.bs, self.bs_hash = self.compute_bs(self.zs, self.zs_hash)
        self.discr = p ** (p**(e - 1) * (p*e - e - 1))


    def embeddings(self, x):
        '''
        returns the canonical embedding function
        '''
        FFT = np.fft.fft(x)
        emb = np.zeros(self.n, dtype=complex)
        for i,j in enumerate(self.G):
            emb[i] = FFT[j]
        return emb

    def Log_embeddings(self, x):
        '''
        returns the log embedding function
        '''
        emb = self.embeddings(x)
        return np.real(np.log(np.abs(emb[:self.n//2])))

    def Exp(self, x):
        v = np.zeros(self.n, dtype=float)
        for i in range(self.n//2):   
            v[i] = exp(x[i])
            v[self.n-1-i] = v[i]
        return v

    def compute_zs(self):
        '''
        returns the list of the z_j's, together with their linear hash.
        '''

        Z_log = []
        Z_log_hash = []

        for i, j in enumerate(self.Gplus):
            z = np.zeros(self.m, dtype=complex)
            z[0] = 1
            z[j] = -1
            Z_log.append(self.Log_embeddings(z))

            Z_log_hash.append(2*getrandbits(220) + 1)

        return Z_log, Z_log_hash

    def compute_bs(self, zs, zs_hash):
        '''
        returns the list of the non-zero bi_j's, together with their linear hash.
        '''
        B_log = []
        B_log_hash = []

        for za, ha in zip(zs, zs_hash):
            for zb, hb in zip(zs, zs_hash):
                if ha==hb:
                    continue
                B_log.append(za - zb)
                B_log_hash.append(ha - hb)
        
        return B_log, B_log_hash


class LogUnitCVP:
   
    def __init__(self, m, iterations):
        '''
        Set-up heuristic_CVP algorithm on the logunit lattice using the pseudo_norm.
        Will measure pseudo_norm for each iter in iterations. 
        (iter=0 means using just Nearest_Plane).


        '''
        self.H = CyclotomicLog(m)
        self.n = self.H.n
        self.Basis = self.H.Basis
        self.BasisInv = np.linalg.pinv(self.Basis)
        self.iterations = iterations
        
        self.Expbs = np.array([np.exp(2*u) for u in self.H.bs])


    def instance_generator(self):
        '''
        Generate a radom points in H that is uniform modulo Lambda.
        '''
        var = 100 + 10 * self.n**4
        log_g = np.random.normal(0, var, self.n//2)
        log_g = proj_orth(np.array(self.n//2*[1.]), log_g)
        return log_g


    def heuristic_CVP(self, g=None):
        '''
        Run the heuristic_CVP algorithm on a random input, and returns length at 
        each of the iterations.
        Compared to the paper:
        - Works with the difference d = t-c rather than with c directly
        - Uses a linear hash for cycle-detection.

        Input: The Log of a generator of an ideal (choose g at random if g=None)
        Output: The Log of a short generator of the same ideal
        '''

        if g is None:
            g = self.instance_generator()

        d = round_off_reduction(self.Basis, self.BasisInv, g)
        
        h = 0
        C = set([])
        l = len(self.H.bs)        
        best = d
        best_norm = pseudo_norm(d)

        for it in range(self.iterations):
            C.add(h)

            # Compute candidate norm with acceleration trick from section 5.1
            e2d = np.exp(2*(d - max(d)))
            candidate_norms = self.Expbs.dot(e2d)
            candidate_visited = [((h + ha) in C) for ha in self.H.bs_hash]

            # Take the best candidate (prioritazing non-visited vectors)
            L = zip(candidate_visited, candidate_norms, range(l))
            (visited, norm, i) = min(L)
            d += self.H.bs[i]
            h += self.H.bs_hash[i]
            d_norm = pseudo_norm(d)

            # A improving solution has been found
            if d_norm < best_norm:
                best, best_norm = d, d_norm

        return best_norm

