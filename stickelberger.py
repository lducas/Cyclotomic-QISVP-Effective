#!/usr/bin/python3 -u

import time
import numpy as np
from fractions import gcd
import fractions
from math import floor
import sys
from random import getrandbits
from common import *

class CyclotomicGroupRing:
    def __init__(self, m, vs_only=False):
        self.m = m

        p, e = decomp_prime_pow(m)

        self.G = [i for i in range(self.m) if fractions.gcd(i,self.m) == 1]
        self.n = len(self.G)
        self.Gplus = self.G[:self.n//2]
        self.vs = self.compute_vs()

        if not vs_only:
            self.ws, self.ws_hash = self.compute_ws(self.vs)

        self.Basis = [(self.vs[0])] 
        self.Basis += [self.vs[i] - self.vs[i-1] for i in range(1, self.n//2)]
        
        for a in self.Gplus:
            v = np.zeros(m, dtype=np.int64)
            v[a] = 1
            v[-a] = 1
            self.Basis += [v]

        self.discr = p ** (p**(e - 1) * (p*e - e - 1))

    def compute_vs(self):
        '''
        returns the list of the w_j's. This is padded with 0 at poisitions i not coprime with m.
        '''
        vs = []

        for a in range(2, self.m+2):
            if not fractions.gcd(a, self.m) == 1:
                continue

            v = np.zeros(self.m, dtype=np.int64)

            for b in self.G:
                v[self.m-b] = floor(a * ((float(b) / self.m) % 1) )

            vs.append(v)

        return vs

    def compact(self, x):
        '''
        Remove the 0 pads, and reduce modulo 1 + tau
        '''
        assert(len(x) == self.m)
        y = np.zeros(self.n//2, dtype=np.int64)
        for i, j in enumerate(self.Gplus):
            y[i] = x[j] - x[-j]

        return y

    def G_orbit(self, x):

        orb = []
        for a in self.G:
            y = np.zeros(self.m, dtype=np.int64)
            for b in self.G:
                y[b] = x[(a * b) % self.m]
            orb.append(y)

        return orb
            
    def compute_ws(self, vs):
        '''
        returns the list of the w_j's and their orbits, and linear hash of them.
        '''

        W_pre = self.G_orbit(vs[0])

        for i in range(1, self.n):
            W_pre += self.G_orbit(vs[i] - vs[i-1])

        hash_coeff = [2*getrandbits(220) + 1 for i in range(self.n//2)]

        W = []
        W_hash = []

        for w_pre in W_pre:
            w = self.compact(w_pre)
            h = w.dot(hash_coeff) 
            if h in W_hash:
                continue
            W.append(w)
            W_hash.append(h)

        return W, W_hash


class StickelbergerCVP:
   
    def __init__(self, m, iterations):
        '''
        Set-up heuristic_CVP algorithm on the logunit lattice using the norm1.
        Will measure norm1 for each iter in iterations. 
        (iter=0 means using just Nearest_Plane).


        '''
        self.R = CyclotomicGroupRing(m)
        self.m = m
        self.n = self.R.n
        self.Basis = self.R.Basis
        self.BasisStar = gram_schmidt_orthogonalization(self.Basis)
        self.iterations = iterations       

    def instance_generator(self):
        '''
        Generate a random representation of a Class that is uniform modulo Stickerberger lattice.
        '''
        var = 100 + 10 * self.n**4
        g = np.array(np.round(np.random.normal(0, var, self.m)), dtype=np.int64)
        g *= np.array([fractions.gcd(i,self.m)==1 for i in range(self.m)], dtype=np.int64)
        return g


    def heuristic_CVP(self, g=None):
        '''
        Run the heuristic_CVP algorithm on a random input, and returns length at 
        each of the iterations.
        Compared to the paper:
        - Works with the difference d = t-c rather than with c directly
        - Uses a linear hash for cycle-detection.

        Input: The representation g of a class
        Output: A short representation of the same class 
        '''

        if g is None:
            g = self.instance_generator()

        d = nearest_plane_reduction(self.Basis, self.BasisStar, g)
        d = self.R.compact(d)
        
        h = 0
        C = set([])
        l = len(self.R.ws)        
        best = d
        best_norm = np.linalg.norm(d, 1)

        for it in range(self.iterations):
            C.add(h)
            # Compute candidate norm with acceleration trick from section TODO REF
            candidate_norms = [np.linalg.norm(d + w, 1) for w in self.R.ws]
            candidate_visited = [((h + ha) in C) for ha in self.R.ws_hash]

            # Take the best candidate (prioritazing non-visited vectors)
            L = zip(candidate_visited, candidate_norms, range(l))
            (visited, norm, i) = min(L)
            d += self.R.ws[i]
            h += self.R.ws_hash[i]
            d_norm = np.linalg.norm(d, 1)

            # A improving solution has been found
            if d_norm < best_norm:
                best, best_norm = d, d_norm

        return best_norm
