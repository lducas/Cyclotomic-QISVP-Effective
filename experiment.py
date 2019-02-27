#!/usr/bin/python3 -u

import begin
from common import *
from stickelberger import *
from logunit import *
import numpy as np
from copy import copy
import logging

def step(answer):
    "What is your favourite colour?"
    logging.info(answer)

@begin.start
@begin.convert(_automatic=True)
def run(step :               "'st' fot stickelberger (step 2) or 'lu' for log-unit (step 4)"    = "st", 
        conductor :          "'prime' for prime conductors, 'pow2' for powers of 2 conductors"  = "prime", 
        min_conductor :      "smallest conductor of the range (inclusive)"                      = 7,
        max_conductor :      "largest conductor of the range (inclusive)"                       = 50,
        samples :            "number of samples to average over"                                = 10, 
        naive :              "use the naive algorithm (otherwise use the HeuristicCVP)"         = False, 
        iteration_exponent : "run HeuristicCVP with n^iteration_exponent iterations"            =.5
        ):
    """
    Run experimental CVP on the log-unit lattice and stickelberger lattice, over a
    range of conductors m. (n = euler_phi(m) below)
    """
    
    if naive:
        iteration_exponent = -1
    datum_base = {'step':step, 'naive': naive, 'samples':samples, 
                  'cond': conductor, 'iter_exp': iteration_exponent}

    for m in range(min_conductor, max_conductor+1):
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

        iterations = int(n**iteration_exponent)

        if step== "lu":
            solver = LogUnitCVP(m, iterations)
        elif step=="st":
            solver = StickelbergerCVP(m, iterations)
        else:
            raise ValueError("step must be 'lu' or 'st'")

        assert n == solver.n

        sampled_data = [solver.heuristic_CVP() for i in range(samples)]
        
        datum = copy(datum_base)

        datum["m"]  =  m
        datum["n"]  =  n
        datum["av"]  =  sum([1.*x for x in sampled_data])/samples
        datum["stddev"] =  sum([1.*x*x for x in sampled_data])/samples
        datum["stddev"] -= datum["av"]**2
        datum["stddev"] = sqrt(datum["stddev"])
        datum["min"]  = min(sampled_data)
        datum["max"]  = max(sampled_data)

        pretty_print(datum)



