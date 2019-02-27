# Cyclotomic-QISVP-Effective
Determining the effective approximation factor of quantum algorithm [CDW17] for SVP in cyclotomic ideals

> python3 experiment.py --help
usage: experiment.py [-h] [--step STEP] [--conductor CONDUCTOR]
                     [--min-conductor MIN_CONDUCTOR]
                     [--max-conductor MAX_CONDUCTOR] [--samples SAMPLES]
                     [--naive] [--no-naive]
                     [--iteration-exponent ITERATION_EXPONENT]

Run experimental CVP on the log-unit lattice and stickelberger lattice, over a
range of conductors m. (n = euler_phi(m) below)

optional arguments:
  -h, --help            show this help message and exit
  --step STEP           'st' fot stickelberger (step 2) or 'lu' for log-unit
                        (step 4) (default: st)
  --conductor CONDUCTOR, -c CONDUCTOR
                        'prime' for prime conductors, 'pow2' for powers of 2
                        conductors (default: prime)
  --min-conductor MIN_CONDUCTOR
                        smallest conductor of the range (inclusive) (default:
                        7)
  --max-conductor MAX_CONDUCTOR, -m MAX_CONDUCTOR
                        largest conductor of the range (inclusive) (default:
                        50)
  --samples SAMPLES, -s SAMPLES
                        number of samples to average over (default: 10)
  --naive               use the naive algorithm (otherwise use the
                        HeuristicCVP) (default: False)
  --no-naive
  --iteration-exponent ITERATION_EXPONENT, -i ITERATION_EXPONENT
                        run HeuristicCVP with n^iteration_exponent iterations
                        (default: 0.5)

