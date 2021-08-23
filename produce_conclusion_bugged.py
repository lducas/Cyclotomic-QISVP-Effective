from math import log as ln 
from math import factorial, exp, pi

e = exp(1)

def rhf_bkz(b):
	return ( (b/(2*pi*e)) * ((b*pi)**(1./b))  )**(1./(2*b - 2))


lll = 1.022
print( "lll :", 1.022)

for b in [80, 120, 160, 300]:
	print( "bkz-%d : %.6f"%(b, rhf_bkz(b)))


for large_p in [False, True]:
	f = open("conclusion"+("-largep" if large_p else "")+".txt", "w")	

	print ("n, naive, heuristic, lowerbound, lowerbound_halved", file=f)

	for m in [10**(1+.1 * x) for x in range(51)]:
		n = m-1
		D = m**((m-2)/(2.*n))
		p = m*ln(m) if large_p else 2*m + 1

		lneta_naive = ln(D) + (0.039 * ln(p) * n**.5 + 0.32 * ln(n)**.5 * n**.5)
		lneta_heuristic = ln(D) + (0.032 * ln(p) * n**.5 + 0.117 * ln(n)**.5 * n**.5)
		lneta_lowerbound = ln(D) + (0.02927 * ln(p) * n**.5 + 0.1839 * n**.5)
		lneta_lowerbound_halved = ln(D) + (0.01463 * ln(p) * n**.5 + 0.1839 * n**.5)

		print( "%.4e, %.4e, %.4e, %.4e, %.4e"%(n, exp(lneta_naive/n), exp(lneta_heuristic/n), exp(lneta_lowerbound/n), exp(lneta_lowerbound_halved/n)), file=f)
