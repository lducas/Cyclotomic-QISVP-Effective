# Compute and print a table for h_minus. Computed according to Theorem 4.17 from [Washington, 2012]
# A character X mod f will be represented by r where X(a) = r^a for r an m-th root of unity
import sys

def conductor(m, r):
	for f in range(1,m+1):
		if r^f == 1:
			return f
	raise ValueError


def B1(X):
	f = X.conductor()
	s = 0
	for a in range(f):
		s+= a * X(a)
	return s/f

def hm(m):
 	Q = 1
	if len(factor(m))>1:
		Q = 2
	
	w = m

	if (m%2)==1:
		w = 2*m

	G = DirichletGroup(m)
	p = 1
	for X in G:
		if X(-1)==1:
			continue
		B1r = B1(X)
		p *= (-1/2) * B1r

	return ZZ(Q* w * p)

for m in range(3,50000):
	if len(factor(m))>1:
		continue

	hminus = hm(m)
	print("h[%d] = %d"%(m, hminus))
	sys.stdout.flush()