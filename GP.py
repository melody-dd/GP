import numpy as np
import matplotlib.pyplot as plt

def K(x, l, sigm = 1):
	return sigm**2*np.exp(-1.0/2.0/l**2*(x**2))

def f(x):
	return x**2-x**3+3+10*x+0.07*x**4

x = np.arange(0,12,0.02)
y = f(x)
plt.plot(x,y)

np.random.seed(42)
randompoint = np.random.random([5])*12.0
f_ = f(randompoint)
plt.scatter(randompoint, f_, marker='x',linewidths=4,c="black")

xsampling = np.arange(0,14,0.2)
ybayes_ = []
sigmabayes_ = []
for x in xsampling:
	f1 = f(randompoint)
	sigm_ = np.std(f1) **2
	f_ = f1 = np.average(f1)
	k = K(x-randompoint, 2, sigm_)
	C = np.zeros([randompoint.shape[0], randompoint.shape[0]])
	Ctilde = np.zeros([randompoint.shape[0]+1, randompoint.shape[0]+1])
	for i1,x1_ in np.ndenumerate(randompoint):
		for i2,x2_ in np.ndenumerate(randompoint):
			C[i1,i2] = K(x1_-x2_, 2, sigm_)
			Ctilde[0,0] = K(0, 2.0, sigm_)
			Ctilde[0,1:randompoint.shape[0]+1] = k.T
			Ctilde[1:,1:] = C
			Ctilde[1:randompoint.shape[0]+1, 0] =k
			mu = np.dot(np.dot(np.transpose(k), np.linalg.inv(C)), f_)
			sigma2 = K(0, 2.0, sigm_) - np.dot(np.dot(np.transpose(k), np.linalg.inv(C)), k )
			ybayes_.append(mu)
			sigmabayes_.append(np.abs(sigma2))
	ybayes = np.asarray(ybayes_) +np.average(f1)
	sigmabayes = np.asarray(sigmabayes_)

