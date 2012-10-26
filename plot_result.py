from pylab import *
import sys

content = open(sys.argv[1],'rb').readlines()

X = []
Y = []

for line in content:
	line = line.rstrip()
	if line=="":continue
	x,y = line.split(" ")
	x = float(x)
	y = float(y)
	X.append(x)
	Y.append(y)

plot(X,Y,'ro')
show()
