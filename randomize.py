import sys
lines = open("iris_proc.data").readlines()
import random
random.shuffle(lines)
out_f = open("iris2.data",'wb')
for line in lines:
	print >> out_f, line,

out_f.close()

