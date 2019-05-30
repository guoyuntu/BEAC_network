import numpy as np

lines = open('../../result/face_4.txt','r').readlines()

acc = 0
count = 0
for line in lines:
	nofacerate = float(line.split()[-1])
	if nofacerate > 0 and nofacerate <= 0.4:
		count += 1
		if line.split()[-2] == 'True':
			acc += 1
print(acc, count, acc/count, count/218.0)

acc = 0
count = 0
for line in lines:
	nofacerate = float(line.split()[-1])
	if nofacerate > 0.4 and nofacerate <= 0.6:
		count += 1
		if line.split()[-2] == 'True':
			acc += 1
print(acc, count, acc/count, count/218.0)

acc = 0
count = 0
for line in lines:
	nofacerate = float(line.split()[-1])
	if nofacerate > 0.6 and nofacerate <= 0.8:
		count += 1
		if line.split()[-2] == 'True':
			acc += 1
print(acc, count, acc/count, count/218.0)

acc = 0
count = 0
for line in lines:
	nofacerate = float(line.split()[-1])
	if nofacerate > 0.8 and nofacerate <= 0.95:
		count += 1
		if line.split()[-2] == 'True':
			acc += 1
print(acc, count, acc/count, count/218.0)

acc = 0
count = 0
for line in lines:
	nofacerate = float(line.split()[-1])
	if nofacerate > 0.95 and nofacerate <= 1.0:
		count += 1
		if line.split()[-2] == 'True':
			acc += 1
print(acc, count, acc/count, count/218.0)