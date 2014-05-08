#sage script
#To execute, enter following at the sage prompt: %runfile sageTree.py

def printDistances(distances):
	for minimum in distances[0]:
		print '%5.1ff,' % minimum,
	
	lastIndex = len(distances[1])
	
	for index in range(lastIndex):
		if index != (lastIndex-1):
			print '%5.1ff,' % distances[1][index],
		else:
			print '%5.1ff' % distances[1][index]

#Calculate minimum and maximum distances for planes of KDOP
def calculateDistances(normals, points, show=False):
	
	minimums = []
	maximums = []
	
	for normal in normals:
		dmin = sys.float_info.max
		dmax = -sys.float_info.max
		for point in points:
			distance = normal * point
			if(distance < dmin): dmin = distance
			if(distance > dmax): dmax = distance
		minimums.append(dmin)
		maximums.append(dmax)
	
	#printing
	if show == True:
		printDistances([minimums, maximums])
	
	return [minimums, maximums]

def merge(distancesA, distancesB, show=False):
	
	minimums = []
	maximums = []
	
	halfK = len(distancesA[0])
	
	for index in range(halfK):
		minimums.append(min(distancesA[0][index], distancesB[0][index]))
		maximums.append(max(distancesA[1][index], distancesB[1][index]))
	
	#printing
	if show == True:
		printDistances([minimums, maximums])
	
	return [minimums, maximums]

def merge4(distancesA, distancesB, distancesC, distancesD, show=False):
	
	mergedDistances = merge(merge(distancesA, distancesB), merge(distancesC, distancesD))
	
	if show == True:
		printDistances(mergedDistances)
	
	return mergedDistances

normals6  = [vector([ 1, 0, 0]),
			 vector([ 0, 1, 0]),
			 vector([ 0, 0, 1])]

normals14 = [vector([ 1, 0, 0]),
			 vector([ 0, 1, 0]),
			 vector([ 0, 0, 1]),
			 vector([ 1, 1, 1]),
			 vector([ 1,-1, 1]),
			 vector([ 1, 1,-1]),
			 vector([ 1,-1,-1])]

normals18 = [vector([ 1, 0, 0]),
			 vector([ 0, 1, 0]),
			 vector([ 0, 0, 1]),
			 vector([ 1, 1, 0]),
			 vector([ 1, 0, 1]),
			 vector([ 0, 1, 1]),
			 vector([ 1,-1, 0]),
			 vector([ 1, 0,-1]),
			 vector([ 0, 1,-1])]

normals26 = [vector([ 1, 0, 0]),
			 vector([ 0, 1, 0]),
			 vector([ 0, 0, 1]),
			 vector([ 1, 1, 1]),
			 vector([ 1,-1, 1]),
			 vector([ 1, 1,-1]),
			 vector([ 1,-1,-1]),
			 vector([ 1, 1, 0]),
			 vector([ 1, 0, 1]),
			 vector([ 0, 1, 1]),
			 vector([ 1,-1, 0]),
			 vector([ 1, 0,-1]),
			 vector([ 0, 1,-1])]

normals = [normals6, normals14, normals18, normals26]

print 'Tree KDOP distances 1 leaf node'

leaf_1_1 = [vector([10.0,  5.8,  7.2]), vector([-4.9, -5.2, -3.0]), vector([-0.9,  7.2,  1.4])]

for norms in normals:
	calculateDistances(norms, leaf_1_1, True)
	print ''

print 'Tree KDOP distances 4 leaf nodes'

leaf_4_1 = [vector([ 8.9, -9.8,  6.6]), vector([-1.1,  6.6, -5.7]), vector([ 3.8,  8.4,  2.9])]
leaf_4_2 = [vector([-3.7,  8.7,  1.9]), vector([-3.7, -8.2, -7.0]), vector([ 5.7,  5.2,  3.1])]
leaf_4_3 = [vector([-1.4, -7.9,  9.4]), vector([ 4.1,  0.5,  7.0]), vector([ 2.9,  6.8,  7.6])]
leaf_4_4 = [vector([-2.3, -1.9, -2.1]), vector([ 6.3, -6.2, -3.3]), vector([ 6.5, -6.8,  6.1])]

for norms in normals:
	merge(merge(calculateDistances(norms, leaf_4_1), calculateDistances(norms, leaf_4_2)), merge(calculateDistances(norms, leaf_4_3), calculateDistances(norms, leaf_4_4)), True)
	calculateDistances(norms, leaf_4_1, True)
	calculateDistances(norms, leaf_4_2, True)
	calculateDistances(norms, leaf_4_3, True)
	calculateDistances(norms, leaf_4_4, True)
	print ''


print 'Tree KDOP distances 5 leaf nodes'

leaf_5_1 = [vector([ 6.6,  0.5, -7.6]), vector([ 6.2, -8.9,  4.8]), vector([-2.2, -6.6,  7.8])]
leaf_5_2 = [vector([-0.3, -4.4,  8.1]), vector([-0.1, -3.6, -3.3]), vector([ 0.8,  5.3, -0.2])]
leaf_5_3 = [vector([ 1.4,  5.7,  7.8]), vector([-6.0, -6.6, -1.4]), vector([ 3.4, -2.8, -2.0])]
leaf_5_4 = [vector([ 8.1, -4.2, -1.5]), vector([ 5.0,  9.9,  9.9]), vector([ 7.8,  5.2,  8.4])]
leaf_5_5 = [vector([ 6.5,  7.7, -7.3]), vector([ 9.4, -7.8,  6.5]), vector([ 6.0, -6.4,  2.4])]

for norms in normals:
	
	merge(merge(merge(calculateDistances(norms, leaf_5_1),calculateDistances(norms, leaf_5_2)), merge(calculateDistances(norms, leaf_5_3),calculateDistances(norms, leaf_5_4))), merge(calculateDistances(norms, leaf_5_5), calculateDistances(norms, leaf_5_5)), True)
	merge(merge(calculateDistances(norms, leaf_5_1),calculateDistances(norms, leaf_5_2)), merge(calculateDistances(norms, leaf_5_3),calculateDistances(norms, leaf_5_4)), True)
	merge(calculateDistances(norms, leaf_5_5), calculateDistances(norms, leaf_5_5), True)
	calculateDistances(norms, leaf_5_1, True)
	calculateDistances(norms, leaf_5_2, True)
	calculateDistances(norms, leaf_5_3, True)
	calculateDistances(norms, leaf_5_4, True)
	calculateDistances(norms, leaf_5_5, True)
	print ''

print 'Tree KDOP distances 16 leaf nodes'

leaf_16_1  = [vector([ 7.6,  1.9, -3.9]), vector([-5.8, -7.3, -7.4]), vector([ 2.9, -6.8,  1.8])]
leaf_16_2  = [vector([-6.8,  3.1,  9.0]), vector([ 6.2, -2.1,  7.9]), vector([-3.3,  6.0, -7.6])]
leaf_16_3  = [vector([-9.2, -2.7,  1.7]), vector([ 4.6,  5.9, -4.0]), vector([-6.4,  8.5,  7.1])]
leaf_16_4  = [vector([ 6.9, -5.0,  7.4]), vector([-7.7, -8.3, -4.8]), vector([-3.8,  6.6,  2.0])]
leaf_16_5  = [vector([ 9.2, -0.6, -2.7]), vector([-2.9,  7.5,  2.7]), vector([ 4.7,  1.7,  2.9])]
leaf_16_6  = [vector([-4.1, -5.7, -7.5]), vector([ 3.7,  3.0, -6.2]), vector([ 7.9,  1.4, -0.2])]
leaf_16_7  = [vector([-1.3,  8.8, -4.6]), vector([-4.9,  0.3,  1.1]), vector([-9.0,  8.1, -1.9])]
leaf_16_8  = [vector([-2.2, -9.5,  0.8]), vector([-9.5,  1.2, -3.4]), vector([ 2.2, -0.7, -2.2])]
leaf_16_9  = [vector([ 1.9,  3.1,  2.3]), vector([-6.9, -5.1, -9.8]), vector([-2.9,  9.3,  5.5])]
leaf_16_10 = [vector([ 2.6,  4.9, -5.9]), vector([-2.1, -7.8,  3.9]), vector([-5.0,  1.3, -3.5])]
leaf_16_11 = [vector([ 5.4,  1.0, -3.0]), vector([-0.2,  9.0, -3.4]), vector([-3.1,  4.2, -3.3])]
leaf_16_12 = [vector([-0.1, -0.7, -1.1]), vector([-7.5,  3.8, -9.0]), vector([-7.3,  7.5, -0.6])]
leaf_16_13 = [vector([-0.6,  2.7,  1.0]), vector([ 0.6, -3.9,  4.8]), vector([-9.1, -4.0,  7.6])]
leaf_16_14 = [vector([ 3.6,  5.2, -9.0]), vector([ 5.1,  7.1, -7.3]), vector([-7.8, -3.4,  7.8])]
leaf_16_15 = [vector([-6.9,  1.6,  4.4]), vector([ 3.5, -0.2,  1.1]), vector([ 3.9,  9.0,  5.0])]
leaf_16_16 = [vector([-1.7,  0.0, -0.2]), vector([ 1.7,  7.7,  9.4]), vector([-3.1,  4.1,  2.0])]


for norms in normals:
	k1  = calculateDistances(norms, leaf_16_1)
	k2  = calculateDistances(norms, leaf_16_2)
	k3  = calculateDistances(norms, leaf_16_3)
	k4  = calculateDistances(norms, leaf_16_4)
	k5  = calculateDistances(norms, leaf_16_5)
	k6  = calculateDistances(norms, leaf_16_6)
	k7  = calculateDistances(norms, leaf_16_7)
	k8  = calculateDistances(norms, leaf_16_8)
	k9  = calculateDistances(norms, leaf_16_9)
	k10 = calculateDistances(norms, leaf_16_10)
	k11 = calculateDistances(norms, leaf_16_11)
	k12 = calculateDistances(norms, leaf_16_12)
	k13 = calculateDistances(norms, leaf_16_13)
	k14 = calculateDistances(norms, leaf_16_14)
	k15 = calculateDistances(norms, leaf_16_15)
	k16 = calculateDistances(norms, leaf_16_16)
	
	k01 = merge4(k1 , k2 , k3 , k4 )
	k02 = merge4(k5 , k6 , k7 , k8 )
	k03 = merge4(k9 , k10, k11, k12)
	k04 = merge4(k13, k14, k15, k16)
	
	k001 = merge4(k01, k02, k03, k04)
	
	printDistances(k001)
	printDistances(k01)
	printDistances(k02)
	printDistances(k03)
	printDistances(k04)
	printDistances(k1)
	printDistances(k2)
	printDistances(k3)
	printDistances(k4)
	printDistances(k5)
	printDistances(k6)
	printDistances(k7)
	printDistances(k8)
	printDistances(k9)
	printDistances(k10)
	printDistances(k11)
	printDistances(k12)
	printDistances(k13)
	printDistances(k14)
	printDistances(k15)
	printDistances(k16)
	
	print ''

print 'Tree KDOP distances 17 leaf nodes'

leaf_17_1  = [vector([-7.9,  1.3,  8.4]), vector([ 3.4,  4.0, -7.8]), vector([ 9.3, -7.5, -6.9])]
leaf_17_2  = [vector([ 6.4,  5.4,  4.8]), vector([ 6.5, -3.9, -5.0]), vector([-5.9, -5.8, -5.5])]
leaf_17_3  = [vector([ 5.6, -1.8, -4.3]), vector([-6.7, -4.7, -4.1]), vector([ 7.2,  9.2,  2.8])]
leaf_17_4  = [vector([ 1.2, -5.6, -5.2]), vector([ 7.5,  9.9,  0.2]), vector([-2.7, -1.4,  4.6])]
leaf_17_5  = [vector([ 2.5,  3.5, -9.4]), vector([-8.4,  7.1,  8.6]), vector([-5.1,  6.3,  3.7])]
leaf_17_6  = [vector([ 3.7, -5.7, -3.0]), vector([-8.8,  7.9,  7.4]), vector([ 9.3,  7.7, -3.3])]
leaf_17_7  = [vector([-8.4,  6.3, -4.2]), vector([-2.2, -0.3, -5.4]), vector([ 9.1,  7.0, -8.8])]
leaf_17_8  = [vector([-7.7,  0.5, -7.4]), vector([ 9.4, -4.1, -3.6]), vector([ 8.8, -4.3, -2.9])]
leaf_17_9  = [vector([-7.1, -7.3, -5.2]), vector([ 3.2, -3.5, -6.2]), vector([ 6.0, -7.6,  1.5])]
leaf_17_10 = [vector([ 2.4, -5.1,  6.1]), vector([ 3.9, -4.4, -5.4]), vector([-5.2,  8.7, -0.3])]
leaf_17_11 = [vector([ 4.5,  4.7,  9.9]), vector([-4.2, -7.4,  0.4]), vector([ 6.7,  3.1, -1.4])]
leaf_17_12 = [vector([ 7.1,  5.3,  5.8]), vector([ 2.1, -2.5, -6.6]), vector([-4.1, -3.3, -3.6])]
leaf_17_13 = [vector([-6.3, -8.5,  9.7]), vector([ 7.2, -5.7,  1.8]), vector([-4.7,  5.6,  0.1])]
leaf_17_14 = [vector([-4.9,  2.4, -6.9]), vector([ 4.5,  6.2, -4.3]), vector([-1.1, -0.2,  4.9])]
leaf_17_15 = [vector([ 7.1, -4.6,  0.1]), vector([-5.1,  1.8,  8.5]), vector([-5.3, -3.3, -5.2])]
leaf_17_16 = [vector([-1.8, -6.3,  6.4]), vector([-2.6, -7.9, -0.9]), vector([ 3.0,  2.8,  4.5])]
leaf_17_17 = [vector([ 7.1, -0.2, -9.2]), vector([ 7.2,  4.3,  4.6]), vector([-4.5,  8.4,  8.5])]

for norms in normals:
	k1  = calculateDistances(norms, leaf_17_1)
	k2  = calculateDistances(norms, leaf_17_2)
	k3  = calculateDistances(norms, leaf_17_3)
	k4  = calculateDistances(norms, leaf_17_4)
	k5  = calculateDistances(norms, leaf_17_5)
	k6  = calculateDistances(norms, leaf_17_6)
	k7  = calculateDistances(norms, leaf_17_7)
	k8  = calculateDistances(norms, leaf_17_8)
	k9  = calculateDistances(norms, leaf_17_9)
	k10 = calculateDistances(norms, leaf_17_10)
	k11 = calculateDistances(norms, leaf_17_11)
	k12 = calculateDistances(norms, leaf_17_12)
	k13 = calculateDistances(norms, leaf_17_13)
	k14 = calculateDistances(norms, leaf_17_14)
	k15 = calculateDistances(norms, leaf_17_15)
	k16 = calculateDistances(norms, leaf_17_16)
	k17 = calculateDistances(norms, leaf_17_17)
	
	k01 = merge4(k1 , k2 , k3 , k4 )
	k02 = merge4(k5 , k6 , k7 , k8 )
	k03 = merge4(k9 , k10, k11, k12)
	k04 = merge4(k13, k14, k15, k16)
	k05 = k17
	
	k001 = merge4(k01, k02, k03, k04)
	k002 = k05
	
	k0001 = merge(k001, k002)
	
	printDistances(k0001)
	printDistances(k001)
	printDistances(k002)
	printDistances(k01)
	printDistances(k02)
	printDistances(k03)
	printDistances(k04)
	printDistances(k05)
	printDistances(k1)
	printDistances(k2)
	printDistances(k3)
	printDistances(k4)
	printDistances(k5)
	printDistances(k6)
	printDistances(k7)
	printDistances(k8)
	printDistances(k9)
	printDistances(k10)
	printDistances(k11)
	printDistances(k12)
	printDistances(k13)
	printDistances(k14)
	printDistances(k15)
	printDistances(k16)
	printDistances(k17)
	
	print ''
