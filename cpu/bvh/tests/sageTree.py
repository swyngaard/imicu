#sage script
#To execute, enter following at the sage prompt: %runfile sageTree.py

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
		for minimum in minimums:
			print '%5.1ff,' % minimum,
		
		lastIndex = len(maximums)
		
		for index in range(lastIndex):
			if index != (lastIndex-1):
				print '%5.1ff,' % maximums[index],
			else:
				print '%5.1ff' % maximums[index]
	
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
		for minimum in minimums:
			print '%5.1ff,' % minimum,
		
		lastIndex = len(maximums)
		
		for index in range(lastIndex):
			if index != (lastIndex-1):
				print '%5.1ff,' % maximums[index],
			else:
				print '%5.1ff' % maximums[index]
	
	return [minimums, maximums]

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

print 'Tree KDOP distances 4 leaf nodes'

leaf_4_1 = [vector([-2.6, -3.2, -4.5]), vector([-9.7, -7.1, -9.0]), vector([-9.8, -9.1, -6.4])]
leaf_4_2 = [vector([ 8.5,  1.9, -6.4]), vector([-1.1, -4.1,  3.8]), vector([ 9.8, -2.9,  7.4])]
leaf_4_3 = [vector([-3.7, -7.5, -2.5]), vector([ 6.4, -5.6,  6.0]), vector([-8.1,  0.1, -9.4])]
leaf_4_4 = [vector([-4.7, -2.0,  6.8]), vector([ 1.3,  7.8, -6.1]), vector([-3.7,  8.6, -3.8])]

for norms in normals:
	
	merge(merge(calculateDistances(norms, leaf_4_1), calculateDistances(norms, leaf_4_2)), merge(calculateDistances(norms, leaf_4_3), calculateDistances(norms, leaf_4_4)), True)
	calculateDistances(norms, leaf_4_1, True)
	calculateDistances(norms, leaf_4_2, True)
	calculateDistances(norms, leaf_4_3, True)
	calculateDistances(norms, leaf_4_4, True)
	print ''

print 'TreeKDOP distances 5 leaf nodes'

leaf_5_1 = [vector([ 5.9,  8.3,  2.1]), vector([ 7.7, -0.8, -4.9]), vector([-3.5,  4.3, -8.6])]
leaf_5_2 = [vector([ 5.3,  5.8,  2.4]), vector([ 1.1, -4.4, -8.1]), vector([ 3.0, -2.6, -2.4])]
leaf_5_3 = [vector([ 3.1,  1.8,  6.7]), vector([ 0.1,  9.6,  8.8]), vector([ 2.6, -7.5, -5.0])]
leaf_5_4 = [vector([-0.8, -0.3,  0.8]), vector([ 9.9,  1.2, -6.5]), vector([ 7.6, -3.7,  0.9])]
leaf_5_5 = [vector([-9.3, -2.7,  9.1]), vector([ 1.8, -5.0,  6.6]), vector([-5.5, -1.8,  6.3])]

for norms in normals:
	
	merge(merge(merge(calculateDistances(norms, leaf_5_1), calculateDistances(norms, leaf_5_2)), merge(calculateDistances(norms, leaf_5_3), calculateDistances(norms, leaf_5_4))), merge(calculateDistances(norms, leaf_5_5), calculateDistances(norms, leaf_5_5)), True)
	merge(merge(calculateDistances(norms, leaf_5_1), calculateDistances(norms, leaf_5_2)), merge(calculateDistances(norms, leaf_5_3), calculateDistances(norms, leaf_5_4)), True)
	merge(calculateDistances(norms, leaf_5_5), calculateDistances(norms, leaf_5_5), True)
	calculateDistances(norms, leaf_5_1, True)
	calculateDistances(norms, leaf_5_2, True)
	calculateDistances(norms, leaf_5_3, True)
	calculateDistances(norms, leaf_5_4, True)
	calculateDistances(norms, leaf_5_5, True)
	print ''

print 'Tree KDOP distances 16 leaf nodes'

leaf_16_1  = [vector([-2.6, -2.8,  6.4]),
			  vector([-6.6,  1.6, -7.3]),
			  vector([ 5.3,  3.6,  3.4])]
leaf_16_2  = [vector([10.0, -2.1,  0.0]),
			  vector([-3.2, -7.9,  7.0]),
			  vector([ 2.2, -8.0, -6.7])]
leaf_16_3  = [vector([ 3.0, -3.5,  8.9]),
			  vector([ 7.0, -0.4,  7.6]),
			  vector([ 2.2,  2.1, -9.3])]
leaf_16_4  = [vector([-3.8, -1.3, -8.5]),
			  vector([-2.9, -5.5, -4.0]),
			  vector([ 5.0, -2.8,  1.4])]
leaf_16_5  = [vector([-2.8,  5.2, -3.3]),
			  vector([ 5.5,  0.1,  3.1]),
			  vector([ 1.1,  0.3,  8.9])]
leaf_16_6  = [vector([ 2.9,  0.2,  0.9]),
			  vector([-3.7, -9.2,  9.7]),
			  vector([-4.6, -5.9,  5.2])]
leaf_16_7  = [vector([-5.4, -8.9, -7.8]),
			  vector([-8.2,  3.7, -1.5]),
			  vector([-8.1, -9.6,  3.7])]
leaf_16_8  = [vector([ 8.6,  9.8, -3.7]),
			  vector([-6.1, -5.7,  7.3]),
			  vector([ 3.3, -5.9, -3.9])]
leaf_16_9  = [vector([-1.0,  9.6,  7.2]),
			  vector([ 5.2, -1.9,  7.7]),
			  vector([-6.6,  9.7,  3.3])]
leaf_16_10 = [vector([-3.5,  2.6,  2.3]),
			  vector([ 7.1, -0.2,  5.5]),
			  vector([-8.8,  9.0, -7.5])]
leaf_16_11 = [vector([-0.3, -5.0, 10.0]),
			  vector([-5.6,  5.1, -1.5]),
			  vector([ 7.4, -7.3, -4.6])]
leaf_16_12 = [vector([ 1.1,  4.9, -3.3]),
			  vector([-9.2,  4.7,  0.5]),
			  vector([-6.4,  4.8, -9.5])]
leaf_16_13 = [vector([-7.9,  9.4,  9.1]),
			  vector([ 0.2, -3.8,  6.0]),
			  vector([ 3.3,  8.9,  9.2])]
leaf_16_14 = [vector([-0.6,  2.6,  2.1]),
			  vector([-9.1, -3.1, -0.5]),
			  vector([-4.2,  3.6, -5.6])]
leaf_16_15 = [vector([-2.1,  5.5, -3.8]),
			  vector([-1.1,  7.5, -0.4]),
			  vector([ 1.0, -4.3, -7.2])]
leaf_16_16 = [vector([-6.6,  9.3, -4.5]),
			  vector([-0.9, -9.7,  3.3]),
			  vector([-6.1, -1.0,  3.3])]

for norms in normals:
	
	merge(merge(merge(merge(calculateDistances(norms, leaf_16_1), calculateDistances(norms, leaf_16_2)),merge(calculateDistances(norms, leaf_16_3), calculateDistances(norms, leaf_16_4))),merge(merge(calculateDistances(norms, leaf_16_5), calculateDistances(norms, leaf_16_6)),merge(calculateDistances(norms, leaf_16_7), calculateDistances(norms, leaf_16_8)))),merge(merge(merge(calculateDistances(norms, leaf_16_9), calculateDistances(norms, leaf_16_10)),merge(calculateDistances(norms, leaf_16_11), calculateDistances(norms, leaf_16_12))),merge(merge(calculateDistances(norms, leaf_16_13), calculateDistances(norms, leaf_16_14)),merge(calculateDistances(norms, leaf_16_15), calculateDistances(norms, leaf_16_16)))),True)
	merge(merge(calculateDistances(norms, leaf_16_1 ), calculateDistances(norms, leaf_16_2 )),merge(calculateDistances(norms, leaf_16_3 ), calculateDistances(norms, leaf_16_4 )), True)
	merge(merge(calculateDistances(norms, leaf_16_5 ), calculateDistances(norms, leaf_16_6 )),merge(calculateDistances(norms, leaf_16_7 ), calculateDistances(norms, leaf_16_8 )), True)
	merge(merge(calculateDistances(norms, leaf_16_9 ), calculateDistances(norms, leaf_16_10)),merge(calculateDistances(norms, leaf_16_11), calculateDistances(norms, leaf_16_12)), True)
	merge(merge(calculateDistances(norms, leaf_16_13), calculateDistances(norms, leaf_16_14)),merge(calculateDistances(norms, leaf_16_15), calculateDistances(norms, leaf_16_16)), True)
	calculateDistances(norms, leaf_16_1, True)
	calculateDistances(norms, leaf_16_2, True)
	calculateDistances(norms, leaf_16_3, True)
	calculateDistances(norms, leaf_16_4, True)
	calculateDistances(norms, leaf_16_5, True)
	calculateDistances(norms, leaf_16_6, True)
	calculateDistances(norms, leaf_16_7, True)
	calculateDistances(norms, leaf_16_8, True)
	calculateDistances(norms, leaf_16_9, True)
	calculateDistances(norms, leaf_16_10, True)
	calculateDistances(norms, leaf_16_11, True)
	calculateDistances(norms, leaf_16_12, True)
	calculateDistances(norms, leaf_16_13, True)
	calculateDistances(norms, leaf_16_14, True)
	calculateDistances(norms, leaf_16_15, True)
	calculateDistances(norms, leaf_16_16, True)
	print ''

print 'Tree KDOP distances 17 leaf nodes'

leaf_17_1  = [vector([ 6.2, -3.2,  3.0]),
			  vector([-3.1, -9.2,  7.7]),
			  vector([ 9.5, -7.1, -2.3])]
leaf_17_2  = [vector([-7.1,  1.3, -5.7]),
			  vector([ 6.2,  0.7,  4.3]),
			  vector([ 4.0,  0.0,  4.6])]
leaf_17_3  = [vector([ 3.3,  5.2, -8.2]),
			  vector([-2.2, -4.6,  0.6]),
			  vector([ 3.1,  1.7, -3.4])]
leaf_17_4  = [vector([-3.7, -8.1,  6.6]),
			  vector([-5.7, -3.5, -0.6]),
			  vector([-2.2,  4.2,  7.8])]
leaf_17_5  = [vector([-9.7,  9.2, -7.1]),
			  vector([ 6.4, -9.9,  9.9]),
			  vector([ 3.6, -1.0,  3.2])]
leaf_17_6  = [vector([-4.9,  2.1,  7.4]),
			  vector([ 2.6,  8.7, -1.3]),
			  vector([-9.1, -0.7, -0.7])]
leaf_17_7  = [vector([-2.6, -1.9,  4.0]),
			  vector([-7.7,  8.9,  5.5]),
			  vector([ 9.1, -8.3,  9.2])]
leaf_17_8  = [vector([-2.0,  1.6, -3.0]),
			  vector([-6.6, -6.8,  6.5]),
			  vector([ 2.9,  5.8, -8.6])]
leaf_17_9  = [vector([ 8.2,  5.1,  2.3]),
			  vector([-8.1,  4.2, -6.3]),
			  vector([-8.4, -5.0, -0.6])]
leaf_17_10 = [vector([ 8.5,  0.4,  4.9]),
			  vector([-7.1, -0.2,  0.3]),
			  vector([ 3.5, -9.9,  3.9])]
leaf_17_11 = [vector([-8.0, -5.9,  5.3]),
			  vector([-3.6,  3.4, -8.2]),
			  vector([-4.2,  0.9, -0.1])]
leaf_17_12 = [vector([-4.0, -4.4, -6.1]),
			  vector([ 9.5, -6.8,  1.3]),
			  vector([-3.5, -8.7,  3.2])]
leaf_17_13 = [vector([ 9.7, -9.4,  4.2]),
			  vector([ 5.7,  6.6, -7.5]),
			  vector([-6.1,  5.4, -0.5])]
leaf_17_14 = [vector([-5.6,  0.2, -4.0]),
			  vector([ 9.8, -2.8,  9.5]),
			  vector([-5.8, -6.3, -6.3])]
leaf_17_15 = [vector([-2.4, -2.2, -2.1]),
			  vector([10.0,  4.8,  2.4]),
			  vector([-8.4, -4.3, -5.7])]
leaf_17_16 = [vector([ 1.0,  9.6,  9.9]),
			  vector([-4.9, -1.3,  6.8]),
			  vector([ 6.0, -9.0,  5.7])]
leaf_17_17 = [vector([ 1.1, -5.6,  5.6]),
			  vector([-5.6, -6.9, -9.7]),
			  vector([-7.7,  2.7, -0.5])]

for norms in normals:
	
	merge(merge(merge(merge(merge(calculateDistances(norms, leaf_17_1 ), calculateDistances(norms, leaf_17_2 )), merge(calculateDistances(norms, leaf_17_3 ), calculateDistances(norms, leaf_17_4 ))), merge(merge(calculateDistances(norms, leaf_17_5 ), calculateDistances(norms, leaf_17_6 )), merge(calculateDistances(norms, leaf_17_7 ), calculateDistances(norms, leaf_17_8 )))), merge(merge(merge(calculateDistances(norms, leaf_17_9 ), calculateDistances(norms, leaf_17_10)), merge(calculateDistances(norms, leaf_17_11), calculateDistances(norms, leaf_17_12))), merge(merge(calculateDistances(norms, leaf_17_13), calculateDistances(norms, leaf_17_14)), merge(calculateDistances(norms, leaf_17_15), calculateDistances(norms, leaf_17_16))))), merge(merge(calculateDistances(norms, leaf_17_17), calculateDistances(norms, leaf_17_17)), merge(calculateDistances(norms, leaf_17_17), calculateDistances(norms, leaf_17_17))), True)
	
	merge(merge(merge(merge(calculateDistances(norms, leaf_17_1 ), calculateDistances(norms, leaf_17_2 )), merge(calculateDistances(norms, leaf_17_3 ), calculateDistances(norms, leaf_17_4 ))), merge(merge(calculateDistances(norms, leaf_17_5 ), calculateDistances(norms, leaf_17_6 )), merge(calculateDistances(norms, leaf_17_7 ), calculateDistances(norms, leaf_17_8 )))), merge(merge(merge(calculateDistances(norms, leaf_17_9 ), calculateDistances(norms, leaf_17_10)), merge(calculateDistances(norms, leaf_17_11), calculateDistances(norms, leaf_17_12))), merge(merge(calculateDistances(norms, leaf_17_13), calculateDistances(norms, leaf_17_14)), merge(calculateDistances(norms, leaf_17_15), calculateDistances(norms, leaf_17_16)))), True)
	merge(merge(calculateDistances(norms, leaf_17_17), calculateDistances(norms, leaf_17_17)), merge(calculateDistances(norms, leaf_17_17), calculateDistances(norms, leaf_17_17)), True)
	
	merge(merge(calculateDistances(norms, leaf_17_1 ), calculateDistances(norms, leaf_17_2 )), merge(calculateDistances(norms, leaf_17_3 ), calculateDistances(norms, leaf_17_4 )), True)
	merge(merge(calculateDistances(norms, leaf_17_5 ), calculateDistances(norms, leaf_17_6 )), merge(calculateDistances(norms, leaf_17_7 ), calculateDistances(norms, leaf_17_8 )), True)
	merge(merge(calculateDistances(norms, leaf_17_9 ), calculateDistances(norms, leaf_17_10)), merge(calculateDistances(norms, leaf_17_11), calculateDistances(norms, leaf_17_12)), True)
	merge(merge(calculateDistances(norms, leaf_17_13), calculateDistances(norms, leaf_17_14)), merge(calculateDistances(norms, leaf_17_15), calculateDistances(norms, leaf_17_16)), True)
	merge(calculateDistances(norms, leaf_17_17), calculateDistances(norms, leaf_17_17), True)
	
	calculateDistances(norms, leaf_17_1, True)
	calculateDistances(norms, leaf_17_2, True)
	calculateDistances(norms, leaf_17_3, True)
	calculateDistances(norms, leaf_17_4, True)
	calculateDistances(norms, leaf_17_5, True)
	calculateDistances(norms, leaf_17_6, True)
	calculateDistances(norms, leaf_17_7, True)
	calculateDistances(norms, leaf_17_8, True)
	calculateDistances(norms, leaf_17_9, True)
	calculateDistances(norms, leaf_17_10, True)
	calculateDistances(norms, leaf_17_11, True)
	calculateDistances(norms, leaf_17_12, True)
	calculateDistances(norms, leaf_17_13, True)
	calculateDistances(norms, leaf_17_14, True)
	calculateDistances(norms, leaf_17_15, True)
	calculateDistances(norms, leaf_17_16, True)
	calculateDistances(norms, leaf_17_17, True)
	
	print ''
