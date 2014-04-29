#sage script
#To execute, enter following at the sage prompt: %runfile testKDOP.py

#Calculate minimum and maximum distances for planes of KDOP
def calculateDistances(normals, points):
	
	minimums = []
	maximums = []
	
	for normal in normals:
		dmin = sys.float_info.max
		dmax = -sys.float_info.max
		for point in points:
			distance = normal * point
			if(distance < dmin): dmin = distance
			if(distance > dmax): dmax = distance
			#print '%7.3f' % (distance),
		#print '\t\tmin: %7.3f\tmax: %7.3f' % (dmin, dmax)
		minimums.append(dmin)
		maximums.append(dmax)
	
	for minimum in minimums:
		print '%5.1ff,' % minimum,
	
	for maximum in maximums:
		print '%5.1ff,' % maximum,
	
	print ''
	
	return [minimums, maximums]

def merge(distancesA, distancesB):
	
	mergedDistances = []
	
	halfK = len(distancesA[0])
	
	for index in range(halfK):
		mergedDistances.append(min(distancesA[0][index], distancesB[0][index]))
	
	for index in range(halfK):
		mergedDistances.append(max(distancesA[1][index], distancesB[1][index]))
	
	numDistances = len(mergedDistances)
	
	for index in range(numDistances):
		if index < (numDistances-1):
			print '%5.1ff,' % mergedDistances[index],
		else:
			print '%5.1ff' % mergedDistances[index],
	
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

red    = [vector([  5.5,   2.3,  -7.9]), vector([ -7.1,   8.8,   0.3]), vector([  1.5,  -6.4,     4])]
blue   = [vector([ 11.1,  10.3,  -9.9]), vector([ 12.9,   7.8,   5.3]), vector([ 17.5,   5.4,     6])]

green  = [vector([-17.4, -24.8,   8.2]), vector([-26.0,  -9.6,   4.5]), vector([-13.4, -16.1,  -3.6])]
purple = [vector([-21.8, -12.1,  -1.6]), vector([-20.0, -14.6,  13.5]), vector([-15.4, -17.0,  14.2])]

black  = [vector([ 23.6, -28.8,   8.2]), vector([ 15.0, -13.6,   4.5]), vector([ 27.6, -20.1,  -3.6])]
yellow = [vector([ 24.6, -21.0,   3.2]), vector([ 18.2, -20.1,   2.4]), vector([ 20.0, -18.6,   2.5])]

orange = [vector([-21.3,  14.7,   9.8]), vector([-19.9,  14.2,   7.9]), vector([-19.0,  16.3,   9.2])]
white  = [vector([-19.9,  16.3,  13.5]), vector([-17.6,  18.3,  -1.6]), vector([-21.9,  11.5,  14.2])]

print 'RED & BLUE: NON-OVERLAPPING'
for norms in normals:
	merge(calculateDistances(norms, red), calculateDistances(norms, blue))
	print ''

print 'GREEN & PURPLE: OVERLAPPING'
for norms in normals:
	merge(calculateDistances(norms, green), calculateDistances(norms, purple))
	print ''

print 'BLACK & YELLOW: ENCLOSING'
for norms in normals:
	merge(calculateDistances(norms, black), calculateDistances(norms, yellow))
	print ''

print 'ORANGE & WHITE: ENCLOSED'
for norms in normals:
	merge(calculateDistances(norms, orange), calculateDistances(norms, white))
	print ''
