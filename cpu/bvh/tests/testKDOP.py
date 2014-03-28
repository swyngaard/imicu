#sage script

#Calculate minimum and maximum distances for planes of KDOP

inputPointA = vector([0.17, -1.2, 5.6])
inputPointB = vector([-1.4, 8.4, -6.4])
inputPointC = vector([2.0, -5.7, 9.4])

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

points = [inputPointA, inputPointB, inputPointC]
normals = [normals6, normals14, normals18, normals26]

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
			print '%7.3f' % (distance),
		print '\t\tmin: %7.3f\tmax: %7.3f' % (dmin, dmax)
		minimums.append(dmin)
		maximums.append(dmax)
	
	for minimum in minimums:
		print '%.3ff,' % minimum,
	
	for maximum in maximums:
		print '%.3ff,' % maximum,
	
	print '\n'


for norms in normals:
	print 'normals ' + repr(len(norms)*2)
	calculateDistances(norms, points)
