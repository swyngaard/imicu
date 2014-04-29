module non_overlapping()
{
	color("red")
	polyhedron(
		points = [
			[5.5, 2.3, -7.9],
			[-7.1, 8.8, 0.3],
			[1.5, -6.4, 4]
		],
		triangles = [[0, 1, 2]]
	);
	
	color("blue")
	polyhedron(
		points = [
			[11.1, 10.3, -9.9],
			[12.9, 7.8, 5.3],
			[17.5, 5.4, 6]
		],
		triangles = [[0, 1, 2]]
	);
}

module overlapping()
{
	color("green")
	polyhedron(
		points = [
			[-17.4, -24.8, 8.2],
			[-26.0, -9.6, 4.5],
			[-13.4, -16.1, -3.6]
		],
		triangles = [[0, 1, 2]]
	);
	
	color("purple")
	polyhedron(
		points = [
			[-21.8, -12.1, -1.6],
			[-20.0, -14.6, 13.5],
			[-15.4, -17.0, 14.2]
		],
		triangles = [[0, 1, 2]]
	);
}

module enclosing()
{
	color("black")
	polyhedron(
		points = [
			[23.6, -28.8, 8.2],
			[15.0, -13.6, 4.5],
			[27.6, -20.1, -3.6]
		],
		triangles = [[0, 1, 2]]
	);
	
	color("yellow")
	polyhedron(
		points = [
			[24.6, -21.0, 3.2],
			[18.2, -20.1, 2.4],
			[20.0, -18.6, 2.5]
		],
		triangles = [[0, 1, 2]]
	);
}

module enclosed()
{
	color("orange")
	polyhedron(
		points = [
			[-21.3, 14.7, 9.8],
			[-19.9, 14.2, 7.9],
			[-19.0, 16.3, 9.2]
		],
		triangles = [[0, 1, 2]]
	);
	
	color("white")
	polyhedron(
		points = [
			[-19.9, 16.3, 13.5],
			[-17.6, 18.3, -1.6],
			[-21.9, 11.5, 14.2]
		],
		triangles = [[0, 1, 2]]
	);
}

non_overlapping();
overlapping();
enclosing();
enclosed();

