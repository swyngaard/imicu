
x0 = 0.25;
x1 = 0.50;
x2 = 0.75;
x3 = 1.00;

v0 = 0.50;
v1 = 0.50;
v2 = 0.50;
v3 = 0.50;

d00 = -x0/abs(x0);
d01 = (x1-x0)/abs(x1-x0);
d10 = (x0-x1)/abs(x0-x1);
d12 = (x2-x1)/abs(x2-x1);
d21 = (x1-x2)/abs(x1-x2);
d23 = (x3-x2)/abs(x3-x2);
d32 = (x2-x3)/abs(x2-x3);

h = 1.25;
g = 0.75;
length = 0.005;

a = matrix([[1+h*d00*d00+h*d01*d01, -h*d01*d01,            0.0],
			[-h*d10*d10,            1+h*d10*d10+h*d12*d12, -h*d12*d12],
			[0.0,                   -h*d21*d21,            1+h*d21*d21]]);

b = vector([v0 + g*(-x0*d00-length)*d00 + g*((x1-x0)*d01-length)*d01,
			v1 + g*((x0-x1)*d10-length)*d10 + g*((x2-x1)*d12-length)*d12, 
			v2 + g*((x1-x2)*d21-length)*d21]);

x = a \ b;

a4 = matrix([[1+h*d00*d00+h*d01*d01, -h*d01*d01,            0.0,                   0.0],
			 [-h*d10*d10,            1+h*d10*d10+h*d12*d12, -h*d12*d12,            0.0],
			 [0.0,                   -h*d21*d21,            1+h*d21*d21+h*d23*d23, -h*d23*d23],
			 [0.0,                   0.0,                   -h*d32*d32,            1+h*d32*d32]]);

b4 = vector([v0 + g*(-x0*d00-length)*d00 + g*((x1-x0)*d01-length)*d01,
			 v1 + g*((x0-x1)*d10-length)*d10 + g*((x2-x1)*d12-length)*d12,
			 v2 + g*((x1-x2)*d21-length)*d21 + g*((x3-x2)*d23-length)*d23,
			 v3 + g*((x2-x3)*d32-length)*d32]);

x4 = a4 \ b4;

