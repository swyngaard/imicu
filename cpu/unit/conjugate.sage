
#Testing 2D conjugate gradient method
aa=matrix([
[     32401.002,           -0.0,     -16200.001,            0.0,            0.0,            0.0,            0.0,            0.0],
[          -0.0,            1.0,            0.0,           -0.0,            0.0,            0.0,            0.0,            0.0],
[    -16200.001,           -0.0,      32401.002,            0.0,     -16200.001,            0.0,            0.0,            0.0],
[          -0.0,           -0.0,            0.0,            1.0,            0.0,           -0.0,            0.0,            0.0],
[           0.0,            0.0,     -16200.001,           -0.0,      32401.002,            0.0,     -16200.001,            0.0],
[           0.0,            0.0,           -0.0,           -0.0,            0.0,            1.0,            0.0,           -0.0],
[           0.0,            0.0,            0.0,            0.0,     -16200.001,           -0.0,      16201.001,            0.0],
[           0.0,            0.0,            0.0,            0.0,           -0.0,           -0.0,            0.0,            1.0]
]);

bb=vector(
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -13500.0, 0.0]
);

xx = aa \ bb;
