
#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#define NUMSTRANDS			1
#define NUMPARTICLES		33 //Needs to be multiples of 7
#define NUMCOMPONENTS		2
#define MASS				0.005f //0.000000001f particle mass is 0.01mg, total strand weight is 1mg
#define K_EDGE				1.5f //(stable value) 0.005f
#define K_BEND				0.004905f
#define K_TWIST				0.004905f
#define K_EXTRA				0.004905f
#define LENGTH				0.005f //5 millimetres separation between particles
#define D_EDGE				1500.0f //-5.5f
#define D_BEND				0.125f
#define D_TWIST				0.125f
#define D_EXTRA				0.125f
#define GRAVITY				-9.81f//(stable value) -9.81f

//Bounding volume constants
#define KDOP_PLANES			26 //other valid values include 6, 14 & 18.

//Strain limiting constants
#define MAX_LENGTH			0.0055f //Maximum length of a spring
#define MAX_LENGTH_SQUARED	0.00003025f //Maximum length of a spring squared

//Geometry collisions constants
#define DOMAIN_DIM		100
#define DOMAIN_WIDTH	0.275f
#define DOMAIN_HALF		0.1375f
#define CELL_WIDTH		0.00275f
#define CELL_HALF		0.001375f

//#define DOMAIN_DIM		200
//#define DOMAIN_WIDTH	0.275f
//#define DOMAIN_HALF		0.1375f
//#define CELL_WIDTH		0.001375f
//#define CELL_HALF		0.0006875f

//#define DOMAIN_DIM		5
//#define DOMAIN_WIDTH	0.275f
//#define DOMAIN_HALF		0.1375f
//#define CELL_WIDTH		0.055f
//#define CELL_HALF		0.0275f

#endif

