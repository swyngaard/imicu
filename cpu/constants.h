
#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#define NUMSTRANDS			1
#define NUMPARTICLES		49 //Needs to be multiples of 7
#define MASS				0.000000001f //0.000000001f particle mass is 0.01mg, total strand weight is 1mg
#define K_EDGE				0.004905f //(stable value) 10000000.0f
#define K_BEND				0.004905f
#define K_TWIST				0.004905f
#define K_EXTRA				0.004905f
#define LENGTH				0.005f //5 millmetres separation between particles
#define D_EDGE				0.000000125f
#define D_BEND				0.000000125f
#define D_TWIST				0.000000125f
#define D_EXTRA				0.0000002f
#define GRAVITY				-0.00981f//(stable value) -0.00981f

#define MAX_LENGTH			0.0055f //Maximum length of a spring
#define MAX_LENGTH_SQUARED	0.00003025f //Maximum length of a spring squared

#define DOMAIN_DIM		100
#define DOMAIN_WIDTH	0.275f
#define DOMAIN_HALF		0.1375f
#define CELL_WIDTH		0.00275f
#define CELL_HALF		0.001375f

//#define DOMAIN_DIM		5
//#define DOMAIN_WIDTH	0.275f
//#define DOMAIN_HALF		0.1375f
//#define CELL_WIDTH		0.055f
//#define CELL_HALF		0.0275f

#endif

