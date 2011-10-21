
#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#define NUMSTRANDS			1
#define NUMPARTICLES		50 //Needs to be multiples of 7
#define MASS				0.000000001f //0.000000001f particle mass is 0.01mg, total strand weight is 1mg
#define K_EDGE				1000000.0f //(stable value) 10000000.0f
#define K_BEND				10000000.0f
#define K_TWIST				10000000.0f
#define K_EXTRA				0.004905f
#define LENGTH				0.005f //5 millmetres separation between particles
#define D_EDGE				0.000000125f
#define D_BEND				0.000000125f
#define D_TWIST				0.000000125f
#define D_EXTRA				0.0000002f
#define GRAVITY				-0.00981f //(stable value) -0.00981f

#define MAX_LENGTH			0.0055f //Maximum length of a spring
#define MAX_LENGTH_SQUARED	((MAX_LENGTH) * (MAX_LENGTH)) //Maximum length of a spring squared

#endif

