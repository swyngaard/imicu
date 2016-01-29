
#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#define NUMSTRANDS			2
#define NUMPARTICLES		9 //Needs to be multiples of 5???
#define NUMCOMPONENTS		3 // 3D vectors
#define NUMSEGMENTS			(NUMPARTICLES-1)
#define MASS				0.002f //0.000000001f particle mass is 0.01mg, total strand weight is 1mg
#define K_EDGE				3.2f //(stable value) 10000000.0f
#define K_BEND				0.08f
#define K_TWIST				0.08f
#define K_EXTRA				0.004905f
#define LENGTH				0.005f //5 millmetres separation between particles
#define LENGTH_EDGE			0.005f  //length between edge springs
#define LENGTH_BEND			0.005f //length between bending springs
#define LENGTH_TWIST		0.005f  //length between twisting springs
#define LENGTH_EXTRA		0.005f
#define D_EDGE				32.0f
#define D_BEND				2.5f
#define D_TWIST				2.5f
#define D_EXTRA				0.125f
#define GRAVITY				-9.81f //(stable value) -0.00981f

//Stiction constants
#define K_STIC				0.01f //Stiction spring coefficient
#define D_STIC				0.2f //Stiction damping coefficient
#define	LEN_STIC			0.0035f //Stiction spring rest length (3.5 millimetres)
#define HALF_LEN_STIC		0.00175f //Half the sticition spring rest length (for KDOP volume calculation)
#define MAX_LEN_STIC		0.005f //Maximum length of stiction spring
#define MAX_SQR_STIC		0.000025f //Maximum length of stiction spring squared

//Bounding volume constants
#define KDOP_PLANES			26 //other valid values include 6, 14 & 18.

#define MAX_LENGTH			0.0055f //Maximum length of a spring
#define MAX_LENGTH_SQUARED	0.00003025f //Maximum length of a spring squared

//Geometry collisions constants
#define DOMAIN_DIM		100
#define DOMAIN_WIDTH	0.275f
#define DOMAIN_HALF		0.1375f
#define CELL_WIDTH		0.00275f
#define CELL_HALF		0.001375f

#endif

