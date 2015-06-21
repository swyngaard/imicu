
#ifndef _HAIR_KERNEL_H_
#define _HAIR_KERNEL_H_

#include "constants.h"

//FIXME return a float3?
__device__
void unitize(float3 &vector)
{
	float length = sqrtf(vector.x*vector.x + vector.y*vector.y + vector.z*vector.z);

	if(length != 0.0f)
	{
		vector.x /= length;
		vector.y /= length;
		vector.z /= length;
	}

	return;
}

__global__
void initialise(int numParticles, const float3* root, float3* normal, float3* position, float3* posc, float3* pos)
{
	//Strand ID
	int sid = blockIdx.x;
	
	int start = numParticles * sid;
	int end = start + numParticles;
	
	for(int i = start, j = 1; i < end; i++, j++)
	{
		unitize(normal[sid]);

		position[i].x = root[sid].x + (normal[sid].x * 0.025f * j);
		position[i].y = root[sid].y + (normal[sid].y * 0.025f * j);
		position[i].z = root[sid].z + (normal[sid].z * 0.025f * j);

		pos[i].x = position[i].x;
		pos[i].y = position[i].y;
		pos[i].z = position[i].z;
		posc[i].x = position[i].x;
		posc[i].y = position[i].y;
		posc[i].z = position[i].z;
	}
//	printf("normal[%d]:\t%.5f\t%.5f\t%.5f\n", sid, normal[sid].x, normal[sid].y, normal[sid].z);
}

__device__
void clearForces_(int numParticles, float3* force)
{
	int startP = blockIdx.x * numParticles;
	int endP = startP + numParticles;
	
	for(int i = startP; i < endP; i++)
	{
		force[i].x = 0.0f;
		force[i].y = 0.0f;
		force[i].z = 0.0f;
	}
}

__device__
float dot(float3 one, float3 two)
{
	return one.x*two.x + one.y*two.y + one.z*two.z;
}

__device__ 
float3 operator-(float3 vector)
{
	return make_float3(-vector.x, -vector.y, -vector.z);
}

__device__
float3 operator-(float3 one, float3 two)
{
	return make_float3(one.x-two.x, one.y-two.y, one.z-two.z);
}

__device__
float3 operator*(float3 vector, float scalar)
{
	return make_float3(vector.x*scalar, vector.y*scalar, vector.z*scalar);
}
__device__
float3 operator/(float3 vector, float scalar)
{
	return make_float3(vector.x/scalar, vector.y/scalar, vector.z/scalar);
}

__device__
float3 operator+(float3 one, float3 two)
{
	return make_float3(one.x+two.x, one.y+two.y, one.z+two.z);
}

__device__
float length_sqr(float3 vector)
{
	return vector.x*vector.x + vector.y*vector.y + vector.z*vector.z;
}

__device__
float length_inverse(float3 vector)
{
	float number = length_sqr(vector);
	
	if(number != 0)
	{
		float xhalf = 0.5f*number;
		int i = *(int*)&number; // get bits for floating value
		i = 0x5f375a86- (i>>1); // gives initial guess y0
		number = *(float*)&i; // convert bits back to float
		number = number*(1.5f-xhalf*number*number); // Newton step, repeating increases accuracy
	}
	
	return number;
}

__device__
void buildAB_(int numParticles,
			  int numComponents,
			  float dt,
			  float mass,
			  float k_edge,
			  float k_bend,
			  float k_twist,
			  float d_edge,
			  float d_bend,
			  float d_twist,
			  float length_e,
			  float length_b,
			  float length_t,
			  float3 &gravity,
			  float3* root,
			  float3* pos,
			  float3* velocity,
			  float* AA,
			  float* bb,
			  float* xx)
{
	//Strand ID
	int sid = blockIdx.x;
	
	//Start and end of square matrix A indices
	int startAA = sid * numParticles * numComponents * numParticles * numComponents;
	int endAA = startAA + numParticles * numComponents * numParticles * numComponents;
	
	//Start and end of particle indices
	int startP = sid * numParticles;
	int endP = startP + numParticles;
	
	//FIXME Check if resetting AA makes a difference
	for(int i = startAA; i < endAA; i++)
	{
		AA[i] = 0.0f;
	}
	


	//Set the 6 different coefficients
	float h_e = dt*dt*k_edge/(4.0f*mass*length_e) + d_edge*dt/(2.0f*mass);
	float h_b = dt*dt*k_bend/(4.0f*mass*length_b) + d_bend*dt/(2.0f*mass);
	float h_t = dt*dt*k_twist/(4.0f*mass*length_t) + d_twist*dt/(2.0f*mass);
	float g_e = dt*k_edge/(2.0f*mass*length_e);
	float g_b = dt*k_bend/(2.0f*mass*length_b);
	float g_t = dt*k_twist/(2.0f*mass*length_t);

	//Set the first 3 particle direction vectors

	//First particle direction vectors
	float3 du0R(root[sid]    -pos[startP]);
	float3 du01(pos[startP+1]-pos[startP]);
	float3 du02(pos[startP+2]-pos[startP]);
	float3 du03(pos[startP+3]-pos[startP]);
	unitize(du0R);
	unitize(du01);
	unitize(du02);
	unitize(du03);
	
	//Second particle direction vectors
	float3 du1R(root[sid]    -pos[startP+1]);
	float3 du10(pos[startP  ]-pos[startP+1]);
	float3 du12(pos[startP+2]-pos[startP+1]);
	float3 du13(pos[startP+3]-pos[startP+1]);
	float3 du14(pos[startP+4]-pos[startP+1]);
	unitize(du1R);
	unitize(du10);
	unitize(du12);
	unitize(du13);
	unitize(du14);
	
	//Third particle direction vectors
	float3 du2R(root[sid]    -pos[startP+2]);
	float3 du20(pos[startP  ]-pos[startP+2]);
	float3 du21(pos[startP+1]-pos[startP+2]);
	float3 du23(pos[startP+3]-pos[startP+2]);
	float3 du24(pos[startP+4]-pos[startP+2]);
	float3 du25(pos[startP+5]-pos[startP+2]);
	unitize(du2R);
	unitize(du20);
	unitize(du21);
	unitize(du23);
	unitize(du24);
	unitize(du25);


	//Set the non-zero entries for the first 3 particles
	
	//Set first twelve entries of the first row of A matrix
	AA[startAA   ] = 1.0f + h_e*du0R.x*du0R.x + h_e*du01.x*du01.x + h_b*du02.x*du02.x + h_t*du03.x*du03.x;
	AA[startAA+ 1] =        h_e*du0R.x*du0R.y + h_e*du01.x*du01.y + h_b*du02.x*du02.y + h_t*du03.x*du03.y;
	AA[startAA+ 2] =        h_e*du0R.x*du0R.z + h_e*du01.x*du01.z + h_b*du02.x*du02.z + h_t*du03.x*du03.z;
	AA[startAA+ 3] = -h_e*du01.x*du01.x;
	AA[startAA+ 4] = -h_e*du01.x*du01.y;
	AA[startAA+ 5] = -h_e*du01.x*du01.z;
	AA[startAA+ 6] = -h_b*du02.x*du02.x;
	AA[startAA+ 7] = -h_b*du02.x*du02.y;
	AA[startAA+ 8] = -h_b*du02.x*du02.z;
	AA[startAA+ 9] = -h_t*du03.x*du03.x;
	AA[startAA+10] = -h_t*du03.x*du03.y;
	AA[startAA+11] = -h_t*du03.x*du03.z;
	
	//Indices for next second and third rows of A
	int row11 = startAA + numParticles * numComponents;
	int row22 = startAA + 2 * numParticles * numComponents;
	
	//Set next twelve non-zero entries of the second row of matrix A
	AA[row11   ] =        h_e*du0R.x*du0R.y + h_e*du01.x*du01.y + h_b*du02.x*du02.y + h_t*du03.x*du03.y;
	AA[row11+1 ] = 1.0f + h_e*du0R.y*du0R.y + h_e*du01.y*du01.y + h_b*du02.y*du02.y + h_t*du03.y*du03.y;
	AA[row11+2 ] =        h_e*du0R.y*du0R.z + h_e*du01.y*du01.z + h_b*du02.y*du02.z + h_t*du03.y*du03.z;
	AA[row11+3 ] = -h_e*du01.x*du01.y;
	AA[row11+4 ] = -h_e*du01.y*du01.y;
	AA[row11+5 ] = -h_e*du01.y*du01.z;
	AA[row11+6 ] = -h_b*du02.x*du02.y;
	AA[row11+7 ] = -h_b*du02.y*du02.y;
	AA[row11+8 ] = -h_b*du02.y*du02.z;
	AA[row11+9 ] = -h_t*du03.x*du03.y;
	AA[row11+10] = -h_t*du03.y*du03.y;
	AA[row11+11] = -h_t*du03.y*du03.z;
	
	//Set the next twelve non-zero entries of the third row of matrix A
	AA[row22   ] =        h_e*du0R.x*du0R.z + h_e*du01.x*du01.z + h_b*du02.x*du02.z + h_t*du03.x*du03.z;
	AA[row22+1 ] =        h_e*du0R.y*du0R.z + h_e*du01.y*du01.z + h_b*du02.y*du02.z + h_t*du03.y*du03.z;
	AA[row22+2 ] = 1.0f + h_e*du0R.z*du0R.z + h_e*du01.z*du01.z + h_b*du02.z*du02.z + h_t*du03.z*du03.z;
	AA[row22+3 ] = -h_e*du01.x*du01.z;
	AA[row22+4 ] = -h_e*du01.y*du01.z;
	AA[row22+5 ] = -h_e*du01.z*du01.z;
	AA[row22+6 ] = -h_b*du02.x*du02.z;
	AA[row22+7 ] = -h_b*du02.y*du02.z;
	AA[row22+8 ] = -h_b*du02.z*du02.z;
	AA[row22+9 ] = -h_t*du03.x*du03.z;
	AA[row22+10] = -h_t*du03.y*du03.z;
	AA[row22+11] = -h_t*du03.z*du03.z;
	
	int row33 = startAA + 3 * numParticles * numComponents;
	int row44 = startAA + 4 * numParticles * numComponents;
	int row55 = startAA + 5 * numParticles * numComponents;
	
	AA[row33   ] = -h_e*du10.x*du10.x;
	AA[row33+1 ] = -h_e*du10.x*du10.y;
	AA[row33+2 ] = -h_e*du10.x*du10.z;
	AA[row33+3 ] = 1.0f + h_b*du1R.x*du1R.x + h_e*du10.x*du10.x + h_e*du12.x*du12.x + h_b*du13.x*du13.x + h_t*du14.x*du14.x;
	AA[row33+4 ] =        h_b*du1R.x*du1R.y + h_e*du10.x*du10.y + h_e*du12.x*du12.y + h_b*du13.x*du13.y + h_t*du14.x*du14.y;
	AA[row33+5 ] =        h_b*du1R.x*du1R.z + h_e*du10.x*du10.z + h_e*du12.x*du12.z + h_b*du13.x*du13.z + h_t*du14.x*du14.z;
	AA[row33+6 ] = -h_e*du12.x*du12.x;
	AA[row33+7 ] = -h_e*du12.x*du12.y;
	AA[row33+8 ] = -h_e*du12.x*du12.z;
	AA[row33+9 ] = -h_b*du13.x*du13.x;
	AA[row33+10] = -h_b*du13.x*du13.y;
	AA[row33+11] = -h_b*du13.x*du13.z;
	AA[row33+12] = -h_t*du14.x*du14.x;
	AA[row33+13] = -h_t*du14.x*du14.y;
	AA[row33+14] = -h_t*du14.x*du14.z;
	
	AA[row44   ] = -h_e*du10.x*du10.y;
	AA[row44+1 ] = -h_e*du10.y*du10.y;
	AA[row44+2 ] = -h_e*du10.y*du10.z;
	AA[row44+3 ] =        h_b*du1R.x*du1R.y + h_e*du10.x*du10.y + h_e*du12.x*du12.y + h_b*du13.x*du13.y + h_t*du14.x*du14.y;
	AA[row44+4 ] = 1.0f + h_b*du1R.y*du1R.y + h_e*du10.y*du10.y + h_e*du12.y*du12.y + h_b*du13.y*du13.y + h_t*du14.y*du14.y;
	AA[row44+5 ] =        h_b*du1R.y*du1R.z + h_e*du10.y*du10.z + h_e*du12.y*du12.z + h_b*du13.y*du13.z + h_t*du14.y*du14.z;
	AA[row44+6 ] = -h_e*du12.x*du12.y;
	AA[row44+7 ] = -h_e*du12.y*du12.y;
	AA[row44+8 ] = -h_e*du12.y*du12.z;
	AA[row44+9 ] = -h_b*du13.x*du13.y;
	AA[row44+10] = -h_b*du13.y*du13.y;
	AA[row44+11] = -h_b*du13.y*du13.z;
	AA[row44+12] = -h_t*du14.x*du14.y;
	AA[row44+13] = -h_t*du14.y*du14.y;
	AA[row44+14] = -h_t*du14.y*du14.z;
	
	AA[row55   ] = -h_e*du10.x*du10.z;
	AA[row55+1 ] = -h_e*du10.y*du10.z;
	AA[row55+2 ] = -h_e*du10.z*du10.z;
	AA[row55+3 ] =        h_b*du1R.x*du1R.z + h_e*du10.x*du10.z + h_e*du12.x*du12.z + h_b*du13.x*du13.z + h_t*du14.x*du14.z;
	AA[row55+4 ] =        h_b*du1R.y*du1R.z + h_e*du10.y*du10.z + h_e*du12.y*du12.z + h_b*du13.y*du13.z + h_t*du14.y*du14.z;
	AA[row55+5 ] = 1.0f + h_b*du1R.z*du1R.z + h_e*du10.z*du10.z + h_e*du12.z*du12.z + h_b*du13.z*du13.z + h_t*du14.z*du14.z;
	AA[row55+6 ] = -h_e*du12.x*du12.z;
	AA[row55+7 ] = -h_e*du12.y*du12.z;
	AA[row55+8 ] = -h_e*du12.z*du12.z;
	AA[row55+9 ] = -h_b*du13.x*du13.z;
	AA[row55+10] = -h_b*du13.y*du13.z;
	AA[row55+11] = -h_b*du13.z*du13.z;
	AA[row55+12] = -h_t*du14.x*du14.z;
	AA[row55+13] = -h_t*du14.y*du14.z;
	AA[row55+14] = -h_t*du14.z*du14.z;
	
	int row66 = startAA + 6 * numParticles * numComponents;
	int row77 = startAA + 7 * numParticles * numComponents;
	int row88 = startAA + 8 * numParticles * numComponents;
	
	AA[row66   ] = -h_b*du20.x*du20.x;
	AA[row66+1 ] = -h_b*du20.x*du20.y;
	AA[row66+2 ] = -h_b*du20.x*du20.z;
	AA[row66+3 ] = -h_e*du21.x*du21.x;
	AA[row66+4 ] = -h_e*du21.x*du21.y;
	AA[row66+5 ] = -h_e*du21.x*du21.z;
	AA[row66+6 ] = 1.0f + h_t*du2R.x*du2R.x + h_b*du20.x*du20.x + h_e*du21.x*du21.x + h_e*du23.x*du23.x + h_b*du24.x*du24.x + h_t*du25.x*du25.x;
	AA[row66+7 ] =        h_t*du2R.x*du2R.y + h_b*du20.x*du20.y + h_e*du21.x*du21.y + h_e*du23.x*du23.y + h_b*du24.x*du24.y + h_t*du25.x*du25.y;
	AA[row66+8 ] =        h_t*du2R.x*du2R.z + h_b*du20.x*du20.z + h_e*du21.x*du21.z + h_e*du23.x*du23.z + h_b*du24.x*du24.z + h_t*du25.x*du25.z;
	AA[row66+9 ] = -h_e*du23.x*du23.x;
	AA[row66+10] = -h_e*du23.x*du23.y;
	AA[row66+11] = -h_e*du23.x*du23.z;
	AA[row66+12] = -h_b*du24.x*du24.x;
	AA[row66+13] = -h_b*du24.x*du24.y;
	AA[row66+14] = -h_b*du24.x*du24.z;
	AA[row66+15] = -h_t*du25.x*du25.x;
	AA[row66+16] = -h_t*du25.x*du25.y;
	AA[row66+17] = -h_t*du25.x*du25.z;
	
	
	AA[row77   ] = -h_b*du20.x*du20.y;
	AA[row77+1 ] = -h_b*du20.y*du20.y;
	AA[row77+2 ] = -h_b*du20.y*du20.z;
	AA[row77+3 ] = -h_e*du21.x*du21.y;
	AA[row77+4 ] = -h_e*du21.y*du21.y;
	AA[row77+5 ] = -h_e*du21.y*du21.z;
	AA[row77+6 ] =        h_t*du2R.x*du2R.y + h_b*du20.x*du20.y + h_e*du21.x*du21.y + h_e*du23.x*du23.y + h_b*du24.x*du24.y + h_t*du25.x*du25.y;
	AA[row77+7 ] = 1.0f + h_t*du2R.y*du2R.y + h_b*du20.y*du20.y + h_e*du21.y*du21.y + h_e*du23.y*du23.y + h_b*du24.y*du24.y + h_t*du25.y*du25.y;
	AA[row77+8 ] =        h_t*du2R.y*du2R.z + h_b*du20.y*du20.z + h_e*du21.y*du21.z + h_e*du23.y*du23.z + h_b*du24.y*du24.z + h_t*du25.y*du25.z;
	AA[row77+9 ] = -h_e*du23.x*du23.y;
	AA[row77+10] = -h_e*du23.y*du23.y;
	AA[row77+11] = -h_e*du23.y*du23.z;
	AA[row77+12] = -h_b*du24.x*du24.y;
	AA[row77+13] = -h_b*du24.y*du24.y;
	AA[row77+14] = -h_b*du24.y*du24.z;
	AA[row77+15] = -h_t*du25.x*du25.y;
	AA[row77+16] = -h_t*du25.y*du25.y;
	AA[row77+17] = -h_t*du25.y*du25.z;
	
	AA[row88   ] = -h_b*du20.x*du20.z;
	AA[row88+1 ] = -h_b*du20.y*du20.z;
	AA[row88+2 ] = -h_b*du20.z*du20.z;
	AA[row88+3 ] = -h_e*du21.x*du21.z;
	AA[row88+4 ] = -h_e*du21.y*du21.z;
	AA[row88+5 ] = -h_e*du21.z*du21.z;
	AA[row88+6 ] =        h_t*du2R.x*du2R.z + h_b*du20.x*du20.z + h_e*du21.x*du21.z + h_e*du23.x*du23.z + h_b*du24.x*du24.z + h_t*du25.x*du25.z;
	AA[row88+7 ] =        h_t*du2R.y*du2R.z + h_b*du20.y*du20.z + h_e*du21.y*du21.z + h_e*du23.y*du23.z + h_b*du24.y*du24.z + h_t*du25.y*du25.z;
	AA[row88+8 ] = 1.0f + h_t*du2R.z*du2R.z + h_b*du20.z*du20.z + h_e*du21.z*du21.z + h_e*du23.z*du23.z + h_b*du24.z*du24.z + h_t*du25.z*du25.z;
	AA[row88+9 ] = -h_e*du23.x*du23.z;
	AA[row88+10] = -h_e*du23.y*du23.z;
	AA[row88+11] = -h_e*du23.z*du23.z;
	AA[row88+12] = -h_b*du24.x*du24.z;
	AA[row88+13] = -h_b*du24.y*du24.z;
	AA[row88+14] = -h_b*du24.z*du24.z;
	AA[row88+15] = -h_t*du25.x*du25.z;
	AA[row88+16] = -h_t*du25.y*du25.z;
	AA[row88+17] = -h_t*du25.z*du25.z;

	int startBB = sid * numParticles * numComponents;
	int endBB = startBB + numParticles * numComponents;

	//Set the first nine entries of the b vector
	bb[startBB  ] = velocity[startP  ].x + g_e*(dot((root[sid]-pos[startP  ]), du0R)-length_e)*du0R.x + g_e*(dot((pos[startP+1]-pos[startP  ]), du01)-length_e)*du01.x + g_b*(dot((pos[startP+2]-pos[startP  ]), du02)-length_b)*du02.x + g_t*(dot((pos[startP+3]-pos[startP  ]), du03)-length_t)*du03.x + gravity.x*(dt/2.0f);
	bb[startBB+1] = velocity[startP  ].y + g_e*(dot((root[sid]-pos[startP  ]), du0R)-length_e)*du0R.y + g_e*(dot((pos[startP+1]-pos[startP  ]), du01)-length_e)*du01.y + g_b*(dot((pos[startP+2]-pos[startP  ]), du02)-length_b)*du02.y + g_t*(dot((pos[startP+3]-pos[startP  ]), du03)-length_t)*du03.y + gravity.y*(dt/2.0f);
	bb[startBB+2] = velocity[startP  ].z + g_e*(dot((root[sid]-pos[startP  ]), du0R)-length_e)*du0R.z + g_e*(dot((pos[startP+1]-pos[startP  ]), du01)-length_e)*du01.z + g_b*(dot((pos[startP+2]-pos[startP  ]), du02)-length_b)*du02.z + g_t*(dot((pos[startP+3]-pos[startP  ]), du03)-length_t)*du03.z + gravity.z*(dt/2.0f);
	
	bb[startBB+3] = velocity[startP+1].x + g_b*(dot((root[sid]-pos[startP+1]), du1R)-length_b)*du1R.x + g_e*(dot((pos[startP  ]-pos[startP+1]), du10)-length_e)*du10.x + g_e*(dot((pos[startP+2]-pos[startP+1]), du12)-length_e)*du12.x + g_b*(dot((pos[startP+3]-pos[startP+1]), du13)-length_b)*du13.x + g_t*(dot((pos[startP+4]-pos[startP+1]), du14)-length_t)*du14.x + gravity.x*(dt/2.0f);
	bb[startBB+4] = velocity[startP+1].y + g_b*(dot((root[sid]-pos[startP+1]), du1R)-length_b)*du1R.y + g_e*(dot((pos[startP  ]-pos[startP+1]), du10)-length_e)*du10.y + g_e*(dot((pos[startP+2]-pos[startP+1]), du12)-length_e)*du12.y + g_b*(dot((pos[startP+3]-pos[startP+1]), du13)-length_b)*du13.y + g_t*(dot((pos[startP+4]-pos[startP+1]), du14)-length_t)*du14.y + gravity.y*(dt/2.0f);
	bb[startBB+5] = velocity[startP+1].z + g_b*(dot((root[sid]-pos[startP+1]), du1R)-length_b)*du1R.z + g_e*(dot((pos[startP  ]-pos[startP+1]), du10)-length_e)*du10.z + g_e*(dot((pos[startP+2]-pos[startP+1]), du12)-length_e)*du12.z + g_b*(dot((pos[startP+3]-pos[startP+1]), du13)-length_b)*du13.z + g_t*(dot((pos[startP+4]-pos[startP+1]), du14)-length_t)*du14.z + gravity.z*(dt/2.0f);
	
	bb[startBB+6] = velocity[startP+2].x + g_t*(dot((root[sid]-pos[startP+2]), du2R)-length_t)*du2R.x + g_b*(dot((pos[startP  ]-pos[startP+2]), du02)-length_b)*du20.x + g_e*(dot((pos[startP+1]-pos[startP+2]), du21)-length_e)*du21.x + g_e*(dot((pos[startP+3]-pos[startP+2]), du23)-length_e)*du23.x + g_b*(dot((pos[startP+4]-pos[startP+2]), du24)-length_b)*du24.x + g_t*(dot((pos[startP+5]-pos[startP+2]), du25)-length_t)*du25.x + gravity.x*(dt/2.0f);
	bb[startBB+7] = velocity[startP+2].y + g_t*(dot((root[sid]-pos[startP+2]), du2R)-length_t)*du2R.y + g_b*(dot((pos[startP  ]-pos[startP+2]), du02)-length_b)*du20.y + g_e*(dot((pos[startP+1]-pos[startP+2]), du21)-length_e)*du21.y + g_e*(dot((pos[startP+3]-pos[startP+2]), du23)-length_e)*du23.y + g_b*(dot((pos[startP+4]-pos[startP+2]), du24)-length_b)*du24.y + g_t*(dot((pos[startP+5]-pos[startP+2]), du25)-length_t)*du25.y + gravity.y*(dt/2.0f);
	bb[startBB+8] = velocity[startP+2].z + g_t*(dot((root[sid]-pos[startP+2]), du2R)-length_t)*du2R.z + g_b*(dot((pos[startP  ]-pos[startP+2]), du02)-length_b)*du20.z + g_e*(dot((pos[startP+1]-pos[startP+2]), du21)-length_e)*du21.z + g_e*(dot((pos[startP+3]-pos[startP+2]), du23)-length_e)*du23.z + g_b*(dot((pos[startP+4]-pos[startP+2]), du24)-length_b)*du24.z + g_t*(dot((pos[startP+5]-pos[startP+2]), du25)-length_t)*du25.z + gravity.z*(dt/2.0f);

	//Build in-between values of matrix A and vector b
	//Loop from fourth to third last particles only
	for(int i = startP+3; i < (endP-3); i++)
	{
		//Current particle position, particle above and particle below
		float3 ui = pos[i];
		
		//Direction vectors for 3 particles above and below the current particle
		float3 uu2 = pos[i-3];
		float3 uu1 = pos[i-2];
		float3 uu0 = pos[i-1];
		float3 ud0 = pos[i+1];
		float3 ud1 = pos[i+2];
		float3 ud2 = pos[i+3];
		
		float3 du2(uu2-ui);
		float3 du1(uu1-ui);
		float3 du0(uu0-ui);
		float3 dd0(ud0-ui);
		float3 dd1(ud1-ui);
		float3 dd2(ud2-ui);
		unitize(du2);
		unitize(du1);
		unitize(du0);
		unitize(dd0);
		unitize(dd1);
		unitize(dd2);

		int row0 = i * numParticles * numComponents * 3 + (i - startP - 3) * numComponents;
		int row1 = row0 + numParticles * numComponents;
		int row2 = row1 + numParticles * numComponents;
		
		AA[row0   ] = -h_t*du2.x*du2.x;
		AA[row0+1 ] = -h_t*du2.x*du2.y;
		AA[row0+2 ] = -h_t*du2.x*du2.z;
		AA[row0+3 ] = -h_b*du1.x*du1.x;
		AA[row0+4 ] = -h_b*du1.x*du1.y;
		AA[row0+5 ] = -h_b*du1.x*du1.z;
		AA[row0+6 ] = -h_e*du0.x*du0.x;
		AA[row0+7 ] = -h_e*du0.x*du0.y;
		AA[row0+8 ] = -h_e*du0.x*du0.z;
		AA[row0+9 ] = 1.0f + h_t*du2.x*du2.x + h_b*du1.x*du1.x + h_e*du0.x*du0.x + h_e*dd0.x*dd0.x + h_b*dd1.x*dd1.x + h_t*dd2.x*dd2.x;
		AA[row0+10] =        h_t*du2.x*du2.y + h_b*du1.x*du1.y + h_e*du0.x*du0.y + h_e*dd0.x*dd0.y + h_b*dd1.x*dd1.y + h_t*dd2.x*dd2.y;
		AA[row0+11] =        h_t*du2.x*du2.z + h_b*du1.x*du1.z + h_e*du0.x*du0.z + h_e*dd0.x*dd0.z + h_b*dd1.x*dd1.z + h_t*dd2.x*dd2.z;
		AA[row0+12] = -h_e*dd0.x*dd0.x;
		AA[row0+13] = -h_e*dd0.x*dd0.y;
		AA[row0+14] = -h_e*dd0.x*dd0.z;
		AA[row0+15] = -h_b*dd1.x*dd1.x;
		AA[row0+16] = -h_b*dd1.x*dd1.y;
		AA[row0+17] = -h_b*dd1.x*dd1.z;
		AA[row0+18] = -h_t*dd2.x*dd2.x;
		AA[row0+19] = -h_t*dd2.x*dd2.y;
		AA[row0+20] = -h_t*dd2.x*dd2.z;
		
		AA[row1   ] = -h_t*du2.x*du2.y;
		AA[row1+1 ] = -h_t*du2.y*du2.y;
		AA[row1+2 ] = -h_t*du2.y*du2.z;
		AA[row1+3 ] = -h_b*du1.x*du1.y;
		AA[row1+4 ] = -h_b*du1.y*du1.y;
		AA[row1+5 ] = -h_b*du1.y*du1.z;
		AA[row1+6 ] = -h_e*du0.x*du0.y;
		AA[row1+7 ] = -h_e*du0.y*du0.y;
		AA[row1+8 ] = -h_e*du0.y*du0.z;
		AA[row1+9 ] =        h_t*du2.x*du2.y + h_b*du1.x*du1.y + h_e*du0.x*du0.y + h_e*dd0.x*dd0.y + h_b*dd1.x*dd1.y + h_t*dd2.x*dd2.y;
		AA[row1+10] = 1.0f + h_t*du2.y*du2.y + h_b*du1.y*du1.y + h_e*du0.y*du0.y + h_e*dd0.y*dd0.y + h_b*dd1.y*dd1.y + h_t*dd2.y*dd2.y;
		AA[row1+11] =        h_t*du2.y*du2.z + h_b*du1.y*du1.z + h_e*du0.y*du0.z + h_e*dd0.y*dd0.z + h_b*dd1.y*dd1.z + h_t*dd2.y*dd2.z;
		AA[row1+12] = -h_e*dd0.x*dd0.y;
		AA[row1+13] = -h_e*dd0.y*dd0.y;
		AA[row1+14] = -h_e*dd0.y*dd0.z;
		AA[row1+15] = -h_b*dd1.x*dd1.y;
		AA[row1+16] = -h_b*dd1.y*dd1.y;
		AA[row1+17] = -h_b*dd1.y*dd1.z;
		AA[row1+18] = -h_t*dd2.x*dd2.y;
		AA[row1+19] = -h_t*dd2.y*dd2.y;
		AA[row1+20] = -h_t*dd2.y*dd2.z;
		
		AA[row2   ] = -h_t*du2.x*du2.z;
		AA[row2+1 ] = -h_t*du2.y*du2.z;
		AA[row2+2 ] = -h_t*du2.z*du2.z;
		AA[row2+3 ] = -h_b*du1.x*du1.z;
		AA[row2+4 ] = -h_b*du1.y*du1.z;
		AA[row2+5 ] = -h_b*du1.z*du1.z;
		AA[row2+6 ] = -h_e*du0.x*du0.z;
		AA[row2+7 ] = -h_e*du0.y*du0.z;
		AA[row2+8 ] = -h_e*du0.z*du0.z;
		AA[row2+9 ] =        h_t*du2.x*du2.z + h_b*du1.x*du1.z + h_e*du0.x*du0.z + h_e*dd0.x*dd0.z + h_b*dd1.x*dd1.z + h_t*dd2.x*dd2.z;
		AA[row2+10] =        h_t*du2.y*du2.z + h_b*du1.y*du1.z + h_e*du0.y*du0.z + h_e*dd0.y*dd0.z + h_b*dd1.y*dd1.z + h_t*dd2.y*dd2.z;
		AA[row2+11] = 1.0f + h_t*du2.z*du2.z + h_b*du1.z*du1.z + h_e*du0.z*du0.z + h_e*dd0.z*dd0.z + h_b*dd1.z*dd1.z + h_t*dd2.z*dd2.z;
		AA[row2+12] = -h_e*dd0.x*dd0.z;
		AA[row2+13] = -h_e*dd0.y*dd0.z;
		AA[row2+14] = -h_e*dd0.z*dd0.z;
		AA[row2+15] = -h_b*dd1.x*dd1.z;
		AA[row2+16] = -h_b*dd1.y*dd1.z;
		AA[row2+17] = -h_b*dd1.z*dd1.z;
		AA[row2+18] = -h_t*dd2.x*dd2.z;
		AA[row2+19] = -h_t*dd2.y*dd2.z;
		AA[row2+20] = -h_t*dd2.z*dd2.z;

		bb[i*numComponents  ] = velocity[i].x + g_t*(dot((uu2-ui), du2)-length_t)*du2.x + g_b*(dot((uu1-ui), du1)-length_b)*du1.x + g_e*(dot((uu0-ui), du0)-length_e)*du0.x + g_e*(dot((ud0-ui), dd0)-length_e)*dd0.x + g_b*(dot((ud1-ui), dd1)-length_b)*dd1.x + g_t*(dot((ud2-ui), dd2)-length_t)*dd2.x + gravity.x*(dt/2.0f);
		bb[i*numComponents+1] = velocity[i].y + g_t*(dot((uu2-ui), du2)-length_t)*du2.y + g_b*(dot((uu1-ui), du1)-length_b)*du1.y + g_e*(dot((uu0-ui), du0)-length_e)*du0.y + g_e*(dot((ud0-ui), dd0)-length_e)*dd0.y + g_b*(dot((ud1-ui), dd1)-length_b)*dd1.y + g_t*(dot((ud2-ui), dd2)-length_t)*dd2.y + gravity.y*(dt/2.0f);
		bb[i*numComponents+2] = velocity[i].z + g_t*(dot((uu2-ui), du2)-length_t)*du2.z + g_b*(dot((uu1-ui), du1)-length_b)*du1.z + g_e*(dot((uu0-ui), du0)-length_e)*du0.z + g_e*(dot((ud0-ui), dd0)-length_e)*dd0.z + g_b*(dot((ud1-ui), dd1)-length_b)*dd1.z + g_t*(dot((ud2-ui), dd2)-length_t)*dd2.z + gravity.z*(dt/2.0f);
	}

	//Calculate direction vectors for the last three particles
	//Third to last particle direction vectors
	float3 du3N1(pos[endP-6]-pos[endP-3]);
	float3 du3N2(pos[endP-5]-pos[endP-3]);
	float3 du3N3(pos[endP-4]-pos[endP-3]);
	float3 du3N5(pos[endP-2]-pos[endP-3]);
	float3 du3N6(pos[endP-1]-pos[endP-3]);
	unitize(du3N1);
	unitize(du3N2);
	unitize(du3N3);
	unitize(du3N5);
	unitize(du3N6);
	
	//Second to last particle direction vectors
	float3 du2N2(pos[endP-5]-pos[endP-2]);
	float3 du2N3(pos[endP-4]-pos[endP-2]);
	float3 du2N4(pos[endP-3]-pos[endP-2]);
	float3 du2N6(pos[endP-1]-pos[endP-2]);
	unitize(du2N2);
	unitize(du2N3);
	unitize(du2N4);
	unitize(du2N6);
	
	//Last particle direction vectors
	float3 du1N3(pos[endP-4]-pos[endP-1]);
	float3 du1N4(pos[endP-3]-pos[endP-1]);
	float3 du1N5(pos[endP-2]-pos[endP-1]);
	unitize(du1N3);
	unitize(du1N4);
	unitize(du1N5);


	int row3N3 = endAA - 8*numParticles*NUMCOMPONENTS - 18;
	int row3N2 = endAA - 7*numParticles*NUMCOMPONENTS - 18;
	int row3N1 = endAA - 6*numParticles*NUMCOMPONENTS - 18;
	
	AA[row3N3   ] = -h_t*du3N1.x*du3N1.x;
	AA[row3N3+1 ] = -h_t*du3N1.x*du3N1.y;
	AA[row3N3+2 ] = -h_t*du3N1.x*du3N1.z;
	AA[row3N3+3 ] = -h_b*du3N2.x*du3N2.x;
	AA[row3N3+4 ] = -h_b*du3N2.x*du3N2.y;
	AA[row3N3+5 ] = -h_b*du3N2.x*du3N2.z;
	AA[row3N3+6 ] = -h_e*du3N3.x*du3N3.x;
	AA[row3N3+7 ] = -h_e*du3N3.x*du3N3.y;
	AA[row3N3+8 ] = -h_e*du3N3.x*du3N3.z;
	AA[row3N3+9 ] = 1.0f + h_t*du3N1.x*du3N1.x + h_b*du3N2.x*du3N2.x + h_e*du3N3.x*du3N3.x + h_e*du3N5.x*du3N5.x + h_b*du3N6.x*du3N6.x;
	AA[row3N3+10] =        h_t*du3N1.x*du3N1.y + h_b*du3N2.x*du3N2.y + h_e*du3N3.x*du3N3.y + h_e*du3N5.x*du3N5.y + h_b*du3N6.x*du3N6.y;
	AA[row3N3+11] =        h_t*du3N1.x*du3N1.z + h_b*du3N2.x*du3N2.z + h_e*du3N3.x*du3N3.z + h_e*du3N5.x*du3N5.z + h_b*du3N6.x*du3N6.z;
	AA[row3N3+12] = -h_e*du3N5.x*du3N5.x;
	AA[row3N3+13] = -h_e*du3N5.x*du3N5.y;
	AA[row3N3+14] = -h_e*du3N5.x*du3N5.z;
	AA[row3N3+15] = -h_b*du3N6.x*du3N6.x;
	AA[row3N3+16] = -h_b*du3N6.x*du3N6.y;
	AA[row3N3+17] = -h_b*du3N6.x*du3N6.z;
	
	AA[row3N2   ] = -h_t*du3N1.x*du3N1.y;
	AA[row3N2+1 ] = -h_t*du3N1.y*du3N1.y;
	AA[row3N2+2 ] = -h_t*du3N1.y*du3N1.z;
	AA[row3N2+3 ] = -h_b*du3N2.x*du3N2.y;
	AA[row3N2+4 ] = -h_b*du3N2.y*du3N2.y;
	AA[row3N2+5 ] = -h_b*du3N2.y*du3N2.z;
	AA[row3N2+6 ] = -h_e*du3N3.x*du3N3.y;
	AA[row3N2+7 ] = -h_e*du3N3.y*du3N3.y;
	AA[row3N2+8 ] = -h_e*du3N3.y*du3N3.z;
	AA[row3N2+9 ] =        h_t*du3N1.x*du3N1.y + h_b*du3N2.x*du3N2.y + h_e*du3N3.x*du3N3.y + h_e*du3N5.x*du3N5.y + h_b*du3N6.x*du3N6.y;
	AA[row3N2+10] = 1.0f + h_t*du3N1.y*du3N1.y + h_b*du3N2.y*du3N2.y + h_e*du3N3.y*du3N3.y + h_e*du3N5.y*du3N5.y + h_b*du3N6.y*du3N6.y;
	AA[row3N2+11] =        h_t*du3N1.y*du3N1.z + h_b*du3N2.y*du3N2.z + h_e*du3N3.y*du3N3.z + h_e*du3N5.y*du3N5.z + h_b*du3N6.y*du3N6.z;
	AA[row3N2+12] = -h_e*du3N5.x*du3N5.y;
	AA[row3N2+13] = -h_e*du3N5.y*du3N5.y;
	AA[row3N2+14] = -h_e*du3N5.y*du3N5.z;
	AA[row3N2+15] = -h_b*du3N6.x*du3N6.y;
	AA[row3N2+16] = -h_b*du3N6.y*du3N6.y;
	AA[row3N2+17] = -h_b*du3N6.y*du3N6.z;
	
	AA[row3N1   ] = -h_t*du3N1.x*du3N1.z;
	AA[row3N1+1 ] = -h_t*du3N1.y*du3N1.z;
	AA[row3N1+2 ] = -h_t*du3N1.z*du3N1.z;
	AA[row3N1+3 ] = -h_b*du3N2.x*du3N2.z;
	AA[row3N1+4 ] = -h_b*du3N2.y*du3N2.z;
	AA[row3N1+5 ] = -h_b*du3N2.z*du3N2.z;
	AA[row3N1+6 ] = -h_e*du3N3.x*du3N3.z;
	AA[row3N1+7 ] = -h_e*du3N3.y*du3N3.z;
	AA[row3N1+8 ] = -h_e*du3N3.z*du3N3.z;
	AA[row3N1+9 ] =        h_t*du3N1.x*du3N1.z + h_b*du3N2.x*du3N2.z + h_e*du3N3.x*du3N3.z + h_e*du3N5.x*du3N5.z + h_b*du3N6.x*du3N6.z;
	AA[row3N1+10] =        h_t*du3N1.y*du3N1.z + h_b*du3N2.y*du3N2.z + h_e*du3N3.y*du3N3.z + h_e*du3N5.y*du3N5.z + h_b*du3N6.y*du3N6.z;
	AA[row3N1+11] = 1.0f + h_t*du3N1.z*du3N1.z + h_b*du3N2.z*du3N2.z + h_e*du3N3.z*du3N3.z + h_e*du3N5.z*du3N5.z + h_b*du3N6.z*du3N6.z;
	AA[row3N1+12] = -h_e*du3N5.x*du3N5.z;
	AA[row3N1+13] = -h_e*du3N5.y*du3N5.z;
	AA[row3N1+14] = -h_e*du3N5.z*du3N5.z;
	AA[row3N1+15] = -h_b*du3N6.x*du3N6.z;
	AA[row3N1+16] = -h_b*du3N6.y*du3N6.z;
	AA[row3N1+17] = -h_b*du3N6.z*du3N6.z;
	
	int row2N3 = endAA - 5*numParticles*NUMCOMPONENTS - 15;
	int row2N2 = endAA - 4*numParticles*NUMCOMPONENTS - 15;
	int row2N1 = endAA - 3*numParticles*NUMCOMPONENTS - 15;
	
	AA[row2N3   ] = -h_t*du2N2.x*du2N2.x;
	AA[row2N3+1 ] = -h_t*du2N2.x*du2N2.y;
	AA[row2N3+2 ] = -h_t*du2N2.x*du2N2.z;
	AA[row2N3+3 ] = -h_b*du2N3.x*du2N3.x;
	AA[row2N3+4 ] = -h_b*du2N3.x*du2N3.y;
	AA[row2N3+5 ] = -h_b*du2N3.x*du2N3.z;
	AA[row2N3+6 ] = -h_e*du2N4.x*du2N4.x;
	AA[row2N3+7 ] = -h_e*du2N4.x*du2N4.y;
	AA[row2N3+8 ] = -h_e*du2N4.x*du2N4.z;
	AA[row2N3+9 ] = 1.0f + h_t*du2N2.x*du2N2.x + h_b*du2N3.x*du2N3.x + h_e*du2N4.x*du2N4.x + h_e*du2N6.x*du2N6.x;
	AA[row2N3+10] =        h_t*du2N2.x*du2N2.y + h_b*du2N3.x*du2N3.y + h_e*du2N4.x*du2N4.y + h_e*du2N6.x*du2N6.y;
	AA[row2N3+11] =        h_t*du2N2.x*du2N2.z + h_b*du2N3.x*du2N3.z + h_e*du2N4.x*du2N4.z + h_e*du2N6.x*du2N6.z;
	AA[row2N3+12] = -h_e*du2N6.x*du2N6.x;
	AA[row2N3+13] = -h_e*du2N6.x*du2N6.y;
	AA[row2N3+14] = -h_e*du2N6.x*du2N6.z;
	
	AA[row2N2   ] = -h_t*du2N2.x*du2N2.y;
	AA[row2N2+1 ] = -h_t*du2N2.y*du2N2.y;
	AA[row2N2+2 ] = -h_t*du2N2.y*du2N2.z;
	AA[row2N2+3 ] = -h_b*du2N3.x*du2N3.y;
	AA[row2N2+4 ] = -h_b*du2N3.y*du2N3.y;
	AA[row2N2+5 ] = -h_b*du2N3.y*du2N3.z;
	AA[row2N2+6 ] = -h_e*du2N4.x*du2N4.y;
	AA[row2N2+7 ] = -h_e*du2N4.y*du2N4.y;
	AA[row2N2+8 ] = -h_e*du2N4.y*du2N4.z;
	AA[row2N2+9 ] =        h_t*du2N2.x*du2N2.y + h_b*du2N3.x*du2N3.y + h_e*du2N4.x*du2N4.y + h_e*du2N6.x*du2N6.y;
	AA[row2N2+10] = 1.0f + h_t*du2N2.y*du2N2.y + h_b*du2N3.y*du2N3.y + h_e*du2N4.y*du2N4.y + h_e*du2N6.y*du2N6.y;
	AA[row2N2+11] =        h_t*du2N2.y*du2N2.z + h_b*du2N3.y*du2N3.z + h_e*du2N4.y*du2N4.z + h_e*du2N6.y*du2N6.z;
	AA[row2N2+12] = -h_e*du2N6.x*du2N6.y;
	AA[row2N2+13] = -h_e*du2N6.y*du2N6.y;
	AA[row2N2+14] = -h_e*du2N6.y*du2N6.z;
	
	AA[row2N1   ] = -h_t*du2N2.x*du2N2.z;
	AA[row2N1+1 ] = -h_t*du2N2.y*du2N2.z;
	AA[row2N1+2 ] = -h_t*du2N2.z*du2N2.z;
	AA[row2N1+3 ] = -h_b*du2N3.x*du2N3.z;
	AA[row2N1+4 ] = -h_b*du2N3.y*du2N3.z;
	AA[row2N1+5 ] = -h_b*du2N3.z*du2N3.z;
	AA[row2N1+6 ] = -h_e*du2N4.x*du2N4.z;
	AA[row2N1+7 ] = -h_e*du2N4.y*du2N4.z;
	AA[row2N1+8 ] = -h_e*du2N4.z*du2N4.z;
	AA[row2N1+9 ] =        h_t*du2N2.x*du2N2.z + h_b*du2N3.x*du2N3.z + h_e*du2N4.x*du2N4.z + h_e*du2N4.x*du2N4.z;
	AA[row2N1+10] =        h_t*du2N2.y*du2N2.z + h_b*du2N3.y*du2N3.z + h_e*du2N4.y*du2N4.z + h_e*du2N4.y*du2N4.z;
	AA[row2N1+11] = 1.0f + h_t*du2N2.z*du2N2.z + h_b*du2N3.z*du2N3.z + h_e*du2N4.z*du2N4.z + h_e*du2N4.z*du2N4.z;
	AA[row2N1+12] = -h_e*du2N4.x*du2N4.z;
	AA[row2N1+13] = -h_e*du2N4.y*du2N4.z;
	AA[row2N1+14] = -h_e*du2N4.z*du2N4.z;
	
	int row1N3 = endAA - 2*numParticles*NUMCOMPONENTS - 12;
	int row1N2 = endAA -   numParticles*NUMCOMPONENTS - 12;
	int row1N1 = endAA - 12;
	
	AA[row1N3   ] = -h_t*du1N3.x*du1N3.x;
	AA[row1N3+1 ] = -h_t*du1N3.x*du1N3.y;
	AA[row1N3+2 ] = -h_t*du1N3.x*du1N3.z;
	AA[row1N3+3 ] = -h_b*du1N4.x*du1N4.x;
	AA[row1N3+4 ] = -h_b*du1N4.x*du1N4.y;
	AA[row1N3+5 ] = -h_b*du1N4.x*du1N4.z;
	AA[row1N3+6 ] = -h_e*du1N5.x*du1N5.x;
	AA[row1N3+7 ] = -h_e*du1N5.x*du1N5.y;
	AA[row1N3+8 ] = -h_e*du1N5.x*du1N5.z;
	AA[row1N3+9 ] = 1.0f + h_t*du1N3.x*du1N3.x + h_b*du1N4.x*du1N4.x + h_e*du1N5.x*du1N5.x;
	AA[row1N3+10] =        h_t*du1N3.x*du1N3.y + h_b*du1N4.x*du1N4.y + h_e*du1N5.x*du1N5.y;
	AA[row1N3+11] =        h_t*du1N3.x*du1N3.z + h_b*du1N4.x*du1N4.z + h_e*du1N5.x*du1N5.z;
	
	AA[row1N2   ] = -h_t*du1N3.x*du1N3.y;
	AA[row1N2+1 ] = -h_t*du1N3.y*du1N3.y;
	AA[row1N2+2 ] = -h_t*du1N3.y*du1N3.z;
	AA[row1N2+3 ] = -h_b*du1N4.x*du1N4.y;
	AA[row1N2+4 ] = -h_b*du1N4.y*du1N4.y;
	AA[row1N2+5 ] = -h_b*du1N4.y*du1N4.z;
	AA[row1N2+6 ] = -h_e*du1N5.x*du1N5.y;
	AA[row1N2+7 ] = -h_e*du1N5.y*du1N5.y;
	AA[row1N2+8 ] = -h_e*du1N5.y*du1N5.z;
	AA[row1N2+9 ] =        h_t*du1N3.x*du1N3.y + h_b*du1N4.x*du1N4.y + h_e*du1N5.x*du1N5.y;
	AA[row1N2+10] = 1.0f + h_t*du1N3.y*du1N3.y + h_b*du1N4.y*du1N4.y + h_e*du1N5.y*du1N5.y;
	AA[row1N2+11] =        h_t*du1N3.y*du1N3.z + h_b*du1N4.y*du1N4.z + h_e*du1N5.y*du1N5.z;
	
	AA[row1N1   ] = -h_t*du1N3.x*du1N3.z;
	AA[row1N1+1 ] = -h_t*du1N3.y*du1N3.z;
	AA[row1N1+2 ] = -h_t*du1N3.z*du1N3.z;
	AA[row1N1+3 ] = -h_b*du1N4.x*du1N4.z;
	AA[row1N1+4 ] = -h_b*du1N4.y*du1N4.z;
	AA[row1N1+5 ] = -h_b*du1N4.z*du1N4.z;
	AA[row1N1+6 ] = -h_e*du1N5.x*du1N5.z;
	AA[row1N1+7 ] = -h_e*du1N5.x*du1N5.z;
	AA[row1N1+8 ] = -h_e*du1N5.x*du1N5.z;
	AA[row1N1+9 ] =        h_t*du1N3.z*du1N3.z + h_b*du1N4.z*du1N4.z + h_e*du1N5.x*du1N5.z;
	AA[row1N1+10] =        h_t*du1N3.z*du1N3.z + h_b*du1N4.z*du1N4.z + h_e*du1N5.x*du1N5.z;
	AA[row1N1+11] = 1.0f + h_t*du1N3.z*du1N3.z + h_b*du1N4.z*du1N4.z + h_e*du1N5.x*du1N5.z;

	//Set the last nine entries of the vector b
	bb[endBB-9] = velocity[endP-3].x + g_t*(dot((pos[endP-6]-pos[endP-3]), du3N1)-length_t)*du3N1.x + g_b*(dot((pos[endP-5]-pos[endP-3]), du3N2)-length_b)*du3N2.x + g_e*(dot((pos[endP-4]-pos[endP-3]), du3N3)-length_e)*du3N3.x + g_e*(dot((pos[endP-2]-pos[endP-3]), du3N5)-length_e)*du3N5.x + g_b*(dot((pos[endP-1]-pos[endP-3]), du3N6)-length_b)*du3N6.x + gravity.x*(dt/2.0f);
	bb[endBB-8] = velocity[endP-3].y + g_t*(dot((pos[endP-6]-pos[endP-3]), du3N1)-length_t)*du3N1.y + g_b*(dot((pos[endP-5]-pos[endP-3]), du3N2)-length_b)*du3N2.y + g_e*(dot((pos[endP-4]-pos[endP-3]), du3N3)-length_e)*du3N3.y + g_e*(dot((pos[endP-2]-pos[endP-3]), du3N5)-length_e)*du3N5.y + g_b*(dot((pos[endP-1]-pos[endP-3]), du3N6)-length_b)*du3N6.y + gravity.y*(dt/2.0f);;
	bb[endBB-7] = velocity[endP-3].z + g_t*(dot((pos[endP-6]-pos[endP-3]), du3N1)-length_t)*du3N1.z + g_b*(dot((pos[endP-5]-pos[endP-3]), du3N2)-length_b)*du3N2.z + g_e*(dot((pos[endP-4]-pos[endP-3]), du3N3)-length_e)*du3N3.z + g_e*(dot((pos[endP-2]-pos[endP-3]), du3N5)-length_e)*du3N5.z + g_b*(dot((pos[endP-1]-pos[endP-3]), du3N6)-length_b)*du3N6.z + gravity.z*(dt/2.0f);;
	bb[endBB-6] = velocity[endP-2].x + g_t*(dot((pos[endP-5]-pos[endP-2]), du2N2)-length_t)*du2N2.x + g_b*(dot((pos[endP-4]-pos[endP-2]), du2N3)-length_b)*du2N3.x + g_e*(dot((pos[endP-3]-pos[endP-2]), du2N4)-length_e)*du2N4.x + g_e*(dot((pos[endP-1]-pos[endP-2]), du2N6)-length_e)*du2N6.x + gravity.x*(dt/2.0f);
	bb[endBB-5] = velocity[endP-2].y + g_t*(dot((pos[endP-5]-pos[endP-2]), du2N2)-length_t)*du2N2.y + g_b*(dot((pos[endP-4]-pos[endP-2]), du2N3)-length_b)*du2N3.y + g_e*(dot((pos[endP-3]-pos[endP-2]), du2N4)-length_e)*du2N4.y + g_e*(dot((pos[endP-1]-pos[endP-2]), du2N6)-length_e)*du2N6.y + gravity.y*(dt/2.0f);
	bb[endBB-4] = velocity[endP-2].z + g_t*(dot((pos[endP-5]-pos[endP-2]), du2N2)-length_t)*du2N2.z + g_b*(dot((pos[endP-4]-pos[endP-2]), du2N3)-length_b)*du2N3.z + g_e*(dot((pos[endP-3]-pos[endP-2]), du2N4)-length_e)*du2N4.z + g_e*(dot((pos[endP-1]-pos[endP-2]), du2N6)-length_e)*du2N6.z + gravity.z*(dt/2.0f);
	bb[endBB-3] = velocity[endP-1].x + g_t*(dot((pos[endP-4]-pos[endP-1]), du1N3)-length_t)*du1N3.x + g_b*(dot((pos[endP-3]-pos[endP-1]), du1N4)-length_b)*du1N4.x + g_e*(dot((pos[endP-2]-pos[endP-1]), du1N5)-length_e)*du1N5.x + gravity.x*(dt/2.0f);
	bb[endBB-2] = velocity[endP-1].y + g_t*(dot((pos[endP-4]-pos[endP-1]), du1N3)-length_t)*du1N3.y + g_b*(dot((pos[endP-3]-pos[endP-1]), du1N4)-length_b)*du1N4.y + g_e*(dot((pos[endP-2]-pos[endP-1]), du1N5)-length_e)*du1N5.y + gravity.y*(dt/2.0f);
	bb[endBB-1] = velocity[endP-1].z + g_t*(dot((pos[endP-4]-pos[endP-1]), du1N3)-length_t)*du1N3.z + g_b*(dot((pos[endP-3]-pos[endP-1]), du1N4)-length_b)*du1N4.z + g_e*(dot((pos[endP-2]-pos[endP-1]), du1N5)-length_e)*du1N5.z + gravity.z*(dt/2.0f);
}

__device__
void conjugate_(int numParticles,
				int numComponents,
				float* AA,
				float* bb,
				float* xx)
{
	int sid = blockIdx.x;
	
	int N = numParticles*numComponents;
	float r[NUMPARTICLES*NUMCOMPONENTS];
	float p[NUMPARTICLES*NUMCOMPONENTS];
	
	int startAA = sid * numParticles * numComponents * numParticles * numComponents;
	int startBB = sid * numParticles * numComponents;
	int startXX = sid * numParticles * numComponents;
	
	for(int i = 0; i < N; i++)
	{
		//r = b - Ax
		r[i] = bb[startBB+i];
		for(int j = 0; j < N; j++)
		{
			r[i] -= AA[startAA+i*N+j]*xx[startXX+j];
		}
	
		//p = r
		p[i] = r[i];
	}

	float rsold = 0.0f;

	for(int i = 0; i < N; i ++)
	{
		rsold += r[i] * r[i];
	}

	for(int i = 0; i < N; i++)
	{
		float Ap[NUMPARTICLES*NUMCOMPONENTS];
	
		for(int j = 0; j < N; j++)
		{
			Ap[j] = 0.0f;
		
			for(int k = 0; k < N; k++)
			{
				Ap[j] += AA[startAA+j*N+k] * p[k];
			}
		}
	
		float abot = 0.0f;
	
		for(int j = 0; j < N; j++)
		{
			abot += p[j] * Ap[j];
		}
	
		float alpha = rsold / abot;
	
		for(int j = 0; j < N; j++)
		{
			xx[startXX+j] = xx[startXX+j] + alpha * p[j];
		
			r[j] = r[j] - alpha * Ap[j];
		}
	
		float rsnew = 0.0f;
	
		for(int j = 0; j < N; j++)
		{
			rsnew += r[j] * r[j];
		}
	
		if(rsnew < 1e-10f)
		{
			break;
		}
		
		for(int j = 0; j < N; j++)
		{
			p[j] = r[j] + rsnew / rsold * p[j];
		}
	
		rsold = rsnew;
	}
}


__device__
void calcVelocities_(int numParticles,
					 int numComponents,
					 float dt,
					 float mass,
					 float k_edge,
					 float k_bend,
					 float k_twist,
					 float d_edge,
					 float d_bend,
					 float d_twist,
					 float length_e,
					 float length_b,
					 float length_t,
					 float3 &gravity,
					 float3* root,
					 float3* pos,
					 float3* velocity,
					 float3* velh,
					 float* AA,
					 float* bb,
					 float* xx)
{
	//Calculate the velocities of each particle
	
	//Build matrix and vector of coefficients of linear equations		
	buildAB_(numParticles, numComponents, dt, mass, k_edge, k_bend, k_twist, d_edge, d_bend, d_twist, length_e, length_b, length_t, gravity, root, pos, velocity, AA, bb, xx);
	
//	int startAA = blockIdx.x * numParticles * numComponents * numParticles * numComponents;
//	int endAA = startAA + numParticles * numComponents * numParticles * numComponents;

//	for(int i = startAA; i < endAA; i++)
//	{
//		printf("AA[%04d]:\t%.7f\n", i, AA[i]);
//	}

//	int startBB = blockIdx.x * numParticles * numComponents;
//	int endBB = startBB + numParticles * numComponents;
//
//	for (int i = startBB; i < endBB; ++i)
//	{
//			printf("bb:\t%d\t%.7f\n",i, bb[i]);
//	}
//	printf("\n\n");


	int startP = blockIdx.x * numParticles;
	int endP = startP + numParticles;
	
	//Set intial solution to previous velocity
	for(int i = startP; i < endP; i++)
	{
		xx[i*numComponents  ] = velocity[i].x;
		xx[i*numComponents+1] = velocity[i].y;
		xx[i*numComponents+2] = velocity[i].z;		
	}
	
	//Solve for velocity using conjugate gradient method
	conjugate_(numParticles, numComponents, AA, bb, xx);
//	conjugate_(hair->numParticles, hair->numComponents, hair->AA_, hair->bb_, hair->xx_);
	
//	for(int i = startP; i < endP; i++)
//	{
//		int idx0 = i*numComponents;
//		int idx1 = idx0 + 1;
//		int idx2 = idx1 + 1;
//
//		printf("xx[%02d]:\t%.7f\n",idx0, xx[idx0]);
//		printf("xx[%02d]:\t%.7f\n",idx1, xx[idx1]);
//		printf("xx[%02d]:\t%.7f\n",idx2, xx[idx2]);
//	}

	//Copy solution to half velocity
	for(int i = startP; i < endP; i++)
	{
		velh[i].x = xx[i*numComponents  ];
		velh[i].y = xx[i*numComponents+1];
		velh[i].z = xx[i*numComponents+2];
	}
}

__device__
void updateSprings_(int numParticles,
					int numComponents,
					float dt,
					float mass,
					float k_edge,
					float k_bend,
					float k_twist,
					float k_extra,
					float d_edge,
					float d_bend,
					float d_twist,
					float d_extra,
					float length_e,
					float length_b,
					float length_t,
					float3 &gravity,
					float3* root,
					float3* pos,
					float3* velh,
					float3* force)
{
	int sid = blockIdx.x;
	
	int startP = sid * numParticles;
	int endP = startP + numParticles;
	
	//calculate the 6 coefficients
	float g_e = k_edge/length_e;
	float g_b = k_bend/length_b;
	float g_t = k_twist/length_t;
	float h_e = dt*k_edge/(2.0f*length_e) + d_edge;
	float h_b = dt*k_bend/(2.0f*length_b) + d_bend;
	float h_t = dt*k_twist/(2.0f*length_t) + d_twist;
	
	//Calculate and apply forces for the first three particles
	float3 uu0R(root[sid]	 -pos[startP]);
	float3 uu01(pos[startP+1]-pos[startP]);
	float3 uu02(pos[startP+2]-pos[startP]);
	float3 uu03(pos[startP+3]-pos[startP]);
	float3 du0R(uu0R);
	float3 du01(uu01);
	float3 du02(uu02);
	float3 du03(uu03);
	unitize(du0R);
	unitize(du01);
	unitize(du02);
	unitize(du03);
	float3 vu0R(			  -velh[startP]);
	float3 vu01(velh[startP+1]-velh[startP]);
	float3 vu02(velh[startP+2]-velh[startP]);
	float3 vu03(velh[startP+3]-velh[startP]);
	
	float3 force0 ( du0R*((dot(uu0R, du0R)-length_e)*g_e) + du0R*((dot(vu0R, du0R))*h_e) +
				    du01*((dot(uu01, du01)-length_e)*g_e) + du01*((dot(vu01, du01))*h_e) +
				    du02*((dot(uu02, du02)-length_b)*g_b) + du02*((dot(vu02, du02))*h_b) +
				    du03*((dot(uu03, du03)-length_t)*g_t) + du03*((dot(vu03, du03))*h_t) );
	
	force[startP] = (force[startP] + force0);
	
	float3 uu1R(root[sid]	 -pos[startP+1]);
	float3 uu10(pos[startP  ]-pos[startP+1]);
	float3 uu12(pos[startP+2]-pos[startP+1]);
	float3 uu13(pos[startP+3]-pos[startP+1]);
	float3 uu14(pos[startP+4]-pos[startP+1]);
	float3 du1R(uu1R);
	float3 du10(uu10);
	float3 du12(uu12);
	float3 du13(uu13);
	float3 du14(uu14);
	unitize(du1R);
	unitize(du10);
	unitize(du12);
	unitize(du13);
	unitize(du14);
	float3 vu1R(			  -velh[startP+1]);
	float3 vu10(velh[startP  ]-velh[startP+1]);
	float3 vu12(velh[startP+2]-velh[startP+1]);
	float3 vu13(velh[startP+3]-velh[startP+1]);
	float3 vu14(velh[startP+4]-velh[startP+1]);
	
	float3 force1 ( du1R*(dot(uu1R, du1R)-length_b)*g_b + du1R*(dot(vu1R, du1R))*h_b + 
					du10*(dot(uu10, du10)-length_e)*g_e + du10*(dot(vu10, du10))*h_e + 
					du12*(dot(uu12, du12)-length_e)*g_e + du12*(dot(vu12, du12))*h_e + 
					du13*(dot(uu13, du13)-length_b)*g_b + du13*(dot(vu13, du13))*h_b + 
					du14*(dot(uu14, du14)-length_t)*g_t + du14*(dot(vu14, du14))*h_t );
	
	force[startP+1] = (force[startP+1] + force1);
	
	float3 uu2R(root[sid]	 -pos[startP+2]);
	float3 uu20(pos[startP  ]-pos[startP+2]);
	float3 uu21(pos[startP+1]-pos[startP+2]);
	float3 uu23(pos[startP+3]-pos[startP+2]);
	float3 uu24(pos[startP+4]-pos[startP+2]);
	float3 uu25(pos[startP+5]-pos[startP+2]);
	float3 du2R(uu2R);
	float3 du20(uu20);
	float3 du21(uu21);
	float3 du23(uu23);
	float3 du24(uu24);
	float3 du25(uu25);
	unitize(du2R);
	unitize(du20);
	unitize(du21);
	unitize(du23);
	unitize(du24);
	unitize(du25);
	float3 vu2R(			  -velh[startP+2]);
	float3 vu20(velh[startP+0]-velh[startP+2]);
	float3 vu21(velh[startP+1]-velh[startP+2]);
	float3 vu23(velh[startP+3]-velh[startP+2]);
	float3 vu24(velh[startP+4]-velh[startP+2]);
	float3 vu25(velh[startP+5]-velh[startP+2]);
	
	float3 force2 ( du2R*(dot(uu2R, du2R)-length_t)*g_t + du2R*(dot(vu2R, du2R))*h_t +
					du20*(dot(uu20, du20)-length_b)*g_b + du20*(dot(vu20, du20))*h_b +
					du21*(dot(uu21, du21)-length_e)*g_e + du21*(dot(vu21, du21))*h_e +
					du23*(dot(uu23, du23)-length_e)*g_e + du23*(dot(vu23, du23))*h_e +
					du24*(dot(uu24, du24)-length_b)*g_b + du24*(dot(vu24, du24))*h_b +
					du25*(dot(uu25, du25)-length_t)*g_t + du25*(dot(vu25, du25))*h_t );
	
	force[startP+2] = (force[startP+2] + force2);
	
	//Calculate force for all particles between first and last
	for(int i = startP+3; i < (endP-3); i++)
	{
		float3 uu3(pos[i-3]-pos[i]);
		float3 uu2(pos[i-2]-pos[i]);
		float3 uu1(pos[i-1]-pos[i]);
		float3 ud1(pos[i+1]-pos[i]);
		float3 ud2(pos[i+2]-pos[i]);
		float3 ud3(pos[i+3]-pos[i]);
		float3 dui3(uu3);
		float3 dui2(uu2);
		float3 dui1(uu1);
		float3 ddi1(ud1);
		float3 ddi2(ud2);
		float3 ddi3(ud3);
		unitize(dui3);
		unitize(dui2);
		unitize(dui1);
		unitize(ddi1);
		unitize(ddi2);
		unitize(ddi3);
		float3 vu3(velh[i-3]-velh[i]);
		float3 vu2(velh[i-2]-velh[i]);
		float3 vu1(velh[i-1]-velh[i]);
		float3 vd1(velh[i+1]-velh[i]);
		float3 vd2(velh[i+2]-velh[i]);
		float3 vd3(velh[i+3]-velh[i]);
		
		float3 forcei ( dui3*(dot(uu3, dui3)-length_t)*g_t + dui3*(dot(vu3, dui3))*h_t + 
						dui2*(dot(uu2, dui2)-length_b)*g_b + dui2*(dot(vu2, dui2))*h_b + 
						dui1*(dot(uu1, dui1)-length_e)*g_e + dui1*(dot(vu1, dui1))*h_e + 
						ddi1*(dot(ud1, ddi1)-length_e)*g_e + ddi1*(dot(vd1, ddi1))*h_e + 
						ddi2*(dot(ud2, ddi2)-length_b)*g_b + ddi2*(dot(vd2, ddi2))*h_b + 
						ddi3*(dot(ud3, ddi3)-length_t)*g_t + ddi3*(dot(vd3, ddi3))*h_t );
		
		force[i] = (force[i] + forcei);
	}
	
	//Calculate and apply forces for last three particles
	float3 uu3N1(pos[endP-6]-pos[endP-3]);
	float3 uu3N2(pos[endP-5]-pos[endP-3]);
	float3 uu3N3(pos[endP-4]-pos[endP-3]);
	float3 uu3N5(pos[endP-2]-pos[endP-3]);
	float3 uu3N6(pos[endP-1]-pos[endP-3]);
	float3 du3N1(uu3N1);
	float3 du3N2(uu3N2);
	float3 du3N3(uu3N3);
	float3 du3N5(uu3N5);
	float3 du3N6(uu3N6);
	unitize(du3N1);
	unitize(du3N2);
	unitize(du3N3);
	unitize(du3N5);
	unitize(du3N6);
	float3 vu3N1(velh[endP-6]-velh[endP-3]);
	float3 vu3N2(velh[endP-5]-velh[endP-3]);
	float3 vu3N3(velh[endP-4]-velh[endP-3]);
	float3 vu3N5(velh[endP-2]-velh[endP-3]);
	float3 vu3N6(velh[endP-1]-velh[endP-3]);
	
	float3 force3N ( du3N1*(dot(uu3N1, du3N1)-length_t)*g_t + du3N1*(dot(vu3N1, du3N1))*h_t +
					 du3N2*(dot(uu3N2, du3N2)-length_b)*g_b + du3N2*(dot(vu3N2, du3N2))*h_b +
					 du3N3*(dot(uu3N3, du3N3)-length_e)*g_e + du3N3*(dot(vu3N3, du3N3))*h_e +
					 du3N5*(dot(uu3N5, du3N5)-length_e)*g_e + du3N5*(dot(vu3N5, du3N5))*h_e +
					 du3N6*(dot(uu3N6, du3N6)-length_b)*g_b + du3N6*(dot(vu3N6, du3N6))*h_b );
	
	force[endP-3] = (force[endP-3] + force3N);
	
	float3 uu2N2(pos[endP-5]-pos[endP-2]);
	float3 uu2N3(pos[endP-4]-pos[endP-2]);
	float3 uu2N4(pos[endP-3]-pos[endP-2]);
	float3 uu2N6(pos[endP-1]-pos[endP-2]);
	float3 du2N2(uu2N2);
	float3 du2N3(uu2N3);
	float3 du2N4(uu2N4);
	float3 du2N6(uu2N6);
	unitize(du2N2);
	unitize(du2N3);
	unitize(du2N4);
	unitize(du2N6);
	float3 vu2N2(velh[endP-5]-velh[endP-2]);
	float3 vu2N3(velh[endP-4]-velh[endP-2]);
	float3 vu2N4(velh[endP-3]-velh[endP-2]);
	float3 vu2N6(velh[endP-1]-velh[endP-2]);
	
	float3 force2N ( du2N2*(dot(uu2N2, du2N2)-length_t)*g_t + du2N2*(dot(vu2N2, du2N2))*h_t +
					 du2N3*(dot(uu2N3, du2N3)-length_t)*g_b + du2N3*(dot(vu2N3, du2N3))*h_b +
					 du2N4*(dot(uu2N4, du2N4)-length_t)*g_e + du2N4*(dot(vu2N4, du2N4))*h_e +
					 du2N6*(dot(uu2N6, du2N6)-length_t)*g_e + du2N6*(dot(vu2N6, du2N6))*h_e );
	
	force[endP-2] = (force[endP-2] + force2N);
	
	float3 uu1N3(pos[endP-4]-pos[endP-1]);
	float3 uu1N4(pos[endP-3]-pos[endP-1]);
	float3 uu1N5(pos[endP-2]-pos[endP-1]);
	float3 du1N3(uu1N3);
	float3 du1N4(uu1N4);
	float3 du1N5(uu1N5);
	unitize(du1N3);
	unitize(du1N4);
	unitize(du1N5);
	float3 vu1N3(velh[endP-4]-velh[endP-1]);
	float3 vu1N4(velh[endP-3]-velh[endP-1]);
	float3 vu1N5(velh[endP-2]-velh[endP-1]);
	
	float3 force1N ( du1N3*(dot(uu1N3, du1N3)-length_t)*g_t + du1N3*(dot(vu1N3, du1N3))*h_t +
					 du1N4*(dot(uu1N4, du1N4)-length_b)*g_b + du1N4*(dot(vu1N4, du1N4))*h_b +
					 du1N5*(dot(uu1N5, du1N5)-length_e)*g_e + du1N5*(dot(vu1N5, du1N5))*h_e );
	
	force[endP-1] = (force[endP-1] + force1N);

//	for(int i = startP; i < endP; i++)
//	{
//		printf("force[%02d]:\t%0.7f\t%0.7f\t%0.7f\n", i, force[i].x, force[i].y, force[i].z);
//	}
}

__device__
void applyForce_(float3 appliedForce, int numParticles, float3 *force)
{
	int startP = blockIdx.x * numParticles;
	int endP = startP + numParticles;
	
	for(int i = startP; i < endP; i++)
	{
		force[i] = (force[i] + appliedForce);
	}
}

__device__
void updateVelocities_(int numParticles, float dt, float3 *velocity, float3 *velh, float3 *force)
{
	int startP = blockIdx.x * numParticles;
	int endP = startP + numParticles;
	
	for(int i = startP; i < endP; i++)
	{
		velh[i] = velocity[i] + (force[i] * (dt / 2.0f));
	}
}

__device__
void updatePositions_(int numParticles, float dt, float3 *position, float3 *posh, float3 *pos, float3 *velh, float3* force)
{
	int startP = blockIdx.x * numParticles;
	int endP = startP + numParticles;
	
	for(int i = startP; i < endP; i++)
	{
		//Save old position
		float3 poso(position[i]);
		
		//Calculate new position
		position[i] = poso + (velh[i] * dt);
		
		//Calculate half position
		posh[i] = (poso + position[i])/2.0f;
		
		//Use half position in current calculations
		pos[i] = posh[i];

		//Reset forces on particles here
		force[i].x = 0.0f;
		force[i].y = 0.0f;
		force[i].z = 0.0f;
	}
}

__device__
void updateParticles_(int numParticles, float dt, float3* position, float3* pos, float3* velocity, float3* velh, float3* force)
{
	int startP = blockIdx.x * numParticles;
	int endP = startP + numParticles;
	
	for(int i = startP; i < endP; i++)
	{
		//Calculate half velocity
		velh[i] = velocity[i] + (force[i] * (dt / 2.0f));
		
		//Extrapolate new velocity
		velocity[i] = (velh[i] * 2) - velocity[i];
		
		//Use previous position in current calculations
		pos[i] = position[i];
		
		//Reset forces on particles
		force[i].x = 0.0f;
		force[i].y = 0.0f;
		force[i].z = 0.0f;
	}
}

__device__
void applyStrainLimiting_(int numParticles, float dt, float3* root, float3* posc, float3* pos, float3* velh)
{
	int sid = blockIdx.x;
	int startP = blockIdx.x * numParticles;
	int endP = startP + numParticles;
	
	for(int i = startP; i < endP; i++)
	{
		//Calculate candidate position using half velocity
		posc[i] = pos[i] + (velh[i] * dt);
		
		//Determine the direction of the spring between the particles
		float3 dir = (i > startP) ? (posc[i] - posc[i-1]) : (posc[i] - root[sid]);
		
		if(length_sqr(dir) > MAX_LENGTH_SQUARED)
		{
			//Find a valid candidate position
			posc[i] = (i > startP) ? (posc[i-1] + (dir * (MAX_LENGTH*length_inverse(dir)))) : (root[sid] + (dir * (MAX_LENGTH*length_inverse(dir)))); //fast length calculation
			
			//~ particle[i]->posc = particle[i-1]->posc + (dir * (MAX_LENGTH/dir.length())); //slower length calculation
			
			//Calculate new half velocity based on valid candidate position, i.e. add a velocity impulse
			velh[i] = (posc[i] - pos[i])/dt;
		}
	}
}

__global__
void update_strands(int numParticles,
					int numStrands,
					int numComponents,
					float dt,
					float mass,
					float k_edge,
					float k_bend,
					float k_twist,
					float k_extra,
					float d_edge,
					float d_bend,
					float d_twist,
					float d_extra,
					float length_e,
					float length_b,
					float length_t,
					float3 gravity,
					float3* root,
					float3* position,
					float3* posc,
					float3* posh,
					float3* pos,
					float3* velocity,
					float3* velh,
					float3* force,
					float* AA,
					float* bb,
					float* xx)
{
	//TODO remove after unit testing
//	dt = 0.008f;
	
	//Calculate candidate velocities
	calcVelocities_(numParticles, numComponents, dt, mass, k_edge, k_bend, k_twist, d_edge, d_bend, d_twist, length_e, length_b, length_t, gravity, root, pos, velocity, velh, AA, bb, xx);


	//Calculate and apply spring forces using previous position
	updateSprings_(numParticles, numComponents, dt, mass, k_edge, k_bend, k_twist, k_extra, d_edge, d_bend, d_twist, d_extra, length_e, length_b, length_t, gravity, root, pos, velh, force);
	
	float3 mgravity;
	mgravity.x = mass * gravity.x;
	mgravity.y = mass * gravity.y;
	mgravity.z = mass * gravity.z;


	//Apply gravity
	applyForce_(mgravity, numParticles, force);

//	int startP = blockIdx.x * numParticles;
//	int endP = startP + numParticles;

//	for(int i = startP; i < endP; i++)
//	{
//		printf("force[%02d]:\t%.7f\t%.7f\t%.7f\n", i, force[i].x, force[i].y, force[i].z);
//	}

	//Calculate half velocities using forces
	updateVelocities_(numParticles, dt, velocity, velh, force);
	
//	for(int i = startP; i < endP; i++)
//	{
//		printf("velh[%02d]:\t%.7f\t%.7f\t%.7f\n", i, velh[i].x, velh[i].y, velh[i].z);
//	}


	applyStrainLimiting_(numParticles, dt, root, posc, pos, velh);

//	for(int i = startP; i < endP; i++)
//	{
//		printf("velh[%02d]:\t%.7f\t%.7f\t%.7f\n", i, velh[i].x, velh[i].y, velh[i].z);
//	}


	//Calculate half position and new position
	updatePositions_(numParticles, dt, position, posh, pos, velh, force);

//	for(int i = startP; i < endP; i++)
//	{
//		printf("position[%02d]:\t%.7f\t%.7f\t%.7f\n", i, position[i].x, position[i].y, position[i].z);
//	}

//	for(int i = startP; i < endP; i++)
//	{
//		printf("posh[%02d]:\t%.7f\t%.7f\t%.7f\n", i, posh[i].x, posh[i].y, posh[i].z);
//	}
	
	//Calculate velocities using half position
	calcVelocities_(numParticles, numComponents, dt, mass, k_edge, k_bend, k_twist, d_edge, d_bend, d_twist, length_e, length_b, length_t, gravity, root, pos, velocity, velh, AA, bb, xx);
	
//	for(int i = startP; i < endP; i++)
//	{
//		printf("velh[%02d]:\t%.7f\t%.7f\t%.7f\n", i, velh[i].x, velh[i].y, velh[i].z);
//	}


	//Calculate and apply spring forces using half position
	updateSprings_(numParticles, numComponents, dt, mass, k_edge, k_bend, k_twist, k_extra, d_edge, d_bend, d_twist, d_extra, length_e, length_b, length_t, gravity, root, pos, velh, force);

//	for(int i = startP; i < endP; i++)
//	{
//		printf("force[%02d]:\t%.7f\t%.7f\t%.7f\n", i, force[i].x, force[i].y, force[i].z);
//	}

	//Apply gravity
	applyForce_(mgravity, numParticles, force);

//	for(int i = startP; i < endP; i++)
//	{
//		printf("force[%02d]:\t%.7f\t%.7f\t%.7f\n", i, force[i].x, force[i].y, force[i].z);
//	}
	
	//Calculate half velocity and new velocity
	updateParticles_(numParticles, dt, position, pos, velocity, velh, force);
	
//	for(int i  = startP; i < endP; i++)
//	{
//		printf("velh[%02d]:\t%.7f\t%.7f\t%.7f\n", i, velh[i].x, velh[i].y, velh[i].z);
//		printf("velocity[%02d]:\t%.7f\t%.7f\t%.7f\n", i, velocity[i].x, velocity[i].y, velocity[i].z);
//		printf("pos[%02d]:\t%.7f\t%.7f\t%.7f\n", i, pos[i].x, pos[i].y, pos[i].z);
//	}
}

#endif
