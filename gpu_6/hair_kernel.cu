
#ifndef _HAIR_KERNEL_H_
#define _HAIR_KERNEL_H_

#include "constants.h"
#include "hair.h"

__global__
void initialise(int numParticles,
				int numStrands,
				int numComponents,
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
				float* AA,
				float* bb,
				float* xx,
				pilar::Vector3f gravity1,
				pilar::Vector3f* root1,
				pilar::Vector3f* normal1,
				pilar::Vector3f* position1,
				pilar::Vector3f* pos1,
				pilar::Vector3f* posc1,
				pilar::Vector3f* posh1,
				pilar::Vector3f* velocity1,
				pilar::Vector3f* velh1,
				pilar::Vector3f* force1,
				pilar::HairState* state)
{
	//Strand ID
	int sid = blockIdx.x;
	
	int start = numParticles * sid;
	int end = start + numParticles;
	
	for(int i = start, j = 1; i < end; i++, j++)
	{
		normal1[sid].unitize();
		position1[i] = root1[sid] + (normal1[sid] * (0.025f * j));
		pos1[i] = position1[i];
		posc1[i] = position1[i];
	}

	state->numParticles = numParticles;
	state->numStrands = numStrands;
	state->numComponents = numComponents;
	state->mass = mass;
	state->k_edge = k_edge;
	state->k_bend = k_bend;
	state->k_twist = k_twist;
	state->k_extra = k_extra;
	state->d_edge = d_edge;
	state->d_bend = d_bend;
	state->d_twist = d_twist;
	state->d_extra = d_extra;
	state->length_e = length_e;
	state->length_b = length_b;
	state->length_t = length_t;
	
	state->AA = AA;
	state->bb = bb;
	state->xx = xx;

	state->gravity1 = gravity1;
	state->root1 = root1;
	state->normal1 = normal1;
	state->position1 = position1;
	state->pos1 = pos1;
	state->posc1 = posc1;
	state->posh1 = posh1;
	state->velocity1 = velocity1;
	state->velh1 = velh1;
	state->force1 = force1;
}

__device__
void buildAB(float dt, pilar::HairState* state)
{
	//State pointers
	pilar::Vector3f* root = state->root1;
	pilar::Vector3f* pos = state->pos1;
	pilar::Vector3f* velocity = state->velocity1;
	float* AA = state->AA;
	float* bb = state->bb;

	//Strand ID
	int sid = blockIdx.x;

//	printf("root[%d]:\t%.7f\t%.7f\t%.7f\n", sid, root[sid].x, root[sid].y, root[sid].z);

	//Start and end of square matrix A indices
	int startAA = sid * state->numParticles * state->numComponents * state->numParticles * state->numComponents;
	int endAA = startAA + state->numParticles * state->numComponents * state->numParticles * state->numComponents;

	//Start and end of particle indices
	int startP = sid * state->numParticles;
	int endP = startP + state->numParticles;

//	for(int i = startP; i < endP; i++)
//	{
//		printf("pos[%03d]:\t%.7f\t%.7f\t%.7f\n", i, pos[i].x, pos[i].y, pos[i].z);
//	}

	//Set AA to zero
	for(int i = startAA; i < endAA; i++)
	{
		AA[i] = 0.0f;
	}

	//Set the 6 different coefficients
	float h_e = dt*dt*state->k_edge/(4.0f*state->mass*state->length_e) + state->d_edge*dt/(2.0f*state->mass);
	float h_b = dt*dt*state->k_bend/(4.0f*state->mass*state->length_b) + state->d_bend*dt/(2.0f*state->mass);
	float h_t = dt*dt*state->k_twist/(4.0f*state->mass*state->length_t) + state->d_twist*dt/(2.0f*state->mass);
	float g_e = dt*state->k_edge/(2.0f*state->mass*state->length_e);
	float g_b = dt*state->k_bend/(2.0f*state->mass*state->length_b);
	float g_t = dt*state->k_twist/(2.0f*state->mass*state->length_t);

	//Set the first 3 particle direction vectors

	//First particle direction vectors
	pilar::Vector3f du0R(root[sid]    -pos[startP]);
	pilar::Vector3f du01(pos[startP+1]-pos[startP]);
	pilar::Vector3f du02(pos[startP+2]-pos[startP]);
	pilar::Vector3f du03(pos[startP+3]-pos[startP]);
	du0R.unitize();
	du01.unitize();
	du02.unitize();
	du03.unitize();

	//Second particle direction vectors
	pilar::Vector3f du1R(root[sid]    -pos[startP+1]);
	pilar::Vector3f du10(pos[startP  ]-pos[startP+1]);
	pilar::Vector3f du12(pos[startP+2]-pos[startP+1]);
	pilar::Vector3f du13(pos[startP+3]-pos[startP+1]);
	pilar::Vector3f du14(pos[startP+4]-pos[startP+1]);
	du1R.unitize();
	du10.unitize();
	du12.unitize();
	du13.unitize();
	du14.unitize();

	//Third particle direction vectors
	pilar::Vector3f du2R(root[sid]    -pos[startP+2]);
	pilar::Vector3f du20(pos[startP  ]-pos[startP+2]);
	pilar::Vector3f du21(pos[startP+1]-pos[startP+2]);
	pilar::Vector3f du23(pos[startP+3]-pos[startP+2]);
	pilar::Vector3f du24(pos[startP+4]-pos[startP+2]);
	pilar::Vector3f du25(pos[startP+5]-pos[startP+2]);
	du2R.unitize();
	du20.unitize();
	du21.unitize();
	du23.unitize();
	du24.unitize();
	du25.unitize();


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
	int row11 = startAA + state->numParticles * state->numComponents;
	int row22 = startAA + 2 * state->numParticles * state->numComponents;

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

	int row33 = startAA + 3 * state->numParticles * state->numComponents;
	int row44 = startAA + 4 * state->numParticles * state->numComponents;
	int row55 = startAA + 5 * state->numParticles * state->numComponents;

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

	int row66 = startAA + 6 * state->numParticles * state->numComponents;
	int row77 = startAA + 7 * state->numParticles * state->numComponents;
	int row88 = startAA + 8 * state->numParticles * state->numComponents;

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

	int startBB = sid * state->numParticles * state->numComponents;
	int endBB = startBB + state->numParticles * state->numComponents;

	//Set the first nine entries of the b vector
	bb[startBB  ] = velocity[startP  ].x + g_e*((root[sid]-pos[startP  ]).dot(du0R)-state->length_e)*du0R.x + g_e*((pos[startP+1]-pos[startP  ]).dot(du01)-state->length_e)*du01.x + g_b*((pos[startP+2]-pos[startP  ]).dot(du02)-state->length_b)*du02.x + g_t*((pos[startP+3]-pos[startP  ]).dot(du03)-state->length_t)*du03.x + state->gravity1.x*(dt/2.0f);
	bb[startBB+1] = velocity[startP  ].y + g_e*((root[sid]-pos[startP  ]).dot(du0R)-state->length_e)*du0R.y + g_e*((pos[startP+1]-pos[startP  ]).dot(du01)-state->length_e)*du01.y + g_b*((pos[startP+2]-pos[startP  ]).dot(du02)-state->length_b)*du02.y + g_t*((pos[startP+3]-pos[startP  ]).dot(du03)-state->length_t)*du03.y + state->gravity1.y*(dt/2.0f);
	bb[startBB+2] = velocity[startP  ].z + g_e*((root[sid]-pos[startP  ]).dot(du0R)-state->length_e)*du0R.z + g_e*((pos[startP+1]-pos[startP  ]).dot(du01)-state->length_e)*du01.z + g_b*((pos[startP+2]-pos[startP  ]).dot(du02)-state->length_b)*du02.z + g_t*((pos[startP+3]-pos[startP  ]).dot(du03)-state->length_t)*du03.z + state->gravity1.z*(dt/2.0f);
	bb[startBB+3] = velocity[startP+1].x + g_b*((root[sid]-pos[startP+1]).dot(du1R)-state->length_b)*du1R.x + g_e*((pos[startP  ]-pos[startP+1]).dot(du10)-state->length_e)*du10.x + g_e*((pos[startP+2]-pos[startP+1]).dot(du12)-state->length_e)*du12.x + g_b*((pos[startP+3]-pos[startP+1]).dot(du13)-state->length_b)*du13.x + g_t*((pos[startP+4]-pos[startP+1]).dot(du14)-state->length_t)*du14.x + state->gravity1.x*(dt/2.0f);
	bb[startBB+4] = velocity[startP+1].y + g_b*((root[sid]-pos[startP+1]).dot(du1R)-state->length_b)*du1R.y + g_e*((pos[startP  ]-pos[startP+1]).dot(du10)-state->length_e)*du10.y + g_e*((pos[startP+2]-pos[startP+1]).dot(du12)-state->length_e)*du12.y + g_b*((pos[startP+3]-pos[startP+1]).dot(du13)-state->length_b)*du13.y + g_t*((pos[startP+4]-pos[startP+1]).dot(du14)-state->length_t)*du14.y + state->gravity1.y*(dt/2.0f);
	bb[startBB+5] = velocity[startP+1].z + g_b*((root[sid]-pos[startP+1]).dot(du1R)-state->length_b)*du1R.z + g_e*((pos[startP  ]-pos[startP+1]).dot(du10)-state->length_e)*du10.z + g_e*((pos[startP+2]-pos[startP+1]).dot(du12)-state->length_e)*du12.z + g_b*((pos[startP+3]-pos[startP+1]).dot(du13)-state->length_b)*du13.z + g_t*((pos[startP+4]-pos[startP+1]).dot(du14)-state->length_t)*du14.z + state->gravity1.z*(dt/2.0f);
	bb[startBB+6] = velocity[startP+2].x + g_t*((root[sid]-pos[startP+2]).dot(du2R)-state->length_t)*du2R.x + g_b*((pos[startP  ]-pos[startP+2]).dot(du02)-state->length_b)*du20.x + g_e*((pos[startP+1]-pos[startP+2]).dot(du21)-state->length_e)*du21.x + g_e*((pos[startP+3]-pos[startP+2]).dot(du23)-state->length_e)*du23.x + g_b*((pos[startP+4]-pos[startP+2]).dot(du24)-state->length_b)*du24.x + g_t*((pos[startP+5]-pos[startP+2]).dot(du25)-state->length_t)*du25.x + state->gravity1.x*(dt/2.0f);
	bb[startBB+7] = velocity[startP+2].y + g_t*((root[sid]-pos[startP+2]).dot(du2R)-state->length_t)*du2R.y + g_b*((pos[startP  ]-pos[startP+2]).dot(du02)-state->length_b)*du20.y + g_e*((pos[startP+1]-pos[startP+2]).dot(du21)-state->length_e)*du21.y + g_e*((pos[startP+3]-pos[startP+2]).dot(du23)-state->length_e)*du23.y + g_b*((pos[startP+4]-pos[startP+2]).dot(du24)-state->length_b)*du24.y + g_t*((pos[startP+5]-pos[startP+2]).dot(du25)-state->length_t)*du25.y + state->gravity1.y*(dt/2.0f);
	bb[startBB+8] = velocity[startP+2].z + g_t*((root[sid]-pos[startP+2]).dot(du2R)-state->length_t)*du2R.z + g_b*((pos[startP  ]-pos[startP+2]).dot(du02)-state->length_b)*du20.z + g_e*((pos[startP+1]-pos[startP+2]).dot(du21)-state->length_e)*du21.z + g_e*((pos[startP+3]-pos[startP+2]).dot(du23)-state->length_e)*du23.z + g_b*((pos[startP+4]-pos[startP+2]).dot(du24)-state->length_b)*du24.z + g_t*((pos[startP+5]-pos[startP+2]).dot(du25)-state->length_t)*du25.z + state->gravity1.z*(dt/2.0f);

	//Build in-between values of matrix A and vector b
	//Loop from fourth to third last particles only
	for(int i = startP+3; i < (endP-3); i++)
	{
		//Current particle position, particle above and particle below
		pilar::Vector3f ui = pos[i];

		//Direction vectors for 3 particles above and below the current particle
		pilar::Vector3f uu2 = pos[i-3];
		pilar::Vector3f uu1 = pos[i-2];
		pilar::Vector3f uu0 = pos[i-1];
		pilar::Vector3f ud0 = pos[i+1];
		pilar::Vector3f ud1 = pos[i+2];
		pilar::Vector3f ud2 = pos[i+3];

		pilar::Vector3f du2(uu2-ui);
		pilar::Vector3f du1(uu1-ui);
		pilar::Vector3f du0(uu0-ui);
		pilar::Vector3f dd0(ud0-ui);
		pilar::Vector3f dd1(ud1-ui);
		pilar::Vector3f dd2(ud2-ui);
		du2.unitize();
		du1.unitize();
		du0.unitize();
		dd0.unitize();
		dd1.unitize();
		dd2.unitize();

		int row0 = i * state->numParticles * state->numComponents * 3 + (i - startP - 3) * state->numComponents;
		int row1 = row0 + state->numParticles * state->numComponents;
		int row2 = row1 + state->numParticles * state->numComponents;

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

		bb[i*state->numComponents  ] = velocity[i].x + g_t*((uu2-ui).dot(du2)-state->length_t)*du2.x + g_b*((uu1-ui).dot(du1)-state->length_b)*du1.x + g_e*((uu0-ui).dot(du0)-state->length_e)*du0.x + g_e*((ud0-ui).dot(dd0)-state->length_e)*dd0.x + g_b*((ud1-ui).dot(dd1)-state->length_b)*dd1.x + g_t*((ud2-ui).dot(dd2)-state->length_t)*dd2.x + state->gravity1.x*(dt/2.0f);
		bb[i*state->numComponents+1] = velocity[i].y + g_t*((uu2-ui).dot(du2)-state->length_t)*du2.y + g_b*((uu1-ui).dot(du1)-state->length_b)*du1.y + g_e*((uu0-ui).dot(du0)-state->length_e)*du0.y + g_e*((ud0-ui).dot(dd0)-state->length_e)*dd0.y + g_b*((ud1-ui).dot(dd1)-state->length_b)*dd1.y + g_t*((ud2-ui).dot(dd2)-state->length_t)*dd2.y + state->gravity1.y*(dt/2.0f);
		bb[i*state->numComponents+2] = velocity[i].z + g_t*((uu2-ui).dot(du2)-state->length_t)*du2.z + g_b*((uu1-ui).dot(du1)-state->length_b)*du1.z + g_e*((uu0-ui).dot(du0)-state->length_e)*du0.z + g_e*((ud0-ui).dot(dd0)-state->length_e)*dd0.z + g_b*((ud1-ui).dot(dd1)-state->length_b)*dd1.z + g_t*((ud2-ui).dot(dd2)-state->length_t)*dd2.z + state->gravity1.z*(dt/2.0f);
	}

	//Calculate direction vectors for the last three particles
	//Third to last particle direction vectors
	pilar::Vector3f du3N1(pos[endP-6]-pos[endP-3]);
	pilar::Vector3f du3N2(pos[endP-5]-pos[endP-3]);
	pilar::Vector3f du3N3(pos[endP-4]-pos[endP-3]);
	pilar::Vector3f du3N5(pos[endP-2]-pos[endP-3]);
	pilar::Vector3f du3N6(pos[endP-1]-pos[endP-3]);
	du3N1.unitize();
	du3N2.unitize();
	du3N3.unitize();
	du3N5.unitize();
	du3N6.unitize();

	//Second to last particle direction vectors
	pilar::Vector3f du2N2(pos[endP-5]-pos[endP-2]);
	pilar::Vector3f du2N3(pos[endP-4]-pos[endP-2]);
	pilar::Vector3f du2N4(pos[endP-3]-pos[endP-2]);
	pilar::Vector3f du2N6(pos[endP-1]-pos[endP-2]);
	du2N2.unitize();
	du2N3.unitize();
	du2N4.unitize();
	du2N6.unitize();

	//Last particle direction vectors
	pilar::Vector3f du1N3(pos[endP-4]-pos[endP-1]);
	pilar::Vector3f du1N4(pos[endP-3]-pos[endP-1]);
	pilar::Vector3f du1N5(pos[endP-2]-pos[endP-1]);
	du1N3.unitize();
	du1N4.unitize();
	du1N5.unitize();


	int row3N3 = endAA - 8*state->numParticles*state->numComponents - 18;
	int row3N2 = endAA - 7*state->numParticles*state->numComponents - 18;
	int row3N1 = endAA - 6*state->numParticles*state->numComponents - 18;

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

	int row2N3 = endAA - 5*state->numParticles*state->numComponents - 15;
	int row2N2 = endAA - 4*state->numParticles*state->numComponents - 15;
	int row2N1 = endAA - 3*state->numParticles*state->numComponents - 15;

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

	int row1N3 = endAA - 2*state->numParticles*state->numComponents - 12;
	int row1N2 = endAA -   state->numParticles*state->numComponents - 12;
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
	bb[endBB-9] = velocity[endP-3].x + g_t*((pos[endP-6]-pos[endP-3]).dot(du3N1)-state->length_t)*du3N1.x + g_b*((pos[endP-5]-pos[endP-3]).dot(du3N2)-state->length_b)*du3N2.x + g_e*((pos[endP-4]-pos[endP-3]).dot(du3N3)-state->length_e)*du3N3.x + g_e*((pos[endP-2]-pos[endP-3]).dot(du3N5)-state->length_e)*du3N5.x + g_b*((pos[endP-1]-pos[endP-3]).dot(du3N6)-state->length_b)*du3N6.x + state->gravity1.x*(dt/2.0f);
	bb[endBB-8] = velocity[endP-3].y + g_t*((pos[endP-6]-pos[endP-3]).dot(du3N1)-state->length_t)*du3N1.y + g_b*((pos[endP-5]-pos[endP-3]).dot(du3N2)-state->length_b)*du3N2.y + g_e*((pos[endP-4]-pos[endP-3]).dot(du3N3)-state->length_e)*du3N3.y + g_e*((pos[endP-2]-pos[endP-3]).dot(du3N5)-state->length_e)*du3N5.y + g_b*((pos[endP-1]-pos[endP-3]).dot(du3N6)-state->length_b)*du3N6.y + state->gravity1.y*(dt/2.0f);;
	bb[endBB-7] = velocity[endP-3].z + g_t*((pos[endP-6]-pos[endP-3]).dot(du3N1)-state->length_t)*du3N1.z + g_b*((pos[endP-5]-pos[endP-3]).dot(du3N2)-state->length_b)*du3N2.z + g_e*((pos[endP-4]-pos[endP-3]).dot(du3N3)-state->length_e)*du3N3.z + g_e*((pos[endP-2]-pos[endP-3]).dot(du3N5)-state->length_e)*du3N5.z + g_b*((pos[endP-1]-pos[endP-3]).dot(du3N6)-state->length_b)*du3N6.z + state->gravity1.z*(dt/2.0f);;
	bb[endBB-6] = velocity[endP-2].x + g_t*((pos[endP-5]-pos[endP-2]).dot(du2N2)-state->length_t)*du2N2.x + g_b*((pos[endP-4]-pos[endP-2]).dot(du2N3)-state->length_b)*du2N3.x + g_e*((pos[endP-3]-pos[endP-2]).dot(du2N4)-state->length_e)*du2N4.x + g_e*((pos[endP-1]-pos[endP-2]).dot(du2N6)-state->length_e)*du2N6.x + state->gravity1.x*(dt/2.0f);
	bb[endBB-5] = velocity[endP-2].y + g_t*((pos[endP-5]-pos[endP-2]).dot(du2N2)-state->length_t)*du2N2.y + g_b*((pos[endP-4]-pos[endP-2]).dot(du2N3)-state->length_b)*du2N3.y + g_e*((pos[endP-3]-pos[endP-2]).dot(du2N4)-state->length_e)*du2N4.y + g_e*((pos[endP-1]-pos[endP-2]).dot(du2N6)-state->length_e)*du2N6.y + state->gravity1.y*(dt/2.0f);
	bb[endBB-4] = velocity[endP-2].z + g_t*((pos[endP-5]-pos[endP-2]).dot(du2N2)-state->length_t)*du2N2.z + g_b*((pos[endP-4]-pos[endP-2]).dot(du2N3)-state->length_b)*du2N3.z + g_e*((pos[endP-3]-pos[endP-2]).dot(du2N4)-state->length_e)*du2N4.z + g_e*((pos[endP-1]-pos[endP-2]).dot(du2N6)-state->length_e)*du2N6.z + state->gravity1.z*(dt/2.0f);
	bb[endBB-3] = velocity[endP-1].x + g_t*((pos[endP-4]-pos[endP-1]).dot(du1N3)-state->length_t)*du1N3.x + g_b*((pos[endP-3]-pos[endP-1]).dot(du1N4)-state->length_b)*du1N4.x + g_e*((pos[endP-2]-pos[endP-1]).dot(du1N5)-state->length_e)*du1N5.x + state->gravity1.x*(dt/2.0f);
	bb[endBB-2] = velocity[endP-1].y + g_t*((pos[endP-4]-pos[endP-1]).dot(du1N3)-state->length_t)*du1N3.y + g_b*((pos[endP-3]-pos[endP-1]).dot(du1N4)-state->length_b)*du1N4.y + g_e*((pos[endP-2]-pos[endP-1]).dot(du1N5)-state->length_e)*du1N5.y + state->gravity1.y*(dt/2.0f);
	bb[endBB-1] = velocity[endP-1].z + g_t*((pos[endP-4]-pos[endP-1]).dot(du1N3)-state->length_t)*du1N3.z + g_b*((pos[endP-3]-pos[endP-1]).dot(du1N4)-state->length_b)*du1N4.z + g_e*((pos[endP-2]-pos[endP-1]).dot(du1N5)-state->length_e)*du1N5.z + state->gravity1.z*(dt/2.0f);
}

__device__
void conjugate(pilar::HairState* state)
{
	//State pointers
	float* AA = state->AA;
	float* bb = state->bb;
	float* xx = state->xx;

	int sid = blockIdx.x;

	int N = state->numParticles * state->numComponents;

	//FIXME research removing constant values
	float r[NUMPARTICLES*NUMCOMPONENTS];
	float p[NUMPARTICLES*NUMCOMPONENTS];

	int startAA = sid * state->numParticles * state->numComponents * state->numParticles * state->numComponents;
	int startBB = sid * state->numParticles * state->numComponents;
	int startXX = sid * state->numParticles * state->numComponents;

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
void calcVelocities(float dt, pilar::HairState* state)
{
	//Calculate the velocities of each particle

	//State pointers
	float* xx = state->xx;
	pilar::Vector3f* velocity = state->velocity1;
	pilar::Vector3f* velh = state->velh1;

	//Build matrix and vector of coefficients of linear equations
	buildAB(dt, state);

//	int startAA = blockIdx.x * state->numParticles * state->numComponents * state->numParticles * state->numComponents;
//	int endAA = startAA + state->numParticles * state->numComponents * state->numParticles * state->numComponents;
//
//	for(int i = startAA; i < endAA; i++)
//	{
//		printf("AA[%04d]:\t%.7f\n", i, state->AA[i]);
//	}

//	int startBB = blockIdx.x * state->numParticles * state->numComponents;
//	int endBB = startBB + state->numParticles * state->numComponents;
//
//	for (int i = startBB; i < endBB; ++i)
//	{
//		printf("bb:\t%d\t%.7f\n",i, state->bb[i]);
//	}
//	printf("\n\n");

	int startP = blockIdx.x * state->numParticles;
	int endP = startP + state->numParticles;

	//Set intial solution to previous velocity
	for(int i = startP; i < endP; i++)
	{
		xx[i*state->numComponents  ] = velocity[i].x;
		xx[i*state->numComponents+1] = velocity[i].y;
		xx[i*state->numComponents+2] = velocity[i].z;
	}

	//Solve for velocity using conjugate gradient method
	conjugate(state);

//	for(int i = startP; i < endP; i++)
//	{
//		int idx0 = i*state->numComponents;
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
		velh[i].x = xx[i*state->numComponents  ];
		velh[i].y = xx[i*state->numComponents+1];
		velh[i].z = xx[i*state->numComponents+2];
	}
}

__device__
void updateSprings(float dt, pilar::HairState* state)
{
	//State pointers
	pilar::Vector3f* root = state->root1;
	pilar::Vector3f* pos = state->pos1;
	pilar::Vector3f* velh = state->velh1;
	pilar::Vector3f* force = state->force1;

	int sid = blockIdx.x;

	int startP = sid * state->numParticles;
	int endP = startP + state->numParticles;

	//calculate the 6 coefficients
	float g_e = state->k_edge/state->length_e;
	float g_b = state->k_bend/state->length_b;
	float g_t = state->k_twist/state->length_t;
	float h_e = dt*state->k_edge/(2.0f*state->length_e) + state->d_edge;
	float h_b = dt*state->k_bend/(2.0f*state->length_b) + state->d_bend;
	float h_t = dt*state->k_twist/(2.0f*state->length_t) + state->d_twist;

	//Calculate and apply forces for the first three particles
	pilar::Vector3f uu0R(root[sid]	 -pos[startP]);
	pilar::Vector3f uu01(pos[startP+1]-pos[startP]);
	pilar::Vector3f uu02(pos[startP+2]-pos[startP]);
	pilar::Vector3f uu03(pos[startP+3]-pos[startP]);
	pilar::Vector3f du0R(uu0R);
	pilar::Vector3f du01(uu01);
	pilar::Vector3f du02(uu02);
	pilar::Vector3f du03(uu03);
	du0R.unitize();
	du01.unitize();
	du02.unitize();
	du03.unitize();
	pilar::Vector3f vu0R(			  -velh[startP]);
	pilar::Vector3f vu01(velh[startP+1]-velh[startP]);
	pilar::Vector3f vu02(velh[startP+2]-velh[startP]);
	pilar::Vector3f vu03(velh[startP+3]-velh[startP]);

	pilar::Vector3f force0 = du0R*((uu0R.dot(du0R)-state->length_e)*g_e) + du0R*((vu0R.dot(du0R))*h_e) +
							 du01*((uu01.dot(du01)-state->length_e)*g_e) + du01*((vu01.dot(du01))*h_e) +
							 du02*((uu02.dot(du02)-state->length_b)*g_b) + du02*((vu02.dot(du02))*h_b) +
							 du03*((uu03.dot(du03)-state->length_t)*g_t) + du03*((vu03.dot(du03))*h_t) ;

	force[startP] += force0;

	pilar::Vector3f uu1R(root[sid]	 -pos[startP+1]);
	pilar::Vector3f uu10(pos[startP  ]-pos[startP+1]);
	pilar::Vector3f uu12(pos[startP+2]-pos[startP+1]);
	pilar::Vector3f uu13(pos[startP+3]-pos[startP+1]);
	pilar::Vector3f uu14(pos[startP+4]-pos[startP+1]);
	pilar::Vector3f du1R(uu1R);
	pilar::Vector3f du10(uu10);
	pilar::Vector3f du12(uu12);
	pilar::Vector3f du13(uu13);
	pilar::Vector3f du14(uu14);
	du1R.unitize();
	du10.unitize();
	du12.unitize();
	du13.unitize();
	du14.unitize();
	pilar::Vector3f vu1R(			  -velh[startP+1]);
	pilar::Vector3f vu10(velh[startP  ]-velh[startP+1]);
	pilar::Vector3f vu12(velh[startP+2]-velh[startP+1]);
	pilar::Vector3f vu13(velh[startP+3]-velh[startP+1]);
	pilar::Vector3f vu14(velh[startP+4]-velh[startP+1]);

	pilar::Vector3f force1 = du1R*(uu1R.dot(du1R)-state->length_b)*g_b + du1R*(vu1R.dot(du1R))*h_b +
							 du10*(uu10.dot(du10)-state->length_e)*g_e + du10*(vu10.dot(du10))*h_e +
							 du12*(uu12.dot(du12)-state->length_e)*g_e + du12*(vu12.dot(du12))*h_e +
							 du13*(uu13.dot(du13)-state->length_b)*g_b + du13*(vu13.dot(du13))*h_b +
							 du14*(uu14.dot(du14)-state->length_t)*g_t + du14*(vu14.dot(du14))*h_t ;

	force[startP+1] += force1;

	pilar::Vector3f uu2R(root[sid]	 -pos[startP+2]);
	pilar::Vector3f uu20(pos[startP  ]-pos[startP+2]);
	pilar::Vector3f uu21(pos[startP+1]-pos[startP+2]);
	pilar::Vector3f uu23(pos[startP+3]-pos[startP+2]);
	pilar::Vector3f uu24(pos[startP+4]-pos[startP+2]);
	pilar::Vector3f uu25(pos[startP+5]-pos[startP+2]);
	pilar::Vector3f du2R(uu2R);
	pilar::Vector3f du20(uu20);
	pilar::Vector3f du21(uu21);
	pilar::Vector3f du23(uu23);
	pilar::Vector3f du24(uu24);
	pilar::Vector3f du25(uu25);
	du2R.unitize();
	du20.unitize();
	du21.unitize();
	du23.unitize();
	du24.unitize();
	du25.unitize();
	pilar::Vector3f vu2R(			  -velh[startP+2]);
	pilar::Vector3f vu20(velh[startP+0]-velh[startP+2]);
	pilar::Vector3f vu21(velh[startP+1]-velh[startP+2]);
	pilar::Vector3f vu23(velh[startP+3]-velh[startP+2]);
	pilar::Vector3f vu24(velh[startP+4]-velh[startP+2]);
	pilar::Vector3f vu25(velh[startP+5]-velh[startP+2]);

	pilar::Vector3f force2 = du2R*(uu2R.dot(du2R)-state->length_t)*g_t + du2R*(vu2R.dot(du2R))*h_t +
							 du20*(uu20.dot(du20)-state->length_b)*g_b + du20*(vu20.dot(du20))*h_b +
							 du21*(uu21.dot(du21)-state->length_e)*g_e + du21*(vu21.dot(du21))*h_e +
							 du23*(uu23.dot(du23)-state->length_e)*g_e + du23*(vu23.dot(du23))*h_e +
							 du24*(uu24.dot(du24)-state->length_b)*g_b + du24*(vu24.dot(du24))*h_b +
							 du25*(uu25.dot(du25)-state->length_t)*g_t + du25*(vu25.dot(du25))*h_t ;

	force[startP+2] += force2;

	//Calculate force for all particles between first and last
	for(int i = startP+3; i < (endP-3); i++)
	{
		pilar::Vector3f uu3(pos[i-3]-pos[i]);
		pilar::Vector3f uu2(pos[i-2]-pos[i]);
		pilar::Vector3f uu1(pos[i-1]-pos[i]);
		pilar::Vector3f ud1(pos[i+1]-pos[i]);
		pilar::Vector3f ud2(pos[i+2]-pos[i]);
		pilar::Vector3f ud3(pos[i+3]-pos[i]);
		pilar::Vector3f dui3(uu3);
		pilar::Vector3f dui2(uu2);
		pilar::Vector3f dui1(uu1);
		pilar::Vector3f ddi1(ud1);
		pilar::Vector3f ddi2(ud2);
		pilar::Vector3f ddi3(ud3);
		dui3.unitize();
		dui2.unitize();
		dui1.unitize();
		ddi1.unitize();
		ddi2.unitize();
		ddi3.unitize();
		pilar::Vector3f vu3(velh[i-3]-velh[i]);
		pilar::Vector3f vu2(velh[i-2]-velh[i]);
		pilar::Vector3f vu1(velh[i-1]-velh[i]);
		pilar::Vector3f vd1(velh[i+1]-velh[i]);
		pilar::Vector3f vd2(velh[i+2]-velh[i]);
		pilar::Vector3f vd3(velh[i+3]-velh[i]);

		pilar::Vector3f forcei = dui3*(uu3.dot(dui3)-state->length_t)*g_t + dui3*(vu3.dot(dui3))*h_t +
								 dui2*(uu2.dot(dui2)-state->length_b)*g_b + dui2*(vu2.dot(dui2))*h_b +
								 dui1*(uu1.dot(dui1)-state->length_e)*g_e + dui1*(vu1.dot(dui1))*h_e +
								 ddi1*(ud1.dot(ddi1)-state->length_e)*g_e + ddi1*(vd1.dot(ddi1))*h_e +
								 ddi2*(ud2.dot(ddi2)-state->length_b)*g_b + ddi2*(vd2.dot(ddi2))*h_b +
								 ddi3*(ud3.dot(ddi3)-state->length_t)*g_t + ddi3*(vd3.dot(ddi3))*h_t ;

		force[i] += forcei;
	}

	//Calculate and apply forces for last three particles
	pilar::Vector3f uu3N1(pos[endP-6]-pos[endP-3]);
	pilar::Vector3f uu3N2(pos[endP-5]-pos[endP-3]);
	pilar::Vector3f uu3N3(pos[endP-4]-pos[endP-3]);
	pilar::Vector3f uu3N5(pos[endP-2]-pos[endP-3]);
	pilar::Vector3f uu3N6(pos[endP-1]-pos[endP-3]);
	pilar::Vector3f du3N1(uu3N1);
	pilar::Vector3f du3N2(uu3N2);
	pilar::Vector3f du3N3(uu3N3);
	pilar::Vector3f du3N5(uu3N5);
	pilar::Vector3f du3N6(uu3N6);
	du3N1.unitize();
	du3N2.unitize();
	du3N3.unitize();
	du3N5.unitize();
	du3N6.unitize();
	pilar::Vector3f vu3N1(velh[endP-6]-velh[endP-3]);
	pilar::Vector3f vu3N2(velh[endP-5]-velh[endP-3]);
	pilar::Vector3f vu3N3(velh[endP-4]-velh[endP-3]);
	pilar::Vector3f vu3N5(velh[endP-2]-velh[endP-3]);
	pilar::Vector3f vu3N6(velh[endP-1]-velh[endP-3]);

	pilar::Vector3f force3N = du3N1*(uu3N1.dot(du3N1)-state->length_t)*g_t + du3N1*(vu3N1.dot(du3N1))*h_t +
							  du3N2*(uu3N2.dot(du3N2)-state->length_b)*g_b + du3N2*(vu3N2.dot(du3N2))*h_b +
							  du3N3*(uu3N3.dot(du3N3)-state->length_e)*g_e + du3N3*(vu3N3.dot(du3N3))*h_e +
							  du3N5*(uu3N5.dot(du3N5)-state->length_e)*g_e + du3N5*(vu3N5.dot(du3N5))*h_e +
							  du3N6*(uu3N6.dot(du3N6)-state->length_b)*g_b + du3N6*(vu3N6.dot(du3N6))*h_b ;

	force[endP-3] += force3N;

	pilar::Vector3f uu2N2(pos[endP-5]-pos[endP-2]);
	pilar::Vector3f uu2N3(pos[endP-4]-pos[endP-2]);
	pilar::Vector3f uu2N4(pos[endP-3]-pos[endP-2]);
	pilar::Vector3f uu2N6(pos[endP-1]-pos[endP-2]);
	pilar::Vector3f du2N2(uu2N2);
	pilar::Vector3f du2N3(uu2N3);
	pilar::Vector3f du2N4(uu2N4);
	pilar::Vector3f du2N6(uu2N6);
	du2N2.unitize();
	du2N3.unitize();
	du2N4.unitize();
	du2N6.unitize();
	pilar::Vector3f vu2N2(velh[endP-5]-velh[endP-2]);
	pilar::Vector3f vu2N3(velh[endP-4]-velh[endP-2]);
	pilar::Vector3f vu2N4(velh[endP-3]-velh[endP-2]);
	pilar::Vector3f vu2N6(velh[endP-1]-velh[endP-2]);

	pilar::Vector3f force2N = du2N2*(uu2N2.dot(du2N2)-state->length_t)*g_t + du2N2*(vu2N2.dot(du2N2))*h_t +
							  du2N3*(uu2N3.dot(du2N3)-state->length_t)*g_b + du2N3*(vu2N3.dot(du2N3))*h_b +
							  du2N4*(uu2N4.dot(du2N4)-state->length_t)*g_e + du2N4*(vu2N4.dot(du2N4))*h_e +
							  du2N6*(uu2N6.dot(du2N6)-state->length_t)*g_e + du2N6*(vu2N6.dot(du2N6))*h_e ;

	force[endP-2] += force2N;

	pilar::Vector3f uu1N3(pos[endP-4]-pos[endP-1]);
	pilar::Vector3f uu1N4(pos[endP-3]-pos[endP-1]);
	pilar::Vector3f uu1N5(pos[endP-2]-pos[endP-1]);
	pilar::Vector3f du1N3(uu1N3);
	pilar::Vector3f du1N4(uu1N4);
	pilar::Vector3f du1N5(uu1N5);
	du1N3.unitize();
	du1N4.unitize();
	du1N5.unitize();
	pilar::Vector3f vu1N3(velh[endP-4]-velh[endP-1]);
	pilar::Vector3f vu1N4(velh[endP-3]-velh[endP-1]);
	pilar::Vector3f vu1N5(velh[endP-2]-velh[endP-1]);

	pilar::Vector3f force1N = du1N3*(uu1N3.dot(du1N3)-state->length_t)*g_t + du1N3*(vu1N3.dot(du1N3))*h_t +
							  du1N4*(uu1N4.dot(du1N4)-state->length_b)*g_b + du1N4*(vu1N4.dot(du1N4))*h_b +
							  du1N5*(uu1N5.dot(du1N5)-state->length_e)*g_e + du1N5*(vu1N5.dot(du1N5))*h_e ;

	force[endP-1] += force1N;
}

__device__
void applyForce(pilar::Vector3f appliedForce, pilar::HairState* state)
{
	int startP = blockIdx.x * state->numParticles;
	int endP = startP + state->numParticles;

	for(int i = startP; i < endP; i++)
	{
		state->force1[i] = state->force1[i] + appliedForce;
	}
}

__device__
void updateVelocities(float dt, pilar::HairState* state)
{
	int startP = blockIdx.x * state->numParticles;
	int endP = startP + state->numParticles;

	for(int i = startP; i < endP; i++)
	{
		state->velh1[i] = state->velocity1[i] + (state->force1[i] * (dt / 2.0f));
	}
}

__device__
void updatePositions(float dt, pilar::HairState* state)
{
	//State pointers
	pilar::Vector3f* position = state->position1;
	pilar::Vector3f* posh = state->posh1;
	pilar::Vector3f* pos = state->pos1;
	pilar::Vector3f* velh = state->velh1;
	pilar::Vector3f* force = state->force1;

	int startP = blockIdx.x * state->numParticles;
	int endP = startP + state->numParticles;

	for(int i = startP; i < endP; i++)
	{
		//Save old position
		pilar::Vector3f poso = position[i];

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
void updateParticles(float dt, pilar::HairState* state)
{
	//State pointers
	pilar::Vector3f* position = state->position1;
	pilar::Vector3f* pos = state->pos1;
	pilar::Vector3f* velocity = state->velocity1;
	pilar::Vector3f* velh = state->velh1;
	pilar::Vector3f* force = state->force1;

	int startP = blockIdx.x * state->numParticles;
	int endP = startP + state->numParticles;

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
void applyStrainLimiting(float dt, pilar::HairState* state)
{
	pilar::Vector3f* root = state->root1;
	pilar::Vector3f* posc = state->posc1;
	pilar::Vector3f* pos = state->pos1;
	pilar::Vector3f* velh = state->velh1;

	int sid = blockIdx.x;
	int startP = blockIdx.x * state->numParticles;
	int endP = startP + state->numParticles;

	for(int i = startP; i < endP; i++)
	{
		//Calculate candidate position using half velocity
		posc[i] = pos[i] + (velh[i] * dt);

		//Determine the direction of the spring between the particles
		pilar::Vector3f dir = (i > startP) ? (posc[i] - posc[i-1]) : (posc[i] - root[sid]);

		if(dir.length_sqr() > MAX_LENGTH_SQUARED)
		{
			//Find a valid candidate position
			posc[i] = (i > startP) ? (posc[i-1] + (dir * (MAX_LENGTH*dir.length_inverse()))) : (root[sid] + (dir * (MAX_LENGTH*dir.length_inverse()))); //fast length calculation

			//~ particle[i]->posc = particle[i-1]->posc + (dir * (MAX_LENGTH/dir.length())); //slower length calculation

			//Calculate new half velocity based on valid candidate position, i.e. add a velocity impulse
			velh[i] = (posc[i] - pos[i])/dt;
		}
	}
}

__global__
void update(float dt, pilar::HairState* state)
{
	//TODO remove after unit testing
//	dt = 0.008f;
	
	//Calculate candidate velocities
	calcVelocities(dt, state);

//	int startP = blockIdx.x * state->numParticles;
//	int endP = startP + state->numParticles;
//
//	for(int i = startP; i < endP; i++)
//	{
//		printf("velh[%02d]:\t%.7f\t%.7f\t%.7f\n", i, state->velh1[i].x, state->velh1[i].y, state->velh1[i].z);
//	}

	//Calculate and apply spring forces using previous position
	updateSprings(dt, state);
	
//	int startP = blockIdx.x * state->numParticles;
//	int endP = startP + state->numParticles;
//
//	for(int i = startP; i < endP; i++)
//	{
//		printf("force[%02d]:\t%.7f\t%.7f\t%.7f\n", i, state->force[i].x, state->force[i].y, state->force[i].z);
//	}

	//Mass multiplied by gravity to get gravitational force
	pilar::Vector3f mgravity1 = state->gravity1 * state->mass;

	//Apply gravity
	applyForce(mgravity1, state);

//	int startP = blockIdx.x * state->numParticles;
//	int endP = startP + state->numParticles;
//
//	for(int i = startP; i < endP; i++)
//	{
//		printf("force[%02d]:\t%.7f\t%.7f\t%.7f\n", i, state->force1[i].x, state->force1[i].y, state->force1[i].z);
//	}

	//Calculate half velocities using forces
	updateVelocities(dt, state);
	
//	int startP = blockIdx.x * state->numParticles;
//	int endP = startP + state->numParticles;
//
//	for(int i = startP; i < endP; i++)
//	{
//		printf("velh[%02d]:\t%.7f\t%.7f\t%.7f\n", i, state->velh1[i].x, state->velh1[i].y, state->velh1[i].z);
//	}

	applyStrainLimiting(dt, state);

//	int startP = blockIdx.x * state->numParticles;
//	int endP = startP + state->numParticles;
//
//	for(int i = startP; i < endP; i++)
//	{
//		printf("velh[%02d]:\t%.7f\t%.7f\t%.7f\n", i, state->velh1[i].x, state->velh1[i].y, state->velh1[i].z);
//	}


	//Calculate half position and new position
	updatePositions(dt,  state);

//	int startP = blockIdx.x * state->numParticles;
//	int endP = startP + state->numParticles;

//	for(int i = startP; i < endP; i++)
//	{
//		printf("position[%02d]:\t%.7f\t%.7f\t%.7f\n", i, state->position1[i].x, state->position1[i].y, state->position1[i].z);
//	}

//	for(int i = startP; i < endP; i++)
//	{
//		printf("posh[%02d]:\t%.7f\t%.7f\t%.7f\n", i, state->posh1[i].x, state->posh1[i].y, state->posh1[i].z);
//	}
	
	//Calculate velocities using half position
	calcVelocities(dt, state);

//	int startP = blockIdx.x * state->numParticles;
//	int endP = startP + state->numParticles;
//
//	for(int i = startP; i < endP; i++)
//	{
//		printf("velh[%02d]:\t%.7f\t%.7f\t%.7f\n", i, state->velh1[i].x, state->velh1[i].y, state->velh1[i].z);
//	}

	//Calculate and apply spring forces using half position
	updateSprings(dt, state);

//	int startP = blockIdx.x * state->numParticles;
//	int endP = startP + state->numParticles;
//
//	for(int i = startP; i < endP; i++)
//	{
//		printf("force[%02d]:\t%.7f\t%.7f\t%.7f\n", i, state->force1[i].x, state->force1[i].y, state->force1[i].z);
//	}

	//Apply gravity
	applyForce(mgravity1, state);

//	int startP = blockIdx.x * state->numParticles;
//	int endP = startP + state->numParticles;
//
//	for(int i = startP; i < endP; i++)
//	{
//		printf("force[%02d]:\t%.7f\t%.7f\t%.7f\n", i, state->force1[i].x, state->force1[i].y, state->force1[i].z);
//	}

	//Calculate half velocity and new velocity
	updateParticles(dt, state);
	
//	int startP = blockIdx.x * state->numParticles;
//	int endP = startP + state->numParticles;
//	
//	for(int i  = startP; i < endP; i++)
//	{
//		printf("velh[%02d]:\t%.7f\t%.7f\t%.7f\n", i, state->velh1[i].x, state->velh1[i].y, state->velh1[i].z);
//		printf("velocity[%02d]:\t%.7f\t%.7f\t%.7f\n", i, state->velocity1[i].x, state->velocity1[i].y, state->velocity1[i].z);
//		printf("pos[%02d]:\t%.7f\t%.7f\t%.7f\n", i, state->pos1[i].x, state->pos1[i].y, state->pos1[i].z);
//	}
}

#endif
