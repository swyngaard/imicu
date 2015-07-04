
#include "hair.h"
#include "constants.h"

#include <iostream>
#include <vector_functions.h>
#include <stdlib.h>

extern "C"
void mallocStrands(pilar::HairState* h_state, pilar::HairState* &d_state);

extern "C"
void freeStrands(pilar::HairState* h_state, pilar::HairState* d_state);

extern "C"
void initPositions(pilar::HairState* h_state, pilar::HairState* d_state);

extern "C"
void updateStrands(float dt, pilar::HairState* h_state, pilar::HairState* d_state);

extern "C"
void copyRoots(const pilar::Vector3f* root3f, const pilar::Vector3f* normal3f, pilar::HairState* h_state);

extern "C"
void copyState(pilar::HairState* h_state, pilar::HairState* d_state);

namespace pilar
{
	Hair::Hair(int numStrands,
			   int numParticles,
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
			   std::vector<Vector3f> &roots,
			   std::vector<Vector3f> &normals)
	{
		h_state = new HairState;
		
		h_state->numStrands = numStrands;
		h_state->numParticles = numParticles;
		h_state->numComponents = numComponents;
		h_state->gravity = Vector3f(0.0f, GRAVITY, 0.0f);
		h_state->mass = mass;
		h_state->k_edge = k_edge;
		h_state->k_bend = k_bend;
		h_state->k_twist = k_twist;
		h_state->k_extra = k_extra;
		h_state->d_edge = d_edge;
		h_state->d_bend = d_bend;
		h_state->d_twist = d_twist;
		h_state->d_extra = d_extra;
		h_state->length_e = length_e;
		h_state->length_b = length_b;
		h_state->length_t = length_t;
		
		d_state = 0;
		
		//Allocate memory on GPU
		mallocStrands(h_state, d_state);
		
		pilar::Vector3f* root3f = (pilar::Vector3f*) calloc(numStrands, sizeof(pilar::Vector3f));
		pilar::Vector3f* normal3f = (pilar::Vector3f*) calloc(numStrands, sizeof(pilar::Vector3f));

		for(int i = 0; i < numStrands; i++)
		{
			root3f[i] = roots[i];
			normal3f[i] = normals[i];
		}

		//Copy root positions and normal directions to GPU
		copyRoots(root3f, normal3f, h_state);
		
		free(root3f);
		free(normal3f);
	}
	
	Hair::~Hair()
	{
		freeStrands(h_state, d_state);

		delete h_state;
	}
	
	void Hair::initialise(Vector3f* position)
	{
		h_state->position = position;
		
		//Copy intialised state to the GPU
		copyState(h_state, d_state);
		
		//Intialise particle positions on the GPU
		initPositions(h_state, d_state);
	}
	
	void Hair::update(float dt, Vector3f* position)
	{
		h_state->position = position;
		
		updateStrands(dt, h_state, d_state);
	}
}
