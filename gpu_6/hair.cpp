
#include "hair.h"

extern "C"
void mallocStrands(pilar::HairState* h_state, pilar::HairState* &d_state, int modelBytes);

extern "C"
void freeStrands(pilar::HairState* h_state, pilar::HairState* d_state);

extern "C"
void initialisePositions(pilar::HairState* h_state, pilar::HairState* d_state);

extern "C"
void updateStrands(float dt, pilar::HairState* h_state, pilar::HairState* d_state);

extern "C"
void copyRoots(pilar::Vector3f* roots, pilar::Vector3f* normals, pilar::HairState* h_state);

extern "C"
void copyState(pilar::HairState* h_state, pilar::HairState* d_state);

extern "C"
void copyModel(ModelOBJ* model, pilar::HairState* h_state);

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
			   Vector3f gravity,
			   Vector3f* roots,
			   Vector3f* normals,
			   ModelOBJ* model)
	{
		h_state = new HairState;
		
		h_state->numStrands = numStrands;
		h_state->numParticles = numParticles;
		h_state->numComponents = numComponents;
		h_state->gravity = gravity;
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
		mallocStrands(h_state, d_state, model->bytes);

		//Copy root positions and normal directions to GPU
		copyRoots(roots, normals, h_state);
		
		//Copy object model data to GPU
		copyModel(model, h_state);
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
		initialisePositions(h_state, d_state);
	}
	
	void Hair::update(float dt, Vector3f* position)
	{
		h_state->position = position;
		
		updateStrands(dt, h_state, d_state);
	}
}
