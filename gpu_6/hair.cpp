
#include "hair.h"
#include "constants.h"

#include <iostream>
#include <vector_functions.h>
#include <stdlib.h>

extern "C"
void mallocStrands(const int &numStrands,
				   const int &numParticles,
				   const int &numComponents,
				   float* &AA,
				   float* &bb,
				   float* &xx,
				   pilar::Vector3f* &root1,
				   pilar::Vector3f* &normal1,
				   pilar::Vector3f* &position1,
				   pilar::Vector3f* &pos1,
				   pilar::Vector3f* &posc1,
				   pilar::Vector3f* &posh1,
				   pilar::Vector3f* &velocity1,
				   pilar::Vector3f* &velh1,
				   pilar::Vector3f* &force1,
				   pilar::HairState* &state);

extern "C"
void freeStrands(float* &AA,
				 float* &bb,
				 float* &xx,
				 pilar::Vector3f* &root1,
				 pilar::Vector3f* &normal1,
				 pilar::Vector3f* &position1,
				 pilar::Vector3f* &pos1,
				 pilar::Vector3f* &posc1,
				 pilar::Vector3f* &posh1,
				 pilar::Vector3f* &velocity1,
				 pilar::Vector3f* &velh1,
				 pilar::Vector3f* &force1,
				 pilar::HairState* &state);

extern "C"
void initPositions(int numParticles,
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
				   pilar::HairState* state);

extern "C"
void updateStrands(float dt, int numStrands, pilar::HairState* state);

extern "C" void copyRoots(int numStrands,
						  const pilar::Vector3f* root3f,
						  const pilar::Vector3f* normal3f,
						  pilar::Vector3f* root1,
						  pilar::Vector3f* normal1);

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
			   std::vector<Vector3f> &root,
			   std::vector<Vector3f> &normal)
	{
		this->numStrands = numStrands;
		this->numParticles = numParticles;
		this->numComponents = numComponents;
		
		this->gravity1 = Vector3f(0.0f, GRAVITY, 0.0f);

		this->mass = mass;
		this->k_edge = k_edge;
		this->k_bend = k_bend;
		this->k_twist = k_twist;
		this->k_extra = k_extra;
		this->d_edge = d_edge;
		this->d_bend = d_bend;
		this->d_twist = d_twist;
		this->d_extra = d_extra;
		this->length_e = length_e;
		this->length_b = length_b;
		this->length_t = length_t;
	}
	
	void Hair::init(std::vector<Vector3f> &root, std::vector<Vector3f> &normal)
	{
		pilar::Vector3f* root3f = (pilar::Vector3f*) calloc(numStrands, sizeof(pilar::Vector3f));
		pilar::Vector3f* normal3f = (pilar::Vector3f*) calloc(numStrands, sizeof(pilar::Vector3f));
		
		for(int i = 0; i < numStrands; i++)
		{
			root3f[i] = root[i];
			
			normal3f[i] = normal[i];
		}
		
		//Allocate memory on GPU
		mallocStrands(numStrands,
					  numParticles,
					  numComponents,
					  AA_,
					  bb_,
					  xx_,
					  root1,
					  normal1,
					  position1,
					  pos1,
					  posc1,
					  posh1,
					  velocity1,
					  velh1,
					  force1,
					  state);
		
		//Copy root positions and normals to GPU
		copyRoots(numStrands, root3f, normal3f, root1, normal1);
		
		//Intialise particle positions on the GPU
		initPositions(numParticles,
					  numStrands,
					  numComponents,
					  mass,
					  k_edge,
					  k_bend,
					  k_twist,
					  k_extra,
					  d_edge,
					  d_bend,
					  d_twist,
					  d_extra,
					  length_e,
					  length_b,
					  length_t,
					  AA_,
					  bb_,
					  xx_,
					  gravity1,
					  root1,
					  normal1,
					  position1,
					  pos1,
					  posc1,
					  posh1,
					  velocity1,
					  velh1,
					  force1,
					  state);
		
		free(root3f);
		free(normal3f);
	}
	
	void Hair::update(float dt)
	{
		updateStrands(dt, numStrands, state);
	}
	
	//Clean up
	void Hair::release()
	{
		freeStrands(AA_,
					bb_,
					xx_,
					root1,
					normal1,
					position1,
					pos1,
					posc1,
					posh1,
					velocity1,
					velh1,
					force1,
					state);
	}
}
