
#include "hair.h"
#include "constants.h"

#include <iostream>
#include <vector_functions.h>
#include <stdlib.h>

extern "C"
void mallocStrands(const int &numStrands,
				   const int &numParticles,
				   const int &numComponents,
				   float3* &root,
				   float3* &normal,
				   float3* &position,
				   float3* &pos,
				   float3* &posc,
				   float3* &posh,
				   float3* &velocity,
				   float3* &velh,
				   float3* &force,
				   float* &AA,
				   float* &bb,
				   float* &xx);

extern "C"
void freeStrands(float3* &root,
				 float3* &normal,
				 float3* &position,
				 float3* &pos,
				 float3* &posc,
				 float3* &posh,
				 float3* &velocity,
				 float3* &velh,
				 float3* &force,
				 float* &AA,
				 float* &bb,
				 float* &xx);

extern "C"
void initPositions(int numStrands, int numParticles, const float3* root, float3* normal, float3* position, float3* posc, float3* pos);

extern "C"
void updateStrandsNew(int numParticles,
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
					  float3 &gravity,
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
					  float* xx);

extern "C" void copyRoots(int numStrands,
						  const float3* root3f,
						  const float3* normal3f,
						  float3* root,
						  float3* normal);

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
			   float length,
			   float length_e,
			   float length_b,
			   float length_t,
			   std::vector<Vector3f> &root,
			   std::vector<Vector3f> &normal)
	{
		this->numStrands = numStrands;
		this->numParticles = numParticles;
		this->numComponents = numComponents;
		
		this->gravity_ = make_float3(0.0f, GRAVITY, 0.0f);
		
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
		float3* root3f = (float3*) calloc(numStrands, sizeof(float3));
		float3* normal3f = (float3*) calloc(numStrands, sizeof(float3));
		
		for(int i = 0; i < numStrands; i++)
		{
			root3f[i].x = root[i].x;
			root3f[i].y = root[i].y;
			root3f[i].z = root[i].z;
			
			normal3f[i].x = normal[i].x;
			normal3f[i].y = normal[i].y;
			normal3f[i].z = normal[i].z;
		}
		
		//Allocate memory on GPU
		mallocStrands(numStrands, numParticles, numComponents, root_, normal_, position_, pos_, posc_, posh_, velocity_, velh_, force_, AA_, bb_, xx_);
		
		//Copy root positions and normals to GPU
		copyRoots(numStrands, root3f, normal3f, root_, normal_);
		
		//Intialise particle positions on the GPU
		initPositions(numStrands, numParticles, root_, normal_, position_, posc_, pos_);
		
		free(root3f);
		free(normal3f);
	}
	
	void Hair::update(float dt)
	{
		updateStrandsNew(numParticles,
						 numStrands,
						 numComponents,
						 dt,
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
						 gravity_,
						 root_,
						 position_,
						 posc_,
						 posh_,
						 pos_,
						 velocity_,
						 velh_,
						 force_,
						 AA_,
						 bb_,
						 xx_);
	}
	
	//Clean up
	void Hair::release()
	{
		freeStrands(root_, normal_, position_, pos_, posc_, posh_, velocity_, velh_, force_, AA_, bb_, xx_);
	}
}
