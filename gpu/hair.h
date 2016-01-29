
#ifndef __HAIR_H__
#define __HAIR_H__

#include "tools.h"
#include <curand_kernel.h>

struct ModelOBJ
{
	float* vertices;
	float* normals;
	float* faces;	//Triangles
	
	long bytes;
	long totalConnectedPoints;
	long totalConnectedTriangles;
};

namespace pilar
{
	struct HairState
	{
		float* AA;
		float* bb;
		float* xx;

		Vector3f* root;
		Vector3f* normal;
		Vector3f* position;
		Vector3f* pos;
		Vector3f* posc;
		Vector3f* posh;
		Vector3f* velocity;
		Vector3f* velh;
		Vector3f* force;

		Vector3f gravity;

		int numStrands;
		int numParticles;
		int numComponents;

		float mass;
		float k_edge;
		float k_bend;
		float k_twist;
		float k_extra;
		float d_edge;
		float d_bend;
		float d_twist;
		float d_extra;
		float length_e;
		float length_b;
		float length_t;
		
		ModelOBJ* model;
		float* vertices;
		float* normals;
		float* faces;
		
		curandStatePhilox4_32_10_t* rng;
	};

	class Hair
	{
	protected:
		HairState* h_state;
		HairState* d_state;
		
	public:
		
		Hair(int numStrands,
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
			 ModelOBJ* model);
		~Hair();
		
		void initialise(Vector3f* position);
		void update(float dt, Vector3f* position);
	};
}

#endif

