
#ifndef __HAIR_H__
#define __HAIR_H__

#include "tools.h"
#include <vector>
#include <vector_types.h>

namespace pilar
{
	class Hair
	{
	protected:
		float3 *root_;
		float3 *normal_;
		float3 *pos_;
		float3 *posc_;
		float3 *posh_;
		float3 *velocity_;
		float3 *velh_;
		float3 *force_;
		

		float3 gravity_;
		
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
		
	public:
		int numStrands;
		int numParticles;
		int numComponents;
		
		float3 *position_;
		float *AA_;
		float *bb_;
		float *xx_;
		
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
			 float length,
			 float length_e,
			 float length_b,
			 float length_t,
			 std::vector<Vector3f> &root,
			 std::vector<Vector3f> &normal);
		
		void update(float dt);
		void init(std::vector<Vector3f> &root, std::vector<Vector3f> &normal);
		void release();
	};
}

#endif

