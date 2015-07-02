
#ifndef __HAIR_H__
#define __HAIR_H__

#include "tools.h"
#include <vector>
#include <vector_types.h>

namespace pilar
{
	struct HairState
	{
		float* AA;
		float* bb;
		float* xx;

		Vector3f* root1;
		Vector3f* normal1;
		Vector3f* position1;
		Vector3f* pos1;
		Vector3f* posc1;
		Vector3f* posh1;
		Vector3f* velocity1;
		Vector3f* velh1;
		Vector3f* force1;

		Vector3f gravity1;

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
	};

	class Hair
	{
	protected:
		Vector3f gravity1;
		
		Vector3f* root1;
		Vector3f* normal1;
		Vector3f* pos1;
		Vector3f* posc1;
		Vector3f* posh1;
		Vector3f* velocity1;
		Vector3f* velh1;
		Vector3f* force1;

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
		
		float* AA_;
		float* bb_;
		float* xx_;

	public:
		int numStrands;
		int numParticles;
		int numComponents;
		
		Vector3f* position1;
		
		HairState* state;

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
			 std::vector<Vector3f> &root,
			 std::vector<Vector3f> &normal);
		
		void update(float dt);
		void init(std::vector<Vector3f> &root, std::vector<Vector3f> &normal);
		void release();
	};
}

#endif

