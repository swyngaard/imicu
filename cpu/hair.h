
#ifndef __HAIR_H__
#define __HAIR_H__

#include "tools.h"
#include "constants.h"
#include "ogl.h"
#include "tree.h"
#include <vector>

namespace pilar
{
	class Particle
	{
	public:
		float mass;
		bool freeze;
		
		Vector3f posh; //half
		Vector3f posc; //candidate
		Vector3f pos;  //current
		Vector3f poso; //old
		Vector3f position; //previous
		
		Vector3f velh;
		Vector3f velocity;
		Vector3f velc;
		
		Vector3f force;
		
		Particle(float mass);
		
		void clearForces();
		void applyForce(Vector3f force);
		void updateVelocity(float dt);
		void updatePosition(float dt);
		void update(float dt);
	};
	
	enum SpringType
	{
		EDGE,
		BEND,
		TWIST,
		EXTRA
	};
	
	class Spring
	{
	protected:
		Particle** particle;
		float k;
		float length;
		float damping;
		SpringType type;
		
		void updateForce(Vector3f p0, Vector3f p1, float dt);
		
	public:
		Spring(Particle* particle1, Particle* particle2, float k, float length, float damping, SpringType type);
		void update1(float dt);
		void update2(float dt);
		void release();
	};
	
	class Strand
	{
	
	protected:
		int numEdges;
		int numBend;
		int numTwist;
		Vector3f root;
		
		float k_edge;
		float k_bend;
		float k_twist;
		float k_extra;
		
		float d_edge;
		float d_bend;
		float d_twist;
		float d_extra;
		
		float length;
		float mass;
		
		float* xx;
		float* AA;
		float* bb;
		
		bool once;
		bool twice;
		bool thrice;
		bool fourth;
		
		Spring** edge;
		Spring** bend;
		Spring** twist;
		
		
		std::vector<KDOP*> leafKDOP;
		
		void buildSprings();
		void clearForces();
		void updateSprings(float dt);
		void updateVelocities(float dt);
		void updatePositions(float dt);
		void updateParticles(float dt);
		void calcVelocities(float dt);
		void objectCollisions(float dt, const float (&grid)[DOMAIN_DIM][DOMAIN_DIM][DOMAIN_DIM]);
		void applyStrainLimiting(float dt);
		void applyStiction(float dt);
		void updateBoundingVolumes();
		
		
		float getA(int i, int j, float dt);
		float getB(int i, float dt);
		void buildAB(float dt);
		void conjugate();
		
	public:
		int numParticles;
		Particle** particle;
		
		Spring* springr0;
		Spring* spring01;
		Spring* spring12;
		Node* bvhTree;
		
		Strand(int numParticles,
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
			   Vector3f root);
		void update(float dt, const float (&grid)[DOMAIN_DIM][DOMAIN_DIM][DOMAIN_DIM]);
		void release();
		void applyForce(Vector3f force);
		
	};
	
	class Hair
	{
	protected:
		Vector3f gravity;
		
		void initDistanceField(Model_OBJ &obj);
		
	public:
		int numStrands;
		Strand** strand;
		float grid[DOMAIN_DIM][DOMAIN_DIM][DOMAIN_DIM];
		
		Hair(int numStrands,
			 int numParticles,
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
			 std::vector<Vector3f> &roots);
		
		Hair(int numStrands,
			 int numParticles,
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
			 std::vector<Vector3f> &roots,
			 Model_OBJ &obj);
			 
		void update(float dt);
		void release();
	};
}

#endif

