
#ifndef __HAIR_H__
#define __HAIR_H__

#include "tools.h"
#include <vector>

namespace pilar
{
	class Particle
	{
	public:
		float mass;
		
		Vector3f posh;
		Vector3f position;
		
		Vector3f velh;
		Vector3f velocity;
		
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
		
		Spring** edge;
		Spring** bend;
		Spring** twist;
		
		buildSprings(k_edge, k_bend, k_twist, k_extra, d_edge, d_bend, d_twist, d_extra, length);
		void clearForces();
		void updateSprings1(float dt);
		void updateSprings2(float dt);
		void updateParticles1(float dt);
		void updateParticles2(float dt);
		
	public:
		int numParticles;
		Particle** particle;
		
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
		void update(float dt);
		void release();
		void applyForce(Vector3f force);
		void applyStrainLimiting(float dt);
	};
	
	class Hair
	{
	protected:
		Vector3f gravity;
		
	public:
		int numStrands;
		Strand** strand;
		
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
		void update(float dt);
		void release();
	};
}

#endif

