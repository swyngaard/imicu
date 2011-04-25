
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
		Vector3f position;
		Vector3f velocity;
		Vector3f force;
		Vector3f vn;
		
		Particle(float mass);
		
		void clearForces();
		void applyForce(Vector3f force);	
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
	private:
		Particle** particle;
		float k;
		float length;
		float damping;
		SpringType type;
		
	public:
		Spring(Particle* particle1, Particle* particle2, float k, float length, float damping, SpringType type);
		void update(float dt);
	};
	
	class Strand
	{
	private:
		int numParticles;
		int numEdges;
		int numBend;
		int numTwist;
		
		
		Vector3f root;
		
		Particle** particle;
		Spring** edge;
		Spring** bend;
		Spring** twist;
		
		
		void buildSprings(float k, float length, float damping);
		void resetParticles();
		void updateSprings(float dt);
		void updateParticles(float dt);
		
	public:
		Strand(int numParticles, float mass, float k, float length, Vector3f root);
		void update(float dt);
		void release();
		void applyForce(Vector3f force);
		void applyStrainLimiting(float dt);
	};
	
	class Hair
	{
	private:
		Strand** strand;
		int numStrands;
		Vector3f gravity;
		
	public:
		Hair(int numStrands, float mass, float k, float length, std::vector<Vector3f> &roots);
		void update(float dt);
		void release();
	};
}

#endif

