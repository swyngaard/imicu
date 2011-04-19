
#ifndef __HAIR_H__
#define __HAIR_H__

#include "tools.h"

namespace pilar
{
	class Particle
	{
	private:
		float mass;
		Vector3f position;
		Vector3f velocity;
		Vector3f force;
		
	public:
		Particle(float mass);
		
		void clearForces();
		void applyForce(Vector3f force);	
		void update(float dt);
	};
}

#endif

