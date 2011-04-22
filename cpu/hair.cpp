
#include "hair.h"

namespace pilar
{

////////////////////////////// Particle Class //////////////////////////////////
	Particle::Particle(float mass)
	{
		this->mass = mass;
	}
	
	void Particle::clearForces()
	{
		force.x = 0.0f;
		force.y = 0.0f;
		force.z = 0.0f;
	}
	
	void Particle::applyForce(Vector3f force)
	{
		this->force += force;
	}
	
	void Particle::update(float dt)
	{
		dt++; //TODO
	}

////////////////////////////// Spring Class ////////////////////////////////////

	Spring::Spring(Particle* particle1, Particle* particle2, float k, float length, float damping, SpringType type)
	{
		this->k = k;
		this->length = length;
		this->damping = damping;
		
		this->particle[0] = particle1;
		this->particle[1] = particle2;
		
	}
	
	void Spring::update(float dt)
	{
		dt++; //TODO
	}
	
////////////////////////////// Strand Class ////////////////////////////////////

	Strand::Strand(int numParticles, float mass, float k, float length, Vector3f root=Vector3f())
	{
		numEdges = numParticles - 1;
		numBend = numParticles - 2;
		numTwist = numTwist - 3;
		
		//TODO
	}
	
	void Strand::buildSprings(float k, float length, float damping)
	{
		edge = new Spring*[numEdges];
		
		for(int i = 0; i < numEdges; i++)
		{
			edge[i] = new Spring(particle[i], particle[i+1], k, length, damping, EDGE);
		}
		
		bend = new Spring*[numBend];
		
		for(int i = 0; i < numBend; i++)
		{
			bend[i] = new Spring(particle[i], particle[i+2], k, length, damping, BEND);
		}
		
		twist = new Spring*[numTwist];
		
		for(int i = 0; i < numTwist; i++)
		{
			twist[i] = new Spring(particle[i], particle[i+3], k, length, damping, TWIST);
		}
	}
	
	void Strand::resetParticles()
	{
		for(int i = 0; i < numParticles; i++)
		{
			particle[i]->clearForces();
		}
	}
	
	void Strand::updateSprings(float dt)
	{
		//TODO
		dt++;
	}
	
	void Strand::updateParticles(float dt)
	{
		//TODO
		dt++;
	}
	
	void Strand::update(float dt)
	{
		//TODO
		dt++;
	}
	
	void Strand::release()
	{
		//TODO
	}
	
	void Strand::applyForce(Vector3f force)
	{
		//TODO
		force = Vector3f();
	}
	
	void Strand::applyStrainLimiting(float dt)
	{
		//TODO
		dt++;
	}

/////////////////////////// Hair Class /////////////////////////////////////////
	
	Hair::Hair(int numParticles, float mass, float k, float length, std::vector<Vector3f> &roots)
	{
		//TODO
	}
	
	void Hair::update(float dt)
	{
		//TODO
	}
	
	void Hair::release()
	{
		//TODO
	}
}

