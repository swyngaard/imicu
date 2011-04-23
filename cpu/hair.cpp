
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
		numTwist = numParticles - 3;
		
		this->numParticles = numParticles;
		
		//Initialise particles
		for(int i = 0; i < numParticles; i++)
		{
			particle[i] = new Particle(mass);
			
			//TODO set the intial particle positions
		}
	}
	
	//Build the three types of spring connections between particles
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
		for(int i = 0; i < numEdges; i++)
		{
			edge[i]->update(dt);
		}
		
		for(int i = 0; i < numBend; i++)
		{
			bend[i]->update(dt);
		}
		
		for(int i = 0; i < numTwist; i++)
		{
			twist[i]->update(dt);
		}
	}
	
	void Strand::updateParticles(float dt)
	{
		//TODO update velocity
		//TODO update position
		//TODO update velocity
	}
	
	void Strand::update(float dt)
	{
		resetParticles();
		updateSprings(dt);
		
		//TODO apply gravity force
		
		updateParticles(dt);
		
	}
	
	//Clean up
	void Strand::release()
	{
		//Edge springs
		for(int i = 0; i < numEdges; i++)
		{
			delete edge[i];
			edge[i] = NULL;
		}
		
		delete [] edge;
		
		//Bending springs
		for(int i = 0; i < numBend; i++)
		{
			delete bend[i];
			bend[i] = NULL;
		}
		
		delete [] bend;
		
		//Torsion springs
		for(int i = 0; i < numTwist; i++)
		{
			delete twist[i];
			twist[i] = NULL;
		}
		
		delete [] twist;
		
		//Particles
		for(int i = 0; i < numParticles; i++)
		{
			delete particle[i];
			particle[i] = NULL;
		}
		
		delete [] particle;
	}
	
	void Strand::applyForce(Vector3f force)
	{
		//TODO apply external forces like gravity here
		force = Vector3f();
	}
	
	void Strand::applyStrainLimiting(float dt)
	{
		//TODO
		dt++;
	}

/////////////////////////// Hair Class /////////////////////////////////////////
	
	Hair::Hair(int numStrands, float mass, float k, float length, std::vector<Vector3f> &roots)
	{
		strand = new Strand*[numStrands];
	
		for(int i = 0; i < numStrands; i++)
		{
			strand[i] = new Strand(numStrands, mass, k, length, roots[i]);
		}
	}
	
	void Hair::update(float dt)
	{
		for(int i = 0; i < numStrands; i++)
		{
			strand[i]->update(dt);
		}
	}
	
	//Clean up
	void Hair::release()
	{
		for(int i = 0; i < numStrands; i++)
		{
			strand[i]->release();
			delete strand[i];
			strand[i] = NULL;
		}
		
		delete [] strand;
	}
}

