#include "hair.h"
#include "constants.h"

#include <Eigen/QR>

#include <iostream>

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
		velocity = velh * 2 - velocity;
	}
	
	void Particle::updateVelocity(float dt)
	{
		velh = velocity + force * (dt / 2.0f);
	}
	
	void Particle::updatePosition(float dt)
	{
		Vector3f newPosition = position + velh * dt;
		posh = (position + newPosition)/2.0f;
		position = newPosition;
	}

////////////////////////////// Spring Class ////////////////////////////////////

	Spring::Spring(Particle* particle1, Particle* particle2, float k, float length, float damping, SpringType type)
	{
		this->k = k;
		this->length = length;
		this->damping = damping;
		
		particle = new Particle*[2];
		
		this->particle[0] = particle1;
		this->particle[1] = particle2;
	}
	
	void Spring::update1(float dt)
	{
		//update spring forces using previous position 
		updateForce(particle[0]->position, particle[1]->position, dt);
	}
	
	void Spring::update2(float dt)
	{
		//Update spring forces using new calculated half positions
		updateForce(particle[0]->posh, particle[1]->posh, dt);
	}
	
	//Calculates the current velocities and applies the spring forces
	void Spring::updateForce(Vector3f p0, Vector3f p1, float dt)
	{
		Vector3f force;
		
		Vector3f xn = p1 - p0;
		Vector3f vn = particle[1]->velocity - particle[0]->velocity;
		Vector3f d = xn.unit();
		
		//Calculate velocity
		float h = dt * dt * k / (4.0f * particle[0]->mass * length);
		float g = dt * k / (2.0f * particle[0]->mass * length);
		float f = k / length * particle[0]->mass;

		Eigen::MatrixXf A(6,6);
		Eigen::VectorXf x(6);
		Eigen::VectorXf b(6);
		
		A <<  h*d.x*d.x+1.0f,  h*d.x*d.y,       h*d.x*d.z,      -h*d.x*d.x,      -h*d.x*d.y,      -h*d.x*d.z,
			  h*d.x*d.y,       h*d.y*d.y+1.0f,  h*d.y*d.z,      -h*d.x*d.y,      -h*d.y*d.y,      -h*d.y*d.z,
			  h*d.x*d.z,	   h*d.y*d.z,       h*d.z*d.z+1.0f, -h*d.x*d.z,      -h*d.y*d.z,      -h*d.z*d.z,
			 -h*d.x*d.x,      -h*d.x*d.y,      -h*d.x*d.z,       h*d.x*d.x+1.0f,  h*d.x*d.y,       h*d.x*d.z,
			 -h*d.x*d.y,   	  -h*d.y*d.y,      -h*d.y*d.z,       h*d.x*d.y,       h*d.y*d.y+1.0f,  h*d.y*d.z,
			 -h*d.x*d.z,	  -h*d.y*d.z,      -h*d.z*d.z,       h*d.x*d.z,       h*d.y*d.z,       h*d.z*d.z+1.0f;
		
		b << particle[0]->velocity.x + g*d.x*((p1.x*d.x + p1.y*d.y + p1.z*d.z) - (p0.x*d.x + p0.y*d.y + p0.z*d.z) - length),
			 particle[0]->velocity.y + g*d.y*((p1.x*d.x + p1.y*d.y + p1.z*d.z) - (p0.x*d.x + p0.y*d.y + p0.z*d.z) - length),
			 particle[0]->velocity.z + g*d.z*((p1.x*d.x + p1.y*d.y + p1.z*d.z) - (p0.x*d.x + p0.y*d.y + p0.z*d.z) - length),
			 particle[1]->velocity.x + g*d.x*((p0.x*d.x + p0.y*d.y + p0.z*d.z) - (p1.x*d.x + p1.y*d.y + p1.z*d.z) - length),
			 particle[1]->velocity.y + g*d.y*((p0.x*d.x + p0.y*d.y + p0.z*d.z) - (p1.x*d.x + p1.y*d.y + p1.z*d.z) - length),
			 particle[1]->velocity.z + g*d.z*((p0.x*d.x + p0.y*d.y + p0.z*d.z) - (p1.x*d.x + p1.y*d.y + p1.z*d.z) - length);
		
		x = A.colPivHouseholderQr().solve(b);
		
		Vector3f v1(x(3)-x(0),x(4)-x(1),x(5)-x(2));
		
		force += d * (f * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt*(v1.x*d.x + v1.y*d.y + v1.z*d.z)));

//		float damp = (dt * k / (length + damping))/particle[0]->mass;
//		Vector3f friction(damp, damp, damp);		
//		force += friction;
		
		particle[0]->applyForce(force*-1.0f);
		particle[1]->applyForce(force);
	}
	
	void Spring::release()
	{
		particle[0] = NULL;
		particle[1] = NULL;
		
		delete [] particle;
	}
	
////////////////////////////// Strand Class ////////////////////////////////////
	
	Strand::Strand(int numParticles,
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
				   Vector3f root=Vector3f())
	{
		numEdges = numParticles - 1;
		numBend  = numParticles - 2;
		numTwist = numParticles - 3;
		
		this->numParticles = numParticles;
		
		particle = new Particle*[numParticles];
		
		//Initialise particles
		for(int i = 0; i < numParticles; i++)
		{
			particle[i] = new Particle(mass);
			
			//TODO set the intial particle positions
			particle[i]->position = (root + Vector3f(length*i, 0.0f, 0.0f));
		}
		
		buildSprings(k_edge, k_bend, k_twist, k_extra, d_edge, d_bend, d_twist, d_extra, length);
	}
	
	//Build the three types of spring connections between particles
	void Strand::buildSprings(float k_edge,
							  float k_bend,
							  float k_twist,
							  float k_extra,
							  float d_edge,
							  float d_bend,
							  float d_twist,
							  float d_extra,
							  float length)
	{
		edge = new Spring*[numEdges];
		
		for(int i = 0; i < numEdges; i++)
		{
			edge[i] = new Spring(particle[i], particle[i+1], k_edge, length, d_edge, EDGE);
		}
		
		bend = new Spring*[numBend];
		
		for(int i = 0; i < numBend; i++)
		{
			bend[i] = new Spring(particle[i], particle[i+2], k_bend, length, d_bend, BEND);
		}
		
		twist = new Spring*[numTwist];
		
		for(int i = 0; i < numTwist; i++)
		{
			twist[i] = new Spring(particle[i], particle[i+3], k_twist, length, d_twist, TWIST);
		}
		
		
		//TODO add extra springs
	}
	
	void Strand::clearForces()
	{
		for(int i = 0; i < numParticles; i++)
		{
			particle[i]->clearForces();
		}
	}
	
	void Strand::updateSprings1(float dt)
	{
		for(int i = 0; i < numEdges; i++)
		{
			edge[i]->update1(dt);
		}
		
		
		for(int i = 0; i < numBend; i++)
		{
			bend[i]->update1(dt);
		}
		
		for(int i = 0; i < numTwist; i++)
		{
			twist[i]->update1(dt);
		}
		
	}
	
	void Strand::updateSprings2(float dt)
	{
		for(int i = 0; i < numEdges; i++)
		{
			edge[i]->update2(dt);
		}
		
		for(int i = 0; i < numBend; i++)
		{
			bend[i]->update2(dt);
		}
		
		for(int i = 0; i < numTwist; i++)
		{
			twist[i]->update2(dt);
		}
		
	}
	
	void Strand::updateParticles1(float dt)
	{
		for(int i = 1; i < numParticles; i++)
		{
			particle[i]->updateVelocity(dt);
			particle[i]->updatePosition(dt);
		}
	}
	
	void Strand::updateParticles2(float dt)
	{
		for(int i = 1; i < numParticles; i++)
		{
			particle[i]->updateVelocity(dt);
			particle[i]->update(dt);
		}
	}
	
	void Strand::applyForce(Vector3f force)
	{
		//Apply external forces like gravity here
		for(int i = 0; i < numParticles; i++)
		{
			particle[i]->applyForce(force);
		}
	}
	
	void Strand::applyStrainLimiting(float dt)
	{
		//TODO
		dt++;
	}
	
	void Strand::update(float dt)
	{
		//Reset forces on particles
		clearForces();
		
		//Calculate and apply spring forces using previous position
		updateSprings1(dt);
		
		//Apply gravity
		applyForce(Vector3f(0.0f, GRAVITY, 0.0f));
		
		//Calculate half velocity, half position and new position
		updateParticles1(dt);
		
		//Reset forces on particles
		clearForces();
		
		//Calculate and apply spring forces using half position
		updateSprings2(dt);
		
		//Apply gravity
		applyForce(Vector3f(0.0f, GRAVITY, 0.0f));
		
		//Calculate half velocity and new velocity
		updateParticles2(dt);
		
//		applyStrainLimiting(dt);		
	}
	
	//Clean up
	void Strand::release()
	{
		//Edge springs
		for(int i = 0; i < numEdges; i++)
		{
			edge[i]->release();
			delete edge[i];
			edge[i] = NULL;
		}
		delete [] edge;
		
		
		//Bending springs
		for(int i = 0; i < numBend; i++)
		{
			bend[i]->release();
			delete bend[i];
			bend[i] = NULL;
		}
		delete [] bend;
		
		//Torsion springs
		for(int i = 0; i < numTwist; i++)
		{
			twist[i]->release();
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
	
/////////////////////////// Hair Class /////////////////////////////////////////
	
	Hair::Hair(int numStrands,
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
			   std::vector<Vector3f> &roots)
	{
		this->numStrands = numStrands;
		
		strand = new Strand*[numStrands];
		
		for(int i = 0; i < numStrands; i++)
		{
			strand[i] = new Strand(numParticles, mass, k_edge, k_bend, k_twist, k_extra, d_edge, d_bend, d_twist, d_extra, length, roots[i]);
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
