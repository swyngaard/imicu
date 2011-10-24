#include "hair.h"
#include "constants.h"

#include <Eigen/QR>
#include <Eigen/Dense>

#include <iostream>
#include <iomanip>

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
		
		this->A = new float[36];
		this->x = new float[6];
		this->b = new float[6];
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
	
	void Spring::conjugate(const float* A, const float* b, float* x)
	{
		
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
		
		Vector3f v1(particle[1]->velc.x-particle[0]->velc.x, particle[1]->velc.y-particle[0]->velc.y, particle[1]->velc.z-particle[0]->velc.z);
		
		force += d * (f * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt*(v1.x*d.x + v1.y*d.y + v1.z*d.z)));

//		float damp = (dt * k / (length + damping))/particle[0]->mass;
//		Vector3f friction(damp, damp, damp);		
//		force += friction;
		
		particle[0]->applyForce(force*-1.0f);
		particle[1]->applyForce(force);
//		particle[0]->applyForce(force);
//		particle[1]->applyForce(force*-1.0f);
	}
	
	void Spring::release()
	{
		particle[0] = NULL;
		particle[1] = NULL;
		
		delete [] particle;
		delete [] A;
		delete [] x;
		delete [] b;
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
		
		this->k_edge = k_edge;
		this->k_bend = k_bend;
		this->k_twist = k_twist;
		this->k_extra = k_extra;
		
		this->d_edge = d_edge;
		this->d_bend = d_bend;
		this->d_twist = d_twist;
		this->d_extra = d_extra;
		
		this->length = length;
		
		particle = new Particle*[numParticles];
		
		//Initialise particles
		for(int i = 0; i < numParticles; i++)
		{
			particle[i] = new Particle(mass);
			
			//TODO set the intial particle positions
			particle[i]->position = (root + Vector3f(length/2.0f*i, 0.0f, 0.0f));
			particle[i]->posc = particle[i]->position;
		}
		
		buildSprings();
	}
	
	//Build the three types of spring connections between particles
	void Strand::buildSprings()
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
	
	void Strand::updateVelocities(float dt)
	{
		for(int i = 1; i < numParticles; i++)
		{
			particle[i]->updateVelocity(dt);
		}
	}
	
	void Strand::updateParticles1(float dt)
	{
		for(int i = 1; i < numParticles; i++)
		{
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
	
	void Strand::calcVelocities(float dt)
	{
		int N = numParticles * 3;
		
		Eigen::MatrixXf AA(N,N);
		Eigen::VectorXf xx(N);
		Eigen::VectorXf bb(N);
		
		//Add edge springs data into A and b
		float h = dt*dt*k_edge/(4.0f*MASS*LENGTH);
		float g = dt*k_edge/(2.0f*MASS*LENGTH);
	
		for(int i = 0; i < numEdges; i++)
		{
			Vector3f d;
			Vector3f e;
		
			d.x = particle[i+1]->position.x - particle[i]->position.x;
			d.y = particle[i+1]->position.y - particle[i]->position.y;
			d.z = particle[i+1]->position.z - particle[i]->position.z;
			e.x = d.x;
			e.y = d.y;
			e.z = d.z;
		
			float length = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
		
			d.x /= length;
			d.y /= length;
			d.z /= length;
		
			AA(i*3,  i*3) += 1+h*d.x*d.x; AA(i*3,  i*3+1) +=   h*d.x*d.y; AA(i*3,  i*3+2) +=   h*d.x*d.z; AA(i*3,  i*3+3) +=  -h*d.x*d.x; AA(i*3,  i*3+4) +=  -h*d.x*d.y; AA(i*3,  i*3+5) +=  -h*d.x*d.z;
			AA(i*3+1,i*3) +=   h*d.x*d.y; AA(i*3+1,i*3+1) += 1+h*d.y*d.y; AA(i*3+1,i*3+2) +=   h*d.y*d.z; AA(i*3+1,i*3+3) +=  -h*d.x*d.y; AA(i*3+1,i*3+4) +=  -h*d.y*d.y; AA(i*3+1,i*3+5) +=  -h*d.y*d.z;
			AA(i*3+2,i*3) +=   h*d.x*d.z; AA(i*3+2,i*3+1) +=   h*d.y*d.z; AA(i*3+2,i*3+2) += 1+h*d.z*d.z; AA(i*3+2,i*3+3) +=  -h*d.x*d.z; AA(i*3+2,i*3+4) +=  -h*d.y*d.z; AA(i*3+2,i*3+5) +=  -h*d.z*d.z;
			AA(i*3+3,i*3) +=  -h*d.x*d.x; AA(i*3+3,i*3+1) +=  -h*d.x*d.y; AA(i*3+3,i*3+2) +=  -h*d.x*d.z; AA(i*3+3,i*3+3) += 1+h*d.x*d.x; AA(i*3+3,i*3+4) +=   h*d.x*d.y; AA(i*3+3,i*3+5) +=   h*d.x*d.z;
			AA(i*3+4,i*3) +=  -h*d.x*d.y; AA(i*3+4,i*3+1) +=  -h*d.y*d.y; AA(i*3+4,i*3+2) +=  -h*d.y*d.z; AA(i*3+4,i*3+3) +=   h*d.x*d.y; AA(i*3+4,i*3+4) += 1+h*d.y*d.y; AA(i*3+4,i*3+5) +=   h*d.y*d.z;
			AA(i*3+5,i*3) +=  -h*d.x*d.z; AA(i*3+5,i*3+1) +=  -h*d.y*d.z; AA(i*3+5,i*3+2) +=  -h*d.z*d.z; AA(i*3+5,i*3+3) +=   h*d.x*d.z; AA(i*3+5,i*3+4) +=   h*d.y*d.z; AA(i*3+5,i*3+5) += 1+h*d.z*d.z;		
		
			float factor = g * ((e.x*d.x+e.y*d.y+e.z*d.z) - (LENGTH));
		
			bb(i*3)   += particle[i]->velocity.x + factor * d.x;
			bb(i*3+1) += particle[i]->velocity.y + factor * d.y;
			bb(i*3+2) += particle[i]->velocity.z + factor * d.z;
			bb(i*3+3) += particle[i+1]->velocity.x - factor * d.x;
			bb(i*3+4) += particle[i+1]->velocity.y - factor * d.y;
			bb(i*3+5) += particle[i+1]->velocity.z - factor * d.z;
		}
		
		//Add bending springs data into A and b
		h = dt*dt*K_BEND/(4*MASS*LENGTH);
		g = dt*K_BEND/(2*MASS*LENGTH);
		
		for(int i = 0; i < numBend; i++)
		{
			Vector3f d;
			
			d.x = particle[i+2]->position.x - particle[i]->position.x;
			d.y = particle[i+2]->position.y - particle[i]->position.y;
			d.z = particle[i+2]->position.z - particle[i]->position.z;
		
			float length = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
		
			d.x /= length;
			d.y /= length;
			d.z /= length;
		
			AA(i*3,  i*3) += 1+h*d.x*d.x; AA(i*3,  i*3+1) +=   h*d.x*d.y; AA(i*3,  i*3+2) +=   h*d.x*d.z; AA(i*3,  i*3+6) +=  -h*d.x*d.x; AA(i*3,  i*3+7) +=  -h*d.x*d.y; AA(i*3,  i*3+8) +=  -h*d.x*d.z;
			AA(i*3+1,i*3) +=   h*d.x*d.y; AA(i*3+1,i*3+1) += 1+h*d.y*d.y; AA(i*3+1,i*3+2) +=   h*d.y*d.z; AA(i*3+1,i*3+6) +=  -h*d.x*d.y; AA(i*3+1,i*3+7) +=  -h*d.y*d.y; AA(i*3+1,i*3+8) +=  -h*d.y*d.z;
			AA(i*3+2,i*3) +=   h*d.x*d.z; AA(i*3+2,i*3+1) +=   h*d.y*d.z; AA(i*3+2,i*3+2) += 1+h*d.z*d.z; AA(i*3+2,i*3+6) +=  -h*d.x*d.z; AA(i*3+2,i*3+7) +=  -h*d.y*d.z; AA(i*3+2,i*3+8) +=  -h*d.z*d.z;
			AA(i*3+6,i*3) +=  -h*d.x*d.x; AA(i*3+6,i*3+1) +=  -h*d.x*d.y; AA(i*3+6,i*3+2) +=  -h*d.x*d.z; AA(i*3+6,i*3+6) += 1+h*d.x*d.x; AA(i*3+6,i*3+7) +=   h*d.x*d.y; AA(i*3+6,i*3+8) +=   h*d.x*d.z;
			AA(i*3+7,i*3) +=  -h*d.x*d.y; AA(i*3+7,i*3+1) +=  -h*d.y*d.y; AA(i*3+7,i*3+2) +=  -h*d.y*d.z; AA(i*3+7,i*3+6) +=   h*d.x*d.y; AA(i*3+7,i*3+7) += 1+h*d.y*d.y; AA(i*3+7,i*3+8) +=   h*d.y*d.z;
			AA(i*3+8,i*3) +=  -h*d.x*d.z; AA(i*3+8,i*3+1) +=  -h*d.y*d.z; AA(i*3+8,i*3+2) +=  -h*d.z*d.z; AA(i*3+8,i*3+6) +=   h*d.x*d.z; AA(i*3+8,i*3+7) +=   h*d.y*d.z; AA(i*3+8,i*3+8) += 1+h*d.z*d.z;
		
			Vector3f e;
		
			e.x = particle[i+2]->position.x - particle[i]->position.x;
			e.y = particle[i+2]->position.y - particle[i]->position.y;
			e.z = particle[i+2]->position.z - particle[i]->position.z;
		
			bb(i*3)   += particle[i]->velocity.x + g * ((e.x*d.x+e.y*d.y+e.z*d.z) - LENGTH) * d.x;
			bb(i*3+1) += particle[i]->velocity.y + g * ((e.x*d.x+e.y*d.y+e.z*d.z) - LENGTH) * d.y;
			bb(i*3+2) += particle[i]->velocity.z + g * ((e.x*d.x+e.y*d.y+e.z*d.z) - LENGTH) * d.z;
			bb(i*3+6) += particle[i+2]->velocity.x + g * (( (-e.x)*(-d.x)+(-e.y)*(-d.y)+(-e.z)*(-d.z) ) - LENGTH) * (-d.x);
			bb(i*3+7) += particle[i+2]->velocity.y + g * (( (-e.x)*(-d.x)+(-e.y)*(-d.y)+(-e.z)*(-d.z) ) - LENGTH) * (-d.y);
			bb(i*3+8) += particle[i+2]->velocity.z + g * (( (-e.x)*(-d.x)+(-e.y)*(-d.y)+(-e.z)*(-d.z) ) - LENGTH) * (-d.z);
		}
	
		//Add twisting springs data into A and b
		h = dt*dt*K_TWIST/(4*MASS*LENGTH);
		g = dt*K_TWIST/(2*MASS*LENGTH);
	
		for(int i = 0; i < numTwist; i++)
		{
			Vector3f d;
		
			d.x = particle[i+3]->position.x - particle[i]->position.x;
			d.y = particle[i+3]->position.y - particle[i]->position.y;
			d.z = particle[i+3]->position.z - particle[i]->position.z;
		
			float length = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
		
			d.x /= length;
			d.y /= length;
			d.z /= length;
		
			AA(i*3,  i*3)  += 1+h*d.x*d.x;  AA(i*3,  i*3+1) +=   h*d.x*d.y; AA(i*3,  i*3+2)  +=   h*d.x*d.z; AA(i*3,  i*3+9)  +=  -h*d.x*d.x; AA(i*3,  i*3+10)  +=  -h*d.x*d.y; AA(i*3,  i*3+11)  +=  -h*d.x*d.z;
			AA(i*3+1,i*3)  +=   h*d.x*d.y;  AA(i*3+1,i*3+1) += 1+h*d.y*d.y; AA(i*3+1,i*3+2)  +=   h*d.y*d.z; AA(i*3+1,i*3+9)  +=  -h*d.x*d.y; AA(i*3+1,i*3+10)  +=  -h*d.y*d.y; AA(i*3+1,i*3+11)  +=  -h*d.y*d.z;
			AA(i*3+2,i*3)  +=   h*d.x*d.z;  AA(i*3+2,i*3+1) +=   h*d.y*d.z; AA(i*3+2,i*3+2)  += 1+h*d.z*d.z; AA(i*3+2,i*3+9)  +=  -h*d.x*d.z; AA(i*3+2,i*3+10)  +=  -h*d.y*d.z; AA(i*3+2,i*3+11)  +=  -h*d.z*d.z;
			AA(i*3+9,i*3)  +=  -h*d.x*d.x;  AA(i*3+9,i*3+1) +=  -h*d.x*d.y; AA(i*3+9,i*3+2)  +=  -h*d.x*d.z; AA(i*3+9,i*3+9)  += 1+h*d.x*d.x; AA(i*3+9,i*3+10)  +=   h*d.x*d.y; AA(i*3+9,i*3+11)  +=   h*d.x*d.z;
			AA(i*3+10,i*3) +=  -h*d.x*d.y; AA(i*3+10,i*3+1) +=  -h*d.y*d.y; AA(i*3+10,i*3+2) +=  -h*d.y*d.z; AA(i*3+10,i*3+9) +=   h*d.x*d.y; AA(i*3+10,i*3+10) += 1+h*d.y*d.y; AA(i*3+10,i*3+11) +=   h*d.y*d.z;
			AA(i*3+11,i*3) +=  -h*d.x*d.z; AA(i*3+11,i*3+1) +=  -h*d.y*d.z; AA(i*3+11,i*3+2) +=  -h*d.z*d.z; AA(i*3+11,i*3+9) +=   h*d.x*d.z; AA(i*3+11,i*3+10) +=   h*d.y*d.z; AA(i*3+11,i*3+11) += 1+h*d.z*d.z;
		
			Vector3f e;
		
			e.x = particle[i+3]->position.x - particle[i]->position.x;
			e.y = particle[i+3]->position.y - particle[i]->position.y;
			e.z = particle[i+3]->position.z - particle[i]->position.z;
		
			bb(i*3)    += particle[i]->velocity.x + g * ((e.x*d.x+e.y*d.y+e.z*d.z) - LENGTH) * d.x;
			bb(i*3+1)  += particle[i]->velocity.y + g * ((e.x*d.x+e.y*d.y+e.z*d.z) - LENGTH) * d.y;
			bb(i*3+2)  += particle[i]->velocity.z + g * ((e.x*d.x+e.y*d.y+e.z*d.z) - LENGTH) * d.z;
			bb(i*3+9)  += particle[i+3]->velocity.x + g * (( (-e.x)*(-d.x)+(-e.y)*(-d.y)+(-e.z)*(-d.z) ) - LENGTH) * (-d.x);
			bb(i*3+10) += particle[i+3]->velocity.y + g * (( (-e.x)*(-d.x)+(-e.y)*(-d.y)+(-e.z)*(-d.z) ) - LENGTH) * (-d.y);
			bb(i*3+11) += particle[i+3]->velocity.z + g * (( (-e.x)*(-d.x)+(-e.y)*(-d.y)+(-e.z)*(-d.z) ) - LENGTH) * (-d.z);
		}
		
		std::cout << AA << std::endl << std::endl;
		
		//	cout << "AA:" <<endl;
		//	std::cout << AA << std::endl;
	
		//	cout << "bb:" <<endl;
		//	std::cout << bb << std::endl;
	
		Eigen::FullPivLU<Eigen::MatrixXf> fplu(AA);
		
		xx = fplu.solve(bb);
		
		for(int i = 0; i < numParticles; i++)
		{
			particle[i]->velc.x = xx(i*3);
			particle[i]->velc.y = xx(i*3+1);
			particle[i]->velc.z = xx(i*3+2);
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
		bool strained = true;
		
		//TODO faster iterative strain limiting!!!
		while(strained)
		{
			strained = false;
			
			for(int i = 1; i < numParticles; i++)
			{
				//Calculate candidate position using half velocity
				particle[i]->posc = particle[i]->position + particle[i]->velh * dt;
			
				//Determine the direction of the spring between the particles
				Vector3f dir = particle[i]->posc - particle[i-1]->posc;
			
				if(dir.length_sqr() > MAX_LENGTH_SQUARED)
				{
					strained = true;
					
					//Find a valid candidate position
					particle[i]->posc = particle[i-1]->posc + (dir * (MAX_LENGTH*dir.length_inverse())); //fast length calculation
//					particle[i]->posc = particle[i-1]->posc + (dir * (MAX_LENGTH/dir.length())); //slower length calculation

					//Calculate new half velocity based on valid candidate position, i.e. add a velocity impulse
					particle[i]->velh = (particle[i]->posc - particle[i]->position)/dt;
				}
			}
		}
	}
	
	void Strand::update(float dt)
	{
		//Reset forces on particles
		clearForces();
		
		calcVelocities(dt);
		
		//Calculate and apply spring forces using previous position
		updateSprings1(dt);
		
		//Apply gravity
//		applyForce(Vector3f(0.0f, GRAVITY, 0.0f));
		
		updateVelocities(dt);		
		
		applyStrainLimiting(dt);
		
		//Calculate half velocity, half position and new position
		updateParticles1(dt);
		
		//Reset forces on particles
		clearForces();
		
		calcVelocities(dt);
		
		//Calculate and apply spring forces using half position
		updateSprings2(dt);
		
		//Apply gravity
//		applyForce(Vector3f(0.0f, GRAVITY, 0.0f));
		
		//Calculate half velocity and new velocity
		updateParticles2(dt);
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
