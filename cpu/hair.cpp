#include "hair.h"
#include "constants.h"

#include <iostream>
#include <iomanip>
//#include <algorithm>
#include <cstring>
#include <cfloat>
#include <cmath>

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
	
//	void Spring::conjugate(const float* A, const float* b, float* x)
//	{
		
//	}
	
	//Calculates the current velocities and applies the spring forces
	void Spring::updateForce(Vector3f p0, Vector3f p1, float dt)
	{
		Vector3f force;
		
		Vector3f xn = p1 - p0;
		Vector3f vn = particle[1]->velocity - particle[0]->velocity;
		Vector3f d = xn * xn.length_inverse();
		
		//Calculate velocity
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
		this->mass  = mass;
		
//		std::cout << mass << " " << length << " " << k_edge << std::endl;
		
		particle = new Particle*[numParticles];
		
		//Initialise particles
		for(int i = 0; i < numParticles; i++)
		{
			particle[i] = new Particle(mass);
			
			//TODO set the intial particle positions
			particle[i]->position = (root + Vector3f(length/2.0f*float(i), 0.0f, 0.0f));
			particle[i]->posc = particle[i]->position;

//			std::cout << particle[i]->position.x << " " << particle[i]->position.y << " " << particle[i]->position.z << std::endl;			
//			std::cout << particle[i]->velh.x << " " << particle[i]->velh.y << " " << particle[i]->velh.z << std::endl;
//			std::cout << particle[i]->velc.x << " " << particle[i]->velc.y << " " << particle[i]->velc.z << std::endl;
		}
		
		this->A = new float[numParticles*3*numParticles*3];
		this->x = new float[numParticles*3];
		this->b = new float[numParticles*3];
		
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
	
	void Strand::conjugate(int N, const float* A, const float* b, float* x)
	{
		float r[N];
		float p[N];
	
		for(int i = 0; i < N; i++)
		{
			//r = b - Ax
			r[i] = b[i];
			for(int j = 0; j < N; j++)
			{
				r[i] -= A[i*N+j]*x[j];
			}
		
			//p = r
			p[i] = r[i];
		}
	
		float rsold = 0.0f;
	
		for(int i = 0; i < N; i ++)
		{
			rsold += r[i] * r[i];
		}
	
		for(int i = 0; i < N; i++)
		{
			float Ap[N];
		
			for(int j = 0; j < N; j++)
			{
				Ap[j] = 0.0f;
			
				for(int k = 0; k < N; k++)
				{
					Ap[j] += A[j*N+k] * p[k];
				}
			}
		
			float abot = 0.0f;
		
			for(int j = 0; j < N; j++)
			{
				abot += p[j] * Ap[j];
			}
		
			float alpha = rsold / abot;
		
			for(int j = 0; j < N; j++)
			{
				x[j] = x[j] + alpha * p[j];
			
				r[j] = r[j] - alpha * Ap[j];
			}
		
			float rsnew = 0.0f;
		
			for(int j = 0; j < N; j++)
			{
				rsnew += r[j] * r[j];
			}
		
			if(rsnew < 1e-10f)
			{
	//			cout << "break " << i << endl;
				break;
			}
		
			for(int j = 0; j < N; j++)
			{
				p[j] = r[j] + rsnew / rsold * p[j];
			}
		
			rsold = rsnew;
		}
	}
	
	void Strand::calcVelocities(float dt)
	{
		int N = numParticles * 3;
		
		memset(A, 0, N*N*sizeof(float));
		memset(b, 0, N*sizeof(float));
		
		for(int i = 0; i < numParticles; i++)
		{
			x[i*3]   = particle[i]->velocity.x;
			x[i*3+1] = particle[i]->velocity.y;
			x[i*3+2] = particle[i]->velocity.z;
		}
		
		//Add edge springs data into A and b
		float h = dt*dt*k_edge/(4.0f*mass*length);
		float g = dt*k_edge/(2.0f*mass*length);
	
		for(int i = 0; i < numEdges; i++)
		{
			Vector3f d(particle[i+1]->position.x-particle[i]->position.x, particle[i+1]->position.y-particle[i]->position.y, particle[i+1]->position.z - particle[i]->position.z);
			Vector3f e = d;
			d = d * d.length_inverse();
		
			A[(i*3)*N  +i*3] += 1+h*d.x*d.x; A[(i*3)*N  +i*3+1] +=   h*d.x*d.y; A[(i*3)*N  +i*3+2] +=   h*d.x*d.z; A[(i*3)*N  +i*3+3] +=  -h*d.x*d.x; A[(i*3)*N  +i*3+4] +=  -h*d.x*d.y; A[(i*3)*N  +i*3+5] +=  -h*d.x*d.z;
			A[(i*3+1)*N+i*3] +=   h*d.x*d.y; A[(i*3+1)*N+i*3+1] += 1+h*d.y*d.y; A[(i*3+1)*N+i*3+2] +=   h*d.y*d.z; A[(i*3+1)*N+i*3+3] +=  -h*d.x*d.y; A[(i*3+1)*N+i*3+4] +=  -h*d.y*d.y; A[(i*3+1)*N+i*3+5] +=  -h*d.y*d.z;
			A[(i*3+2)*N+i*3] +=   h*d.x*d.z; A[(i*3+2)*N+i*3+1] +=   h*d.y*d.z; A[(i*3+2)*N+i*3+2] += 1+h*d.z*d.z; A[(i*3+2)*N+i*3+3] +=  -h*d.x*d.z; A[(i*3+2)*N+i*3+4] +=  -h*d.y*d.z; A[(i*3+2)*N+i*3+5] +=  -h*d.z*d.z;
			A[(i*3+3)*N+i*3] +=  -h*d.x*d.x; A[(i*3+3)*N+i*3+1] +=  -h*d.x*d.y; A[(i*3+3)*N+i*3+2] +=  -h*d.x*d.z; A[(i*3+3)*N+i*3+3] += 1+h*d.x*d.x; A[(i*3+3)*N+i*3+4] +=   h*d.x*d.y; A[(i*3+3)*N+i*3+5] +=   h*d.x*d.z;
			A[(i*3+4)*N+i*3] +=  -h*d.x*d.y; A[(i*3+4)*N+i*3+1] +=  -h*d.y*d.y; A[(i*3+4)*N+i*3+2] +=  -h*d.y*d.z; A[(i*3+4)*N+i*3+3] +=   h*d.x*d.y; A[(i*3+4)*N+i*3+4] += 1+h*d.y*d.y; A[(i*3+4)*N+i*3+5] +=   h*d.y*d.z;
			A[(i*3+5)*N+i*3] +=  -h*d.x*d.z; A[(i*3+5)*N+i*3+1] +=  -h*d.y*d.z; A[(i*3+5)*N+i*3+2] +=  -h*d.z*d.z; A[(i*3+5)*N+i*3+3] +=   h*d.x*d.z; A[(i*3+5)*N+i*3+4] +=   h*d.y*d.z; A[(i*3+5)*N+i*3+5] += 1+h*d.z*d.z;
		
			float factor = g * ((e.x*d.x+e.y*d.y+e.z*d.z) - (length));
		
			b[i*3]   += particle[i]->velocity.x + factor * d.x;
			b[i*3+1] += particle[i]->velocity.y + factor * d.y;
			b[i*3+2] += particle[i]->velocity.z + factor * d.z;
			b[i*3+3] += particle[i+1]->velocity.x - factor * d.x;
			b[i*3+4] += particle[i+1]->velocity.y - factor * d.y;
			b[i*3+5] += particle[i+1]->velocity.z - factor * d.z;
		}
		
		//Add bending springs data into A and b
		h = dt*dt*k_bend/(4.0f*mass*length);
		g = dt*k_bend/(2.0f*mass*length);
		
		for(int i = 0; i < numBend; i++)
		{
			Vector3f d(particle[i+2]->position.x-particle[i]->position.x, particle[i+2]->position.y-particle[i]->position.y, particle[i+2]->position.z - particle[i]->position.z);
			Vector3f e = d;
			d = d * d.length_inverse();
			
			A[(i*3  )*N+i*3] += 1+h*d.x*d.x; A[(i*3  )*N+i*3+1] +=   h*d.x*d.y; A[(i*3  )*N+i*3+2] +=   h*d.x*d.z; A[(i*3  )*N+i*3+6] +=  -h*d.x*d.x; A[(i*3  )*N+i*3+7] +=  -h*d.x*d.y; A[(i*3  )*N+i*3+8] +=  -h*d.x*d.z;
			A[(i*3+1)*N+i*3] +=   h*d.x*d.y; A[(i*3+1)*N+i*3+1] += 1+h*d.y*d.y; A[(i*3+1)*N+i*3+2] +=   h*d.y*d.z; A[(i*3+1)*N+i*3+6] +=  -h*d.x*d.y; A[(i*3+1)*N+i*3+7] +=  -h*d.y*d.y; A[(i*3+1)*N+i*3+8] +=  -h*d.y*d.z;
			A[(i*3+2)*N+i*3] +=   h*d.x*d.z; A[(i*3+2)*N+i*3+1] +=   h*d.y*d.z; A[(i*3+2)*N+i*3+2] += 1+h*d.z*d.z; A[(i*3+2)*N+i*3+6] +=  -h*d.x*d.z; A[(i*3+2)*N+i*3+7] +=  -h*d.y*d.z; A[(i*3+2)*N+i*3+8] +=  -h*d.z*d.z;
			A[(i*3+6)*N+i*3] +=  -h*d.x*d.x; A[(i*3+6)*N+i*3+1] +=  -h*d.x*d.y; A[(i*3+6)*N+i*3+2] +=  -h*d.x*d.z; A[(i*3+6)*N+i*3+6] += 1+h*d.x*d.x; A[(i*3+6)*N+i*3+7] +=   h*d.x*d.y; A[(i*3+6)*N+i*3+8] +=   h*d.x*d.z;
			A[(i*3+7)*N+i*3] +=  -h*d.x*d.y; A[(i*3+7)*N+i*3+1] +=  -h*d.y*d.y; A[(i*3+7)*N+i*3+2] +=  -h*d.y*d.z; A[(i*3+7)*N+i*3+6] +=   h*d.x*d.y; A[(i*3+7)*N+i*3+7] += 1+h*d.y*d.y; A[(i*3+7)*N+i*3+8] +=   h*d.y*d.z;
			A[(i*3+8)*N+i*3] +=  -h*d.x*d.z; A[(i*3+8)*N+i*3+1] +=  -h*d.y*d.z; A[(i*3+8)*N+i*3+2] +=  -h*d.z*d.z; A[(i*3+8)*N+i*3+6] +=   h*d.x*d.z; A[(i*3+8)*N+i*3+7] +=   h*d.y*d.z; A[(i*3+8)*N+i*3+8] += 1+h*d.z*d.z;
			
			float factor = g * ((e.x*d.x+e.y*d.y+e.z*d.z) - (length));
			
			b[i*3  ] += particle[i]->velocity.x + factor * d.x;
			b[i*3+1] += particle[i]->velocity.y + factor * d.y;
			b[i*3+2] += particle[i]->velocity.z + factor * d.z;
			b[i*3+6] += particle[i+2]->velocity.x - factor * d.x;
			b[i*3+7] += particle[i+2]->velocity.y - factor * d.y;
			b[i*3+8] += particle[i+2]->velocity.z - factor * d.z;
		}
	
		//Add twisting springs data into A and b
		h = dt*dt*k_twist/(4.0f*mass*length);
		g = dt*k_twist/(2.0f*mass*length);
	
		for(int i = 0; i < numTwist; i++)
		{
			Vector3f d(particle[i+3]->position.x-particle[i]->position.x, particle[i+3]->position.y-particle[i]->position.y, particle[i+3]->position.z - particle[i]->position.z);
			Vector3f e = d;
			d = d * d.length_inverse();
		
			A[(i*3   )*N+i*3] += 1+h*d.x*d.x; A[(i*3   )*N+i*3+1] +=   h*d.x*d.y; A[(i*3   )*N+i*3+2] +=   h*d.x*d.z; A[(i*3   )*N+i*3+9] +=  -h*d.x*d.x; A[(i*3   )*N+i*3+10] +=  -h*d.x*d.y; A[(i*3   )*N+i*3+11] +=  -h*d.x*d.z;
			A[(i*3+1 )*N+i*3] +=   h*d.x*d.y; A[(i*3+1 )*N+i*3+1] += 1+h*d.y*d.y; A[(i*3+1 )*N+i*3+2] +=   h*d.y*d.z; A[(i*3+1 )*N+i*3+9] +=  -h*d.x*d.y; A[(i*3+1 )*N+i*3+10] +=  -h*d.y*d.y; A[(i*3+1 )*N+i*3+11] +=  -h*d.y*d.z;
			A[(i*3+2 )*N+i*3] +=   h*d.x*d.z; A[(i*3+2 )*N+i*3+1] +=   h*d.y*d.z; A[(i*3+2 )*N+i*3+2] += 1+h*d.z*d.z; A[(i*3+2 )*N+i*3+9] +=  -h*d.x*d.z; A[(i*3+2 )*N+i*3+10] +=  -h*d.y*d.z; A[(i*3+2 )*N+i*3+11] +=  -h*d.z*d.z;
			A[(i*3+9 )*N+i*3] +=  -h*d.x*d.x; A[(i*3+9 )*N+i*3+1] +=  -h*d.x*d.y; A[(i*3+9 )*N+i*3+2] +=  -h*d.x*d.z; A[(i*3+9 )*N+i*3+9] += 1+h*d.x*d.x; A[(i*3+9 )*N+i*3+10] +=   h*d.x*d.y; A[(i*3+9 )*N+i*3+11] +=   h*d.x*d.z;
			A[(i*3+10)*N+i*3] +=  -h*d.x*d.y; A[(i*3+10)*N+i*3+1] +=  -h*d.y*d.y; A[(i*3+10)*N+i*3+2] +=  -h*d.y*d.z; A[(i*3+10)*N+i*3+9] +=   h*d.x*d.y; A[(i*3+10)*N+i*3+10] += 1+h*d.y*d.y; A[(i*3+10)*N+i*3+11] +=   h*d.y*d.z;
			A[(i*3+11)*N+i*3] +=  -h*d.x*d.z; A[(i*3+11)*N+i*3+1] +=  -h*d.y*d.z; A[(i*3+11)*N+i*3+2] +=  -h*d.z*d.z; A[(i*3+11)*N+i*3+9] +=   h*d.x*d.z; A[(i*3+11)*N+i*3+10] +=   h*d.y*d.z; A[(i*3+11)*N+i*3+11] += 1+h*d.z*d.z;
			
			float factor = g * ((e.x*d.x+e.y*d.y+e.z*d.z) - (length));
			
			b[i*3   ] += particle[i]->velocity.x + factor * d.x;
			b[i*3+1 ] += particle[i]->velocity.y + factor * d.y;
			b[i*3+2 ] += particle[i]->velocity.z + factor * d.z;
			b[i*3+9 ] += particle[i+3]->velocity.x + factor * d.x;
			b[i*3+10] += particle[i+3]->velocity.y + factor * d.y;
			b[i*3+11] += particle[i+3]->velocity.z + factor * d.z;
		}
		
//		for(int i = 0; i < N; i++)
//		{
//			for(int j = 0; j < N; j++)
//			{
//				std::cout << std::setw(6) << A[i * N + j] << " "; 
//			}
//			std::cout << std::endl;
//		}
//		std::cout << std::endl;
		
		conjugate(N, A, b, x);
		
		for(int i = 0; i < numParticles; i++)
		{
			particle[i]->velc.x = x[i*3];
			particle[i]->velc.y = x[i*3+1];
			particle[i]->velc.z = x[i*3+2];
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
	
	void Strand::objectCollisions()
	{
		//Transform particle coordinates to collision grid coordinates
//		for(int i = 1; i < numParticles; i++)
//		{
//			Vector3f position;
//			position.x = (particle[i]->position.x + DOMAIN_HALF-CELL_HALF)/CELL_WIDTH;
//			position.y = (particle[i]->position.y + DOMAIN_HALF+0.125f-CELL_HALF)/CELL_WIDTH;
//			position.z = (particle[i]->position.z + DOMAIN_HALF-CELL_HALF)/CELL_WIDTH;
//		}
		
		Vector3f position;
		position.x = (particle[25]->position.x + DOMAIN_HALF-CELL_HALF)/CELL_WIDTH;
		position.y = (particle[25]->position.y + DOMAIN_HALF+0.125f-CELL_HALF)/CELL_WIDTH;
		position.z = (particle[25]->position.z + DOMAIN_HALF-CELL_HALF)/CELL_WIDTH;
		
		std::cout << position.x << " " << position.y << " " << position.z << std::endl;
		
		Vector3f cube[8];
		cube[0].x = std::floor(position.x);
		cube[0].y = std::floor(position.y);
		cube[0].z = std::floor(position.z);
		
		cube[1].x = std::ceil(position.x);
		cube[1].y = std::floor(position.y);
		cube[1].z = std::floor(position.z);
		
		cube[2].x = std::floor(position.x);
		cube[2].y = std::floor(position.y);
		cube[2].z = std::ceil(position.z);
		
		cube[3].x = std::ceil(position.x);
		cube[3].y = std::floor(position.y);
		cube[3].z = std::ceil(position.z);
		
		cube[4].x = std::floor(position.x);
		cube[4].y = std::ceil(position.y);
		cube[4].z = std::floor(position.z);
		
		cube[5].x = std::ceil(position.x);
		cube[5].y = std::ceil(position.y);
		cube[5].z = std::floor(position.z);
		
		cube[6].x = std::floor(position.x);
		cube[6].y = std::ceil(position.y);
		cube[6].z = std::ceil(position.z);
		
		cube[7].x = std::ceil(position.x);
		cube[7].y = std::ceil(position.y);
		cube[7].z = std::ceil(position.z);
		
	}
	
	void Strand::update(float dt)
	{
		//Reset forces on particles
		clearForces();
		
		calcVelocities(dt);
		
		//Calculate and apply spring forces using previous position
		updateSprings1(dt);
		
		//Apply gravity
		applyForce(Vector3f(0.0f, GRAVITY, 0.0f));
		
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
		applyForce(Vector3f(0.0f, GRAVITY, 0.0f));
		
		//Calculate half velocity and new velocity
		updateParticles2(dt);
		
		//Check geometry collisions and adjust velocities and positions
		objectCollisions();
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
		
		delete [] A;
		delete [] x;
		delete [] b;
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
			   std::vector<Vector3f> &roots,
			   Model_OBJ &obj)
	{
		this->numStrands = numStrands;
		
		strand = new Strand*[numStrands];
		
		for(int i = 0; i < numStrands; i++)
		{
			strand[i] = new Strand(numParticles, mass, k_edge, k_bend, k_twist, k_extra, d_edge, d_bend, d_twist, d_extra, length, roots[i]);
		}
		
		initDistanceField(obj);
	}
	
	void Hair::initDistanceField(Model_OBJ &obj)
	{
		//Initialise distance field to inifinity
		for(int xx = 0; xx < DOMAIN_DIM; xx++)
			for(int yy = 0; yy < DOMAIN_DIM; yy++)
				for(int zz = 0; zz < DOMAIN_DIM; zz++)
					grid[xx][yy][zz] = FLT_MAX;
		
		//calculate triangle normal scaling factor
		float delta = 0.25f;
		float echo = CELL_WIDTH * delta;
		
		int numVertices = obj.TotalConnectedPoints / POINTS_PER_VERTEX;
		int numTriangles = obj.TotalConnectedTriangles / TOTAL_FLOATS_IN_TRIANGLE;
		
//		std::cout << numVertices << std::endl;
//		std::cout << numTriangles << std::endl;
		
		//read in each triangle with its normal data
		for(int i = 0; i < numTriangles; i++)
		{
			//print triangle normals
//			int index = i * TOTAL_FLOATS_IN_TRIANGLE;
			
			float triangle[3][POINTS_PER_VERTEX];
			for(int j = 0; j < POINTS_PER_VERTEX; j++)
			{
				triangle[j][0] = obj.Faces_Triangles[i*TOTAL_FLOATS_IN_TRIANGLE+j*3];
				triangle[j][1] = obj.Faces_Triangles[i*TOTAL_FLOATS_IN_TRIANGLE+j*3+1];
				triangle[j][2] = obj.Faces_Triangles[i*TOTAL_FLOATS_IN_TRIANGLE+j*3+2];
			}
			
			float normal[POINTS_PER_VERTEX];
			normal[0] = obj.normals[i*TOTAL_FLOATS_IN_TRIANGLE];
			normal[1] = obj.normals[i*TOTAL_FLOATS_IN_TRIANGLE+1];
			normal[2] = obj.normals[i*TOTAL_FLOATS_IN_TRIANGLE+2];
			
//			std::cout << "tri: " << i << std::endl;
//			std::cout << "0: " << triangle[0][0] << " " << triangle[0][1] << " " << triangle[0][2] << std::endl;
//			std::cout << "1: " << triangle[1][0] << " " << triangle[1][1] << " " << triangle[1][2] << std::endl;
//			std::cout << "2: " << triangle[2][0] << " " << triangle[2][1] << " " << triangle[2][2] << std::endl;
//			std::cout << "n: " << normal[0] << " " << normal[1] << " " << normal[2] << std::endl;
//			std::cout << std::endl;
			
			//build prism
			float prism[6][POINTS_PER_VERTEX];
			
			for(int j = 0; j < POINTS_PER_VERTEX; j++)
			{
				prism[j][0]   = triangle[j][0] + echo * normal[0];
				prism[j][1]   = triangle[j][1] + echo * normal[1];
				prism[j][2]   = triangle[j][2] + echo * normal[2];
				prism[j+3][0] = triangle[j][0] - echo * normal[0];
				prism[j+3][1] = triangle[j][1] - echo * normal[1];
				prism[j+3][2] = triangle[j][2] - echo * normal[2];
			}
			
//			for(int j = 0; j < 6; j++)
//			{
//				std::cout << j << ": " << prism[j][0] << " " << prism[j][1] << " " << prism[j][2] << std::endl;
//			}
//			std::cout << std::endl;
			
			//Axis-aligned bounding box
			float aabb[2][POINTS_PER_VERTEX]; //-x,-y,-z,+x,+y,+z
			aabb[0][0] =  FLT_MAX;
			aabb[0][1] =  FLT_MAX;
			aabb[0][2] =  FLT_MAX;
			aabb[1][0] = -FLT_MAX;
			aabb[1][1] = -FLT_MAX;
			aabb[1][2] = -FLT_MAX;
			
			//Build the aabb using the minimum and maximum values of the prism
			for(int j = 0; j < 6; j++)
			{
				//minimum x, y, z
				aabb[0][0] = std::min(prism[j][0], aabb[0][0]);
				aabb[0][1] = std::min(prism[j][1], aabb[0][1]);
				aabb[0][2] = std::min(prism[j][2], aabb[0][2]);
				
				//maximum x, y, z
				aabb[1][0] = std::max(prism[j][0], aabb[1][0]);
				aabb[1][1] = std::max(prism[j][1], aabb[1][1]);
				aabb[1][2] = std::max(prism[j][2], aabb[1][2]);
			}
			
//			std::cout << "min: " << std::setw(10) << std::setprecision(8) << aabb[0][0] << " " << aabb[0][1] << " " << aabb[0][2] << std::endl;
//			std::cout << "max: " << std::setw(10) << std::setprecision(8) << aabb[1][0] << " " << aabb[1][1] << " " << aabb[1][2] << std::endl;
//			std::cout << std::endl;
			
			//normalise to the grid
			aabb[0][0] = (aabb[0][0] + DOMAIN_HALF-CELL_HALF)/CELL_WIDTH;
			aabb[0][1] = (aabb[0][1] + DOMAIN_HALF+0.125f-CELL_HALF)/CELL_WIDTH;
			aabb[0][2] = (aabb[0][2] + DOMAIN_HALF-CELL_HALF)/CELL_WIDTH;
			aabb[1][0] = (aabb[1][0] + DOMAIN_HALF-CELL_HALF)/CELL_WIDTH;
			aabb[1][1] = (aabb[1][1] + DOMAIN_HALF+0.125f-CELL_HALF)/CELL_WIDTH;
			aabb[1][2] = (aabb[1][2] + DOMAIN_HALF-CELL_HALF)/CELL_WIDTH;
			
//			std::cout << "nmin: " << std::setw(10) << std::setprecision(8) << aabb[0][0] << " " << aabb[0][1] << " " << aabb[0][2] << std::endl;
//			std::cout << "nmax: " << std::setw(10) << std::setprecision(8) << aabb[1][0] << " " << aabb[1][1] << " " << aabb[1][2] << std::endl;
//			std::cout << std::endl;
			
			//round aabb
			aabb[0][0] = floor(aabb[0][0]);
			aabb[0][1] = floor(aabb[0][1]);
			aabb[0][2] = floor(aabb[0][2]);
			aabb[1][0] = ceil(aabb[1][0]);
			aabb[1][1] = ceil(aabb[1][1]);
			aabb[1][2] = ceil(aabb[1][2]);
			
//			std::cout << "nmin: " << std::setw(10) << std::setprecision(8) << aabb[0][0] << " " << aabb[0][1] << " " << aabb[0][2] << std::endl;
//			std::cout << "nmax: " << std::setw(10) << std::setprecision(8) << aabb[1][0] << " " << aabb[1][1] << " " << aabb[1][2] << std::endl;
//			std::cout << std::endl;
			
			int iaabb[2][POINTS_PER_VERTEX];
			iaabb[0][0] = int(aabb[0][0]);
			iaabb[0][1] = int(aabb[0][1]);
			iaabb[0][2] = int(aabb[0][2]);
			iaabb[1][0] = int(aabb[1][0]);
			iaabb[1][1] = int(aabb[1][1]);
			iaabb[1][2] = int(aabb[1][2]);
			
//			std::cout << "nmin: " << std::setw(10) << std::setprecision(8) << iaabb[0][0] << " " << iaabb[0][1] << " " << iaabb[0][2] << std::endl;
//			std::cout << "nmax: " << std::setw(10) << std::setprecision(8) << iaabb[1][0] << " " << iaabb[1][1] << " " << iaabb[1][2] << std::endl;
//			std::cout << std::endl;
			
			//build edge vectors
			float edge[3][POINTS_PER_VERTEX];
			edge[0][0] = triangle[1][0] - triangle[0][0];
			edge[0][1] = triangle[1][1] - triangle[0][1];
			edge[0][2] = triangle[1][2] - triangle[0][2];
			edge[1][0] = triangle[2][0] - triangle[1][0];
			edge[1][1] = triangle[2][1] - triangle[1][1];
			edge[1][2] = triangle[2][2] - triangle[1][2];
			edge[2][0] = triangle[0][0] - triangle[2][0];
			edge[2][1] = triangle[0][1] - triangle[2][0];
			edge[2][2] = triangle[0][2] - triangle[2][0];
			
			//build edge normal vectors by cross product with triangle normal
			float edgeNormal[3][POINTS_PER_VERTEX];
			edgeNormal[0][0] = normal[1] * edge[0][2] - normal[2] * edge[0][1];
			edgeNormal[0][1] = normal[2] * edge[0][0] - normal[0] * edge[0][2];
			edgeNormal[0][2] = normal[0] * edge[0][1] - normal[1] * edge[0][0];
			edgeNormal[1][0] = normal[1] * edge[1][2] - normal[2] * edge[1][1];
			edgeNormal[1][1] = normal[2] * edge[1][0] - normal[0] * edge[1][2];
			edgeNormal[1][2] = normal[0] * edge[1][1] - normal[1] * edge[1][0];
			edgeNormal[2][0] = normal[1] * edge[2][2] - normal[2] * edge[2][1];
			edgeNormal[2][1] = normal[2] * edge[2][0] - normal[0] * edge[2][2];
			edgeNormal[2][2] = normal[0] * edge[2][1] - normal[1] * edge[2][0];
			
			for(int xx = iaabb[0][0]; xx <= iaabb[1][0]; xx++)
			{
				for(int yy = iaabb[0][1]; yy <= iaabb[1][1]; yy++)
				{	
					for(int zz = iaabb[0][2]; zz <= iaabb[1][2]; zz++)
					{
						//Denormalise from grid to centre of cell
						float xpos = xx * CELL_WIDTH - DOMAIN_HALF + CELL_HALF;
						float ypos = yy * CELL_WIDTH - DOMAIN_HALF - 0.125f + CELL_HALF;
						float zpos = zz * CELL_WIDTH - DOMAIN_HALF + CELL_HALF;
						
						//dot product between gridpoint and triangle normal
						float dvalue = (xpos - triangle[0][0]) * normal[0] + (ypos - triangle[0][1]) * normal[1] + (zpos - triangle[0][2]) * normal[2];
						
						//Test whether the point lies within the triangle voronoi region
						float planeTest[3];
						planeTest[0] = xpos*edgeNormal[0][0] + ypos*edgeNormal[0][1] + zpos*edgeNormal[0][2] - triangle[0][0]*edgeNormal[0][0] - triangle[0][1]*edgeNormal[0][1] - triangle[0][2]*edgeNormal[0][2];
						planeTest[1] = xpos*edgeNormal[1][0] + ypos*edgeNormal[1][1] + zpos*edgeNormal[1][2] - triangle[1][0]*edgeNormal[1][0] - triangle[1][1]*edgeNormal[1][1] - triangle[1][2]*edgeNormal[1][2];
						planeTest[2] = xpos*edgeNormal[2][0] + ypos*edgeNormal[2][1] + zpos*edgeNormal[2][2] - triangle[2][0]*edgeNormal[2][0] - triangle[2][1]*edgeNormal[2][1] - triangle[2][2]*edgeNormal[2][2];
						
						if(!(planeTest[0] < 0.0f && planeTest[1] < 0.0f && planeTest[2] < 0.0f))
						{
							//Cross products
							float regionNormal[3][POINTS_PER_VERTEX];
							regionNormal[0][0] = normal[1] * edgeNormal[0][2] - normal[2] * edgeNormal[0][1];
							regionNormal[0][1] = normal[2] * edgeNormal[0][0] - normal[0] * edgeNormal[0][2];
							regionNormal[0][2] = normal[0] * edgeNormal[0][1] - normal[1] * edgeNormal[0][0];
							regionNormal[1][0] = normal[1] * edgeNormal[1][2] - normal[2] * edgeNormal[1][1];
							regionNormal[1][1] = normal[2] * edgeNormal[1][0] - normal[0] * edgeNormal[1][2];
							regionNormal[1][2] = normal[0] * edgeNormal[1][1] - normal[1] * edgeNormal[1][0];
							regionNormal[2][0] = normal[1] * edgeNormal[2][2] - normal[2] * edgeNormal[2][1];
							regionNormal[2][1] = normal[2] * edgeNormal[2][0] - normal[0] * edgeNormal[2][2];
							regionNormal[2][2] = normal[0] * edgeNormal[2][1] - normal[1] * edgeNormal[2][0];
							
							float regionTest[3][2];
							//Test if the point lies between the planes that define the first edge's voronoi region.
							regionTest[0][0] = -xpos*regionNormal[0][0] - ypos*regionNormal[0][1] - zpos*regionNormal[0][2] + triangle[0][0]*regionNormal[0][0] + triangle[0][1]*regionNormal[0][1] + triangle[0][2]*regionNormal[0][2];
							regionTest[0][1] =  xpos*regionNormal[0][0] + ypos*regionNormal[0][1] + zpos*regionNormal[0][2] - triangle[1][0]*regionNormal[0][0] - triangle[1][1]*regionNormal[0][1] - triangle[1][2]*regionNormal[0][2];
							//Test if the point lies between the planes that define the second edge's voronoi region.
							regionTest[1][0] = -xpos*regionNormal[1][0] - ypos*regionNormal[1][1] - zpos*regionNormal[1][2] + triangle[1][0]*regionNormal[1][0] + triangle[1][1]*regionNormal[1][1] + triangle[1][2]*regionNormal[1][2];
							regionTest[1][1] =  xpos*regionNormal[1][0] + ypos*regionNormal[1][1] + zpos*regionNormal[1][2] - triangle[2][0]*regionNormal[1][0] - triangle[2][1]*regionNormal[1][1] - triangle[2][2]*regionNormal[1][2];
							//Test if the point lies between the planes that define the third edge's voronoi region.
							regionTest[2][0] = -xpos*regionNormal[2][0] - ypos*regionNormal[2][1] - zpos*regionNormal[2][2] + triangle[2][0]*regionNormal[1][0] + triangle[2][1]*regionNormal[1][1] + triangle[2][2]*regionNormal[1][2];
							regionTest[2][1] =  xpos*regionNormal[2][0] + ypos*regionNormal[2][1] + zpos*regionNormal[2][2] - triangle[0][0]*regionNormal[1][0] - triangle[0][1]*regionNormal[1][1] - triangle[0][2]*regionNormal[1][2];
							
							if(planeTest[0] >= 0.0f && regionTest[0][0] < 0.0f && regionTest[0][1] < 0.0f)
							{
								float aa[POINTS_PER_VERTEX];
								float bb[POINTS_PER_VERTEX];
								float cc[POINTS_PER_VERTEX];
								float dd[POINTS_PER_VERTEX];
								
								aa[0] = xpos - triangle[0][0];
								aa[1] = ypos - triangle[0][1];
								aa[2] = zpos - triangle[0][2];
								bb[0] = xpos - triangle[1][0];
								bb[1] = ypos - triangle[1][1];
								bb[2] = zpos - triangle[1][2];
								cc[0] = triangle[1][0] - triangle[0][0];
								cc[1] = triangle[1][1] - triangle[0][1];
								cc[2] = triangle[1][2] - triangle[0][2];
								
								dd[0] = aa[1]*bb[2] - aa[2]*bb[1];
								dd[1] = aa[2]*bb[0] - aa[0]*bb[2];
								dd[2] = aa[0]*bb[1] - aa[1]*bb[0];
								
								float dist = sqrtf(dd[0]*dd[0]+dd[1]*dd[1]+dd[2]*dd[2])/sqrtf(cc[0]*cc[0]+cc[1]*cc[1]+cc[2]*cc[2]);
								
								dvalue = (dvalue >= 0.0f) ? dist : -dist;
								
							}
							else if(planeTest[1] >= 0.0f && regionTest[1][0] < 0.0f && regionTest[1][1] < 0.0f)
							{
								float aa[POINTS_PER_VERTEX];
								float bb[POINTS_PER_VERTEX];
								float cc[POINTS_PER_VERTEX];
								float dd[POINTS_PER_VERTEX];
								
								aa[0] = xpos - triangle[1][0];
								aa[1] = ypos - triangle[1][1];
								aa[2] = zpos - triangle[1][2];
								bb[0] = xpos - triangle[2][0];
								bb[1] = ypos - triangle[2][1];
								bb[2] = zpos - triangle[2][2];
								cc[0] = triangle[2][0] - triangle[1][0];
								cc[1] = triangle[2][1] - triangle[1][1];
								cc[2] = triangle[2][2] - triangle[1][2];
								
								dd[0] = aa[1]*bb[2] - aa[2]*bb[1];
								dd[1] = aa[2]*bb[0] - aa[0]*bb[2];
								dd[2] = aa[0]*bb[1] - aa[1]*bb[0];
								
								float dist = sqrtf(dd[0]*dd[0]+dd[1]*dd[1]+dd[2]*dd[2])/sqrtf(cc[0]*cc[0]+cc[1]*cc[1]+cc[2]*cc[2]);
								
								dvalue = (dvalue >= 0.0f) ? dist : -dist;
							}
							else if(planeTest[2] >= 0.0f && regionTest[2][0] < 0.0f && regionTest[2][1] < 0.0f)
							{
								float aa[POINTS_PER_VERTEX];
								float bb[POINTS_PER_VERTEX];
								float cc[POINTS_PER_VERTEX];
								float dd[POINTS_PER_VERTEX];
								
								aa[0] = xpos - triangle[2][0];
								aa[1] = ypos - triangle[2][1];
								aa[2] = zpos - triangle[2][2];
								bb[0] = xpos - triangle[0][0];
								bb[1] = ypos - triangle[0][1];
								bb[2] = zpos - triangle[0][2];
								cc[0] = triangle[0][0] - triangle[2][0];
								cc[1] = triangle[0][1] - triangle[2][1];
								cc[2] = triangle[0][2] - triangle[2][2];
								
								dd[0] = aa[1]*bb[2] - aa[2]*bb[1];
								dd[1] = aa[2]*bb[0] - aa[0]*bb[2];
								dd[2] = aa[0]*bb[1] - aa[1]*bb[0];
								
								float dist = sqrtf(dd[0]*dd[0]+dd[1]*dd[1]+dd[2]*dd[2])/sqrtf(cc[0]*cc[0]+cc[1]*cc[1]+cc[2]*cc[2]);
								
								dvalue = (dvalue >= 0.0f) ? dist : -dist;
							}
							else
							{
								float dist[3];
								dist[0] = sqrtf( (xpos-triangle[0][0])*(xpos - triangle[0][0]) + (ypos-triangle[0][1])*(ypos-triangle[0][1]) + (zpos-triangle[0][2])*(zpos-triangle[0][2]));
								dist[1] = sqrtf( (xpos-triangle[1][0])*(xpos - triangle[1][0]) + (ypos-triangle[1][1])*(ypos-triangle[1][1]) + (zpos-triangle[1][2])*(zpos-triangle[1][2]));
								dist[2] = sqrtf( (xpos-triangle[2][0])*(xpos - triangle[2][0]) + (ypos-triangle[2][1])*(ypos-triangle[2][1]) + (zpos-triangle[2][2])*(zpos-triangle[2][2]));
								
								dvalue = (dvalue >= 0.0f) ? std::min(dist[0], std::min(dist[1], dist[2])) : -std::min(dist[0], std::min(dist[1], dist[2]));
							}
						}
						
						if(grid[xx][yy][zz] < FLT_MAX)
						{
							if(std::abs(dvalue) < std::abs(grid[xx][yy][zz]) && dvalue > 0.0f && grid[xx][yy][zz] < 0.0f)
							{
								grid[xx][yy][zz] = dvalue;
							}
							else if(std::abs(dvalue) < std::abs(grid[xx][yy][zz]) && dvalue >= 0.0f && grid[xx][yy][zz] > 0.0f)
							{
								grid[xx][yy][zz] = dvalue;
							}
							else if(std::abs(dvalue) < std::abs(grid[xx][yy][zz]) && dvalue <= 0.0f && grid[xx][yy][zz] < 0.0f)
							{
								grid[xx][yy][zz] = dvalue;
							}
						}
						else
						{
							grid[xx][yy][zz] = dvalue;
						}
						
					}
				}
			}
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

