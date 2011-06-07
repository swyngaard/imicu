
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
		Vector3f force;
		
		Vector3f xn = particle[0]->position - particle[1]->position;
		Vector3f vn = particle[0]->velocity - particle[1]->velocity;
		Vector3f dn = xn.unit();
		
		force += dn * ((k / length) * (xn.x * dn.x + xn.y * dn.y + xn.z * dn.z - length));
		force += dn * ((dt * k / length) * (vn.x * dn.x + vn.y * dn.y + vn.z * dn.z - length));
		
		particle[0]->applyForce(force);
		particle[1]->applyForce(-force);
	}
	
	void Spring::conjugate(float* A, float* b, float* x)
	{
		float r[6];
		float p[6];
		
		//r = b - A * x
		r[0] = b[0] - (A[0]*x[0] +A[1]*x[1] +A[2]*x[2] +A[3]*x[3] +A[4]*x[4] +A[5]*x[5]);
		r[1] = b[1] - (A[6]*x[0] +A[7]*x[1] +A[8]*x[2] +A[9]*x[3] +A[10]*x[4]+A[11]*x[5]);
		r[2] = b[2] - (A[12]*x[0]+A[13]*x[1]+A[14]*x[2]+A[15]*x[3]+A[16]*x[4]+A[17]*x[5]);
		r[3] = b[3] - (A[18]*x[0]+A[19]*x[1]+A[20]*x[2]+A[21]*x[3]+A[22]*x[4]+A[23]*x[5]);
		r[4] = b[4] - (A[24]*x[0]+A[25]*x[1]+A[26]*x[2]+A[27]*x[3]+A[28]*x[4]+A[29]*x[5]);
		r[5] = b[5] - (A[30]*x[0]+A[31]*x[1]+A[32]*x[2]+A[33]*x[3]+A[34]*x[4]+A[35]*x[5]);
		
		//p = r
		p[0] = r[0];
		p[1] = r[1];
		p[2] = r[2];
		p[3] = r[3];
		p[4] = r[4];
		p[5] = r[5];
		
		//rsold = r dot r
		float rsold = r[0]*r[0]+r[1]*r[1]+r[2]*r[2]+r[3]*r[3]+r[4]*r[4]+r[5]*r[5];
		float rsnew = rsold;
		
		float Ap[6];
		
		for(int i = 0; i < 6; i++)
		{
			Ap[0] = A[0]*p[0] +A[1]*p[1] +A[2]*p[2] +A[3]*p[3] +A[4]*p[4] +A[5]*p[5];
			Ap[1] = A[6]*p[0] +A[7]*p[1] +A[8]*p[2] +A[9]*p[3] +A[10]*p[4]+A[11]*p[5];
			Ap[2] = A[12]*p[0]+A[13]*p[1]+A[14]*p[2]+A[15]*p[3]+A[16]*p[4]+A[17]*p[5];
			Ap[3] = A[18]*p[0]+A[19]*p[1]+A[20]*p[2]+A[21]*p[3]+A[22]*p[4]+A[23]*p[5];
			Ap[4] = A[24]*p[0]+A[25]*p[1]+A[26]*p[2]+A[27]*p[3]+A[28]*p[4]+A[29]*p[5];
			Ap[5] = A[30]*p[0]+A[31]*p[1]+A[32]*p[2]+A[33]*p[3]+A[34]*p[4]+A[35]*p[5];
			
			float alpha = rsold / (p[0]*Ap[0]+p[1]*Ap[1]+p[2]*Ap[2]+p[3]*Ap[3]+p[4]*Ap[4]+p[5]*Ap[5]);
			
			x[0] += alpha * p[0];
			x[1] += alpha * p[1];
			x[2] += alpha * p[2];
			x[3] += alpha * p[3];
			x[4] += alpha * p[4];
			x[5] += alpha * p[5];
			
			r[0] -= alpha * Ap[0];
			r[1] -= alpha * Ap[1];
			r[2] -= alpha * Ap[2];
			r[3] -= alpha * Ap[3];
			r[4] -= alpha * Ap[4];
			r[5] -= alpha * Ap[5];
			
			rsnew = r[0]*r[0]+r[1]*r[1]+r[2]*r[2]+r[3]*r[3]+r[4]*r[4]+r[5]*r[5];
			
			if(rsnew < 1e-20f) break;
			
			p[0] = r[0] + rsnew / rsold * p[0];
			p[1] = r[1] + rsnew / rsold * p[1];
			p[2] = r[2] + rsnew / rsold * p[2];
			p[3] = r[3] + rsnew / rsold * p[3];
			p[4] = r[4] + rsnew / rsold * p[4];
			p[5] = r[5] + rsnew / rsold * p[5];
			
			rsold = rsnew;
		}
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
		
		//Solve half velocity for each particle and store it
		//Calculate and apply spring forces between appropriate particles
		updateSprings(dt);
		
		
		//Calculate and store the new half velocity
		//Calculate and store the new position
		//Calculate and store the new half position
		
		//Calculate and store the new half velocity
		//Calculate and store the new velocity
		
		
		
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

