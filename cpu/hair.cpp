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
		Vector3f force;
		
		Vector3f p0 = particle[0]->position;
		Vector3f p1 = particle[1]->position;
		
		Vector3f xn = p1 - p0;
		Vector3f vn = particle[1]->velocity - particle[0]->velocity;
		Vector3f d = xn.unit();
		
		//Calculate velocity
		float A[36];
		float x[6];
		float b[6];
		
		float h = dt * dt * k / (4.0f * particle[0]->mass * length);
		float g = dt * k / (2.0f * particle[0]->mass * length);
		float f = k / length * particle[0]->mass;
		
		//Build Matrix A
		A[0] = h*d.x*d.x+1; A[1] = h*d.x*d.y;   A[2] = h*d.x*d.z;   A[3] =-h*d.x*d.x;   A[4] =-h*d.x*d.y;   A[5] =-h*d.x*d.z;
		A[6] = h*d.x*d.y;   A[7] = h*d.y*d.y+1; A[8] = h*d.y*d.z;   A[9] =-h*d.x*d.y;   A[10]=-h*d.y*d.y;   A[11]=-h*d.y*d.z;
		A[12]= h*d.x*d.z;   A[13]= h*d.y*d.z;   A[14]= h*d.z*d.z+1; A[15]=-h*d.x*d.z;   A[16]=-h*d.y*d.z;   A[17]=-h*d.z*d.z;
		A[18]=-h*d.x*d.x;   A[19]=-h*d.x*d.y;   A[20]=-h*d.x*d.z;   A[21]= h*d.x*d.x+1; A[22]= h*d.x*d.y;   A[23]= h*d.x*d.z;
		A[24]=-h*d.x*d.y;   A[25]=-h*d.y*d.y;   A[26]=-h*d.y*d.z;   A[27]= h*d.x*d.y;   A[28]= h*d.y*d.y+1; A[29]= h*d.y*d.z;
		A[30]=-h*d.x*d.z;   A[31]=-h*d.y*d.z;   A[32]=-h*d.z*d.z;   A[33]= h*d.x*d.z;   A[34]= h*d.y*d.z;   A[35]= h*d.z*d.z+1;
		
		//Store known values in vector b
		b[0] = particle[0]->velocity.x + g*d.x*((p1.x*d.x + p1.y*d.y + p1.z*d.z) - (p0.x*d.x + p0.y*d.y + p0.z*d.z) - length);
		b[1] = particle[0]->velocity.y + g*d.y*((p1.x*d.x + p1.y*d.y + p1.z*d.z) - (p0.x*d.x + p0.y*d.y + p0.z*d.z) - length);
		b[2] = particle[0]->velocity.z + g*d.z*((p1.x*d.x + p1.y*d.y + p1.z*d.z) - (p0.x*d.x + p0.y*d.y + p0.z*d.z) - length);
		b[3] = particle[1]->velocity.x + g*d.x*((p0.x*d.x + p0.y*d.y + p0.z*d.z) - (p1.x*d.x + p1.y*d.y + p1.z*d.z) - length);
		b[4] = particle[1]->velocity.y + g*d.y*((p0.x*d.x + p0.y*d.y + p0.z*d.z) - (p1.x*d.x + p1.y*d.y + p1.z*d.z) - length);
		b[5] = particle[1]->velocity.z + g*d.z*((p0.x*d.x + p0.y*d.y + p0.z*d.z) - (p1.x*d.x + p1.y*d.y + p1.z*d.z) - length);
		
		//Predict a solution and store in x
		x[0] = f * d.x * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt * (vn.x*d.x + vn.y*d.y + vn.z*d.z));
		x[1] = f * d.y * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt * (vn.x*d.x + vn.y*d.y + vn.z*d.z));
		x[2] = f * d.z * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt * (vn.x*d.x + vn.y*d.y + vn.z*d.z));
		x[3] = f * d.x * (-xn.x*d.x - xn.y*d.y - xn.z*d.z - length - dt * (vn.x*d.x + vn.y*d.y + vn.z*d.z));
		x[4] = f * d.y * (-xn.x*d.x - xn.y*d.y - xn.z*d.z - length - dt * (vn.x*d.x + vn.y*d.y + vn.z*d.z));
		x[5] = f * d.z * (-xn.x*d.x - xn.y*d.y - xn.z*d.z - length - dt * (vn.x*d.x + vn.y*d.y + vn.z*d.z));
		
		Vector3f v1(x[3]-x[0],x[4]-x[1],x[5]-x[2]);
		
		force += d * (f * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt*(v1.x*d.x + v1.y*d.y + v1.z*d.z)));
		
		particle[0]->applyForce(force);
		particle[1]->applyForce(-force);
	}
	
	void Spring::update2(float dt)
	{
		Vector3f force;
		
		Vector3f p0 = particle[0]->posh;
		Vector3f p1 = particle[1]->posh;
		
		Vector3f xn = p1 - p0;
		Vector3f vn = particle[1]->velocity - particle[0]->velocity;
		Vector3f d = xn.unit();
		
		//Calculate velocity
		float A[36];
		float x[6];
		float b[6];
		
		float h = dt * dt * k / (4.0f * particle[0]->mass * length);
		float g = dt * k / (2.0f * particle[0]->mass * length);
		float f = k / length * particle[0]->mass;
		
		//Build Matrix A
		A[0] = h*d.x*d.x+1; A[1] = h*d.x*d.y;   A[2] = h*d.x*d.z;   A[3] =-h*d.x*d.x;   A[4] =-h*d.x*d.y;   A[5] =-h*d.x*d.z;
		A[6] = h*d.x*d.y;   A[7] = h*d.y*d.y+1; A[8] = h*d.y*d.z;   A[9] =-h*d.x*d.y;   A[10]=-h*d.y*d.y;   A[11]=-h*d.y*d.z;
		A[12]= h*d.x*d.z;   A[13]= h*d.y*d.z;   A[14]= h*d.z*d.z+1; A[15]=-h*d.x*d.z;   A[16]=-h*d.y*d.z;   A[17]=-h*d.z*d.z;
		A[18]=-h*d.x*d.x;   A[19]=-h*d.x*d.y;   A[20]=-h*d.x*d.z;   A[21]= h*d.x*d.x+1; A[22]= h*d.x*d.y;   A[23]= h*d.x*d.z;
		A[24]=-h*d.x*d.y;   A[25]=-h*d.y*d.y;   A[26]=-h*d.y*d.z;   A[27]= h*d.x*d.y;   A[28]= h*d.y*d.y+1; A[29]= h*d.y*d.z;
		A[30]=-h*d.x*d.z;   A[31]=-h*d.y*d.z;   A[32]=-h*d.z*d.z;   A[33]= h*d.x*d.z;   A[34]= h*d.y*d.z;   A[35]= h*d.z*d.z+1;
		
		//Store known values in vector b
		b[0] = particle[0]->velocity.x + g*d.x*((p1.x*d.x + p1.y*d.y + p1.z*d.z) - (p0.x*d.x + p0.y*d.y + p0.z*d.z) - length);
		b[1] = particle[0]->velocity.y + g*d.y*((p1.x*d.x + p1.y*d.y + p1.z*d.z) - (p0.x*d.x + p0.y*d.y + p0.z*d.z) - length);
		b[2] = particle[0]->velocity.z + g*d.z*((p1.x*d.x + p1.y*d.y + p1.z*d.z) - (p0.x*d.x + p0.y*d.y + p0.z*d.z) - length);
		b[3] = particle[1]->velocity.x + g*d.x*((p0.x*d.x + p0.y*d.y + p0.z*d.z) - (p1.x*d.x + p1.y*d.y + p1.z*d.z) - length);
		b[4] = particle[1]->velocity.y + g*d.y*((p0.x*d.x + p0.y*d.y + p0.z*d.z) - (p1.x*d.x + p1.y*d.y + p1.z*d.z) - length);
		b[5] = particle[1]->velocity.z + g*d.z*((p0.x*d.x + p0.y*d.y + p0.z*d.z) - (p1.x*d.x + p1.y*d.y + p1.z*d.z) - length);
		
		//Predict a solution and store in x
		x[0] = f * d.x * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt * (vn.x*d.x + vn.y*d.y + vn.z*d.z));
		x[1] = f * d.y * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt * (vn.x*d.x + vn.y*d.y + vn.z*d.z));
		x[2] = f * d.z * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt * (vn.x*d.x + vn.y*d.y + vn.z*d.z));
		x[3] = f * d.x * (-xn.x*d.x - xn.y*d.y - xn.z*d.z - length - dt * (vn.x*d.x + vn.y*d.y + vn.z*d.z));
		x[4] = f * d.y * (-xn.x*d.x - xn.y*d.y - xn.z*d.z - length - dt * (vn.x*d.x + vn.y*d.y + vn.z*d.z));
		x[5] = f * d.z * (-xn.x*d.x - xn.y*d.y - xn.z*d.z - length - dt * (vn.x*d.x + vn.y*d.y + vn.z*d.z));
		
		Vector3f v1(x[3]-x[0],x[4]-x[1],x[5]-x[2]);
		
		force += d * (f * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt*(v1.x*d.x + v1.y*d.y + v1.z*d.z)));
		
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
	
	void Spring::release()
	{
		particle[0] = NULL;
		particle[1] = NULL;
		
		delete [] particle;
	}
	
////////////////////////////// Strand Class ////////////////////////////////////
	
	Strand::Strand(int numParticles, float mass, float k, float length, Vector3f root=Vector3f())
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
		
//		std::cout << "particles init" << std::endl;
		
		buildSprings(k, length, 0.05f);
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
			particle[i]->force += force;
		}
	}
	
	void Strand::applyStrainLimiting(float dt)
	{
		//TODO
		dt++;
	}
	
	void Strand::update(float dt)
	{
		clearForces();
		
		updateSprings1(dt);
		
		//Apply gravity
		applyForce(Vector3f(0.0f, -9.8f, 0.0f));
		
		updateParticles1(dt);
		
		updateSprings2(dt);
		
		//Apply gravity
		applyForce(Vector3f(0.0f, -9.8f, 0.0f));
		
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
	
	Hair::Hair(int numStrands, float mass, float k, float length, std::vector<Vector3f> &roots)
	{
		this->numStrands = numStrands;
		
		strand = new Strand*[numStrands];
		
		for(int i = 0; i < numStrands; i++)
		{
			strand[i] = new Strand(50, mass, k, length, roots[i]);
//			std::cout << i << std::endl;
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

