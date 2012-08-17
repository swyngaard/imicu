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
			
			//Use previous position in current calculations
			pos = position;
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
			
			//Use half position in current calculations
			pos = posh;
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
//		Vector3f vn = particle[1]->velocity - particle[0]->velocity;
		Vector3f d = xn * xn.length_inverse();
		
		//Calculate velocity
		float f = k / length * particle[0]->mass;
		
		Vector3f v1(particle[1]->velc.x-particle[0]->velc.x, particle[1]->velc.y-particle[0]->velc.y, particle[1]->velc.z-particle[0]->velc.z);
		
//		force += d * (f * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt*(v1.x*d.x + v1.y*d.y + v1.z*d.z)));
		
		force.y += f * (xn.y * d.y - length) * d.y + dt * f * (v1.y * d.y) * d.y;
		
//		float damp = (dt * k / (length + damping))/particle[0]->mass;
//		Vector3f friction(damp, damp, damp);		
//		force += friction;
		
//		particle[0]->applyForce(force*-1.0f); //pull top particle down
//		particle[1]->applyForce(force); //pull bottom particle up
		particle[0]->applyForce(force);
		particle[1]->applyForce(force*-1.0f);
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
		
		particle = new Particle*[numParticles];
		
		for(int i = 0; i < numParticles; i++)
		{
			particle[i] = new Particle(mass);
			particle[i]->position = Vector3f(0.0f, (i+1.0f)*(-length/2.0f), 0.0f);
			particle[i]->posc = particle[i]->position;
			particle[i]->pos = particle[i]->position;
		}
		
//		buildSprings();		
	}
	
	//Build the three types of spring connections between particles
	void Strand::buildSprings()
	{
//		edge = new Spring*[numEdges];
		
//		for(int i = 0; i < numEdges; i++)
//		{
//			edge[i] = new Spring(particle[i], particle[i+1], k_edge, length, d_edge, EDGE);
//			std::cout << "edge: " << i+1 << " & " << i+2 << std::endl;
//		}
		
//		bend = new Spring*[numBend];
		
//		for(int i = 0; i < numBend; i++)
//		{
//			bend[i] = new Spring(particle[i], particle[i+2], k_bend, length, d_bend, BEND);
//			std::cout << "bend: " << i+1 << " & " << i+3 << std::endl;
//		}
		
//		twist = new Spring*[numTwist];
		
//		for(int i = 0; i < numTwist; i++)
//		{
//			twist[i] = new Spring(particle[i], particle[i+3], k_twist, length, d_twist, TWIST);
//			std::cout << "twist: " << i+1 << " & " << i+4 << std::endl;
//		}
		
		
		//TODO add extra springs
	}
	
	void Strand::clearForces()
	{
		for(int i = 0; i < numParticles; i++)
		{
			particle[i]->clearForces();
		}
	}
	
	float Strand::getA(int i, int j, float dt)
	{
		int N = numParticles;
		
		if(i == j)
		{
			float h = dt*dt*k_edge/(4.0f*mass*length);
			float d_above = (i == 0) ? -particle[i]->position.y/fabs(-particle[i]->position.y) : (particle[i-1]->position.y-particle[i]->position.y)/fabs(particle[i-1]->position.y-particle[i]->position.y);
			float d_below = (i == (N-1)) ? 0.0f : (particle[i+1]->position.y-particle[i]->position.y)/fabs(particle[i+1]->position.y-particle[i]->position.y);
			
			return 1.0f + h*d_above*d_above + h*d_below*d_below;
		}
		else if(i != 0 && (i - j) == 1)
		{
			float h = dt*dt*k_edge/(4.0f*mass*length);
			float d_above = (particle[i-1]->position.y-particle[i]->position.y)/fabs(particle[i-1]->position.y-particle[i]->position.y);
			
			return -h*d_above*d_above;
		}
		else if(i != (N-1) && (i - j) == -1)
		{
			float h = dt*dt*k_edge/(4.0f*mass*length);
			float d_below = (particle[i+1]->position.y-particle[i]->position.y)/fabs(particle[i+1]->position.y-particle[i]->position.y);
			
			return -h*d_below*d_below;
		}
		
		return 0.0f;
	}
	
	void Strand::calcVelocities(float dt)
	{
		//Calculate the velocities of each particle
		float g = dt*k_edge/(2.0f*mass*length);
		float h = dt*dt*k_edge/(4.0f*mass*length);
		
		float x0 = particle[0]->pos.y;
		float x1 = particle[1]->pos.y;
		float x2 = particle[2]->pos.y;
		
		float v0 = particle[0]->velocity.y;
		float v1 = particle[1]->velocity.y;
		float v2 = particle[2]->velocity.y;
		
		float d0 = -x0/fabs(x0);
		float d01 = (x1-x0)/fabs(x1-x0);
		float d10 = (x0-x1)/fabs(x0-x1);
		float d12 = (x2-x1)/fabs(x2-x1);
		float d21 = (x1-x2)/fabs(x1-x2);
		
		int n = numParticles;
		
		float* bb = new float[n];
		float* xx = new float[n];
		
		bb[0] = v0 + g*(-x0*d0-length)*d0 + g*((x1-x0)*d01-length)*d01;
		bb[1] = v1 + g*((x0-x1)*d10-length)*d10 + g*((x2-x1)*d12-length)*d12;
		bb[2] = v2 + g*((x1-x2)*d21-length)*d21;
		
		xx[0] = v0;
		xx[1] = v1;
		xx[2] = v2;
		
		conjugate(bb, xx, dt);
		
		//~ float a  = 1 + h*d0*d0 + h*d01*d01;
		//~ float b  = -h*d01*d01;
		//~ float d  = -h*d10*d10;
		//~ float e  = 1 + h*d10*d10 + h*d12*d12;
		//~ float f  = -h*d12*d12;
		//~ float hh = -h*d21*d21;
		//~ float k  = 1 + h*d21*d21;
		//~ float det = a*(e*k-f*hh) - b*(k*d);
		
		particle[0]->velh.y = xx[0];
		particle[1]->velh.y = xx[1];
		particle[2]->velh.y = xx[2];
		
		//~ particle[0]->velh.y = (bb[0]*(e*k-f*hh) + bb[1]*(-b*k)  + bb[2]*(b*f)    )/det;
		//~ particle[1]->velh.y = (bb[0]*(-d*k)     + bb[1]*(a*k)   + bb[2]*(-a*f)   )/det;
		//~ particle[2]->velh.y = (bb[0]*(d*hh)     + bb[1]*(-a*hh) + bb[2]*(a*e-b*d))/det;
		
		delete [] bb;
		delete [] xx;
	}
	
	void Strand::updateSprings(float dt)
	{
		float x0 = particle[0]->pos.y;
		float x1 = particle[1]->pos.y;
		float x2 = particle[2]->pos.y;
		
		float v0 = particle[0]->velh.y;
		float v1 = particle[1]->velh.y;
		float v2 = particle[2]->velh.y;
		
		float d0 = -x0/fabs(x0);
		float d01 = (x1-x0)/fabs(x1-x0);
		float d10 = (x0-x1)/fabs(x0-x1);
		float d12 = (x2-x1)/fabs(x2-x1);
		float d21 = (x1-x2)/fabs(x1-x2);
		
		float g = k_edge/length;
		float h = dt*k_edge/(2.0f*length);
		
		float force0 = g*(-x0*d0-length)*d0 - h*v0*d0*d0 + g*((x1-x0)*d01-length)*d01 + h*(v1-v0)*d01*d01;
		float force1 = g*((x0-x1)*d10-length)*d10 + h*(v0-v1)*d10*d10 + g*((x2-x1)*d12-length)*d12 + h*(v2-v1)*d12*d12;
		float force2 = g*((x1-x2)*d21-length)*d21 + h*(v1-v2)*d21*d21;
		
		particle[0]->applyForce(Vector3f(0.0f, force0, 0.0f));
		particle[1]->applyForce(Vector3f(0.0f, force1, 0.0f));
		particle[2]->applyForce(Vector3f(0.0f, force2, 0.0f));
	}
	
	void Strand::updateVelocities(float dt)
	{
		for(int i = 0; i < numParticles; i++)
		{
			particle[i]->updateVelocity(dt);
		}
	}
	
	void Strand::updatePositions(float dt)
	{
		for(int i = 0; i < numParticles; i++)
		{
			particle[i]->updatePosition(dt);
		}
	}
	
	void Strand::updateParticles(float dt)
	{
		for(int i = 0; i < numParticles; i++)
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
	
	void Strand::update(float dt, const float (&grid)[DOMAIN_DIM][DOMAIN_DIM][DOMAIN_DIM])
	{
		//Reset forces on particles
		clearForces();
		
		//Calculate candidate velocities
		calcVelocities(dt);
		
		//Calculate and apply spring forces using previous position
		updateSprings(dt);
		
		//Apply gravity
		applyForce(Vector3f(0.0f, mass*GRAVITY, 0.0f));
		
		//Calculate half velocities using forces
		updateVelocities(dt);
		
//		applyStrainLimiting(dt);
		
		//Stiction calculations
		
		//Calculate half position and new position
		updatePositions(dt);
		
		//Check geometry collisions and adjust velocities and positions
//		objectCollisions(grid);
		
		//Self collisions
		
		//Reset forces on particles
		clearForces();
		
		//Calculate velocities using half position
		calcVelocities(dt);
		
		//Calculate and apply spring forces using half position
		updateSprings(dt);
		
		//Apply gravity
		applyForce(Vector3f(0.0f, mass*GRAVITY, 0.0f));
		
		//Calculate half velocity and new velocity
		updateParticles(dt);
		
		//Self Collisions
	}
	
	void Strand::conjugate(const float* b, float* x, float dt)
	{
		int N = numParticles;
		float r[N];
		float p[N];

		for(int i = 0; i < N; i++)
		{
			//r = b - Ax
			r[i] = b[i];
			for(int j = 0; j < N; j++)
			{
				r[i] -= getA(i,j,dt)*x[j];
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
					Ap[j] += getA(j,k,dt) * p[k];
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
	
	void Strand::objectCollisions(const float (&grid)[DOMAIN_DIM][DOMAIN_DIM][DOMAIN_DIM])
	{
		//Transform particle coordinates to collision grid coordinates
		Vector3f position;
		position.x = (particle[25]->position.x + DOMAIN_HALF-CELL_HALF)/CELL_WIDTH;
		position.y = (particle[25]->position.y + DOMAIN_HALF+0.125f-CELL_HALF)/CELL_WIDTH;
		position.z = (particle[25]->position.z + DOMAIN_HALF-CELL_HALF)/CELL_WIDTH;
		
		Vector3i min (int(position.x), int(position.y), int(position.z));
		Vector3i max (min.x+1, min.y+1, min.z+1);
		
		float v000 = grid[min.x][min.y][min.z];
		float v100 = grid[max.x][min.y][min.z];
		float v001 = grid[min.x][min.y][max.z];
		float v101 = grid[max.x][min.y][max.z];
		float v010 = grid[min.x][max.y][min.z];
		float v110 = grid[max.x][max.y][min.z];
		float v011 = grid[min.x][max.y][max.z];
		float v111 = grid[max.x][max.y][max.z];
		
		Vector3f d;
		d.x = (position.x - min.x)/(max.x - min.x);
		d.y = (position.y - min.y)/(max.y - min.y);
		d.z = (position.z - min.z)/(max.z - min.z);
		
		float c0 = v000 * (1 - d.x) + v100 * d.x;
		float c1 = v001 * (1 - d.x) + v101 * d.x;
		float c2 = v010 * (1 - d.x) + v110 * d.x;
		float c3 = v011 * (1 - d.x) + v111 * d.x;
		
		float c00 = c0 * (1 - d.z) + c1 * d.z;
		float c11 = c2 * (1 - d.z) + c3 * d.z;
		
		float c000 = c00 * (1 - d.y) + c11 * d.y;
		
		//Check for surface collision
		if(c000 < 0.002f)
		{
			//Calculate normal
			Vector3f normal;
			
			normal.x = -v000*d.y*d.z + v001*d.y*d.z + v010*d.y*d.z - v011*d.y*d.z + v100*d.y*d.z - v101*d.y*d.z - v110*d.y*d.z + v111*d.y*d.z + v000*d.y + v000*d.z - v001*d.y - v010*d.z - v100*d.y - v100*d.z + v101*d.y + v110*d.z - v000 + v100;
			normal.y = -v000*d.x*d.z + v001*d.x*d.z + v010*d.x*d.z - v011*d.x*d.z + v100*d.x*d.z - v101*d.x*d.z - v110*d.x*d.z + v111*d.x*d.z + v000*d.x + v000*d.z - v001*d.x - v001*d.z - v010*d.z + v011*d.z - v100*d.x + v101*d.x - v000 + v001;
			normal.z = -v000*d.x*d.y + v001*d.x*d.y + v010*d.x*d.y - v011*d.x*d.y + v100*d.x*d.y - v101*d.x*d.y - v110*d.x*d.y + v111*d.x*d.y + v000*d.x + v000*d.y - v001*d.y - v010*d.x - v010*d.y + v011*d.y - v100*d.x + v110*d.x - v000 + v010;
			
			//Normalise
			normal.unitize();
			
//			particle[25]->freeze = true;			
						
//			particle[25]->velc = Vector3f();
//			particle[25]->velh = Vector3f();
//			particle[25]->velocity = Vector3f();
//			particle[25]->posc = Vector3f();
//			particle[25]->posh = Vector3f();
//			particle[25]->position = Vector3f();
			
			//Move particle to surface of mesh
			
			//Change direction of velocity
			
		}
		
	}
	
	//Clean up
	void Strand::release()
	{	
		//Edge springs
//		for(int i = 0; i < numEdges; i++)
//		{
//			edge[i]->release();
//			delete edge[i];
//			edge[i] = NULL;
//		}
//		delete [] edge;
		
		
		//Bending springs
//		for(int i = 0; i < numBend; i++)
//		{
//			bend[i]->release();
//			delete bend[i];
//			bend[i] = NULL;
//		}
//		delete [] bend;
		
		//Torsion springs
//		for(int i = 0; i < numTwist; i++)
//		{
//			twist[i]->release();
//			delete twist[i];
//			twist[i] = NULL;
//		}
//		delete [] twist;
		
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
			strand[i]->update(dt, grid);
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

