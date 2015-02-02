#include "hair.h"
#include "constants.h"
#include "tree.h"

#include <iostream>
#include <cstring>
#include <cfloat>
#include <cmath>
#include <ctime>
#include <cstdlib>

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
		//Extrapolate new velocity
		velocity = velh * 2 - velocity;
		
		//Use previous position in current calculations
		pos = position;
	}
	
	void Particle::updateVelocity(float dt)
	{
		//Calculate half velocity
		velh = velocity + force * (dt / 2.0f);
	}
	
	void Particle::updatePosition(float dt)
	{
		//Save old position
		poso = position;
		
		//Calculate new position
		position = poso + velh * dt;
		
		//Calculate half position
		posh = (poso + position)/2.0f;
		
		//Use half position in current calculations
		pos = posh;
	}
	
////////////////////////////// Strand Class ////////////////////////////////////
	
	Strand::Strand(int numParticles,
				   int strandID,
				   int numStrands,
				   float mass,
				   float k_edge,
				   float k_bend,
				   float k_twist,
				   float k_extra,
				   float d_edge,
				   float d_bend,
				   float d_twist,
				   float d_extra,
				   float length_e,
				   float length_b,
				   float length_t,
				   Vector3f root,
				   Vector3f normal)
	{
		numEdges = numParticles - 1;
		numBend  = numParticles - 2;
		numTwist = numParticles - 3;
		
		this->numParticles = numParticles;
		
		//Unique strand ID number
		this->strandID = strandID;
		
		//Total number of strands
		this->numStrands = numStrands;
		
		this->k_edge = k_edge;
		this->k_bend = k_bend;
		this->k_twist = k_twist;
		this->k_extra = k_extra;
		
		this->d_edge = d_edge;
		this->d_bend = d_bend;
		this->d_twist = d_twist;
		this->d_extra = d_extra;
		
		this->length_e = length_e;
		this->length_b = length_b;
		this->length_t = length_t;
		
		this->mass  = mass;
		
		this->root = root;
		
		normal.unitize();
		
		xx = new float[numParticles*NUMCOMPONENTS];
		bb = new float[numParticles*NUMCOMPONENTS];
		AA = new float[numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS];
		
		std::list<Node*> leafNodes;
		
		particle = new Particle*[numParticles];
		
		for(int i = 0; i < numParticles; i++)
		{
			particle[i] = new Particle(mass);
			particle[i]->position = root;
			
			//Project next particle given distance from root or previous particle along normal direction
			particle[i]->position += (normal*(0.025f*(i+1)));
			particle[i]->posc = particle[i]->position;
			particle[i]->pos = particle[i]->position;
			
			//Create KDOP and Node for every two particles and build leaf node
			if(i > 0)
			{
				//Create list of vertices that comprise the strand segment
				std::vector<Vector3f> vertices;
				vertices.push_back(particle[i-1]->position);
				vertices.push_back(particle[i]->position);
				
				//Create additional ghost vertices to approximate a cylindrical volume
				addVolumeVertices(vertices);
				
				//Create KDOP bounding volume from the two vertices
				KDOP* kdop = new KDOP(vertices, KDOP_PLANES);
				
				//Build leaf node and KDOP from vertices then add the node to the list of leaves
				//Each leaf node has its depth set to 0 and is assigned a unique identity number
				leafNodes.push_back(new Node(kdop, 0, i-1, strandID));
				
				//Save the KDOP to a list of leaf KDOPs for later updating
				leafKDOP.push_back(kdop);
			}
		}
		
		//Create BVH Tree and save root pointer
		bvhTree = Node::buildTree(leafNodes);	
	}
	
	void Strand::clearForces()
	{
		for(int i = 0; i < numParticles; i++)
		{
			particle[i]->clearForces();
		}
	}
	
	void Strand::conjugate()
	{
		int N = numParticles*NUMCOMPONENTS;
		float r[N];
		float p[N];

		for(int i = 0; i < N; i++)
		{
			//r = b - Ax
			//~ r[i] = getB(i,dt);
			r[i] = bb[i];
			for(int j = 0; j < N; j++)
			{
				r[i] -= AA[i*N+j]*xx[j];
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
					Ap[j] += AA[j*N+k] * p[k];
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
				xx[j] = xx[j] + alpha * p[j];
			
				r[j] = r[j] - alpha * Ap[j];
			}
		
			float rsnew = 0.0f;
		
			for(int j = 0; j < N; j++)
			{
				rsnew += r[j] * r[j];
			}
		
			if(rsnew < 1e-10f)
			{
				break;
			}
			
			for(int j = 0; j < N; j++)
			{
				p[j] = r[j] + rsnew / rsold * p[j];
			}
		
			rsold = rsnew;
		}
	}
	
	void Strand::buildAB(float dt)
	{
		memset(AA, 0, sizeof(float)*numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS);
		
		Vector3f gravity(0.0f, GRAVITY, 0.0f);
		
		//Set the 6 different coefficients
		float h_e = dt*dt*k_edge/(4.0f*mass*length_e) + d_edge*dt/(2.0f*mass);
		float h_b = dt*dt*k_bend/(4.0f*mass*length_b) + d_bend*dt/(2.0f*mass);
		float h_t = dt*dt*k_twist/(4.0f*mass*length_t) + d_twist*dt/(2.0f*mass);
		float g_e = dt*k_edge/(2.0f*mass*length_e);
		float g_b = dt*k_bend/(2.0f*mass*length_b);
		float g_t = dt*k_twist/(2.0f*mass*length_t);
		
		//Set the first 3 particle direction vectors
		
		//First particle direction vectors
		Vector3f du0R(root-particle[0]->pos);
		Vector3f du01(particle[1]->pos-particle[0]->pos);
		Vector3f du02(particle[2]->pos-particle[0]->pos);
		Vector3f du03(particle[3]->pos-particle[0]->pos);
		du0R.unitize();
		du01.unitize();
		du02.unitize();
		du03.unitize();
		
		//Second particle direction vectors
		Vector3f du1R(root-particle[1]->pos);
		Vector3f du10(particle[0]->pos-particle[1]->pos);
		Vector3f du12(particle[2]->pos-particle[1]->pos);
		Vector3f du13(particle[3]->pos-particle[1]->pos);
		Vector3f du14(particle[4]->pos-particle[1]->pos);
		du1R.unitize();
		du10.unitize();
		du12.unitize();
		du13.unitize();
		du14.unitize();
		
		//Third particle direction vectors
		Vector3f du2R(root-particle[2]->pos);
		Vector3f du20(particle[0]->pos-particle[2]->pos);
		Vector3f du21(particle[1]->pos-particle[2]->pos);
		Vector3f du23(particle[3]->pos-particle[2]->pos);
		Vector3f du24(particle[4]->pos-particle[2]->pos);
		Vector3f du25(particle[5]->pos-particle[2]->pos);
		du2R.unitize();
		du20.unitize();
		du21.unitize();
		du23.unitize();
		du24.unitize();
		du25.unitize();
		
		//Set the non-zero entries for the first 3 particles
		
		//Set first twelve entries of the first row of A matrix
		AA[0 ] = 1.0f + h_e*du0R.x*du0R.x + h_e*du01.x*du01.x + h_b*du02.x*du02.x + h_t*du03.x*du03.x;
		AA[1 ] =        h_e*du0R.x*du0R.y + h_e*du01.x*du01.y + h_b*du02.x*du02.y + h_t*du03.x*du03.y;
		AA[2 ] =        h_e*du0R.x*du0R.z + h_e*du01.x*du01.z + h_b*du02.x*du02.z + h_t*du03.x*du03.z;
		AA[3 ] = -h_e*du01.x*du01.x;
		AA[4 ] = -h_e*du01.x*du01.y;
		AA[5 ] = -h_e*du01.x*du01.z;
		AA[6 ] = -h_b*du02.x*du02.x;
		AA[7 ] = -h_b*du02.x*du02.y;
		AA[8 ] = -h_b*du02.x*du02.z;
		AA[9 ] = -h_t*du03.x*du03.x;
		AA[10] = -h_t*du03.x*du03.y;
		AA[11] = -h_t*du03.x*du03.z;
		
		//Indices for next second and third rows of A
		int row11 = numParticles * NUMCOMPONENTS;
		int row22 = 2 * numParticles * NUMCOMPONENTS;
		
		//Set next twelve non-zero entries of the second row of matrix A
		AA[row11   ] =        h_e*du0R.x*du0R.y + h_e*du01.x*du01.y + h_b*du02.x*du02.y + h_t*du03.x*du03.y;
		AA[row11+1 ] = 1.0f + h_e*du0R.y*du0R.y + h_e*du01.y*du01.y + h_b*du02.y*du02.y + h_t*du03.y*du03.y;
		AA[row11+2 ] =        h_e*du0R.y*du0R.z + h_e*du01.y*du01.z + h_b*du02.y*du02.z + h_t*du03.y*du03.z;
		AA[row11+3 ] = -h_e*du01.x*du01.y;
		AA[row11+4 ] = -h_e*du01.y*du01.y;
		AA[row11+5 ] = -h_e*du01.y*du01.z;
		AA[row11+6 ] = -h_b*du02.x*du02.y;
		AA[row11+7 ] = -h_b*du02.y*du02.y;
		AA[row11+8 ] = -h_b*du02.y*du02.z;
		AA[row11+9 ] = -h_t*du03.x*du03.y;
		AA[row11+10] = -h_t*du03.y*du03.y;
		AA[row11+11] = -h_t*du03.y*du03.z;
		
		//Set the next twelve non-zero entries of the third row of matrix A
		AA[row22   ] =        h_e*du0R.x*du0R.z + h_e*du01.x*du01.z + h_b*du02.x*du02.z + h_t*du03.x*du03.z;
		AA[row22+1 ] =        h_e*du0R.y*du0R.z + h_e*du01.y*du01.z + h_b*du02.y*du02.z + h_t*du03.y*du03.z;
		AA[row22+2 ] = 1.0f + h_e*du0R.z*du0R.z + h_e*du01.z*du01.z + h_b*du02.z*du02.z + h_t*du03.z*du03.z;
		AA[row22+3 ] = -h_e*du01.x*du01.z;
		AA[row22+4 ] = -h_e*du01.y*du01.z;
		AA[row22+5 ] = -h_e*du01.z*du01.z;
		AA[row22+6 ] = -h_b*du02.x*du02.z;
		AA[row22+7 ] = -h_b*du02.y*du02.z;
		AA[row22+8 ] = -h_b*du02.z*du02.z;
		AA[row22+9 ] = -h_t*du03.x*du03.z;
		AA[row22+10] = -h_t*du03.y*du03.z;
		AA[row22+11] = -h_t*du03.z*du03.z;
		
		int row33 = 3 * numParticles * NUMCOMPONENTS;
		int row44 = 4 * numParticles * NUMCOMPONENTS;
		int row55 = 5 * numParticles * NUMCOMPONENTS;
		
		AA[row33   ] = -h_e*du10.x*du10.x;
		AA[row33+1 ] = -h_e*du10.x*du10.y;
		AA[row33+2 ] = -h_e*du10.x*du10.z;
		AA[row33+3 ] = 1.0f + h_b*du1R.x*du1R.x + h_e*du10.x*du10.x + h_e*du12.x*du12.x + h_b*du13.x*du13.x + h_t*du14.x*du14.x;
		AA[row33+4 ] =        h_b*du1R.x*du1R.y + h_e*du10.x*du10.y + h_e*du12.x*du12.y + h_b*du13.x*du13.y + h_t*du14.x*du14.y;
		AA[row33+5 ] =        h_b*du1R.x*du1R.z + h_e*du10.x*du10.z + h_e*du12.x*du12.z + h_b*du13.x*du13.z + h_t*du14.x*du14.z;
		AA[row33+6 ] = -h_e*du12.x*du12.x;
		AA[row33+7 ] = -h_e*du12.x*du12.y;
		AA[row33+8 ] = -h_e*du12.x*du12.z;
		AA[row33+9 ] = -h_b*du13.x*du13.x;
		AA[row33+10] = -h_b*du13.x*du13.y;
		AA[row33+11] = -h_b*du13.x*du13.z;
		AA[row33+12] = -h_t*du14.x*du14.x;
		AA[row33+13] = -h_t*du14.x*du14.y;
		AA[row33+14] = -h_t*du14.x*du14.z;
		
		AA[row44   ] = -h_e*du10.x*du10.y;
		AA[row44+1 ] = -h_e*du10.y*du10.y;
		AA[row44+2 ] = -h_e*du10.y*du10.z;
		AA[row44+3 ] =        h_b*du1R.x*du1R.y + h_e*du10.x*du10.y + h_e*du12.x*du12.y + h_b*du13.x*du13.y + h_t*du14.x*du14.y;
		AA[row44+4 ] = 1.0f + h_b*du1R.y*du1R.y + h_e*du10.y*du10.y + h_e*du12.y*du12.y + h_b*du13.y*du13.y + h_t*du14.y*du14.y;
		AA[row44+5 ] =        h_b*du1R.y*du1R.z + h_e*du10.y*du10.z + h_e*du12.y*du12.z + h_b*du13.y*du13.z + h_t*du14.y*du14.z;
		AA[row44+6 ] = -h_e*du12.x*du12.y;
		AA[row44+7 ] = -h_e*du12.y*du12.y;
		AA[row44+8 ] = -h_e*du12.y*du12.z;
		AA[row44+9 ] = -h_b*du13.x*du13.y;
		AA[row44+10] = -h_b*du13.y*du13.y;
		AA[row44+11] = -h_b*du13.y*du13.z;
		AA[row44+12] = -h_t*du14.x*du14.y;
		AA[row44+13] = -h_t*du14.y*du14.y;
		AA[row44+14] = -h_t*du14.y*du14.z;
		
		AA[row55   ] = -h_e*du10.x*du10.z;
		AA[row55+1 ] = -h_e*du10.y*du10.z;
		AA[row55+2 ] = -h_e*du10.z*du10.z;
		AA[row55+3 ] =        h_b*du1R.x*du1R.z + h_e*du10.x*du10.z + h_e*du12.x*du12.z + h_b*du13.x*du13.z + h_t*du14.x*du14.z;
		AA[row55+4 ] =        h_b*du1R.y*du1R.z + h_e*du10.y*du10.z + h_e*du12.y*du12.z + h_b*du13.y*du13.z + h_t*du14.y*du14.z;
		AA[row55+5 ] = 1.0f + h_b*du1R.z*du1R.z + h_e*du10.z*du10.z + h_e*du12.z*du12.z + h_b*du13.z*du13.z + h_t*du14.z*du14.z;
		AA[row55+6 ] = -h_e*du12.x*du12.z;
		AA[row55+7 ] = -h_e*du12.y*du12.z;
		AA[row55+8 ] = -h_e*du12.z*du12.z;
		AA[row55+9 ] = -h_b*du13.x*du13.z;
		AA[row55+10] = -h_b*du13.y*du13.z;
		AA[row55+11] = -h_b*du13.z*du13.z;
		AA[row55+12] = -h_t*du14.x*du14.z;
		AA[row55+13] = -h_t*du14.y*du14.z;
		AA[row55+14] = -h_t*du14.z*du14.z;
		
		int row66 = 6 * numParticles * NUMCOMPONENTS;
		int row77 = 7 * numParticles * NUMCOMPONENTS;
		int row88 = 8 * numParticles * NUMCOMPONENTS;
		
		AA[row66   ] = -h_b*du20.x*du20.x;
		AA[row66+1 ] = -h_b*du20.x*du20.y;
		AA[row66+2 ] = -h_b*du20.x*du20.z;
		AA[row66+3 ] = -h_e*du21.x*du21.x;
		AA[row66+4 ] = -h_e*du21.x*du21.y;
		AA[row66+5 ] = -h_e*du21.x*du21.z;
		AA[row66+6 ] = 1.0f + h_t*du2R.x*du2R.x + h_b*du20.x*du20.x + h_e*du21.x*du21.x + h_e*du23.x*du23.x + h_b*du24.x*du24.x + h_t*du25.x*du25.x;
		AA[row66+7 ] =        h_t*du2R.x*du2R.y + h_b*du20.x*du20.y + h_e*du21.x*du21.y + h_e*du23.x*du23.y + h_b*du24.x*du24.y + h_t*du25.x*du25.y;
		AA[row66+8 ] =        h_t*du2R.x*du2R.z + h_b*du20.x*du20.z + h_e*du21.x*du21.z + h_e*du23.x*du23.z + h_b*du24.x*du24.z + h_t*du25.x*du25.z;
		AA[row66+9 ] = -h_e*du23.x*du23.x;
		AA[row66+10] = -h_e*du23.x*du23.y;
		AA[row66+11] = -h_e*du23.x*du23.z;
		AA[row66+12] = -h_b*du24.x*du24.x;
		AA[row66+13] = -h_b*du24.x*du24.y;
		AA[row66+14] = -h_b*du24.x*du24.z;
		AA[row66+15] = -h_t*du25.x*du25.x;
		AA[row66+16] = -h_t*du25.x*du25.y;
		AA[row66+17] = -h_t*du25.x*du25.z;
		
		
		AA[row77   ] = -h_b*du20.x*du20.y;
		AA[row77+1 ] = -h_b*du20.y*du20.y;
		AA[row77+2 ] = -h_b*du20.y*du20.z;
		AA[row77+3 ] = -h_e*du21.x*du21.y;
		AA[row77+4 ] = -h_e*du21.y*du21.y;
		AA[row77+5 ] = -h_e*du21.y*du21.z;
		AA[row77+6 ] =        h_t*du2R.x*du2R.y + h_b*du20.x*du20.y + h_e*du21.x*du21.y + h_e*du23.x*du23.y + h_b*du24.x*du24.y + h_t*du25.x*du25.y;
		AA[row77+7 ] = 1.0f + h_t*du2R.y*du2R.y + h_b*du20.y*du20.y + h_e*du21.y*du21.y + h_e*du23.y*du23.y + h_b*du24.y*du24.y + h_t*du25.y*du25.y;
		AA[row77+8 ] =        h_t*du2R.y*du2R.z + h_b*du20.y*du20.z + h_e*du21.y*du21.z + h_e*du23.y*du23.z + h_b*du24.y*du24.z + h_t*du25.y*du25.z;
		AA[row77+9 ] = -h_e*du23.x*du23.y;
		AA[row77+10] = -h_e*du23.y*du23.y;
		AA[row77+11] = -h_e*du23.y*du23.z;
		AA[row77+12] = -h_b*du24.x*du24.y;
		AA[row77+13] = -h_b*du24.y*du24.y;
		AA[row77+14] = -h_b*du24.y*du24.z;
		AA[row77+15] = -h_t*du25.x*du25.y;
		AA[row77+16] = -h_t*du25.y*du25.y;
		AA[row77+17] = -h_t*du25.y*du25.z;
		
		AA[row88   ] = -h_b*du20.x*du20.z;
		AA[row88+1 ] = -h_b*du20.y*du20.z;
		AA[row88+2 ] = -h_b*du20.z*du20.z;
		AA[row88+3 ] = -h_e*du21.x*du21.z;
		AA[row88+4 ] = -h_e*du21.y*du21.z;
		AA[row88+5 ] = -h_e*du21.z*du21.z;
		AA[row88+6 ] =        h_t*du2R.x*du2R.z + h_b*du20.x*du20.z + h_e*du21.x*du21.z + h_e*du23.x*du23.z + h_b*du24.x*du24.z + h_t*du25.x*du25.z;
		AA[row88+7 ] =        h_t*du2R.y*du2R.z + h_b*du20.y*du20.z + h_e*du21.y*du21.z + h_e*du23.y*du23.z + h_b*du24.y*du24.z + h_t*du25.y*du25.z;
		AA[row88+8 ] = 1.0f + h_t*du2R.z*du2R.z + h_b*du20.z*du20.z + h_e*du21.z*du21.z + h_e*du23.z*du23.z + h_b*du24.z*du24.z + h_t*du25.z*du25.z;
		AA[row88+9 ] = -h_e*du23.x*du23.z;
		AA[row88+10] = -h_e*du23.y*du23.z;
		AA[row88+11] = -h_e*du23.z*du23.z;
		AA[row88+12] = -h_b*du24.x*du24.z;
		AA[row88+13] = -h_b*du24.y*du24.z;
		AA[row88+14] = -h_b*du24.z*du24.z;
		AA[row88+15] = -h_t*du25.x*du25.z;
		AA[row88+16] = -h_t*du25.y*du25.z;
		AA[row88+17] = -h_t*du25.z*du25.z;
		
		//Set the first nine entries of the b vector
		bb[0] = particle[0]->velocity.x + g_e*((root-particle[0]->pos).dot(du0R)-length_e)*du0R.x + g_e*((particle[1]->pos-particle[0]->pos).dot(du01)-length_e)*du01.x + g_b*((particle[2]->pos-particle[0]->pos).dot(du02)-length_b)*du02.x + g_t*((particle[3]->pos-particle[0]->pos).dot(du03)-length_t)*du03.x + gravity.x*(dt/2.0f);
		bb[1] = particle[0]->velocity.y + g_e*((root-particle[0]->pos).dot(du0R)-length_e)*du0R.y + g_e*((particle[1]->pos-particle[0]->pos).dot(du01)-length_e)*du01.y + g_b*((particle[2]->pos-particle[0]->pos).dot(du02)-length_b)*du02.y + g_t*((particle[3]->pos-particle[0]->pos).dot(du03)-length_t)*du03.y + gravity.y*(dt/2.0f);
		bb[2] = particle[0]->velocity.z + g_e*((root-particle[0]->pos).dot(du0R)-length_e)*du0R.z + g_e*((particle[1]->pos-particle[0]->pos).dot(du01)-length_e)*du01.z + g_b*((particle[2]->pos-particle[0]->pos).dot(du02)-length_b)*du02.z + g_t*((particle[3]->pos-particle[0]->pos).dot(du03)-length_t)*du03.z + gravity.z*(dt/2.0f);
		
		bb[3] = particle[1]->velocity.x + g_b*((root-particle[1]->pos).dot(du1R)-length_b)*du1R.x + g_e*((particle[0]->pos-particle[1]->pos).dot(du10)-length_e)*du10.x + g_e*((particle[2]->pos-particle[1]->pos).dot(du12)-length_e)*du12.x + g_b*((particle[3]->pos-particle[1]->pos).dot(du13)-length_b)*du13.x + g_t*((particle[4]->pos-particle[1]->pos).dot(du14)-length_t)*du14.x + gravity.x*(dt/2.0f);
		bb[4] = particle[1]->velocity.y + g_b*((root-particle[1]->pos).dot(du1R)-length_b)*du1R.y + g_e*((particle[0]->pos-particle[1]->pos).dot(du10)-length_e)*du10.y + g_e*((particle[2]->pos-particle[1]->pos).dot(du12)-length_e)*du12.y + g_b*((particle[3]->pos-particle[1]->pos).dot(du13)-length_b)*du13.y + g_t*((particle[4]->pos-particle[1]->pos).dot(du14)-length_t)*du14.y + gravity.y*(dt/2.0f);
		bb[5] = particle[1]->velocity.z + g_b*((root-particle[1]->pos).dot(du1R)-length_b)*du1R.z + g_e*((particle[0]->pos-particle[1]->pos).dot(du10)-length_e)*du10.z + g_e*((particle[2]->pos-particle[1]->pos).dot(du12)-length_e)*du12.z + g_b*((particle[3]->pos-particle[1]->pos).dot(du13)-length_b)*du13.z + g_t*((particle[4]->pos-particle[1]->pos).dot(du14)-length_t)*du14.z + gravity.z*(dt/2.0f);
		
		bb[6] = particle[2]->velocity.x + g_t*((root-particle[2]->pos).dot(du2R)-length_t)*du2R.x + g_b*((particle[0]->pos-particle[2]->pos).dot(du02)-length_b)*du20.x + g_e*((particle[1]->pos-particle[2]->pos).dot(du21)-length_e)*du21.x + g_e*((particle[3]->pos-particle[2]->pos).dot(du23)-length_e)*du23.x + g_b*((particle[4]->pos-particle[2]->pos).dot(du24)-length_b)*du24.x + g_t*((particle[5]->pos-particle[2]->pos).dot(du25)-length_t)*du25.x + gravity.x*(dt/2.0f);
		bb[7] = particle[2]->velocity.y + g_t*((root-particle[2]->pos).dot(du2R)-length_t)*du2R.y + g_b*((particle[0]->pos-particle[2]->pos).dot(du02)-length_b)*du20.y + g_e*((particle[1]->pos-particle[2]->pos).dot(du21)-length_e)*du21.y + g_e*((particle[3]->pos-particle[2]->pos).dot(du23)-length_e)*du23.y + g_b*((particle[4]->pos-particle[2]->pos).dot(du24)-length_b)*du24.y + g_t*((particle[5]->pos-particle[2]->pos).dot(du25)-length_t)*du25.y + gravity.y*(dt/2.0f);
		bb[8] = particle[2]->velocity.z + g_t*((root-particle[2]->pos).dot(du2R)-length_t)*du2R.z + g_b*((particle[0]->pos-particle[2]->pos).dot(du02)-length_b)*du20.z + g_e*((particle[1]->pos-particle[2]->pos).dot(du21)-length_e)*du21.z + g_e*((particle[3]->pos-particle[2]->pos).dot(du23)-length_e)*du23.z + g_b*((particle[4]->pos-particle[2]->pos).dot(du24)-length_b)*du24.z + g_t*((particle[5]->pos-particle[2]->pos).dot(du25)-length_t)*du25.z + gravity.z*(dt/2.0f);
		
		//Build in-between values of matrix A and vector b
		//Loop from fourth to third last particles only
		for(int i = 3; i < (numParticles-3); i++)
		{
			//Current particle position, particle above and particle below
			Vector3f ui = particle[i]->pos;
			Vector3f uu = particle[i-1]->pos;
			Vector3f ud = particle[i+1]->pos;
			
			//Direction vectors for 3 particles above and below the current particle
			Vector3f uu2 = particle[i-3]->pos;
			Vector3f uu1 = particle[i-2]->pos;
			Vector3f uu0 = particle[i-1]->pos;
			Vector3f ud0 = particle[i+1]->pos;
			Vector3f ud1 = particle[i+2]->pos;
			Vector3f ud2 = particle[i+3]->pos;
			
			Vector3f du2(uu2-ui);
			Vector3f du1(uu1-ui);
			Vector3f du0(uu0-ui);
			Vector3f dd0(ud0-ui);
			Vector3f dd1(ud1-ui);
			Vector3f dd2(ud2-ui);
			du2.unitize();
			du1.unitize();
			du0.unitize();
			dd0.unitize();
			dd1.unitize();
			dd2.unitize();
			
			int row0 = i * numParticles * NUMCOMPONENTS *  NUMCOMPONENTS + (i - 3) * NUMCOMPONENTS;
			int row1 = row0 + numParticles * NUMCOMPONENTS;
			int row2 = row1 + numParticles * NUMCOMPONENTS;
			
			AA[row0   ] = -h_t*du2.x*du2.x;
			AA[row0+1 ] = -h_t*du2.x*du2.y;
			AA[row0+2 ] = -h_t*du2.x*du2.z;
			AA[row0+3 ] = -h_b*du1.x*du1.x;
			AA[row0+4 ] = -h_b*du1.x*du1.y;
			AA[row0+5 ] = -h_b*du1.x*du1.z;
			AA[row0+6 ] = -h_e*du0.x*du0.x;
			AA[row0+7 ] = -h_e*du0.x*du0.y;
			AA[row0+8 ] = -h_e*du0.x*du0.z;
			AA[row0+9 ] = 1.0f + h_t*du2.x*du2.x + h_b*du1.x*du1.x + h_e*du0.x*du0.x + h_e*dd0.x*dd0.x + h_b*dd1.x*dd1.x + h_t*dd2.x*dd2.x;
			AA[row0+10] =        h_t*du2.x*du2.y + h_b*du1.x*du1.y + h_e*du0.x*du0.y + h_e*dd0.x*dd0.y + h_b*dd1.x*dd1.y + h_t*dd2.x*dd2.y;
			AA[row0+11] =        h_t*du2.x*du2.z + h_b*du1.x*du1.z + h_e*du0.x*du0.z + h_e*dd0.x*dd0.z + h_b*dd1.x*dd1.z + h_t*dd2.x*dd2.z;
			AA[row0+12] = -h_e*dd0.x*dd0.x;
			AA[row0+13] = -h_e*dd0.x*dd0.y;
			AA[row0+14] = -h_e*dd0.x*dd0.z;
			AA[row0+15] = -h_b*dd1.x*dd1.x;
			AA[row0+16] = -h_b*dd1.x*dd1.y;
			AA[row0+17] = -h_b*dd1.x*dd1.z;
			AA[row0+18] = -h_t*dd2.x*dd2.x;
			AA[row0+19] = -h_t*dd2.x*dd2.y;
			AA[row0+20] = -h_t*dd2.x*dd2.z;
			
			AA[row1   ] = -h_t*du2.x*du2.y;
			AA[row1+1 ] = -h_t*du2.y*du2.y;
			AA[row1+2 ] = -h_t*du2.y*du2.z;
			AA[row1+3 ] = -h_b*du1.x*du1.y;
			AA[row1+4 ] = -h_b*du1.y*du1.y;
			AA[row1+5 ] = -h_b*du1.y*du1.z;
			AA[row1+6 ] = -h_e*du0.x*du0.y;
			AA[row1+7 ] = -h_e*du0.y*du0.y;
			AA[row1+8 ] = -h_e*du0.y*du0.z;
			AA[row1+9 ] =        h_t*du2.x*du2.y + h_b*du1.x*du1.y + h_e*du0.x*du0.y + h_e*dd0.x*dd0.y + h_b*dd1.x*dd1.y + h_t*dd2.x*dd2.y;
			AA[row1+10] = 1.0f + h_t*du2.y*du2.y + h_b*du1.y*du1.y + h_e*du0.y*du0.y + h_e*dd0.y*dd0.y + h_b*dd1.y*dd1.y + h_t*dd2.y*dd2.y;
			AA[row1+11] =        h_t*du2.y*du2.z + h_b*du1.y*du1.z + h_e*du0.y*du0.z + h_e*dd0.y*dd0.z + h_b*dd1.y*dd1.z + h_t*dd2.y*dd2.z;
			AA[row1+12] = -h_e*dd0.x*dd0.y;
			AA[row1+13] = -h_e*dd0.y*dd0.y;
			AA[row1+14] = -h_e*dd0.y*dd0.z;
			AA[row1+15] = -h_b*dd1.x*dd1.y;
			AA[row1+16] = -h_b*dd1.y*dd1.y;
			AA[row1+17] = -h_b*dd1.y*dd1.z;
			AA[row1+18] = -h_t*dd2.x*dd2.y;
			AA[row1+19] = -h_t*dd2.y*dd2.y;
			AA[row1+20] = -h_t*dd2.y*dd2.z;
			
			AA[row2   ] = -h_t*du2.x*du2.z;
			AA[row2+1 ] = -h_t*du2.y*du2.z;
			AA[row2+2 ] = -h_t*du2.z*du2.z;
			AA[row2+3 ] = -h_b*du1.x*du1.z;
			AA[row2+4 ] = -h_b*du1.y*du1.z;
			AA[row2+5 ] = -h_b*du1.z*du1.z;
			AA[row2+6 ] = -h_e*du0.x*du0.z;
			AA[row2+7 ] = -h_e*du0.y*du0.z;
			AA[row2+8 ] = -h_e*du0.z*du0.z;
			AA[row2+9 ] =        h_t*du2.x*du2.z + h_b*du1.x*du1.z + h_e*du0.x*du0.z + h_e*dd0.x*dd0.z + h_b*dd1.x*dd1.z + h_t*dd2.x*dd2.z;
			AA[row2+10] =        h_t*du2.y*du2.z + h_b*du1.y*du1.z + h_e*du0.y*du0.z + h_e*dd0.y*dd0.z + h_b*dd1.y*dd1.z + h_t*dd2.y*dd2.z;
			AA[row2+11] = 1.0f + h_t*du2.z*du2.z + h_b*du1.z*du1.z + h_e*du0.z*du0.z + h_e*dd0.z*dd0.z + h_b*dd1.z*dd1.z + h_t*dd2.z*dd2.z;
			AA[row2+12] = -h_e*dd0.x*dd0.z;
			AA[row2+13] = -h_e*dd0.y*dd0.z;
			AA[row2+14] = -h_e*dd0.z*dd0.z;
			AA[row2+15] = -h_b*dd1.x*dd1.z;
			AA[row2+16] = -h_b*dd1.y*dd1.z;
			AA[row2+17] = -h_b*dd1.z*dd1.z;
			AA[row2+18] = -h_t*dd2.x*dd2.z;
			AA[row2+19] = -h_t*dd2.y*dd2.z;
			AA[row2+20] = -h_t*dd2.z*dd2.z;
			
			bb[i*NUMCOMPONENTS  ] = particle[i]->velocity.x + g_t*((uu2-ui).dot(du2)-length_t)*du2.x + g_b*((uu1-ui).dot(du1)-length_b)*du1.x + g_e*((uu0-ui).dot(du0)-length_e)*du0.x + g_e*((ud0-ui).dot(dd0)-length_e)*dd0.x + g_b*((ud1-ui).dot(dd1)-length_b)*dd1.x + g_t*((ud2-ui).dot(dd2)-length_t)*dd2.x + gravity.x*(dt/2.0f);
			bb[i*NUMCOMPONENTS+1] = particle[i]->velocity.y + g_t*((uu2-ui).dot(du2)-length_t)*du2.y + g_b*((uu1-ui).dot(du1)-length_b)*du1.y + g_e*((uu0-ui).dot(du0)-length_e)*du0.y + g_e*((ud0-ui).dot(dd0)-length_e)*dd0.y + g_b*((ud1-ui).dot(dd1)-length_b)*dd1.y + g_t*((ud2-ui).dot(dd2)-length_t)*dd2.y + gravity.y*(dt/2.0f);
			bb[i*NUMCOMPONENTS+2] = particle[i]->velocity.z + g_t*((uu2-ui).dot(du2)-length_t)*du2.z + g_b*((uu1-ui).dot(du1)-length_b)*du1.z + g_e*((uu0-ui).dot(du0)-length_e)*du0.z + g_e*((ud0-ui).dot(dd0)-length_e)*dd0.z + g_b*((ud1-ui).dot(dd1)-length_b)*dd1.z + g_t*((ud2-ui).dot(dd2)-length_t)*dd2.z + gravity.z*(dt/2.0f);
		}
		
		//Calculate direction vectors for the last three particles
		//Third to last particle direction vectors
		Vector3f du3N1(particle[numParticles-6]->pos-particle[numParticles-3]->pos);
		Vector3f du3N2(particle[numParticles-5]->pos-particle[numParticles-3]->pos);
		Vector3f du3N3(particle[numParticles-4]->pos-particle[numParticles-3]->pos);
		Vector3f du3N5(particle[numParticles-2]->pos-particle[numParticles-3]->pos);
		Vector3f du3N6(particle[numParticles-1]->pos-particle[numParticles-3]->pos);
		du3N1.unitize();
		du3N2.unitize();
		du3N3.unitize();
		du3N5.unitize();
		du3N6.unitize();
		
		//Second to last particle direction vectors
		Vector3f du2N2(particle[numParticles-5]->pos-particle[numParticles-2]->pos);
		Vector3f du2N3(particle[numParticles-4]->pos-particle[numParticles-2]->pos);
		Vector3f du2N4(particle[numParticles-3]->pos-particle[numParticles-2]->pos);
		Vector3f du2N6(particle[numParticles-1]->pos-particle[numParticles-2]->pos);
		du2N2.unitize();
		du2N3.unitize();
		du2N4.unitize();
		du2N6.unitize();
		
		//Last particle direction vectors
		Vector3f du1N3(particle[numParticles-4]->pos-particle[numParticles-1]->pos);
		Vector3f du1N4(particle[numParticles-3]->pos-particle[numParticles-1]->pos);
		Vector3f du1N5(particle[numParticles-2]->pos-particle[numParticles-1]->pos);
		du1N3.unitize();
		du1N4.unitize();
		du1N5.unitize();
		
		int row3N3 = numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS - 8*numParticles*NUMCOMPONENTS - 18;
		int row3N2 = numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS - 7*numParticles*NUMCOMPONENTS - 18;
		int row3N1 = numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS - 6*numParticles*NUMCOMPONENTS - 18;
		
		AA[row3N3   ] = -h_t*du3N1.x*du3N1.x;
		AA[row3N3+1 ] = -h_t*du3N1.x*du3N1.y;
		AA[row3N3+2 ] = -h_t*du3N1.x*du3N1.z;
		AA[row3N3+3 ] = -h_b*du3N2.x*du3N2.x;
		AA[row3N3+4 ] = -h_b*du3N2.x*du3N2.y;
		AA[row3N3+5 ] = -h_b*du3N2.x*du3N2.z;
		AA[row3N3+6 ] = -h_e*du3N3.x*du3N3.x;
		AA[row3N3+7 ] = -h_e*du3N3.x*du3N3.y;
		AA[row3N3+8 ] = -h_e*du3N3.x*du3N3.z;
		AA[row3N3+9 ] = 1.0f + h_t*du3N1.x*du3N1.x + h_b*du3N2.x*du3N2.x + h_e*du3N3.x*du3N3.x + h_e*du3N5.x*du3N5.x + h_b*du3N6.x*du3N6.x;
		AA[row3N3+10] =        h_t*du3N1.x*du3N1.y + h_b*du3N2.x*du3N2.y + h_e*du3N3.x*du3N3.y + h_e*du3N5.x*du3N5.y + h_b*du3N6.x*du3N6.y;
		AA[row3N3+11] =        h_t*du3N1.x*du3N1.z + h_b*du3N2.x*du3N2.z + h_e*du3N3.x*du3N3.z + h_e*du3N5.x*du3N5.z + h_b*du3N6.x*du3N6.z;
		AA[row3N3+12] = -h_e*du3N5.x*du3N5.x;
		AA[row3N3+13] = -h_e*du3N5.x*du3N5.y;
		AA[row3N3+14] = -h_e*du3N5.x*du3N5.z;
		AA[row3N3+15] = -h_b*du3N6.x*du3N6.x;
		AA[row3N3+16] = -h_b*du3N6.x*du3N6.y;
		AA[row3N3+17] = -h_b*du3N6.x*du3N6.z;
		
		AA[row3N2   ] = -h_t*du3N1.x*du3N1.y;
		AA[row3N2+1 ] = -h_t*du3N1.y*du3N1.y;
		AA[row3N2+2 ] = -h_t*du3N1.y*du3N1.z;
		AA[row3N2+3 ] = -h_b*du3N2.x*du3N2.y;
		AA[row3N2+4 ] = -h_b*du3N2.y*du3N2.y;
		AA[row3N2+5 ] = -h_b*du3N2.y*du3N2.z;
		AA[row3N2+6 ] = -h_e*du3N3.x*du3N3.y;
		AA[row3N2+7 ] = -h_e*du3N3.y*du3N3.y;
		AA[row3N2+8 ] = -h_e*du3N3.y*du3N3.z;
		AA[row3N2+9 ] =        h_t*du3N1.x*du3N1.y + h_b*du3N2.x*du3N2.y + h_e*du3N3.x*du3N3.y + h_e*du3N5.x*du3N5.y + h_b*du3N6.x*du3N6.y;
		AA[row3N2+10] = 1.0f + h_t*du3N1.y*du3N1.y + h_b*du3N2.y*du3N2.y + h_e*du3N3.y*du3N3.y + h_e*du3N5.y*du3N5.y + h_b*du3N6.y*du3N6.y;
		AA[row3N2+11] =        h_t*du3N1.y*du3N1.z + h_b*du3N2.y*du3N2.z + h_e*du3N3.y*du3N3.z + h_e*du3N5.y*du3N5.z + h_b*du3N6.y*du3N6.z;
		AA[row3N2+12] = -h_e*du3N5.x*du3N5.y;
		AA[row3N2+13] = -h_e*du3N5.y*du3N5.y;
		AA[row3N2+14] = -h_e*du3N5.y*du3N5.z;
		AA[row3N2+15] = -h_b*du3N6.x*du3N6.y;
		AA[row3N2+16] = -h_b*du3N6.y*du3N6.y;
		AA[row3N2+17] = -h_b*du3N6.y*du3N6.z;
		
		AA[row3N1   ] = -h_t*du3N1.x*du3N1.z;
		AA[row3N1+1 ] = -h_t*du3N1.y*du3N1.z;
		AA[row3N1+2 ] = -h_t*du3N1.z*du3N1.z;
		AA[row3N1+3 ] = -h_b*du3N2.x*du3N2.z;
		AA[row3N1+4 ] = -h_b*du3N2.y*du3N2.z;
		AA[row3N1+5 ] = -h_b*du3N2.z*du3N2.z;
		AA[row3N1+6 ] = -h_e*du3N3.x*du3N3.z;
		AA[row3N1+7 ] = -h_e*du3N3.y*du3N3.z;
		AA[row3N1+8 ] = -h_e*du3N3.z*du3N3.z;
		AA[row3N1+9 ] =        h_t*du3N1.x*du3N1.z + h_b*du3N2.x*du3N2.z + h_e*du3N3.x*du3N3.z + h_e*du3N5.x*du3N5.z + h_b*du3N6.x*du3N6.z;
		AA[row3N1+10] =        h_t*du3N1.y*du3N1.z + h_b*du3N2.y*du3N2.z + h_e*du3N3.y*du3N3.z + h_e*du3N5.y*du3N5.z + h_b*du3N6.y*du3N6.z;
		AA[row3N1+11] = 1.0f + h_t*du3N1.z*du3N1.z + h_b*du3N2.z*du3N2.z + h_e*du3N3.z*du3N3.z + h_e*du3N5.z*du3N5.z + h_b*du3N6.z*du3N6.z;
		AA[row3N1+12] = -h_e*du3N5.x*du3N5.z;
		AA[row3N1+13] = -h_e*du3N5.y*du3N5.z;
		AA[row3N1+14] = -h_e*du3N5.z*du3N5.z;
		AA[row3N1+15] = -h_b*du3N6.x*du3N6.z;
		AA[row3N1+16] = -h_b*du3N6.y*du3N6.z;
		AA[row3N1+17] = -h_b*du3N6.z*du3N6.z;
		
		int row2N3 = numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS - 5*numParticles*NUMCOMPONENTS - 15;
		int row2N2 = numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS - 4*numParticles*NUMCOMPONENTS - 15;
		int row2N1 = numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS - 3*numParticles*NUMCOMPONENTS - 15;
		
		AA[row2N3   ] = -h_t*du2N2.x*du2N2.x;
		AA[row2N3+1 ] = -h_t*du2N2.x*du2N2.y;
		AA[row2N3+2 ] = -h_t*du2N2.x*du2N2.z;
		AA[row2N3+3 ] = -h_b*du2N3.x*du2N3.x;
		AA[row2N3+4 ] = -h_b*du2N3.x*du2N3.y;
		AA[row2N3+5 ] = -h_b*du2N3.x*du2N3.z;
		AA[row2N3+6 ] = -h_e*du2N4.x*du2N4.x;
		AA[row2N3+7 ] = -h_e*du2N4.x*du2N4.y;
		AA[row2N3+8 ] = -h_e*du2N4.x*du2N4.z;
		AA[row2N3+9 ] = 1.0f + h_t*du2N2.x*du2N2.x + h_b*du2N3.x*du2N3.x + h_e*du2N4.x*du2N4.x + h_e*du2N6.x*du2N6.x;
		AA[row2N3+10] =        h_t*du2N2.x*du2N2.y + h_b*du2N3.x*du2N3.y + h_e*du2N4.x*du2N4.y + h_e*du2N6.x*du2N6.y;
		AA[row2N3+11] =        h_t*du2N2.x*du2N2.z + h_b*du2N3.x*du2N3.z + h_e*du2N4.x*du2N4.z + h_e*du2N6.x*du2N6.z;
		AA[row2N3+12] = -h_e*du2N6.x*du2N6.x;
		AA[row2N3+13] = -h_e*du2N6.x*du2N6.y;
		AA[row2N3+14] = -h_e*du2N6.x*du2N6.z;
		
		AA[row2N2   ] = -h_t*du2N2.x*du2N2.y;
		AA[row2N2+1 ] = -h_t*du2N2.y*du2N2.y;
		AA[row2N2+2 ] = -h_t*du2N2.y*du2N2.z;
		AA[row2N2+3 ] = -h_b*du2N3.x*du2N3.y;
		AA[row2N2+4 ] = -h_b*du2N3.y*du2N3.y;
		AA[row2N2+5 ] = -h_b*du2N3.y*du2N3.z;
		AA[row2N2+6 ] = -h_e*du2N4.x*du2N4.y;
		AA[row2N2+7 ] = -h_e*du2N4.y*du2N4.y;
		AA[row2N2+8 ] = -h_e*du2N4.y*du2N4.z;
		AA[row2N2+9 ] =        h_t*du2N2.x*du2N2.y + h_b*du2N3.x*du2N3.y + h_e*du2N4.x*du2N4.y + h_e*du2N6.x*du2N6.y;
		AA[row2N2+10] = 1.0f + h_t*du2N2.y*du2N2.y + h_b*du2N3.y*du2N3.y + h_e*du2N4.y*du2N4.y + h_e*du2N6.y*du2N6.y;
		AA[row2N2+11] =        h_t*du2N2.y*du2N2.z + h_b*du2N3.y*du2N3.z + h_e*du2N4.y*du2N4.z + h_e*du2N6.y*du2N6.z;
		AA[row2N2+12] = -h_e*du2N6.x*du2N6.y;
		AA[row2N2+13] = -h_e*du2N6.y*du2N6.y;
		AA[row2N2+14] = -h_e*du2N6.y*du2N6.z;
		
		AA[row2N1   ] = -h_t*du2N2.x*du2N2.z;
		AA[row2N1+1 ] = -h_t*du2N2.y*du2N2.z;
		AA[row2N1+2 ] = -h_t*du2N2.z*du2N2.z;
		AA[row2N1+3 ] = -h_b*du2N3.x*du2N3.z;
		AA[row2N1+4 ] = -h_b*du2N3.y*du2N3.z;
		AA[row2N1+5 ] = -h_b*du2N3.z*du2N3.z;
		AA[row2N1+6 ] = -h_e*du2N4.x*du2N4.z;
		AA[row2N1+7 ] = -h_e*du2N4.y*du2N4.z;
		AA[row2N1+8 ] = -h_e*du2N4.z*du2N4.z;
		AA[row2N1+9 ] =        h_t*du2N2.x*du2N2.z + h_b*du2N3.x*du2N3.z + h_e*du2N4.x*du2N4.z + h_e*du2N4.x*du2N4.z;
		AA[row2N1+10] =        h_t*du2N2.y*du2N2.z + h_b*du2N3.y*du2N3.z + h_e*du2N4.y*du2N4.z + h_e*du2N4.y*du2N4.z;
		AA[row2N1+11] = 1.0f + h_t*du2N2.z*du2N2.z + h_b*du2N3.z*du2N3.z + h_e*du2N4.z*du2N4.z + h_e*du2N4.z*du2N4.z;
		AA[row2N1+12] = -h_e*du2N4.x*du2N4.z;
		AA[row2N1+13] = -h_e*du2N4.y*du2N4.z;
		AA[row2N1+14] = -h_e*du2N4.z*du2N4.z;
		
		int row1N3 = numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS - 2*numParticles*NUMCOMPONENTS - 12;
		int row1N2 = numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS -   numParticles*NUMCOMPONENTS - 12;
		int row1N1 = numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS -12;
		
		AA[row1N3   ] = -h_t*du1N3.x*du1N3.x;
		AA[row1N3+1 ] = -h_t*du1N3.x*du1N3.y;
		AA[row1N3+2 ] = -h_t*du1N3.x*du1N3.z;
		AA[row1N3+3 ] = -h_b*du1N4.x*du1N4.x;
		AA[row1N3+4 ] = -h_b*du1N4.x*du1N4.y;
		AA[row1N3+5 ] = -h_b*du1N4.x*du1N4.z;
		AA[row1N3+6 ] = -h_e*du1N5.x*du1N5.x;
		AA[row1N3+7 ] = -h_e*du1N5.x*du1N5.y;
		AA[row1N3+8 ] = -h_e*du1N5.x*du1N5.z;
		AA[row1N3+9 ] = 1.0f + h_t*du1N3.x*du1N3.x + h_b*du1N4.x*du1N4.x + h_e*du1N5.x*du1N5.x;
		AA[row1N3+10] =        h_t*du1N3.x*du1N3.y + h_b*du1N4.x*du1N4.y + h_e*du1N5.x*du1N5.y;
		AA[row1N3+11] =        h_t*du1N3.x*du1N3.z + h_b*du1N4.x*du1N4.z + h_e*du1N5.x*du1N5.z;
		
		AA[row1N2   ] = -h_t*du1N3.x*du1N3.y;
		AA[row1N2+1 ] = -h_t*du1N3.y*du1N3.y;
		AA[row1N2+2 ] = -h_t*du1N3.y*du1N3.z;
		AA[row1N2+3 ] = -h_b*du1N4.x*du1N4.y;
		AA[row1N2+4 ] = -h_b*du1N4.y*du1N4.y;
		AA[row1N2+5 ] = -h_b*du1N4.y*du1N4.z;
		AA[row1N2+6 ] = -h_e*du1N5.x*du1N5.y;
		AA[row1N2+7 ] = -h_e*du1N5.y*du1N5.y;
		AA[row1N2+8 ] = -h_e*du1N5.y*du1N5.z;
		AA[row1N2+9 ] =        h_t*du1N3.x*du1N3.y + h_b*du1N4.x*du1N4.y + h_e*du1N5.x*du1N5.y;
		AA[row1N2+10] = 1.0f + h_t*du1N3.y*du1N3.y + h_b*du1N4.y*du1N4.y + h_e*du1N5.y*du1N5.y;
		AA[row1N2+11] =        h_t*du1N3.y*du1N3.z + h_b*du1N4.y*du1N4.z + h_e*du1N5.y*du1N5.z;
		
		AA[row1N1   ] = -h_t*du1N3.x*du1N3.z;
		AA[row1N1+1 ] = -h_t*du1N3.y*du1N3.z;
		AA[row1N1+2 ] = -h_t*du1N3.z*du1N3.z;
		AA[row1N1+3 ] = -h_b*du1N4.x*du1N4.z;
		AA[row1N1+4 ] = -h_b*du1N4.y*du1N4.z;
		AA[row1N1+5 ] = -h_b*du1N4.z*du1N4.z;
		AA[row1N1+6 ] = -h_e*du1N5.x*du1N5.z;
		AA[row1N1+7 ] = -h_e*du1N5.x*du1N5.z;
		AA[row1N1+8 ] = -h_e*du1N5.x*du1N5.z;
		AA[row1N1+9 ] =        h_t*du1N3.z*du1N3.z + h_b*du1N4.z*du1N4.z + h_e*du1N5.x*du1N5.z;
		AA[row1N1+10] =        h_t*du1N3.z*du1N3.z + h_b*du1N4.z*du1N4.z + h_e*du1N5.x*du1N5.z;
		AA[row1N1+11] = 1.0f + h_t*du1N3.z*du1N3.z + h_b*du1N4.z*du1N4.z + h_e*du1N5.x*du1N5.z;
		
		//Set the last nine entries of the vector b
		int bin = numParticles * NUMCOMPONENTS;
		bb[bin-9] = particle[numParticles-3]->velocity.x + g_t*((particle[numParticles-6]->pos-particle[numParticles-3]->pos).dot(du3N1)-length_t)*du3N1.x + g_b*((particle[numParticles-5]->pos-particle[numParticles-3]->pos).dot(du3N2)-length_b)*du3N2.x + g_e*((particle[numParticles-4]->pos-particle[numParticles-3]->pos).dot(du3N3)-length_e)*du3N3.x + g_e*((particle[numParticles-2]->pos-particle[numParticles-3]->pos).dot(du3N5)-length_e)*du3N5.x + g_b*((particle[numParticles-1]->pos-particle[numParticles-3]->pos).dot(du3N6)-length_b)*du3N6.x + gravity.x*(dt/2.0f);
		bb[bin-8] = particle[numParticles-3]->velocity.y + g_t*((particle[numParticles-6]->pos-particle[numParticles-3]->pos).dot(du3N1)-length_t)*du3N1.y + g_b*((particle[numParticles-5]->pos-particle[numParticles-3]->pos).dot(du3N2)-length_b)*du3N2.y + g_e*((particle[numParticles-4]->pos-particle[numParticles-3]->pos).dot(du3N3)-length_e)*du3N3.y + g_e*((particle[numParticles-2]->pos-particle[numParticles-3]->pos).dot(du3N5)-length_e)*du3N5.y + g_b*((particle[numParticles-1]->pos-particle[numParticles-3]->pos).dot(du3N6)-length_b)*du3N6.y + gravity.y*(dt/2.0f);;
		bb[bin-7] = particle[numParticles-3]->velocity.z + g_t*((particle[numParticles-6]->pos-particle[numParticles-3]->pos).dot(du3N1)-length_t)*du3N1.z + g_b*((particle[numParticles-5]->pos-particle[numParticles-3]->pos).dot(du3N2)-length_b)*du3N2.z + g_e*((particle[numParticles-4]->pos-particle[numParticles-3]->pos).dot(du3N3)-length_e)*du3N3.z + g_e*((particle[numParticles-2]->pos-particle[numParticles-3]->pos).dot(du3N5)-length_e)*du3N5.z + g_b*((particle[numParticles-1]->pos-particle[numParticles-3]->pos).dot(du3N6)-length_b)*du3N6.z + gravity.z*(dt/2.0f);;
		bb[bin-6] = particle[numParticles-2]->velocity.x + g_t*((particle[numParticles-5]->pos-particle[numParticles-2]->pos).dot(du2N2)-length_t)*du2N2.x + g_b*((particle[numParticles-4]->pos-particle[numParticles-2]->pos).dot(du2N3)-length_b)*du2N3.x + g_e*((particle[numParticles-3]->pos-particle[numParticles-2]->pos).dot(du2N4)-length_e)*du2N4.x + g_e*((particle[numParticles-1]->pos-particle[numParticles-2]->pos).dot(du2N6)-length_e)*du2N6.x + gravity.x*(dt/2.0f);
		bb[bin-5] = particle[numParticles-2]->velocity.y + g_t*((particle[numParticles-5]->pos-particle[numParticles-2]->pos).dot(du2N2)-length_t)*du2N2.y + g_b*((particle[numParticles-4]->pos-particle[numParticles-2]->pos).dot(du2N3)-length_b)*du2N3.y + g_e*((particle[numParticles-3]->pos-particle[numParticles-2]->pos).dot(du2N4)-length_e)*du2N4.y + g_e*((particle[numParticles-1]->pos-particle[numParticles-2]->pos).dot(du2N6)-length_e)*du2N6.y + gravity.y*(dt/2.0f);
		bb[bin-4] = particle[numParticles-2]->velocity.z + g_t*((particle[numParticles-5]->pos-particle[numParticles-2]->pos).dot(du2N2)-length_t)*du2N2.z + g_b*((particle[numParticles-4]->pos-particle[numParticles-2]->pos).dot(du2N3)-length_b)*du2N3.z + g_e*((particle[numParticles-3]->pos-particle[numParticles-2]->pos).dot(du2N4)-length_e)*du2N4.z + g_e*((particle[numParticles-1]->pos-particle[numParticles-2]->pos).dot(du2N6)-length_e)*du2N6.z + gravity.z*(dt/2.0f);
		bb[bin-3] = particle[numParticles-1]->velocity.x + g_t*((particle[numParticles-4]->pos-particle[numParticles-1]->pos).dot(du1N3)-length_t)*du1N3.x + g_b*((particle[numParticles-3]->pos-particle[numParticles-1]->pos).dot(du1N4)-length_b)*du1N4.x + g_e*((particle[numParticles-2]->pos-particle[numParticles-1]->pos).dot(du1N5)-length_e)*du1N5.x + gravity.x*(dt/2.0f);
		bb[bin-2] = particle[numParticles-1]->velocity.y + g_t*((particle[numParticles-4]->pos-particle[numParticles-1]->pos).dot(du1N3)-length_t)*du1N3.y + g_b*((particle[numParticles-3]->pos-particle[numParticles-1]->pos).dot(du1N4)-length_b)*du1N4.y + g_e*((particle[numParticles-2]->pos-particle[numParticles-1]->pos).dot(du1N5)-length_e)*du1N5.y + gravity.y*(dt/2.0f);
		bb[bin-1] = particle[numParticles-1]->velocity.z + g_t*((particle[numParticles-4]->pos-particle[numParticles-1]->pos).dot(du1N3)-length_t)*du1N3.z + g_b*((particle[numParticles-3]->pos-particle[numParticles-1]->pos).dot(du1N4)-length_b)*du1N4.z + g_e*((particle[numParticles-2]->pos-particle[numParticles-1]->pos).dot(du1N5)-length_e)*du1N5.z + gravity.z*(dt/2.0f);
		
	}
	
	void Strand::calcVelocities(float dt)
	{
		//Calculate the velocities of each particle
		
		//Build matrix and vector of coefficients of linear equations		
		buildAB(dt);
		
		//Set intial solution to previous velocity
		for(int i = 0; i < numParticles; i++)
		{
			xx[i*NUMCOMPONENTS  ] = particle[i]->velocity.x;
			xx[i*NUMCOMPONENTS+1] = particle[i]->velocity.y;
			xx[i*NUMCOMPONENTS+2] = particle[i]->velocity.z;
		}
		
		//Solve for velocity using conjugate gradient method
		conjugate();
		
		//Copy solution to half velocity
		for(int i = 0; i < numParticles; i++)
		{
			particle[i]->velh.x = xx[i*NUMCOMPONENTS  ];
			particle[i]->velh.y = xx[i*NUMCOMPONENTS+1];
			particle[i]->velh.z = xx[i*NUMCOMPONENTS+2];
		}
	}
	
	void Strand::updateSprings(float dt)
	{
		//calculate the 6 coefficients
		float g_e = k_edge/length_e;
		float g_b = k_bend/length_b;
		float g_t = k_twist/length_t;
		float h_e = dt*k_edge/(2.0f*length_e) + d_edge;
		float h_b = dt*k_bend/(2.0f*length_b) + d_bend;
		float h_t = dt*k_twist/(2.0f*length_t) + d_twist;
		
		//Calculate and apply forces for the first three particles
		Vector3f uu0R(root-particle[0]->pos);
		Vector3f uu01(particle[1]->pos-particle[0]->pos);
		Vector3f uu02(particle[2]->pos-particle[0]->pos);
		Vector3f uu03(particle[3]->pos-particle[0]->pos);
		Vector3f du0R(uu0R.unit());
		Vector3f du01(uu01.unit());
		Vector3f du02(uu02.unit());
		Vector3f du03(uu03.unit());
		Vector3f vu0R(-particle[0]->velh);
		Vector3f vu01(particle[1]->velh-particle[0]->velh);
		Vector3f vu02(particle[2]->velh-particle[0]->velh);
		Vector3f vu03(particle[3]->velh-particle[0]->velh);
		
		Vector3f force0 = du0R*(uu0R.dot(du0R)-length_e)*g_e + du0R*(vu0R.dot(du0R))*h_e +
						  du01*(uu01.dot(du01)-length_e)*g_e + du01*(vu01.dot(du01))*h_e +
						  du02*(uu02.dot(du02)-length_b)*g_b + du02*(vu02.dot(du02))*h_b +
						  du03*(uu03.dot(du03)-length_t)*g_t + du03*(vu03.dot(du03))*h_t;
		
		particle[0]->applyForce(force0);
		
		Vector3f uu1R(root-particle[1]->pos);
		Vector3f uu10(particle[0]->pos-particle[1]->pos);
		Vector3f uu12(particle[2]->pos-particle[1]->pos);
		Vector3f uu13(particle[3]->pos-particle[1]->pos);
		Vector3f uu14(particle[4]->pos-particle[1]->pos);
		Vector3f du1R(uu1R.unit());
		Vector3f du10(uu10.unit());
		Vector3f du12(uu12.unit());
		Vector3f du13(uu13.unit());
		Vector3f du14(uu14.unit());
		Vector3f vu1R(-particle[1]->velh);
		Vector3f vu10(particle[0]->velh-particle[1]->velh);
		Vector3f vu12(particle[2]->velh-particle[1]->velh);
		Vector3f vu13(particle[3]->velh-particle[1]->velh);
		Vector3f vu14(particle[4]->velh-particle[1]->velh);
		
		Vector3f force1 = du1R*(uu1R.dot(du1R)-length_b)*g_b + du1R*(vu1R.dot(du1R))*h_b + 
						  du10*(uu10.dot(du10)-length_e)*g_e + du10*(vu10.dot(du10))*h_e + 
						  du12*(uu12.dot(du12)-length_e)*g_e + du12*(vu12.dot(du12))*h_e + 
						  du13*(uu13.dot(du13)-length_b)*g_b + du13*(vu13.dot(du13))*h_b + 
						  du14*(uu14.dot(du14)-length_t)*g_t + du14*(vu14.dot(du14))*h_t; 
		
		particle[1]->applyForce(force1);
		
		Vector3f uu2R(root-particle[2]->pos);
		Vector3f uu20(particle[0]->pos-particle[2]->pos);
		Vector3f uu21(particle[1]->pos-particle[2]->pos);
		Vector3f uu23(particle[3]->pos-particle[2]->pos);
		Vector3f uu24(particle[4]->pos-particle[2]->pos);
		Vector3f uu25(particle[5]->pos-particle[2]->pos);
		Vector3f du2R(uu2R.unit());
		Vector3f du20(uu20.unit());
		Vector3f du21(uu21.unit());
		Vector3f du23(uu23.unit());
		Vector3f du24(uu24.unit());
		Vector3f du25(uu25.unit());
		Vector3f vu2R(-particle[2]->velh);
		Vector3f vu20(particle[0]->velh-particle[2]->velh);
		Vector3f vu21(particle[1]->velh-particle[2]->velh);
		Vector3f vu23(particle[3]->velh-particle[2]->velh);
		Vector3f vu24(particle[4]->velh-particle[2]->velh);
		Vector3f vu25(particle[5]->velh-particle[2]->velh);
		
		Vector3f force2 = du2R*(uu2R.dot(du2R)-length_t)*g_t + du2R*(vu2R.dot(du2R))*h_t +
						  du20*(uu20.dot(du20)-length_b)*g_b + du20*(vu20.dot(du20))*h_b +
						  du21*(uu21.dot(du21)-length_e)*g_e + du21*(vu21.dot(du21))*h_e +
						  du23*(uu23.dot(du23)-length_e)*g_e + du23*(vu23.dot(du23))*h_e +
						  du24*(uu24.dot(du24)-length_b)*g_b + du24*(vu24.dot(du24))*h_b +
						  du25*(uu25.dot(du25)-length_t)*g_t + du25*(vu25.dot(du25))*h_t;
		
		particle[2]->applyForce(force2);
		
		//Calculate force for all particles between first and last
		for(int i = 3; i < (numParticles-3); i++)
		{
			Vector3f uu3(particle[i-3]->pos-particle[i]->pos);
			Vector3f uu2(particle[i-2]->pos-particle[i]->pos);
			Vector3f uu1(particle[i-1]->pos-particle[i]->pos);
			Vector3f ud1(particle[i+1]->pos-particle[i]->pos);
			Vector3f ud2(particle[i+2]->pos-particle[i]->pos);
			Vector3f ud3(particle[i+3]->pos-particle[i]->pos);
			Vector3f dui3(uu3.unit());
			Vector3f dui2(uu2.unit());
			Vector3f dui1(uu1.unit());
			Vector3f ddi1(ud1.unit());
			Vector3f ddi2(ud2.unit());
			Vector3f ddi3(ud3.unit());
			Vector3f vu3(particle[i-3]->velh-particle[i]->velh);
			Vector3f vu2(particle[i-2]->velh-particle[i]->velh);
			Vector3f vu1(particle[i-1]->velh-particle[i]->velh);
			Vector3f vd1(particle[i+1]->velh-particle[i]->velh);
			Vector3f vd2(particle[i+2]->velh-particle[i]->velh);
			Vector3f vd3(particle[i+3]->velh-particle[i]->velh);
			
			Vector3f force = dui3*(uu3.dot(dui3)-length_t)*g_t + dui3*(vu3.dot(dui3))*h_t + 
							 dui2*(uu2.dot(dui2)-length_b)*g_b + dui2*(vu2.dot(dui2))*h_b + 
							 dui1*(uu1.dot(dui1)-length_e)*g_e + dui1*(vu1.dot(dui1))*h_e + 
							 ddi1*(ud1.dot(ddi1)-length_e)*g_e + ddi1*(vd1.dot(ddi1))*h_e + 
							 ddi2*(ud2.dot(ddi2)-length_b)*g_b + ddi2*(vd2.dot(ddi2))*h_b + 
							 ddi3*(ud3.dot(ddi3)-length_t)*g_t + ddi3*(vd3.dot(ddi3))*h_t;
			
			particle[i]->applyForce(force);
		}
		
		//Calculate and apply forces for last three particles
		Vector3f uu3N1(particle[numParticles-6]->pos-particle[numParticles-3]->pos);
		Vector3f uu3N2(particle[numParticles-5]->pos-particle[numParticles-3]->pos);
		Vector3f uu3N3(particle[numParticles-4]->pos-particle[numParticles-3]->pos);
		Vector3f uu3N5(particle[numParticles-2]->pos-particle[numParticles-3]->pos);
		Vector3f uu3N6(particle[numParticles-1]->pos-particle[numParticles-3]->pos);
		Vector3f du3N1(uu3N1.unit());
		Vector3f du3N2(uu3N2.unit());
		Vector3f du3N3(uu3N3.unit());
		Vector3f du3N5(uu3N5.unit());
		Vector3f du3N6(uu3N6.unit());
		Vector3f vu3N1(particle[numParticles-6]->velh-particle[numParticles-3]->velh);
		Vector3f vu3N2(particle[numParticles-5]->velh-particle[numParticles-3]->velh);
		Vector3f vu3N3(particle[numParticles-4]->velh-particle[numParticles-3]->velh);
		Vector3f vu3N5(particle[numParticles-2]->velh-particle[numParticles-3]->velh);
		Vector3f vu3N6(particle[numParticles-1]->velh-particle[numParticles-3]->velh);
		
		Vector3f force3N = du3N1*(uu3N1.dot(du3N1)-length_t)*g_t + du3N1*(vu3N1.dot(du3N1))*h_t +
						   du3N2*(uu3N2.dot(du3N2)-length_b)*g_b + du3N2*(vu3N2.dot(du3N2))*h_b +
						   du3N3*(uu3N3.dot(du3N3)-length_e)*g_e + du3N3*(vu3N3.dot(du3N3))*h_e +
						   du3N5*(uu3N5.dot(du3N5)-length_e)*g_e + du3N5*(vu3N5.dot(du3N5))*h_e +
						   du3N6*(uu3N6.dot(du3N6)-length_b)*g_b + du3N6*(vu3N6.dot(du3N6))*h_b;
		
		particle[numParticles-3]->applyForce(force3N);
		
		Vector3f uu2N2(particle[numParticles-5]->pos-particle[numParticles-2]->pos);
		Vector3f uu2N3(particle[numParticles-4]->pos-particle[numParticles-2]->pos);
		Vector3f uu2N4(particle[numParticles-3]->pos-particle[numParticles-2]->pos);
		Vector3f uu2N6(particle[numParticles-1]->pos-particle[numParticles-2]->pos);
		Vector3f du2N2(uu2N2.unit());
		Vector3f du2N3(uu2N3.unit());
		Vector3f du2N4(uu2N4.unit());
		Vector3f du2N6(uu2N6.unit());
		Vector3f vu2N2(particle[numParticles-5]->velh-particle[numParticles-2]->velh);
		Vector3f vu2N3(particle[numParticles-4]->velh-particle[numParticles-2]->velh);
		Vector3f vu2N4(particle[numParticles-3]->velh-particle[numParticles-2]->velh);
		Vector3f vu2N6(particle[numParticles-1]->velh-particle[numParticles-2]->velh);
		
		Vector3f force2N = du2N2*(uu2N2.dot(du2N2)-length_t)*g_t + du2N2*(vu2N2.dot(du2N2))*h_t +
						   du2N3*(uu2N3.dot(du2N3)-length_t)*g_b + du2N3*(vu2N3.dot(du2N3))*h_b +
						   du2N4*(uu2N4.dot(du2N4)-length_t)*g_e + du2N4*(vu2N4.dot(du2N4))*h_e +
						   du2N6*(uu2N6.dot(du2N6)-length_t)*g_e + du2N6*(vu2N6.dot(du2N6))*h_e;
		
		particle[numParticles-2]->applyForce(force2N);
		
		Vector3f uu1N3(particle[numParticles-4]->pos-particle[numParticles-1]->pos);
		Vector3f uu1N4(particle[numParticles-3]->pos-particle[numParticles-1]->pos);
		Vector3f uu1N5(particle[numParticles-2]->pos-particle[numParticles-1]->pos);
		Vector3f du1N3(uu1N3.unit());
		Vector3f du1N4(uu1N4.unit());
		Vector3f du1N5(uu1N5.unit());
		Vector3f vu1N3(particle[numParticles-4]->velh-particle[numParticles-1]->velh);
		Vector3f vu1N4(particle[numParticles-3]->velh-particle[numParticles-1]->velh);
		Vector3f vu1N5(particle[numParticles-2]->velh-particle[numParticles-1]->velh);
		
		Vector3f force1N = du1N3*(uu1N3.dot(du1N3)-length_t)*g_t + du1N3*(vu1N3.dot(du1N3))*h_t +
						   du1N4*(uu1N4.dot(du1N4)-length_b)*g_b + du1N4*(vu1N4.dot(du1N4))*h_b +
						   du1N5*(uu1N5.dot(du1N5)-length_e)*g_e + du1N5*(vu1N5.dot(du1N5))*h_e;
		
		particle[numParticles-1]->applyForce(force1N);
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
	
	void Strand::update(float dt, const float (&grid)[DOMAIN_DIM][DOMAIN_DIM][DOMAIN_DIM], Strand** strand, std::vector<Collision> (&collision)[NUMSTRANDS][NUMSEGMENTS])
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
		
		applyStrainLimiting(dt);
		
		//Detect segment collisions, calculate stiction forces and apply stiction to half velocity
		applyStiction(dt, strand, collision);
		
		//Calculate half position and new position
		updatePositions(dt);
		
		//Check geometry collisions and adjust velocities and positions
		objectCollisions(dt, grid);
		
		//Self collisions
		
		//Reset forces on particles
		clearForces();
		
		//Calculate velocities using half position
		calcVelocities(dt);
		
		//Calculate and apply spring forces using half position
		updateSprings(dt);
		
		//Apply gravity
		applyForce(Vector3f(0.0f, mass*GRAVITY, 0.0f));
		
		//FIXME Apply other forces such as wind
		
		//Calculate half velocity and new velocity
		updateParticles(dt);
		
		//Detect segment collisions, calculate stiction forces and apply stiction to velocity
		applyStiction2(dt, strand, collision);
		
		//FIXME Check when is best to update bounding volumes
		//FIXME Check if bounding volumes need updating between half time steps
		//Update bounding volume tree values using newly calculated particle positions
		updateBoundingVolumes();
		
		//Self Collisions
	}
	
	void Strand::applyStiction(float dt, Strand** strand, std::vector<Collision> (&collision)[NUMSTRANDS][NUMSEGMENTS])
	{
		//Break existing springs based on distance between segments
		//Search through list of connections with this strand
		for(int i = 0; i < (numParticles-1); i++)
		{
			if(collision[strandID][i].size() > 0)
			{
				for(int j = 0; j < collision[strandID][i].size(); j++)
				{
					//Get particle indices
					int sid0 = strandID;
					int sid1 = collision[strandID][i][j].getStrandID();
					
					int seg0 = i;
					int seg1 = collision[strandID][i][j].getSegmentID();
					
					int pid00 = i;
					int pid01 = i + 1;
					int pid10 = collision[strandID][i][j].getParticleOneID();
					int pid11 = collision[strandID][i][j].getParticleTwoID();
					
					//Get particle data
					Particle * particle00 = strand[sid0]->particle[pid00];
					Particle * particle01 = strand[sid0]->particle[pid01];
					Particle * particle10 = strand[sid1]->particle[pid10];
					Particle * particle11 = strand[sid1]->particle[pid11];
					
					//Interpolated candidate position
					Vector3f posc0 = (particle00->posc + particle01->posc)/2;
					Vector3f posc1 = (particle10->posc + particle11->posc)/2;
					
					//Calculate vector along the midpoints
					Vector3f d01(posc1-posc0); //From first mid-point to second mid-point
					
					//Calculate squared distance between the two segments
					float distance = d01.length_sqr();
					
					//Check if squared distance is greater than threshold
					if(distance > MAX_SQR_STIC)
					{
						//Find the collision object from the other strand's segment's collision list
						for(int k = 0; k < collision[sid1][seg1].size(); k++)
						{
							if(collision[sid1][seg1][k].getStrandID() == sid0 && collision[sid1][seg1][k].getSegmentID() == seg0)
							{
								//Remove the collision object from the other strand's segment's collision list
								collision[sid1][seg1].erase(collision[sid1][seg1].begin() + k);
							}
						}
						
						//Remove collision object from this strand's segment collision list
						collision[sid0][seg0].erase(collision[sid0][seg0].begin() + j);
						
						//Set inner loop index back by one increment to compensate for the removed element.
						j = j - 1;
					}
				}
			}
		}
		
		//New colliding pairs of segments will be saved here
		std::vector<NodePair> pairs;
		
		//Register any new collisions with other strands
		for(int i = strandID+1; i < numStrands; i++)
		{
			//Check for collisions against other strand BVH Trees
			//Identify which particles are affected by collisions
			Node::collides(bvhTree, strand[i]->getTree(), pairs);
		}
		
		//Save new collisions globally
		for(int i = 0; i < pairs.size(); i++)
		{
			int sid0 = pairs[i].one->getStrandID();
			int sid1 = pairs[i].two->getStrandID();
			
			int seg0 = pairs[i].one->getID();
			int seg1 = pairs[i].two->getID();
			
			//Check collision with this strand doesn't already exist before adding it to the list
			bool exists = false;
			
			for(int j = 0; j < collision[sid0][seg0].size(); j++)
			{
				if(collision[sid0][seg0][j].getStrandID() == sid1 && collision[sid0][seg0][j].getSegmentID() == seg1)
				{
					exists = true;
					break;
				}
			}
			
			if(!exists)
			{
				collision[sid0][seg0].push_back(Collision(sid1, seg1));
			}
			
			//Reset the flag that saves the already-exists state
			exists = false;
			
			for(int j = 0; j < collision[sid1][seg1].size(); j++)
			{
				if(collision[sid1][seg1][j].getStrandID() == sid0 && collision[sid1][seg1][j].getSegmentID() == seg0)
				{
					exists = true;
					break;
				}
			}
			
			if(!exists)
			{
				collision[sid1][seg1].push_back(Collision(sid0, seg0));
			}
		}
		
		//Apply velocity impulse to every applicable segment in this strand only
		for(int segmentID = 0; segmentID < NUMSEGMENTS; segmentID++)
		{
			for(int i = 0; i < collision[strandID][segmentID].size(); i++)
			{
				//Retrieve indices for appropriate strands and particles frm global system index
				int sid0 = strandID;
				int sid1 = collision[strandID][segmentID][i].getStrandID();
				
				int pid00 = segmentID;
				int pid01 = segmentID + 1;
				
				int pid10 = collision[strandID][segmentID][i].getParticleOneID();
				int pid11 = collision[strandID][segmentID][i].getParticleTwoID();
				
				//Get particle data
				Particle * particle00 = strand[sid0]->particle[pid00];
				Particle * particle01 = strand[sid0]->particle[pid01];
				Particle * particle10 = strand[sid1]->particle[pid10];
				Particle * particle11 = strand[sid1]->particle[pid11];
				
				//Interpolated candidate position
				Vector3f posc0 = (particle00->posc + particle01->posc)/2;
				Vector3f posc1 = (particle10->posc + particle11->posc)/2;
				
				//Calculate vectors along the midpoints
				Vector3f d01(posc1-posc0); //From first mid-point to second mid-point
				
				//Unitize above direction
				Vector3f du01(d01.unit());
				
				//Interpolated velocity
				Vector3f velh0 = (particle00->velh + particle01->velh)/2;
				Vector3f velh1 = (particle10->velh + particle11->velh)/2;
				
				//Simplify half velocity calculations for later use
				Vector3f vu01 = velh1 - velh0;
				
				float g = k_edge/LEN_STIC;
				float h = dt*k_edge/(2.0f*LEN_STIC) + D_STIC; //Damping coefficient added
				
				//Calculate spring force acting on this segment's two particles
				Vector3f force = du01 * g * (d01.dot(du01)-LEN_STIC) + du01 * h * (vu01.dot(du01));
				
				//Modify half velocity for each particle in this segment
				particle00->velh += force * (dt/2.0f);
				particle01->velh += force * (dt/2.0f);
			}
		}
	}
	
	void Strand::applyStiction2(float dt, Strand** strand, std::vector<Collision> (&collision)[NUMSTRANDS][NUMSEGMENTS])
	{
		
		//Break existing springs based on distance between segments
		//Search through list of connections with this strand
		for(int i = 0; i < (numParticles-1); i++)
		{
			if(collision[strandID][i].size() > 0)
			{
				for(int j = 0; j < collision[strandID][i].size(); j++)
				{
					//Get particle indices
					int sid0 = strandID;
					int sid1 = collision[strandID][i][j].getStrandID();
					
					int seg0 = i;
					int seg1 = collision[strandID][i][j].getSegmentID();
					
					int pid00 = i;
					int pid01 = i + 1;
					int pid10 = collision[strandID][i][j].getParticleOneID();
					int pid11 = collision[strandID][i][j].getParticleTwoID();
					
					//Get particle data
					Particle * particle00 = strand[sid0]->particle[pid00];
					Particle * particle01 = strand[sid0]->particle[pid01];
					Particle * particle10 = strand[sid1]->particle[pid10];
					Particle * particle11 = strand[sid1]->particle[pid11];
					
					//Interpolated position
					Vector3f pos0 = (particle00->pos + particle01->pos)/2;
					Vector3f pos1 = (particle10->pos + particle11->pos)/2;
					
					//Calculate vector along the midpoints
					Vector3f d01(pos1-pos0); //From first mid-point to second mid-point
					
					//Calculate squared distance between the two segments
					float distance = d01.length_sqr();
					
					//Check if squared distance is greater than threshold
					if(distance > MAX_SQR_STIC)
					{
						//Find the collision object from the other strand's segment's collision list
						for(int k = 0; k < collision[sid1][seg1].size(); k++)
						{
							if(collision[sid1][seg1][k].getStrandID() == sid0 && collision[sid1][seg1][k].getSegmentID() == seg0)
							{
								//Remove the collision object from the other strand's segment's collision list
								collision[sid1][seg1].erase(collision[sid1][seg1].begin() + k);
							}
						}
						
						//Remove collision object from this strand's segment collision list
						collision[sid0][seg0].erase(collision[sid0][seg0].begin() + j);
						
						//Set inner loop index back by one increment to compensate for the removed element.
						j = j - 1;
					}
				}
			}
		}
		
		
		//Colliding pairs will be saved here
		std::vector<NodePair> pairs;
		
		//Detect for collision with other strands
		for(int i = strandID+1; i < numStrands; i++)
		{
			//Check for collisions against other strand BVH Trees
			//Identify which particles are affected by collisions
			Node::collides(bvhTree, strand[i]->getTree(), pairs);
		}
		
		//Save new collisions globally
		for(int i = 0; i < pairs.size(); i++)
		{
			int sid0 = pairs[i].one->getStrandID();
			int sid1 = pairs[i].two->getStrandID();
			
			int seg0 = pairs[i].one->getID();
			int seg1 = pairs[i].two->getID();
			
			//Check collision doesn't already exist before adding it to the list
			bool exists = false;
			
			for(int j = 0; j < collision[sid0][seg0].size(); j++)
			{
				if(collision[sid0][seg0][j].getStrandID() == sid1 && collision[sid0][seg0][j].getSegmentID() == seg1)
				{
					exists = true;
					break;
				}
			}
			
			//Register new collision with the other strand
			if(!exists)
			{
				collision[sid0][seg0].push_back(Collision(sid1, seg1));
			}
			
			//Reset the flag that saves the already-exists state
			exists = false;
			
			for(int j = 0; j < collision[sid1][seg1].size(); j++)
			{
				if(collision[sid1][seg1][j].getStrandID() == sid0 && collision[sid1][seg1][j].getSegmentID() == seg0)
				{
					exists = true;
					break;
				}
			}
			
			//Register new collision with the other strand
			if(!exists)
			{
				collision[sid1][seg1].push_back(Collision(sid0, seg0));
			}
		}
		
		//Apply velocity impulse to every applicable segment in this strand only
		for(int segmentID = 0; segmentID < NUMSEGMENTS; segmentID++)
		{
			for(int i = 0; i < collision[strandID][segmentID].size(); i++)
			{
				//Retrieve indices for appropriate strands and particles
				int sid0 = strandID;
				int sid1 = collision[strandID][segmentID][i].getStrandID();
				
				int pid00 = segmentID;
				int pid01 = segmentID + 1;
				
				int pid10 = collision[strandID][segmentID][i].getParticleOneID();
				int pid11 = collision[strandID][segmentID][i].getParticleTwoID();
				
				//Get particle data
				Particle * particle00 = strand[sid0]->particle[pid00];
				Particle * particle01 = strand[sid0]->particle[pid01];
				Particle * particle10 = strand[sid1]->particle[pid10];
				Particle * particle11 = strand[sid1]->particle[pid11];
				
				//Interpolated candidate position
				Vector3f pos0 = (particle00->pos + particle01->pos)/2;
				Vector3f pos1 = (particle10->pos + particle11->pos)/2;
				
				//Calculate vectors along the midpoints
				Vector3f d01(pos1-pos0); //From first mid-point to second mid-point
				
				//Unitize above directions
				Vector3f du01(d01.unit());
				
				//Interpolated velocity
				Vector3f vel0 = (particle00->velocity + particle01->velocity)/2;
				Vector3f vel1 = (particle10->velocity + particle11->velocity)/2;
				
				//Simplify velocity calculation
				Vector3f vu01 = vel1 - vel0;
				
				float g = k_edge/LEN_STIC;
				float h = dt*k_edge/(2.0f*LEN_STIC) + D_STIC;
				
				//Calculate spring force acting on this segment's two particles
				Vector3f force = du01 * g * (d01.dot(du01)-LEN_STIC) + du01 * h * (vu01.dot(du01));
				
				//Modify velocity for each particle in each strand
				particle00->velocity += force * (dt/2.0f);
				particle01->velocity += force * (dt/2.0f);
			}
		}
	}
	
	Node* Strand::getTree()
	{
		return bvhTree;
	}
	
	void Strand::updateBoundingVolumes()
	{
		//Update the leaf nodes with new positions
		for(int i = 1; i < numParticles; i++)
		{
			//Build segment vertices
			std::vector<Vector3f> vertices;
			vertices.push_back(particle[i-1]->pos);
			vertices.push_back(particle[i]->pos);
			
			//Create additional ghost vertices to approximate a cylindrical volume
			addVolumeVertices(vertices);
			
			leafKDOP[i-1]->update(vertices);
		}
		
		//Update the internal nodes
		Node::updateTree(bvhTree);
	}
	
	void Strand::addVolumeVertices(std::vector<Vector3f> &vertices)
	{
		//Vector from first point to second point
		Vector3f n(vertices[1] - vertices[0]);
		
		//Store the perpendicular vector here
		Vector3f r;
		
		do
		{
			//Generate random point
			Vector3f p = Vector3f::random(-1.0f, 1.0f);
			
			//Vector from first point to randomly generated point
			Vector3f np(p - vertices[0]);
			
			//Calculate the vector perpendicular to n and np
			r = n.cross(np);
		}
		while(r.x == 0.0f && r.y == 0.0f && r.z == 0.0f);
		
		//Calculate a third vector that is perpendicular to the n vector and the r vector
		Vector3f s = r.cross(n);
		
		//Normalise and scale the perpendicular vectors
		r = r.unit() * HALF_LEN_STIC;
		s = s.unit() * HALF_LEN_STIC;
		
		//Calculate and add new approximate volume vertices by offsetting original two vertices
		vertices.push_back(vertices[0] + r);
		vertices.push_back(vertices[0] - r);
		vertices.push_back(vertices[0] + s);
		vertices.push_back(vertices[0] - s);
		vertices.push_back(vertices[1] + r);
		vertices.push_back(vertices[1] - r);
		vertices.push_back(vertices[1] + s);
		vertices.push_back(vertices[1] - s);
	}
	
	void Strand::applyStrainLimiting(float dt)
	{
		for(int i = 0; i < numParticles; i++)
		{
			//Calculate candidate position using half velocity
			particle[i]->posc = particle[i]->pos + particle[i]->velh * dt;
			
			//Determine the direction of the spring between the particles
			Vector3f dir = (i > 0) ? (particle[i]->posc - particle[i-1]->posc) : (particle[i]->posc - root);
			
			if(dir.length_sqr() > MAX_LENGTH_SQUARED)
			{
				//Find a valid candidate position
				particle[i]->posc = (i > 0) ? (particle[i-1]->posc + (dir * (MAX_LENGTH*dir.length_inverse()))) : (root + (dir * (MAX_LENGTH*dir.length_inverse()))); //fast length calculation
				
				//~ particle[i]->posc = particle[i-1]->posc + (dir * (MAX_LENGTH/dir.length())); //slower length calculation
				
				//Calculate new half velocity based on valid candidate position, i.e. add a velocity impulse
				particle[i]->velh = (particle[i]->posc - particle[i]->pos)/dt;
			}
		}
	}
	
	void Strand::objectCollisions(float dt, const float (&grid)[DOMAIN_DIM][DOMAIN_DIM][DOMAIN_DIM])
	{		
		for(int i = 0; i < numParticles; i++)
		{			
			//Transform particle coordinates to collision grid coordinates
			Vector3f position;
			position.x = (particle[i]->position.x + DOMAIN_HALF-CELL_HALF)/CELL_WIDTH;
			position.y = (particle[i]->position.y + DOMAIN_HALF+0.125f-CELL_HALF)/CELL_WIDTH;
			position.z = (particle[i]->position.z + DOMAIN_HALF-CELL_HALF)/CELL_WIDTH;
			
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
			
			float c000 =  c00 * (1 - d.y) + c11 * d.y;
			
			//Calculate normal
			Vector3f normal;
			
			normal.x = -v000*d.y*d.z + v001*d.y*d.z + v010*d.y*d.z - v011*d.y*d.z + v100*d.y*d.z - v101*d.y*d.z - v110*d.y*d.z + v111*d.y*d.z + v000*d.y + v000*d.z - v001*d.y - v010*d.z - v100*d.y - v100*d.z + v101*d.y + v110*d.z - v000 + v100;
			normal.y = -v000*d.x*d.z + v001*d.x*d.z + v010*d.x*d.z - v011*d.x*d.z + v100*d.x*d.z - v101*d.x*d.z - v110*d.x*d.z + v111*d.x*d.z + v000*d.x + v000*d.z - v001*d.x - v001*d.z - v010*d.z + v011*d.z - v100*d.x + v101*d.x - v000 + v001;
			normal.z = -v000*d.x*d.y + v001*d.x*d.y + v010*d.x*d.y - v011*d.x*d.y + v100*d.x*d.y - v101*d.x*d.y - v110*d.x*d.y + v111*d.x*d.y + v000*d.x + v000*d.y - v001*d.y - v010*d.x - v010*d.y + v011*d.y - v100*d.x + v110*d.x - v000 + v010;
			
			//Normalise
			normal.unitize();
			
			float phi = c000 + dt * (particle[i]->velocity.dot(normal));
			
			//Check for surface collision
			if(phi < 0.0f)
			{
				//~ std::cout << "hit" << std::endl;
				//~ std::cout << c000 << std::endl;
				
				//~ particle[13]->freeze = true;
				
				//Move particle along normal to surface of mesh
				//~ particle[i]->position = particle[i]->position + normal*(0.002f-c000);
				
				//Calculate new half position
				//~ particle[i]->posh = (particle[i]->position + particle[i]->poso)/2.0f;
				
				//~ particle[i]->pos = particle[i]->posh;
				
				//Reflect direction of previous velocity across normal
				//~ particle[i]->velocity = particle[i]->velocity - normal*(2.0f*(particle[i]->velocity.x*normal.x+particle[i]->velocity.y*normal.y+particle[i]->velocity.z*normal.z));
				
				float vn = particle[i]->velocity.dot(normal);
				Vector3f vt = particle[i]->velocity - normal*vn;
								
				float vnew = vn - phi/dt;
				float friction = 1.0f - 0.3f*(vnew-vn)/vt.length();
				Vector3f vrel = (0.0f > friction) ? vt*0.0f : vt*friction;
				
				particle[i]->velocity = normal*vnew + vrel;
			}
			
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
		
		//Delete BVH Tree and KDOPs
		delete bvhTree;
		
		//Particles
		for(int i = 0; i < numParticles; i++)
		{
			delete particle[i];
			particle[i] = NULL;
		}
		delete [] particle;
		
		delete [] xx;
		delete [] bb;
		delete [] AA;
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
			   float length_e,
			   float length_b,
			   float length_t,
			   std::vector<Vector3f> &roots,
			   std::vector<Vector3f> &normals,
			   Model_OBJ &obj)
	{
		//Seed random number generation
		srand (static_cast <unsigned> (time(0)));
		
		this->numStrands = numStrands;
		
		strand = new Strand*[numStrands];
		
		for(int i = 0; i < numStrands; i++)
		{
			strand[i] = new Strand(numParticles, i, numStrands, mass, k_edge, k_bend, k_twist, k_extra, d_edge, d_bend, d_twist, d_extra, length_e, length_b, length_t, roots[i], normals[i]); 
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
			strand[i]->update(dt, grid, strand, collision);
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

