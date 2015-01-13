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
				   float length,
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
		
		this->length = length;
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
	/*
	void Strand::buildAB(float dt)
	{
		memset(AA, 0, sizeof(float)*numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS);
		
		//TODO Find reasonable value for damping coefficient (d_edge) that allows for a more stable system
		float h = dt*dt*k_edge/(4.0f*mass*length) + d_edge*dt/(2.0f*mass);
		float g = dt*k_edge/(2.0f*mass*length);
		Vector3f gravity(0.0f, GRAVITY, 0.0f);
		
		//First particle direction vectors
		Vector3f du0R(root-particle[0]->pos);
		Vector3f du01(particle[1]->pos-particle[0]->pos);
		du0R.unitize();
		du01.unitize();
		
		//TODO Set first six entries of the first row of A matrix
		AA[0] = 1.0f + h*du0R.x*du0R.x + h*du01.x*du01.x;
		AA[1] = h*du0R.x*du0R.y + h*du01.x*du01.y;
		AA[2] = h*du0R.x*du0R.z + h*du01.x*du01.z;
		AA[3] = -h*du01.x*du01.x;
		AA[4] = -h*du01.x*du01.y;
		AA[5] = -h*du01.x*du01.z;
		
		//TODO Set next six non-zero entries of the second row of matrix A
		AA[numParticles*NUMCOMPONENTS  ] = h*du0R.x*du0R.y + h*du01.x*du01.y;
		AA[numParticles*NUMCOMPONENTS+1] = 1.0f + h*du0R.y*du0R.y + h*du01.y*du01.y;
		AA[numParticles*NUMCOMPONENTS+2] = h*du0R.y*du0R.z + h*du01.y*du01.z;
		AA[numParticles*NUMCOMPONENTS+3] = -h*du01.x*du01.y;
		AA[numParticles*NUMCOMPONENTS+4] = -h*du01.y*du01.y;
		AA[numParticles*NUMCOMPONENTS+5] = -h*du01.y*du01.z;
		
		//TODO Set the next six non-zero entries of the third row of matrix A
		AA[2*numParticles*NUMCOMPONENTS  ] = h*du0R.x*du0R.z + h*du01.x*du01.z;
		AA[2*numParticles*NUMCOMPONENTS+1] = h*du0R.y*du0R.z + h*du01.y*du01.z;
		AA[2*numParticles*NUMCOMPONENTS+2] = 1.0f + h*du0R.z*du0R.z + h*du01.z*du01.z;
		AA[2*numParticles*NUMCOMPONENTS+3] = -h*du01.x*du01.z;
		AA[2*numParticles*NUMCOMPONENTS+4] = -h*du01.y*du01.z;
		AA[2*numParticles*NUMCOMPONENTS+5] = -h*du01.z*du01.z;
		
		//TODO Set the first three entries of the b vector
		bb[0] = particle[0]->velocity.x + g*((root-particle[0]->pos).dot(du0R)-length)*du0R.x + g*((particle[1]->pos-particle[0]->pos).dot(du01)-length)*du01.x + gravity.x*(dt/2.0f);
		bb[1] = particle[0]->velocity.y + g*((root-particle[0]->pos).dot(du0R)-length)*du0R.y + g*((particle[1]->pos-particle[0]->pos).dot(du01)-length)*du01.y + gravity.y*(dt/2.0f);
		bb[2] = particle[0]->velocity.z + g*((root-particle[0]->pos).dot(du0R)-length)*du0R.z + g*((particle[1]->pos-particle[0]->pos).dot(du01)-length)*du01.z + gravity.z*(dt/2.0f);
		
		//Build in-between values of matrix A and vector b
		for(int i = 1; i < (numParticles-1); i++)
		{
			//Current particle position, particle above and particle below
			Vector3f ui = particle[i]->pos;
			Vector3f uu = particle[i-1]->pos;
			Vector3f ud = particle[i+1]->pos;
			
			Vector3f du(uu-particle[i]->pos);
			Vector3f dd(ud-particle[i]->pos);
			du.unitize();
			dd.unitize();
			
			int row0 = i * numParticles * NUMCOMPONENTS *  NUMCOMPONENTS + (i - 1) * NUMCOMPONENTS;
			int row1 = row0 + numParticles * NUMCOMPONENTS;
			int row2 = row1 + numParticles * NUMCOMPONENTS;
			
			//TODO Set first row along diagonal
			AA[row0  ] = -h*du.x*du.x;
			AA[row0+1] = -h*du.x*du.y;
			AA[row0+2] = -h*dd.x*dd.z;
			AA[row0+3] =  1.0f + h*du.x*du.x + h*dd.x*dd.x; //Diagonal
			AA[row0+4] =  h*du.x*du.y + h*dd.x*dd.y;
			AA[row0+5] =  h*du.x*du.z + h*dd.x*dd.z;
			AA[row0+6] = -h*dd.x*dd.x;
			AA[row0+7] = -h*dd.x*dd.y;
			AA[row0+8] = -h*dd.x*dd.z;
			
			//TODO Set second row along diagonal
			AA[row1  ] = -h*du.x*du.y;
			AA[row1+1] = -h*du.y*du.y;
			AA[row1+2] = -h*du.y*du.z;
			AA[row1+3] = h*du.x*du.y + h*dd.x*dd.y;
			AA[row1+4] = 1.0f + h*du.y*du.y + h*dd.y*dd.y; //Diagonal
			AA[row1+5] = h*du.y*du.z + h*dd.y*dd.z;
			AA[row1+6] = -h*dd.x*dd.y;
			AA[row1+7] = -h*dd.y*dd.y;
			AA[row1+8] = -h*dd.y*dd.z;
			
			//TODO Set third row along diagonal
			AA[row2  ] = -h*du.x*du.z;
			AA[row2+1] = -h*du.y*du.z;
			AA[row2+2] = -h*du.z*du.z;
			AA[row2+3] = h*du.x*du.z + h*dd.x*dd.z;
			AA[row2+4] = h*du.y*du.z + h*dd.y*dd.z;
			AA[row2+5] = 1.0f + h*du.z*du.z + h*dd.z*dd.z; //Diagonal
			AA[row2+6] = -h*dd.x*dd.z;
			AA[row2+7] = -h*dd.y*dd.z;
			AA[row2+8] = -h*dd.z*dd.z;
			
			//TODO Set the three appropriate entries in b vector
			bb[i*NUMCOMPONENTS  ] = particle[i]->velocity.x + g*((uu-ui).dot(du)-length)*du.x + g*((ud-ui).dot(dd)-length)*dd.x + gravity.x*(dt/2.0f);
			bb[i*NUMCOMPONENTS+1] = particle[i]->velocity.y + g*((uu-ui).dot(du)-length)*du.y + g*((ud-ui).dot(dd)-length)*dd.y + gravity.y*(dt/2.0f);
			bb[i*NUMCOMPONENTS+2] = particle[i]->velocity.z + g*((uu-ui).dot(du)-length)*du.z + g*((ud-ui).dot(dd)-length)*dd.z + gravity.z*(dt/2.0f);
		}
		
		//Last particle boundary condition
		Vector3f duN(particle[numParticles-2]->pos-particle[numParticles-1]->pos);
		duN.unitize();
		
		//TODO Set third to last row of matrix A
		AA[(numParticles-2)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-6] = -h*duN.x*duN.x;
		AA[(numParticles-2)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-5] = -h*duN.x*duN.y;
		AA[(numParticles-2)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-4] = -h*duN.x*duN.z;
		AA[(numParticles-2)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-3] = 1.0f + h*duN.x*duN.x;
		AA[(numParticles-2)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-2] = h*duN.x*duN.y;
		AA[(numParticles-2)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-1] = h*duN.x*duN.z;
		
		//TODO Set second to last row of matrix A
		AA[(numParticles-1)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-6] = -h*duN.x*duN.y;
		AA[(numParticles-1)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-5] = -h*duN.y*duN.y;
		AA[(numParticles-1)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-4] = -h*duN.y*duN.z;
		AA[(numParticles-1)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-3] = h*duN.x*duN.y;
		AA[(numParticles-1)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-2] = 1.0f + h*duN.y*duN.y;
		AA[(numParticles-1)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-1] = h*duN.y*duN.z;
		
		//TODO Set last row of matrix A
		AA[numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-6] = -h*duN.x*duN.z;
		AA[numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-5] = -h*duN.y*duN.z;
		AA[numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-4] = -h*duN.z*duN.z;
		AA[numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-3] = h*duN.x*duN.z;
		AA[numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-2] = h*duN.y*duN.z;
		AA[numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS-1] = 1.0f + h*duN.z*duN.z;
		
		for(int i = 0; i < numParticles * NUMCOMPONENTS; i++)
		{
			for(int j = 0; j < numParticles * NUMCOMPONENTS; j++)
			{
				std::cout << AA[i*numParticles*NUMCOMPONENTS + j] << " " << std::ends;
			}
			
			std::cout << std::endl;
		}
		
		//TODO Set last three entries of vector b
		bb[numParticles*NUMCOMPONENTS-3] = particle[numParticles-1]->velocity.x + g*((particle[numParticles-2]->pos-particle[numParticles-1]->pos).dot(duN)-length)*duN.x + gravity.x*(dt/2.0f);
		bb[numParticles*NUMCOMPONENTS-2] = particle[numParticles-1]->velocity.y + g*((particle[numParticles-2]->pos-particle[numParticles-1]->pos).dot(duN)-length)*duN.y + gravity.y*(dt/2.0f);
		bb[numParticles*NUMCOMPONENTS-1] = particle[numParticles-1]->velocity.z + g*((particle[numParticles-2]->pos-particle[numParticles-1]->pos).dot(duN)-length)*duN.z + gravity.z*(dt/2.0f);
		
		for(int i = 0; i < numParticles; i++)
		{
			for(int j = 0; j < NUMCOMPONENTS; j++)
			{
				std::cout << bb[i*NUMCOMPONENTS + j] << std::endl;
			}
			std::cout << std::endl;
		}
	}
	*/
	
	void Strand::buildAB(float dt)
	{
		memset(AA, 0, sizeof(float)*numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS);
		
		//TODO Find reasonable value for damping coefficient (d_edge) that allows for a more stable system
		float h = dt*dt*k_edge/(4.0f*mass*length) + d_edge*dt/(2.0f*mass);
		float g = dt*k_edge/(2.0f*mass*length);
		Vector3f gravity(0.0f, GRAVITY, 0.0f);
		
		//First particle direction vectors
		Vector3f du0R(root-particle[0]->pos);
		Vector3f du01(particle[1]->pos-particle[0]->pos);
		du0R.unitize();
		du01.unitize();
		
		//TODO Set first six entries of the first row of A matrix
		AA[0] = 1.0f + h*du0R.x*du0R.x + h*du01.x*du01.x;
		AA[1] = h*du0R.x*du0R.y + h*du01.x*du01.y;
		AA[2] = -h*du01.x*du01.x;
		AA[3] = -h*du01.x*du01.y;
		
		//TODO Set next six non-zero entries of the second row of matrix A
		AA[numParticles*NUMCOMPONENTS  ] = h*du0R.x*du0R.y + h*du01.x*du01.y;
		AA[numParticles*NUMCOMPONENTS+1] = 1.0f + h*du0R.y*du0R.y + h*du01.y*du01.y;
		AA[numParticles*NUMCOMPONENTS+2] = -h*du01.x*du01.y;
		AA[numParticles*NUMCOMPONENTS+3] = -h*du01.y*du01.y;
		
		//TODO Set the next six non-zero entries of the third row of matrix A
		
		//TODO Set the first three entries of the b vector
		bb[0] = particle[0]->velocity.x + g*((root-particle[0]->pos).dot(du0R)-length)*du0R.x + g*((particle[1]->pos-particle[0]->pos).dot(du01)-length)*du01.x + gravity.x*(dt/2.0f);
		bb[1] = particle[0]->velocity.y + g*((root-particle[0]->pos).dot(du0R)-length)*du0R.y + g*((particle[1]->pos-particle[0]->pos).dot(du01)-length)*du01.y + gravity.y*(dt/2.0f);
		
		//Build in-between values of matrix A and vector b
		for(int i = 1; i < (numParticles-1); i++)
		{
			//Current particle position, particle above and particle below
			Vector3f ui = particle[i]->pos;
			Vector3f uu = particle[i-1]->pos;
			Vector3f ud = particle[i+1]->pos;
			
			Vector3f du(uu-particle[i]->pos);
			Vector3f dd(ud-particle[i]->pos);
			du.unitize();
			dd.unitize();
			
			//TODO Set first row along diagonal
			AA[(i*NUMCOMPONENTS)*NUMCOMPONENTS*numParticles + i*NUMCOMPONENTS-2] = -h*du.x*du.x;
			AA[(i*NUMCOMPONENTS)*NUMCOMPONENTS*numParticles + i*NUMCOMPONENTS-1] = -h*du.x*du.y;
			AA[(i*NUMCOMPONENTS)*NUMCOMPONENTS*numParticles + i*NUMCOMPONENTS  ] = 1.0f + h*du.x*du.x + h*dd.x*dd.x; //Diagonal
			AA[(i*NUMCOMPONENTS)*NUMCOMPONENTS*numParticles + i*NUMCOMPONENTS+1] = h*du.x*du.y + h*dd.x*dd.y;
			AA[(i*NUMCOMPONENTS)*NUMCOMPONENTS*numParticles + i*NUMCOMPONENTS+2] = -h*dd.x*dd.x;
			AA[(i*NUMCOMPONENTS)*NUMCOMPONENTS*numParticles + i*NUMCOMPONENTS+3] = -h*dd.x*dd.y;
			
			//TODO Set second row along diagonal
			AA[(i*NUMCOMPONENTS+1)*NUMCOMPONENTS*numParticles + i*NUMCOMPONENTS-2] = -h*du.x*du.y;
			AA[(i*NUMCOMPONENTS+1)*NUMCOMPONENTS*numParticles + i*NUMCOMPONENTS-1] = -h*du.y*du.y;
			AA[(i*NUMCOMPONENTS+1)*NUMCOMPONENTS*numParticles + i*NUMCOMPONENTS  ] = h*du.x*du.y + h*dd.x*dd.y;
			AA[(i*NUMCOMPONENTS+1)*NUMCOMPONENTS*numParticles + i*NUMCOMPONENTS+1] = 1.0f + h*du.y*du.y + h*dd.y*dd.y; //Diagonal
			AA[(i*NUMCOMPONENTS+1)*NUMCOMPONENTS*numParticles + i*NUMCOMPONENTS+2] = -h*dd.x*dd.y;
			AA[(i*NUMCOMPONENTS+1)*NUMCOMPONENTS*numParticles + i*NUMCOMPONENTS+3] = -h*dd.y*dd.y;
			
			//TODO Set third row along diagonal
			
			//TODO Set the three appropriate entries in b vector
			bb[i*NUMCOMPONENTS  ] = particle[i]->velocity.x + g*((uu-ui).dot(du)-length)*du.x + g*((ud-ui).dot(dd)-length)*dd.x + gravity.x*(dt/2.0f);
			bb[i*NUMCOMPONENTS+1] = particle[i]->velocity.y + g*((uu-ui).dot(du)-length)*du.y + g*((ud-ui).dot(dd)-length)*dd.y + gravity.y*(dt/2.0f);
		}
		
		//Last particle boundary condition
		Vector3f duN(particle[numParticles-2]->pos-particle[numParticles-1]->pos);
		duN.unitize();
		
		//TODO Set third to last row of matrix A
		
		//TODO Set second to last row of matrix A
		AA[(numParticles-1)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS + numParticles*NUMCOMPONENTS-4] = -h*duN.x*duN.x;
		AA[(numParticles-1)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS + numParticles*NUMCOMPONENTS-3] = -h*duN.x*duN.y;
		AA[(numParticles-1)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS + numParticles*NUMCOMPONENTS-2] = 1.0f + h*duN.x*duN.x;
		AA[(numParticles-1)*NUMCOMPONENTS*numParticles*NUMCOMPONENTS + numParticles*NUMCOMPONENTS-1] = h*duN.x*duN.y;
		
		//TODO Set last row of matrix A
		AA[numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS - 4] = -h*duN.x*duN.y;
		AA[numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS - 3] = -h*duN.y*duN.y;
		AA[numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS - 2] = h*duN.x*duN.y;
		AA[numParticles*NUMCOMPONENTS*numParticles*NUMCOMPONENTS - 1] = 1.0f + h*duN.y*duN.y;
		
		//TODO Set last three entries of vector b
		bb[numParticles*NUMCOMPONENTS-2] = particle[numParticles-1]->velocity.x + g*((particle[numParticles-2]->pos-particle[numParticles-1]->pos).dot(duN)-length)*duN.x + gravity.x*(dt/2.0f);
		bb[numParticles*NUMCOMPONENTS-1] = particle[numParticles-1]->velocity.y + g*((particle[numParticles-2]->pos-particle[numParticles-1]->pos).dot(duN)-length)*duN.y + gravity.y*(dt/2.0f);
	}
	
	void Strand::calcVelocities(float dt)
	{
		//Calculate the velocities of each particle
		
		//Build matrix and vector of coefficients of linear equations		
		buildAB(dt);
		
		//TODO Set intial solution to previous velocity
		for(int i = 0; i < numParticles; i++)
		{
			xx[i*NUMCOMPONENTS  ] = particle[i]->velocity.x;
			xx[i*NUMCOMPONENTS+1] = particle[i]->velocity.y;
			//~ xx[i*NUMCOMPONENTS+2] = particle[i]->velocity.z;
		}
		
		//Solve for velocity using conjugate gradient method
		conjugate();
		
		//TODO Copy solution to half velocity
		for(int i = 0; i < numParticles; i++)
		{
			particle[i]->velh.x = xx[i*NUMCOMPONENTS  ];
			particle[i]->velh.y = xx[i*NUMCOMPONENTS+1];
			//~ particle[i]->velh.z = xx[i*NUMCOMPONENTS+2];
		}
	}
	
	void Strand::updateSprings(float dt)
	{
		float g = k_edge/length;
		float h = dt*k_edge/(2.0f*length) + d_edge;
		
		//Calculate force for first particle
		Vector3f uu0(root-particle[0]->pos);
		Vector3f ud0(particle[1]->pos-particle[0]->pos);
		Vector3f du0(uu0.unit());
		Vector3f dd0(ud0.unit());
		Vector3f vu0(-particle[0]->velh);
		Vector3f vd0(particle[1]->velh-particle[0]->velh);
		
		Vector3f force0 = du0*(g*(uu0.dot(du0)-length) + h*(vu0.dot(du0))) + dd0*(g*(ud0.dot(dd0)-length) + h*(vd0.dot(dd0)));
		
		particle[0]->applyForce(force0);
		
		//Calculate force for all particles between first and last
		for(int i = 1; i < (numParticles-1); i++)
		{
			Vector3f uu(particle[i-1]->pos-particle[i]->pos);
			Vector3f ud(particle[i+1]->pos-particle[i]->pos);
			Vector3f du(uu.unit());
			Vector3f dd(ud.unit());
			Vector3f vu(particle[i-1]->velh-particle[i]->velh);
			Vector3f vd(particle[i+1]->velh-particle[i]->velh);
			
			Vector3f force = du*(g*(uu.dot(du)-length) + h*(vu.dot(du))) + dd*(g*(ud.dot(dd)-length) + h*(vd.dot(dd)));
			
			particle[i]->applyForce(force);
		}
		
		//Calculate force for last particle
		Vector3f uuN(particle[numParticles-2]->pos-particle[numParticles-1]->pos);
		Vector3f duN(uuN.unit());
		Vector3f vuN(particle[numParticles-2]->velh-particle[numParticles-1]->velh);
		
		Vector3f forceN = duN*(g*(uuN.dot(duN)-length) + h*(vuN.dot(duN)));
		
		particle[numParticles-1]->applyForce(forceN);
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
		
		//TODO Apply other forces such as wind
		
		//Calculate half velocity and new velocity
		updateParticles(dt);
		
		//Detect segment collisions, calculate stiction forces and apply stiction to velocity
		applyStiction2(dt, strand, collision);
		
		//TODO Check when is best to update bounding volumes
		//TODO Check if bounding volumes need updating between half time steps
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
			   float length,
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
			strand[i] = new Strand(numParticles, i, numStrands, mass, k_edge, k_bend, k_twist, k_extra, d_edge, d_bend, d_twist, d_extra, length, roots[i], normals[i]); 
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

