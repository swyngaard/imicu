
#ifndef __HAIR_H__
#define __HAIR_H__

#include "tools.h"
#include "constants.h"
#include "ogl.h"
#include "tree.h"
#include <vector>

namespace pilar
{
	class Particle
	{
	public:
		float mass;
		bool freeze;
		
		Vector3f posh; //half
		Vector3f posc; //candidate
		Vector3f pos;  //current
		Vector3f position; //previous
		
		Vector3f velh;
		Vector3f velocity;
		
		Vector3f force;
		
		Particle(float mass);
		
		void clearForces();
		void applyForce(Vector3f force);
		void updateVelocity(float dt);
		void updatePosition(float dt);
		void update(float dt);
	};
	
	class Collision
	{
	protected:
		int strandID;
		int segmentID;
		int particleID[2];
	public:
		
		Collision(int strandID_, int segmentID_):strandID(strandID_),segmentID(segmentID_)
		{
			particleID[0] = segmentID;
			particleID[1] = segmentID_ + 1;
		}
		
		int getStrandID() { return strandID; }
		int getSegmentID() { return segmentID; }
		int getParticleOneID() { return particleID[0]; }
		int getParticleTwoID() { return particleID[1]; }
	};
	
	class Strand
	{
	
	protected:
		int numEdges;
		int numBend;
		int numTwist;
		
		float k_edge;
		float k_bend;
		float k_twist;
		float k_extra;
		
		float d_edge;
		float d_bend;
		float d_twist;
		float d_extra;
		
		float length_e;
		float length_b;
		float length_t;
		
		float mass;
		
		float* xx;
		float* AA;
		
		
		std::vector<KDOP*> leafKDOP;
		Node* bvhTree;
		
		int strandID;
		int numStrands;
		
		void clearForces();
		void updateSprings(float dt);
		void updateVelocities(float dt);
		void updatePositions(float dt);
		void updateParticles(float dt);
		void calcVelocities(float dt);
		void objectCollisions(float dt, const float (&grid)[DOMAIN_DIM][DOMAIN_DIM][DOMAIN_DIM]);
		void applyStrainLimiting(float dt);
		void applyStiction(float dt, Strand** strand, std::vector<Collision> (&collision)[NUMSTRANDS][NUMSEGMENTS]);
		void applyStiction2(float dt, Strand** strand, std::vector<Collision> (&collision)[NUMSTRANDS][NUMSEGMENTS]);
		void updateBoundingVolumes();
		void addVolumeVertices(std::vector<Vector3f> &vertices);
		
		void buildAB(float dt);
		void conjugate();
		
		bool printOnce;
		
	public:
		int numParticles;
		Particle** particle;
		Vector3f root;
		float* bb;
		
		Strand(int numParticles,
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
			   Vector3f normal);
		
		void update(float dt, const float (&grid)[DOMAIN_DIM][DOMAIN_DIM][DOMAIN_DIM], Strand** strand, std::vector<Collision> (&collision)[NUMSTRANDS][NUMSEGMENTS]);
		void release();
		void applyForce(Vector3f force);
				
		Node* getTree();
		
	};
	
	class Hair
	{
	protected:
		Vector3f gravity;
		
		void initDistanceField(Model_OBJ &obj);
		
	public:
		int numStrands;
		Strand** strand;
		float grid[DOMAIN_DIM][DOMAIN_DIM][DOMAIN_DIM];
		
		std::vector<Collision> collision[NUMSTRANDS][NUMSEGMENTS];
		
		Hair(int numStrands,
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
			 Model_OBJ &obj);
			 
		void update(float dt);
		void release();
	};
}

#endif

