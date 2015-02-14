
#ifndef __HAIR_H__
#define __HAIR_H__

#include "tools.h"
#include <vector>
#include <vector_types.h>

namespace pilar
{
	class Particle
	{
	public:
		float mass;
		
		Vector3f posh;
		Vector3f posc;
		Vector3f position;
		
		
		Vector3f velh;
		Vector3f velocity;
		
		Vector3f force;
		
		Particle(float mass);
		
		void clearForces();
		void applyForce(Vector3f force);
		void updateVelocity(float dt);
		void updatePosition(float dt);
		void update();
	};
	
	class Spring
	{
	protected:
		Particle** particle;
		float k;
		float length;
		float damping;
		
		void updateForce(Vector3f p0, Vector3f p1, float dt);
		
	public:
		Spring(Particle* particle1, Particle* particle2, float k, float length, float damping);
		void update1(float dt);
		void update2(float dt);
		void release();
	};
	
	class Strand
	{
	protected:
		int numEdges;
		int numBend;
		int numTwist;
		Vector3f root;
		
		Spring** edge;
		Spring** bend;
		Spring** twist;
		
		void buildSprings(float k_edge, float k_bend, float k_twist, float k_extra, float d_edge, float d_bend, float d_twist, float d_extra, float length);
		void clearForces();
		void updateSprings1(float dt);
		void updateSprings2(float dt);
		void updateVelocities(float dt);
		void updateParticles1(float dt);
		void updateParticles2(float dt);
		
	public:
		int numParticles;
		Particle** particle;
		
		Strand(int numParticles,
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
			   Vector3f root);
		void update(float dt);
		void release();
		void applyForce(Vector3f force);
		void applyStrainLimiting(float dt);
	};
	
	class Hair
	{
	protected:
		Vector3f gravity;
		
		float3 *pos;
		float3 *posc;
		float3 *posh;
		
		float3 *velocity;
		float3 *velc;
		float3 *velh;
		
		float3 *force;
		
		
		float3 *root_;
		float3 *normal_;
		float3 *position_;
		float3 *pos_;
		float3 *posc_;
		float3 *posh_;
		float3 *velocity_;
		float3 *velh_;
		float3 *force_;
		float *AA_;
		float *bb_;
		float *xx_;
		
		float *A;
		float *b;
		float *x;
		float *r;
		float *p;
		float *Ap;
		
		float4 mlgt; //Mass of particle, maximum length of a spring, gravity and change in time (dt)
		float4 k;  //Spring constants
		float4 d;  //Dampening constants
		
	public:
		int numStrands;
		int numParticles;
		int numComponents;
		
		float3 *position;
		
		Strand** strand;
		
		Hair(int numStrands,
			 int numParticles,
			 int numComponents,
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
			 float length_e,
			 float length_b,
			 float length_t,
			 std::vector<Vector3f> &roots,
			 std::vector<Vector3f> &normals);
		
		void update(float dt);
		void init();
		void release();
	};
}

#endif

