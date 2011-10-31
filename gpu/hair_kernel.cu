
#ifndef _HAIR_KERNEL_H_
#define _HAIR_KERNEL_H_

#include "constants.h"

__device__
void clearForces(int numParticles, float3* force)
{
	for(int i = 0; i < numParticles; i++)
	{
		force[i].x = 0.0f;
		force[i].y = 0.0f;
		force[i].z = 0.0f;
	}
}

__device__
float inv_sqrt(float number)
{
	float xhalf = 0.5f*number;
	int i = *(int*)&number; // get bits for floating value
	i = 0x5f375a86- (i>>1); // gives initial guess y0
	number = *(float*)&i; // convert bits back to float
	number = number*(1.5f-xhalf*number*number); // Newton step, repeating increases accuracy
	
  	return number;
}

__device__
void updateSprings(int numSprings,
				   int stride,
				   float mass,
				   float length,
				   float dt,
				   float k,
				   float3* position,
				   float3* velocity,
				   float3* force)
{
	for(int i = 0; i < numSprings; i++)
	{
		float3 xn = make_float3(position[i+stride].x-position[i].x, position[i+stride].y-position[i].y, position[i+stride].z-position[i].z);
		float3 vn = make_float3(velocity[i+stride].x-velocity[i].x, velocity[i+stride].y-velocity[i].y, velocity[i+stride].z-velocity[i].z);
		
		float inv_length = inv_sqrt(xn.x*xn.x + xn.y*xn.y + xn.z*xn.z);
		float3 d = make_float3(xn.x*inv_length, xn.y*inv_length, xn.z*inv_length);
		
		float f = k / length * mass;
		
		//TODO CONJUGATE GRADIENT!!!!!
		
		//Store resulting velocity here
		float3 v1 = make_float3(0.0f, 0.0f, 0.0f);
		
		//Store resulting force here
		float3 result;
		
		result.x = d.x * (f * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt * (v1.x*d.x + v1.y*d.y + v1.z*d.z)));
		result.y = d.y * (f * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt * (v1.x*d.x + v1.y*d.y + v1.z*d.z)));
		result.z = d.z * (f * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt * (v1.x*d.x + v1.y*d.y + v1.z*d.z)));
		
		force[i].x += result.x * -1.0f; 
		force[i].y += result.y * -1.0f;
		force[i].z += result.z * -1.0f;
		force[i+stride].x += result.x; 
		force[i+stride].y += result.y;
		force[i+stride].z += result.z;
	}
}

__device__
void applyForce(int numParticles, float3 f, float3* force)
{
	for(int i = 1; i < numParticles; i++)
	{
		force[i].x += f.x;
		force[i].y += f.y;
		force[i].z += f.z;
	}
}

__device__
void updateVelocities(int numParticles,
					  float dt,
					  float3* velocity,
					  float3* velh,
					  float3* force)
{
	for(int i = 1; i < numParticles; i++)
	{
		velh[i].x = velocity[i].x + force[i].x * (dt / 2.0f);
		velh[i].y = velocity[i].y + force[i].y * (dt / 2.0f);
		velh[i].z = velocity[i].z + force[i].z * (dt / 2.0f);
	}
}

__device__
void updatePositions(int numParticles,
					 float dt,
					 float3* position,
					 float3* posh,
					 float3* velh)
{
	for(int i = 1; i < numParticles; i++)
	{
		float3 newPosition = make_float3(position[i].x + velh[i].x * dt,
										 position[i].y + velh[i].y * dt,
										 position[i].z + velh[i].z * dt);
		
		posh[i].x = (position[i].x + newPosition.x)/2.0f;
		posh[i].y = (position[i].y + newPosition.y)/2.0f;
		posh[i].z = (position[i].z + newPosition.z)/2.0f;
		
		position[i].x = newPosition.x;
		position[i].y = newPosition.y;
		position[i].z = newPosition.z;
	}
}

__device__
void updateParticles(int numParticles, float dt, float3* velocity, float3* velh, float3* force)
{
	for(int i = 1; i < numParticles; i++)
	{
		velh[i].x = velocity[i].x + force[i].x * (dt / 2.0f);
		velh[i].y = velocity[i].y + force[i].y * (dt / 2.0f);
		velh[i].z = velocity[i].z + force[i].z * (dt / 2.0f);
		
		velocity[i].x = velh[i].x * 2.0f - velocity[i].x;
		velocity[i].y = velh[i].y * 2.0f - velocity[i].y;
		velocity[i].z = velh[i].z * 2.0f - velocity[i].z;
	}
}

__device__
void conjugate(int N, const float* A, const float* b, float* x, float* r, float* p, float* Ap)
{
//	float r[NUMPARTICLES*3];
//	float p[NUMPARTICLES*3];

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
//		float Ap[NUMPARTICLES*3];
	
		for(int j = 0; j < N; j++)
		{
			Ap[j] = 0.0f;
		
//			for(int k = 0; k < N; k++)
//			{
//				Ap[j] += A[j*N+k] * p[k];
//			}
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
	
		if(rsnew < 1e-10f) break;
	
		for(int j = 0; j < N; j++)
		{
			p[j] = r[j] + rsnew / rsold * p[j];
		}
	
		rsold = rsnew;
	}
}

__device__
void calcVelocities(int numParticles,
					float dt,
					float mass,
					float length,
					float4 k,
					float3* position,
					float3* velocity,
					float3* velc,
					float* A,
					float* b,
					float* x,
					float* r,
					float* p,
					float* Ap)
{
	int N = numParticles * 3;
	int numEdges = numParticles - 1;
	int numBend  = numParticles - 2;
	int numTwist = numParticles - 3;
	
	for(int i = 0; i < numParticles; i++)
	{
		int xx = i * 3;
		int yy = xx + 1;
		int zz = yy + 1;
		
		A[xx] = 0.0f;
		A[yy] = 0.0f;
		A[zz] = 0.0f;
		
		b[xx] = 0.0f;
		b[yy] = 0.0f;
		b[zz] = 0.0f;
		
		x[xx] = velocity[i].x;
		x[yy] = velocity[i].y;
		x[zz] = velocity[i].z;
	}
	
	float h = dt * dt * k.x / ( 4.0f * mass * length );
	float g = dt * k.x / (2.0f * mass * length);
	
	for(int i = 0; i < numEdges; i++)
	{
		float3 d = make_float3(position[i+1].x-position[i].x,position[i+1].y-position[i].y,position[i+1].z-position[i].z);
		float3 e = make_float3(d.x, d.y, d.z);
		
		float dlength = inv_sqrt(d.x*d.x+d.y*d.y+d.z*d.z);
		d.x = d.x * dlength;
		d.y = d.y * dlength;
		d.z = d.z * dlength;
	
		A[(i*3)*N  +i*3] += 1+h*d.x*d.x; A[(i*3)*N  +i*3+1] +=   h*d.x*d.y; A[(i*3)*N  +i*3+2] +=   h*d.x*d.z; A[(i*3)*N  +i*3+3] +=  -h*d.x*d.x; A[(i*3)*N  +i*3+4] +=  -h*d.x*d.y; A[(i*3)*N  +i*3+5] +=  -h*d.x*d.z;
		A[(i*3+1)*N+i*3] +=   h*d.x*d.y; A[(i*3+1)*N+i*3+1] += 1+h*d.y*d.y; A[(i*3+1)*N+i*3+2] +=   h*d.y*d.z; A[(i*3+1)*N+i*3+3] +=  -h*d.x*d.y; A[(i*3+1)*N+i*3+4] +=  -h*d.y*d.y; A[(i*3+1)*N+i*3+5] +=  -h*d.y*d.z;
		A[(i*3+2)*N+i*3] +=   h*d.x*d.z; A[(i*3+2)*N+i*3+1] +=   h*d.y*d.z; A[(i*3+2)*N+i*3+2] += 1+h*d.z*d.z; A[(i*3+2)*N+i*3+3] +=  -h*d.x*d.z; A[(i*3+2)*N+i*3+4] +=  -h*d.y*d.z; A[(i*3+2)*N+i*3+5] +=  -h*d.z*d.z;
		A[(i*3+3)*N+i*3] +=  -h*d.x*d.x; A[(i*3+3)*N+i*3+1] +=  -h*d.x*d.y; A[(i*3+3)*N+i*3+2] +=  -h*d.x*d.z; A[(i*3+3)*N+i*3+3] += 1+h*d.x*d.x; A[(i*3+3)*N+i*3+4] +=   h*d.x*d.y; A[(i*3+3)*N+i*3+5] +=   h*d.x*d.z;
		A[(i*3+4)*N+i*3] +=  -h*d.x*d.y; A[(i*3+4)*N+i*3+1] +=  -h*d.y*d.y; A[(i*3+4)*N+i*3+2] +=  -h*d.y*d.z; A[(i*3+4)*N+i*3+3] +=   h*d.x*d.y; A[(i*3+4)*N+i*3+4] += 1+h*d.y*d.y; A[(i*3+4)*N+i*3+5] +=   h*d.y*d.z;
		A[(i*3+5)*N+i*3] +=  -h*d.x*d.z; A[(i*3+5)*N+i*3+1] +=  -h*d.y*d.z; A[(i*3+5)*N+i*3+2] +=  -h*d.z*d.z; A[(i*3+5)*N+i*3+3] +=   h*d.x*d.z; A[(i*3+5)*N+i*3+4] +=   h*d.y*d.z; A[(i*3+5)*N+i*3+5] += 1+h*d.z*d.z;
	
		float factor = g * ((e.x*d.x+e.y*d.y+e.z*d.z) - (length));
	
		b[i*3]   += velocity[i].x   + factor * d.x;
		b[i*3+1] += velocity[i].y   + factor * d.y;
		b[i*3+2] += velocity[i].z   + factor * d.z;
		b[i*3+3] += velocity[i+1].x - factor * d.x;
		b[i*3+4] += velocity[i+1].y - factor * d.y;
		b[i*3+5] += velocity[i+1].z - factor * d.z;
	}
	
	//Add bending springs data into A and b
	h = dt * dt * k.y / (4.0f * mass * length);
	g = dt * k.y / (2.0f * mass * length);
	
	for(int i = 0; i < numBend; i++)
	{
		float3 d = make_float3(position[i+2].x-position[i].x,position[i+2].y-position[i].y,position[i+2].z-position[i].z);
		float3 e = make_float3(d.x, d.y, d.z);
		
		float dlength = inv_sqrt(d.x*d.x+d.y*d.y+d.z*d.z);
		d.x = d.x * dlength;
		d.y = d.y * dlength;
		d.z = d.z * dlength;
		
		A[(i*3  )*N+i*3] += 1+h*d.x*d.x; A[(i*3  )*N+i*3+1] +=   h*d.x*d.y; A[(i*3  )*N+i*3+2] +=   h*d.x*d.z; A[(i*3  )*N+i*3+6] +=  -h*d.x*d.x; A[(i*3  )*N+i*3+7] +=  -h*d.x*d.y; A[(i*3  )*N+i*3+8] +=  -h*d.x*d.z;
		A[(i*3+1)*N+i*3] +=   h*d.x*d.y; A[(i*3+1)*N+i*3+1] += 1+h*d.y*d.y; A[(i*3+1)*N+i*3+2] +=   h*d.y*d.z; A[(i*3+1)*N+i*3+6] +=  -h*d.x*d.y; A[(i*3+1)*N+i*3+7] +=  -h*d.y*d.y; A[(i*3+1)*N+i*3+8] +=  -h*d.y*d.z;
		A[(i*3+2)*N+i*3] +=   h*d.x*d.z; A[(i*3+2)*N+i*3+1] +=   h*d.y*d.z; A[(i*3+2)*N+i*3+2] += 1+h*d.z*d.z; A[(i*3+2)*N+i*3+6] +=  -h*d.x*d.z; A[(i*3+2)*N+i*3+7] +=  -h*d.y*d.z; A[(i*3+2)*N+i*3+8] +=  -h*d.z*d.z;
		A[(i*3+6)*N+i*3] +=  -h*d.x*d.x; A[(i*3+6)*N+i*3+1] +=  -h*d.x*d.y; A[(i*3+6)*N+i*3+2] +=  -h*d.x*d.z; A[(i*3+6)*N+i*3+6] += 1+h*d.x*d.x; A[(i*3+6)*N+i*3+7] +=   h*d.x*d.y; A[(i*3+6)*N+i*3+8] +=   h*d.x*d.z;
		A[(i*3+7)*N+i*3] +=  -h*d.x*d.y; A[(i*3+7)*N+i*3+1] +=  -h*d.y*d.y; A[(i*3+7)*N+i*3+2] +=  -h*d.y*d.z; A[(i*3+7)*N+i*3+6] +=   h*d.x*d.y; A[(i*3+7)*N+i*3+7] += 1+h*d.y*d.y; A[(i*3+7)*N+i*3+8] +=   h*d.y*d.z;
		A[(i*3+8)*N+i*3] +=  -h*d.x*d.z; A[(i*3+8)*N+i*3+1] +=  -h*d.y*d.z; A[(i*3+8)*N+i*3+2] +=  -h*d.z*d.z; A[(i*3+8)*N+i*3+6] +=   h*d.x*d.z; A[(i*3+8)*N+i*3+7] +=   h*d.y*d.z; A[(i*3+8)*N+i*3+8] += 1+h*d.z*d.z;
		
		float factor = g * ((e.x*d.x+e.y*d.y+e.z*d.z) - (length));
		
		b[i*3]   += velocity[i].x   + factor * d.x;
		b[i*3+1] += velocity[i].y   + factor * d.y;
		b[i*3+2] += velocity[i].z   + factor * d.z;
		b[i*3+6] += velocity[i+2].x - factor * d.x;
		b[i*3+7] += velocity[i+2].y - factor * d.y;
		b[i*3+8] += velocity[i+2].z - factor * d.z;
	}
	
	//Add twisting springs data into A and b
	h = dt * dt * k.z / (4.0f * mass * length);
	g = dt * k.z / (2.0f * mass * length);

	for(int i = 0; i < numTwist; i++)
	{
		float3 d = make_float3(position[i+3].x-position[i].x,position[i+3].y-position[i].y,position[i+3].z-position[i].z);
		float3 e = make_float3(d.x, d.y, d.z);
		
		float dlength = inv_sqrt(d.x*d.x+d.y*d.y+d.z*d.z);
		d.x = d.x * dlength;
		d.y = d.y * dlength;
		d.z = d.z * dlength;
	
		A[(i*3   )*N+i*3] += 1+h*d.x*d.x; A[(i*3   )*N+i*3+1] +=   h*d.x*d.y; A[(i*3   )*N+i*3+2] +=   h*d.x*d.z; A[(i*3   )*N+i*3+9] +=  -h*d.x*d.x; A[(i*3   )*N+i*3+10] +=  -h*d.x*d.y; A[(i*3   )*N+i*3+11] +=  -h*d.x*d.z;
		A[(i*3+1 )*N+i*3] +=   h*d.x*d.y; A[(i*3+1 )*N+i*3+1] += 1+h*d.y*d.y; A[(i*3+1 )*N+i*3+2] +=   h*d.y*d.z; A[(i*3+1 )*N+i*3+9] +=  -h*d.x*d.y; A[(i*3+1 )*N+i*3+10] +=  -h*d.y*d.y; A[(i*3+1 )*N+i*3+11] +=  -h*d.y*d.z;
		A[(i*3+2 )*N+i*3] +=   h*d.x*d.z; A[(i*3+2 )*N+i*3+1] +=   h*d.y*d.z; A[(i*3+2 )*N+i*3+2] += 1+h*d.z*d.z; A[(i*3+2 )*N+i*3+9] +=  -h*d.x*d.z; A[(i*3+2 )*N+i*3+10] +=  -h*d.y*d.z; A[(i*3+2 )*N+i*3+11] +=  -h*d.z*d.z;
		A[(i*3+9 )*N+i*3] +=  -h*d.x*d.x; A[(i*3+9 )*N+i*3+1] +=  -h*d.x*d.y; A[(i*3+9 )*N+i*3+2] +=  -h*d.x*d.z; A[(i*3+9 )*N+i*3+9] += 1+h*d.x*d.x; A[(i*3+9 )*N+i*3+10] +=   h*d.x*d.y; A[(i*3+9 )*N+i*3+11] +=   h*d.x*d.z;
		A[(i*3+10)*N+i*3] +=  -h*d.x*d.y; A[(i*3+10)*N+i*3+1] +=  -h*d.y*d.y; A[(i*3+10)*N+i*3+2] +=  -h*d.y*d.z; A[(i*3+10)*N+i*3+9] +=   h*d.x*d.y; A[(i*3+10)*N+i*3+10] += 1+h*d.y*d.y; A[(i*3+10)*N+i*3+11] +=   h*d.y*d.z;
		A[(i*3+11)*N+i*3] +=  -h*d.x*d.z; A[(i*3+11)*N+i*3+1] +=  -h*d.y*d.z; A[(i*3+11)*N+i*3+2] +=  -h*d.z*d.z; A[(i*3+11)*N+i*3+9] +=   h*d.x*d.z; A[(i*3+11)*N+i*3+10] +=   h*d.y*d.z; A[(i*3+11)*N+i*3+11] += 1+h*d.z*d.z;
		
		float factor = g * ((e.x*d.x+e.y*d.y+e.z*d.z) - (length));
		
		b[i*3   ] += velocity[i].x   + factor * d.x;
		b[i*3+1 ] += velocity[i].y   + factor * d.y;
		b[i*3+2 ] += velocity[i].z   + factor * d.z;
		b[i*3+9 ] += velocity[i+3].x - factor * d.x;
		b[i*3+10] += velocity[i+3].y - factor * d.y;
		b[i*3+11] += velocity[i+3].z - factor * d.z;
	}
	
//	conjugate(N, A, b, x, r, p, Ap);
	
	for(int i = 0; i < numParticles; i++)
	{
		velc[i].x = x[i*3  ];
		velc[i].y = x[i*3+1];
		velc[i].z = x[i*3+2];
	}
}

__device__
void applyStrainLimiting(int numParticles, float dt, float3* position, float3* posc, float3* velh)
{
	bool strained = true;
	
	while(strained)
	{
		strained = false;
		
		for(int i = 1; i < numParticles; i++)
		{
			posc[i].x = position[i].x + velh[i].x * dt;
			posc[i].y = position[i].y + velh[i].y * dt;
			posc[i].z = position[i].z + velh[i].z * dt;
			
			float3 dir = make_float3(posc[i].x-posc[i-1].x, posc[i].y-posc[i-1].y, posc[i].z-posc[i-1].z);
			float length_sqr = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
			
			if(length_sqr > MAX_LENGTH_SQUARED)
			{
				strained = true;
				
				float length_inv = inv_sqrt(length_sqr);
				
				posc[i].x = posc[i-1].x + (dir.x * (MAX_LENGTH*length_inv));
				posc[i].y = posc[i-1].y + (dir.y * (MAX_LENGTH*length_inv));
				posc[i].z = posc[i-1].z + (dir.z * (MAX_LENGTH*length_inv));
				
				velh[i].x = (posc[i].x - position[i].x)/dt;
				velh[i].y = (posc[i].y - position[i].y)/dt;
				velh[i].z = (posc[i].z - position[i].z)/dt;
			}
		}
	}
}

__global__
void update(const int numParticles,
			float4 mlgt,
			const float4 k,
			const float4 d,
			float3* position,
			float3* posc,
			float3* posh,
			float3* velocity,
			float3* velc,
			float3* velh,
			float3* force,
			float* A,
			float* b,
			float* x,
			float* r,
			float* p,
			float* Ap)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	int start = numParticles * tid; //first particle index
//	int end = start + numParticles; //last particle index
	
	clearForces(numParticles, force+start);
	
	calcVelocities(numParticles,
				   mlgt.w,
				   mlgt.x,
				   mlgt.y,
				   k,
				   position+start,
				   velocity+start,
				   velc+start,
				   A,
				   b,
				   x,
				   r,
				   p,
				   Ap);
	
	//Update edge springs
	updateSprings(numParticles-1, 1, mlgt.x, mlgt.y, mlgt.w, k.x, position+start, velocity+start, force+start);
	
	//Update bend springs
	updateSprings(numParticles-2, 2, mlgt.x, mlgt.y, mlgt.w, k.y, position+start, velocity+start, force+start);
	
	//Update twist springs
	updateSprings(numParticles-3, 3, mlgt.x, mlgt.y, mlgt.w, k.z, position+start, velocity+start, force+start);
	
	//TODO Update extra springs
	
	//Apply gravity
	applyForce(numParticles, make_float3(0.0f,mlgt.z, 0.0f), force+start);
	
	updateVelocities(numParticles, mlgt.w, velocity+start, velh+start, force+start);
	
	//TODO apply strain limiting
	applyStrainLimiting(numParticles, mlgt.w, position+start, posc+start, velh+start);
	
	//Calculate half position and new position
	updatePositions(numParticles, mlgt.w, position+start, posh+start, velh+start);
	
	//Reset forces on particles
	clearForces(numParticles, force+start);
	
	calcVelocities(numParticles,
				   mlgt.w,
				   mlgt.x,
				   mlgt.y,
				   k,
				   position+start,
				   velocity+start,
				   velc+start,
				   A,
				   b,
				   x,
				   r,
				   p,
				   Ap);
	
	//Calculate and apply spring forces using half position
	//Update edge springs
	updateSprings(numParticles-1, 1, mlgt.x, mlgt.y, mlgt.w, k.x, posh+start, velocity+start, force+start);
	
	//Update bend springs
	updateSprings(numParticles-2, 2, mlgt.x, mlgt.y, mlgt.w, k.y, posh+start, velocity+start, force+start);
	
	//Update twist springs
	updateSprings(numParticles-3, 3, mlgt.x, mlgt.y, mlgt.w, k.z, posh+start, velocity+start, force+start);
	
	//Apply gravity
	applyForce(numParticles, make_float3(0.0f,mlgt.z, 0.0f), force+start);
	
	//Calculate half velocity and new velocity
	updateParticles(numParticles, mlgt.w, velocity+start, velh+start, force+start);
}

#endif

