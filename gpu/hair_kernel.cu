
#ifndef _HAIR_KERNEL_H_
#define _HAIR_KERNEL_H_

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
void updateEdgeSprings(int numParticles,
					   float mass,
					   float length,
					   float dt,
					   float k,
					   float3* position,
					   float3* velocity,
					   float3* force)
{
	int numEdges = numParticles - 1;
	
	for(int i = 0; i < numEdges; i++)
	{
		float3 xn = make_float3(position[i+1].x-position[i].x, position[i+1].y-position[i].y, position[i+1].z-position[i].z);
		float3 vn = make_float3(velocity[i+1].x-velocity[i].x, velocity[i+1].y-velocity[i].y, velocity[i+1].z-velocity[i].z);
		
		float inv_length = inv_sqrt(xn.x*xn.x + xn.y*xn.y + xn.z*xn.z);
		float3 d = make_float3(xn.x*inv_length, xn.y*inv_length, xn.z*inv_length);
		
		float h = dt * dt * k / (4.0f * mass * length);
		float g = dt * k / (2.0f * mass * length);
		float f = k / length * mass;
		
		//TODO CONJUGATE GRADIENT!!!!!
		
		//Store resulting velocity here
		float3 v1;
		
		//Store resulting force here
		float3 result;
		
		result.x = d.x * (f * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt * (v1.x*d.x + v1.y*d.y + v1.z*d.z)));
		result.y = d.y * (f * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt * (v1.x*d.x + v1.y*d.y + v1.z*d.z)));
		result.z = d.z * (f * (xn.x*d.x + xn.y*d.y + xn.z*d.z - length + dt * (v1.x*d.x + v1.y*d.y + v1.z*d.z)));
		
		force[i].x += result.x * -1.0f; 
		force[i].y += result.y * -1.0f;
		force[i].z += result.z * -1.0f;
		force[i+1].x += result.x; 
		force[i+1].y += result.y;
		force[i+1].z += result.z;
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
			float3* velh,
			float3* force)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	int start = numParticles * tid; //first particle index
//	int end = start + numParticles; //last particle index
	
	clearForces(numParticles, force+start);
	
	//Update edge springs
	updateEdgeSprings(numParticles,
					  mlgt.x,
					  mlgt.y,
					  mlgt.w,
					  k.x,
					  position+start,
					  velocity+start,
					  force+start);
	
	//Update bend springs
	//Update twist springs
	
	
	clearForces(numParticles, force+start);
}

#endif

