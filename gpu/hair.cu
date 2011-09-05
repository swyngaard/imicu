
#include <cutil_inline.h>

#include <hair_kernel.cu>


static float3* init(const int size)
{
	float3* d_vec;
	cutilSafeCall(cudaMalloc((void**)&d_vec, size));
	cutilSafeCall(cudaMemset(d_vec, 0, size));
	
	return d_vec;
}

extern "C"
void initStrands(int numStrands,
				 int numParticles,
				 float length,
				 const float3 *root,
				 float3* &position,
				 float3* &posc,
				 float3* &posh,
				 float3* &velocity,
				 float3* &velh,
				 float3* &force)
{
	int size = numStrands*numParticles*sizeof(float3);
	
	position = init(size);
	posc = init(size);
	posh = init(size);
	velocity = init(size);
	velh = init(size);
	force = init(size);
	
	float3* position_h = (float3*) calloc(numStrands*numParticles, sizeof(float3));
	
	for(int i = 0; i < numStrands; i++)
	{
		for(int j = 0; j < numParticles; j++)
		{
			int index = i*numParticles + j;
			position_h[index].x = root[i].x + j * length / 2.0f;
			position_h[index].y = root[i].y;
			position_h[index].z = root[i].z;
			
//			printf("%f %f %f\n", position_h[index].x, position_h[index].y, position_h[index].z);
		}
	}
	
	cutilSafeCall(cudaMemcpy(position, position_h, size, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(posc, position_h, size, cudaMemcpyHostToDevice));
	
	free(position_h);
}

extern "C"
void releaseStrands(float3* &position,
				 	float3* &posc,
				 	float3* &posh,
				 	float3* &velocity,
				 	float3* &velh,
				 	float3* &force)
//				 	int numStrands,
//				 	int numParticles)
{
	/*
	float3* position_h = (float3*) calloc(numStrands*numParticles, sizeof(float3));
	
	cutilSafeCall(cudaMemcpy(position_h, position, numStrands*numParticles*sizeof(float3), cudaMemcpyDeviceToHost));
	
	for(int i = 0; i < numStrands; i++)
	{
		for(int j = 0; j < numParticles; j++)
		{
			int index = i * numParticles + j;
			
			printf("%f %f %f\n", position_h[index].x, position_h[index].y, position_h[index].z);
		}
	}
	
	free(position_h);
	*/
	
	cutilSafeCall(cudaFree(position));
	cutilSafeCall(cudaFree(posc));
	cutilSafeCall(cudaFree(posh));
	cutilSafeCall(cudaFree(velocity));
	cutilSafeCall(cudaFree(velh));
	cutilSafeCall(cudaFree(force));
}

extern "C"
void updateStrands(const int numParticles,
				   float4 &mlgt,
				   const float4 k,
				   const float4 d,
				   float3* &position,
				   float3* &posc,
				   float3* &posh,
				   float3* &velocity,
				   float3* &velh,
				   float3* &force)
{
	dim3 grid(1,1,1);
	dim3 block(1,1,1);
	
	update<<<grid, block>>>(numParticles,
							mlgt,
							k,
							d,
							position,
							posc,
							posh,
							velocity,
							velh,
							force);
	
	cudaThreadSynchronize();
}



