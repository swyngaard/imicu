
#include <cutil_inline.h>

#include <hair_kernel.cu>


static float3* init(const int size)
{
	float3* d_vec;
	cutilSafeCall(cudaMalloc((void**)&d_vec, size));
	cutilSafeCall(cudaMemset(d_vec, 0, size));
	
	return d_vec;
}

static float* init2(const int size)
{
	float* d_vec;
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
				 float3* &velc,
				 float3* &velh,
				 float3* &force,
				 float* &A,
				 float* &b,
				 float* &x,
				 float* &r,
				 float* &p,
				 float* &Ap)
{
	int size = numStrands*numParticles*sizeof(float3);
	
//	position = init(size);
	posc = init(size);
	posh = init(size);
	velocity = init(size);
	velh = init(size);
	velc = init(size);
	force = init(size);
	A = init2(numStrands*numParticles*3*numParticles*3*sizeof(float));
	b = init2(numStrands*numParticles*3*sizeof(float));
	x = init2(numStrands*numParticles*3*sizeof(float));
	r = init2(numStrands*numParticles*3*sizeof(float));
	p = init2(numStrands*numParticles*3*sizeof(float));
	Ap = init2(numStrands*numParticles*3*sizeof(float));
	
	/*
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
	
	printf("before memcpy\n");
	cutilSafeCall(cudaMemcpy(position, position_h, size, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(posc, position, size, cudaMemcpyDeviceToDevice));
	printf("after memcpy\n");
	
	free(position_h);
	*/
}

extern "C"
void copyMem(const int numStrands,
			 const int numParticles,
			 float3* &position,
			 float3* &posc)
{
	cutilSafeCall(cudaMemcpy(posc, position, numStrands*numParticles*sizeof(float3), cudaMemcpyDeviceToDevice));
}


extern "C"
void releaseStrands(float3* &position,
				 	float3* &posc,
				 	float3* &posh,
				 	float3* &velocity,
				 	float3* &velc,
				 	float3* &velh,
				 	float3* &force,
				 	float* &A,
				 	float* &b,
				 	float* &x,
				 	float* &r,
				 	float* &p,
				 	float* &Ap)
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
	
	cutilSafeCall(cudaFree(posc));
	cutilSafeCall(cudaFree(posh));
	cutilSafeCall(cudaFree(velocity));
	cutilSafeCall(cudaFree(velc));
	cutilSafeCall(cudaFree(velh));
	cutilSafeCall(cudaFree(force));
	cutilSafeCall(cudaFree(A));
	cutilSafeCall(cudaFree(b));
	cutilSafeCall(cudaFree(x));
	cutilSafeCall(cudaFree(r));
	cutilSafeCall(cudaFree(p));
	cutilSafeCall(cudaFree(Ap));
//	cutilSafeCall(cudaFree(position));
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
				   float3* &velc,
				   float3* &velh,
				   float3* &force,
				   float* &A,
				   float* &b,
				   float* &x,
				   float* &r,
				   float* &p,
				   float* &Ap)
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
							velc,
							velh,
							force,
							A,
							b,
							x,
							r,
							p,
							Ap);
	
	cudaThreadSynchronize();
}



