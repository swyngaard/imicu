
//#include <cutil_inline.h>

#include <helper_cuda.h>

#include "hair_kernel.cu"


static float* mallocFloat(const int bytes)
{
	float* pointer;
	
	checkCudaErrors(cudaMalloc((void**)&pointer, bytes));
	checkCudaErrors(cudaMemset(pointer, 0, bytes));
	
	return pointer;
}

static float3* mallocFloat3(const int bytes)
{
	float3* pointer;
	
	checkCudaErrors(cudaMalloc((void**)&pointer, bytes));
	checkCudaErrors(cudaMemset(pointer, 0, bytes));
	
	return pointer;
}

extern "C"
void mallocStrands(const int &numStrands,
				   const int &numParticles,
				   const int &numComponents,
				   float3* &root,
				   float3* &normal,
				   float3* &position,
				   float3* &pos,
				   float3* &posc,
				   float3* &posh,
				   float3* &velocity,
				   float3* &velh,
				   float3* &force,
				   float* &AA,
				   float* &bb,
				   float* &xx)
{
	int bytes1D = numParticles * numStrands * numComponents * sizeof(float);
	int bytes2D = numParticles * numStrands * numComponents * numParticles * numStrands * numComponents * sizeof(float);
	int bytes3fR = numStrands * sizeof(float3);
	int bytes3f1D = numParticles * numStrands * sizeof(float3);
	
	root	 = mallocFloat3(bytes3fR);
	normal	 = mallocFloat3(bytes3fR);
	//~ position = mallocFloat3(bytes3f1D);
	pos		 = mallocFloat3(bytes3f1D);
	posc	 = mallocFloat3(bytes3f1D);
	posh	 = mallocFloat3(bytes3f1D);
	velocity = mallocFloat3(bytes3f1D);
	velh	 = mallocFloat3(bytes3f1D);
	force	 = mallocFloat3(bytes3f1D);
	
	AA = mallocFloat(bytes2D);
	
	bb = mallocFloat(bytes1D);
	xx = mallocFloat(bytes1D);
}

extern "C"
void freeStrands(float3* &root,
				 float3* &normal,
				 float3* &position,
				 float3* &pos,
				 float3* &posc,
				 float3* &posh,
				 float3* &velocity,
				 float3* &velh,
				 float3* &force,
				 float* &AA,
				 float* &bb,
				 float* &xx)
{
	checkCudaErrors(cudaFree(root));
	checkCudaErrors(cudaFree(normal));
	//~ checkCudaErrors(cudaFree(position));
	checkCudaErrors(cudaFree(pos));
	checkCudaErrors(cudaFree(posc));
	checkCudaErrors(cudaFree(posh));
	checkCudaErrors(cudaFree(velocity));
	checkCudaErrors(cudaFree(velh));
	checkCudaErrors(cudaFree(force));
	checkCudaErrors(cudaFree(AA));
	checkCudaErrors(cudaFree(bb));
	checkCudaErrors(cudaFree(xx));
}

extern "C"
void copyRoots(int numStrands, const float3* root3f, const float3* normal3f, float3* root, float3* normal)
{
	int size = numStrands * sizeof(float3);
	checkCudaErrors(cudaMemcpy(root, root3f, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(normal, normal3f, size, cudaMemcpyHostToDevice));
}

static float3* init(const int size)
{
	float3* d_vec;
	checkCudaErrors(cudaMalloc((void**)&d_vec, size));
	checkCudaErrors(cudaMemset(d_vec, 0, size));
	
	return d_vec;
}

static float* init2(const int size)
{
	float* d_vec;
	checkCudaErrors(cudaMalloc((void**)&d_vec, size));
	checkCudaErrors(cudaMemset(d_vec, 0, size));
	
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
	checkCudaErrors(cudaMemcpy(position, position_h, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(posc, position, size, cudaMemcpyDeviceToDevice));
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
	checkCudaErrors(cudaMemcpy(posc, position, numStrands*numParticles*sizeof(float3), cudaMemcpyDeviceToDevice));
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
	
	checkCudaErrors(cudaMemcpy(position_h, position, numStrands*numParticles*sizeof(float3), cudaMemcpyDeviceToHost));
	
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
	
	checkCudaErrors(cudaFree(posc));
	checkCudaErrors(cudaFree(posh));
	checkCudaErrors(cudaFree(velocity));
	checkCudaErrors(cudaFree(velc));
	checkCudaErrors(cudaFree(velh));
	checkCudaErrors(cudaFree(force));
	checkCudaErrors(cudaFree(A));
	checkCudaErrors(cudaFree(b));
	checkCudaErrors(cudaFree(x));
	checkCudaErrors(cudaFree(r));
	checkCudaErrors(cudaFree(p));
	checkCudaErrors(cudaFree(Ap));
//	checkCudaErrors(cudaFree(position));
}

extern "C"
void initPositions(int numStrands, int numParticles, const float3* root, const float3* normal, float3* position)
{
	dim3 grid(numStrands, 1, 1);
	dim3 block(1, 1, 1);
	
	initialise<<<grid,block>>>(numParticles, root, normal, position);
	
	cudaThreadSynchronize();
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



