
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

extern "C"
void initPositions(int numStrands, int numParticles, const float3* root, float3* normal, float3* position, float3* posc, float3* pos)
{
	dim3 grid(numStrands, 1, 1);
	dim3 block(1, 1, 1);
	
	initialise<<<grid,block>>>(numParticles, root, normal, position, posc, pos);
	
	cudaThreadSynchronize();
}

extern "C"
void updateStrandsNew(int numParticles,
					  int numStrands,
					  int numComponents,
					  float dt,
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
					  float3 &gravity,
					  float3* root,
					  float3* position,
					  float3* posc,
					  float3* posh,
					  float3* pos,
					  float3* velocity,
					  float3* velh,
					  float3* force,
					  float* AA,
					  float* bb,
					  float* xx)
{
	dim3 grid(numStrands, 1, 1);
	dim3 block(1, 1, 1);
//	static bool once = false;
	
//	if(!once)
//	{
		update_strands<<<grid,block>>>(numParticles,
				numStrands,
				numComponents,
				dt,
				mass,
				k_edge,
				k_bend,
				k_twist,
				k_extra,
				d_edge,
				d_bend,
				d_twist,
				d_extra,
				length_e,
				length_b,
				length_t,
				gravity,
				root,
				position,
				posc,
				posh,
				pos,
				velocity,
				velh,
				force,
				AA,
				bb,
				xx);

		cudaThreadSynchronize();

//		once = true;
//	}
}
