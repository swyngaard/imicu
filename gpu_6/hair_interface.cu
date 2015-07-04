
#include <helper_cuda.h>

#include "hair_kernel.cu"
#include "hair.h"

static void* mallocBytes(int bytes)
{
	void* pointer;
	
	//Allocate bytes of memory
	checkCudaErrors(cudaMalloc((void**)&pointer, bytes));
	
	//Set memory to zero
	checkCudaErrors(cudaMemset(pointer, 0, bytes));

	return pointer;
}

extern "C"
void mallocStrands(pilar::HairState* h_state, pilar::HairState* &d_state)
{
	h_state->AA = (float*) mallocBytes(h_state->numParticles * h_state->numStrands * h_state->numComponents * h_state->numParticles * h_state->numStrands * h_state->numComponents * sizeof(float));
	h_state->bb = (float*) mallocBytes(h_state->numParticles * h_state->numStrands * h_state->numComponents * sizeof(float));
	h_state->xx = (float*) mallocBytes(h_state->numParticles * h_state->numStrands * h_state->numComponents * sizeof(float));

	h_state->root	  = (pilar::Vector3f*) mallocBytes(h_state->numStrands * sizeof(pilar::Vector3f));
	h_state->normal	  = (pilar::Vector3f*) mallocBytes(h_state->numStrands * sizeof(pilar::Vector3f));
	//~ h_state->position = (pilar::Vector3f*) mallocBytes(numParticles * numStrands * sizeof(pilar::Vector3f));
	h_state->pos	  = (pilar::Vector3f*) mallocBytes(h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f));
	h_state->posc	  = (pilar::Vector3f*) mallocBytes(h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f));
	h_state->posh	  = (pilar::Vector3f*) mallocBytes(h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f));
	h_state->velocity = (pilar::Vector3f*) mallocBytes(h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f));
	h_state->velh 	  = (pilar::Vector3f*) mallocBytes(h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f));
	h_state->force	  = (pilar::Vector3f*) mallocBytes(h_state->numParticles * h_state->numStrands * sizeof(pilar::Vector3f));

	d_state = (pilar::HairState*) mallocBytes(sizeof(pilar::HairState));
}

extern "C"
void freeStrands(pilar::HairState* h_state, pilar::HairState* d_state)
{
	checkCudaErrors(cudaFree(h_state->AA));
	checkCudaErrors(cudaFree(h_state->bb));
	checkCudaErrors(cudaFree(h_state->xx));

	checkCudaErrors(cudaFree(h_state->root));
	checkCudaErrors(cudaFree(h_state->normal));
	//~ checkCudaErrors(cudaFree(h_state->position));
	checkCudaErrors(cudaFree(h_state->pos));
	checkCudaErrors(cudaFree(h_state->posc));
	checkCudaErrors(cudaFree(h_state->posh));
	checkCudaErrors(cudaFree(h_state->velocity));
	checkCudaErrors(cudaFree(h_state->velh));
	checkCudaErrors(cudaFree(h_state->force));

	checkCudaErrors(cudaFree(d_state));
}

extern "C"
void copyRoots(const pilar::Vector3f* root3f, const pilar::Vector3f* normal3f, pilar::HairState* h_state)
{
	int bytes = h_state->numStrands * sizeof(pilar::Vector3f);
	checkCudaErrors(cudaMemcpy(h_state->root, root3f, bytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(h_state->normal, normal3f, bytes, cudaMemcpyHostToDevice));
}

extern "C"
void copyState(pilar::HairState* h_state, pilar::HairState* d_state)
{
	checkCudaErrors(cudaMemcpy(d_state, h_state, sizeof(pilar::HairState), cudaMemcpyHostToDevice));
}

extern "C"
void initPositions(pilar::HairState* h_state, pilar::HairState* d_state)
{
	dim3 grid(h_state->numStrands, 1, 1);
	dim3 block(1, 1, 1);
	
	initialise<<<grid,block>>>(d_state);
	
	cudaThreadSynchronize();
}

extern "C"
void updateStrands(float dt, pilar::HairState* h_state, pilar::HairState* d_state)
{
	dim3 grid(h_state->numStrands, 1, 1);
	dim3 block(1, 1, 1);
//	static bool once = false;
//
//	if(!once)
//	{
		update<<<grid,block>>>(dt, d_state);
		
		cudaThreadSynchronize();

//		once = true;
//	}
}
