
#include <helper_cuda.h>

#include "hair.h"
#include "hair_kernel.cu"

static
void* mallocBytes(int bytes)
{
	void* pointer;
	
	//Allocate bytes of memory
	checkCudaErrors(cudaMalloc((void**)&pointer, bytes));
	
	//Set memory to zero
	checkCudaErrors(cudaMemset(pointer, 0, bytes));
	
	return pointer;
}

extern "C"
void mallocStrands(pilar::HairState* h_state, pilar::HairState* &d_state, int modelBytes)
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
	
	h_state->rng	  = (curandStatePhilox4_32_10_t*) mallocBytes(h_state->numStrands * sizeof(curandStatePhilox4_32_10_t));
	
	h_state->vertices = (float*) mallocBytes(modelBytes);
	h_state->normals  = (float*) mallocBytes(modelBytes*sizeof(float));
	h_state->faces	  = (float*) mallocBytes(modelBytes*sizeof(float));
	h_state->model	  = (ModelOBJ*) mallocBytes(sizeof(ModelOBJ));
	
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
	
	checkCudaErrors(cudaFree(h_state->rng));
	
	checkCudaErrors(cudaFree(h_state->vertices));
	checkCudaErrors(cudaFree(h_state->normals));
	checkCudaErrors(cudaFree(h_state->faces));
	checkCudaErrors(cudaFree(h_state->model));
	
	checkCudaErrors(cudaFree(d_state));
}

extern "C"
void copyRoots(pilar::Vector3f* roots, pilar::Vector3f* normals, pilar::HairState* h_state)
{
	checkCudaErrors(cudaMemcpy(h_state->root,   roots,   h_state->numStrands * sizeof(*roots), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(h_state->normal, normals, h_state->numStrands * sizeof(*normals), cudaMemcpyHostToDevice));
}

extern "C"
void copyState(pilar::HairState* h_state, pilar::HairState* d_state)
{
	checkCudaErrors(cudaMemcpy(d_state, h_state, sizeof(*h_state), cudaMemcpyHostToDevice));
}

extern "C"
void copyModel(ModelOBJ* model, pilar::HairState* h_state)
{
	checkCudaErrors(cudaMemcpy(h_state->model,	  model,		   sizeof(*model), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(h_state->vertices, model->vertices, model->bytes,			   cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(h_state->normals,  model->normals,  model->bytes*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(h_state->faces,	  model->faces,    model->bytes*sizeof(float), cudaMemcpyHostToDevice));
}

extern "C"
void initialisePositions(pilar::HairState* h_state, pilar::HairState* d_state)
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
