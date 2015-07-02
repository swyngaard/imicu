
#include <helper_cuda.h>

#include "hair_kernel.cu"
#include "hair.h"

static void* mallocBytes(int bytes)
{
	void* pointer;

	checkCudaErrors(cudaMalloc((void**)&pointer, bytes));
	checkCudaErrors(cudaMemset(pointer, 0, bytes));

	return pointer;
}

extern "C"
void mallocStrands(const int &numStrands,
				   const int &numParticles,
				   const int &numComponents,
				   float* &AA,
				   float* &bb,
				   float* &xx,
				   pilar::Vector3f* &root1,
				   pilar::Vector3f* &normal1,
				   pilar::Vector3f* &position1,
				   pilar::Vector3f* &pos1,
				   pilar::Vector3f* &posc1,
				   pilar::Vector3f* &posh1,
				   pilar::Vector3f* &velocity1,
				   pilar::Vector3f* &velh1,
				   pilar::Vector3f* &force1,
				   pilar::HairState* &state)
{
	AA = (float*) mallocBytes(numParticles * numStrands * numComponents * numParticles * numStrands * numComponents * sizeof(float));
	bb = (float*) mallocBytes(numParticles * numStrands * numComponents * sizeof(float));
	xx = (float*) mallocBytes(numParticles * numStrands * numComponents * sizeof(float));

	root1	  = (pilar::Vector3f*) mallocBytes(numStrands * sizeof(pilar::Vector3f));
	normal1	  = (pilar::Vector3f*) mallocBytes(numStrands * sizeof(pilar::Vector3f));
	//~ position1 = (pilar::Vector3f*) mallocBytes(numParticles * numStrands * sizeof(pilar::Vector3f));
	pos1	  = (pilar::Vector3f*) mallocBytes(numParticles * numStrands * sizeof(pilar::Vector3f));
	posc1	  = (pilar::Vector3f*) mallocBytes(numParticles * numStrands * sizeof(pilar::Vector3f));
	posh1	  = (pilar::Vector3f*) mallocBytes(numParticles * numStrands * sizeof(pilar::Vector3f));
	velocity1 = (pilar::Vector3f*) mallocBytes(numParticles * numStrands * sizeof(pilar::Vector3f));
	velh1 	  = (pilar::Vector3f*) mallocBytes(numParticles * numStrands * sizeof(pilar::Vector3f));
	force1	  = (pilar::Vector3f*) mallocBytes(numParticles * numStrands * sizeof(pilar::Vector3f));

	state = (pilar::HairState*) mallocBytes(sizeof(pilar::HairState));
}

extern "C"
void freeStrands(float* &AA,
				 float* &bb,
				 float* &xx,
				 pilar::Vector3f* &root1,
				 pilar::Vector3f* &normal1,
				 pilar::Vector3f* &position1,
				 pilar::Vector3f* &pos1,
				 pilar::Vector3f* &posc1,
				 pilar::Vector3f* &posh1,
				 pilar::Vector3f* &velocity1,
				 pilar::Vector3f* &velh1,
				 pilar::Vector3f* &force1,
				 pilar::HairState* &state)
{
	checkCudaErrors(cudaFree(AA));
	checkCudaErrors(cudaFree(bb));
	checkCudaErrors(cudaFree(xx));

	checkCudaErrors(cudaFree(root1));
	checkCudaErrors(cudaFree(normal1));
	//~ checkCudaErrors(cudaFree(position1));
	checkCudaErrors(cudaFree(pos1));
	checkCudaErrors(cudaFree(posc1));
	checkCudaErrors(cudaFree(posh1));
	checkCudaErrors(cudaFree(velocity1));
	checkCudaErrors(cudaFree(velh1));
	checkCudaErrors(cudaFree(force1));

	checkCudaErrors(cudaFree(state));
}

extern "C"
void copyRoots(int numStrands, const pilar::Vector3f* root3f, const pilar::Vector3f* normal3f, pilar::Vector3f* root1, pilar::Vector3f* normal1)
{
	int bytes = numStrands * sizeof(pilar::Vector3f);
	checkCudaErrors(cudaMemcpy(root1, root3f, bytes, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(normal1, normal3f, bytes, cudaMemcpyHostToDevice));
}

extern "C"
void initPositions(int numParticles,
				   int numStrands,
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
				   float length_e,
				   float length_b,
				   float length_t,
				   float* AA,
				   float* bb,
				   float* xx,
				   pilar::Vector3f gravity1,
				   pilar::Vector3f* root1,
				   pilar::Vector3f* normal1,
				   pilar::Vector3f* position1,
				   pilar::Vector3f* pos1,
				   pilar::Vector3f* posc1,
				   pilar::Vector3f* posh1,
				   pilar::Vector3f* velocity1,
				   pilar::Vector3f* velh1,
				   pilar::Vector3f* force1,
				   pilar::HairState* state)
{
	dim3 grid(numStrands, 1, 1);
	dim3 block(1, 1, 1);
	
	initialise<<<grid,block>>>(numParticles,
							   numStrands,
							   numComponents,
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
							   AA,
							   bb,
							   xx,
							   gravity1,
							   root1,
							   normal1,
							   position1,
							   pos1,
							   posc1,
							   posh1,
							   velocity1,
							   velh1,
							   force1,
							   state);
	
	cudaThreadSynchronize();
}

extern "C"
void updateStrands(float dt, int numStrands, pilar::HairState* state)
{
	dim3 grid(numStrands, 1, 1);
	dim3 block(1, 1, 1);
//	static bool once = false;
//
//	if(!once)
//	{
		update<<<grid,block>>>(dt, state);
		
		cudaThreadSynchronize();

//		once = true;
//	}
}
