
#include <hair_kernel.cu>

extern "C"
void updateStrands()
{
	dim3 grid(1,1,1);
	dim3 block(1,1,1);
	
	kernel<<<grid, block>>>();
}

