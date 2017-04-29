
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "CombinationIt.h"
#include <list>

#define BLOCKS 709;
#define THREADS 512;

__global__ void addKernel(int * c);

cudaError_t addWithCuda(int *c);

__device__ void factorial(long * number) {
	int n = (int) *number;
	if (n == 3) {
		n = 3;
	}
	while (n-1 > 0) {
		n--;
		*number *= n;
	}
}

__device__ bool magicSquare(int *c) {
	int r1 = c[0] + c[1] + c[2];
	int r2 = c[3] + c[4] + c[5];
	int r3 = c[6] + c[7] + c[8];

	int c1 = c[0] + c[3] + c[6];
	int c2 = c[1] + c[4] + c[7];
	int c3 = c[2] + c[5] + c[8];

	int d1 = c[0] + c[4] + c[8];
	int d2 = c[6] + c[4] + c[2];

	bool temp = r1 == r2 && r2 == r3 && r1 == c1 && c1 == c2 && c2 == c3;
	temp = temp && (r1 == d1 || r1 == d2);

	return temp;
}

__device__ void getIth(int *c, int *index) {
	int count = 0;
	int firstPositiveElement = 0;
	for (int i = 0; i < 9; i++) {
		if (c[i] >= 0) {
			firstPositiveElement = i;
			break;
		}
	}
	int i = firstPositiveElement;
	for (; i < 9 && count < *index; i++) {
		if (c[i] != -1) {
			count++;
		}
	}
	
	*index = c[i];
	c[i] = -1;

}

/*
give y and x calculate c such that c*y!<x, for the largest c possible
*/
__device__ int* largestC(int y, int x,int * factorial) {
	int *c=new int;
	*c = 1;
	while (x > *c*factorial[y]) {
		(*c)++;
	}
	*c -= 1;
	return c;
}


__device__ void getPermutation(int * permu, int *index, int* factorial) {
	int temp[] = { 0,1,2,3,4,5,6,7,8 };
	int *ordering = new int[9];
	for (int i = 0; i < 9; i++) {
		int *t = largestC(8 - i, *index, factorial);
		*index -= *t * factorial[8 - i];
		getIth(temp, t);
		ordering[i] = *t;
	}

	//order the actual array into the correct order
	int *reordered = new int[9];
	for (int i = 0; i < 9; i++) {
		reordered[i] = permu[ordering[i]];
	}
	for (int i = 0; i < 9; i++) {
		permu[i] = reordered[i];
	}
	
}

__device__ void initialize(int c[]) {
	c[0] = 1;
	for (int i = 1; i < 9; i++) {
		c[i] = c[i - 1] * i;
	}
}

__global__ void addKernel(int *c,int * factorial)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int *c2 = &c[index * 9];
	if (index < 362880) {//ignore cases which are outside of the 9! range
		getPermutation(c2, &index, factorial);
	}
}

int main()
{
	int c[9] = { 0,1,2,3,4,5,6,7,8 };
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4],c[5],c[6],c[7],c[8]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	//testing
	init(20, 9, choose(20, 9));/*
	while (hasNext()) {
		int *t= next();
		printf("{%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
			t[0], t[1 ], t[2 ], t[3 ], t[4 ], t[5 ], t[6 ], t[7 ], t[8 ]);
	}*/
    return 0;
}


cudaError_t addWithCuda(int *c)
{
	
	int t =  BLOCKS;
	t *= THREADS;
	int *ccopy = new int[t*9];
	

	for (int i = 0; i < t; i++) {
		for (int j = 0; j < 9; j++) {
			ccopy[i * 9 + j] = c[j];
		}
	}

	printf("here");

    int *dev_c = 0;
    cudaError_t cudaStatus;

	int *d_factorial=0;
	int factorial[9] = { 0 };
	factorial[0] = 1;
	for (int i = 1; i < 9; i++) {
		factorial[i] = factorial[i - 1] * i;
	}

	
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c,t*9*sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	// Allocate GPU buffers for three vectors(two input, one output)    .
		cudaStatus = cudaMalloc((void**)&d_factorial, 9 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffers for three vectors(two input, one output)    .
	cudaStatus = cudaMemcpy(d_factorial,factorial, 9 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffers for three vectors(two input, one output)    .
	cudaStatus = cudaMemcpy(dev_c, ccopy, t *9*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<709, 512>>>(dev_c,d_factorial);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(ccopy, dev_c, t*9 * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
	cudaFree(d_factorial);
    
    return cudaStatus;
}
