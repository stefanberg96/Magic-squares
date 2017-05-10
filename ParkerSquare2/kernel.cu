
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "CombinationIt.h"
#include <list>
#include "device_functions.h"

#define BLOCKS 709;
#define THREADS 512;
#define N  1000;//amount of results


__global__ void addKernel(int * c);
static void reportMemStatus();
cudaError_t addWithCuda(int *c);


__device__ int myPushBack(int* result,int *results, int index) {
	/*if (results[0] >= 1000) {
		return;
	}
	int pointer= atomicAdd(&results[0],1);
	if (pointer >= 1000) {
		return;
	}*/
	int pointer;
	if (index < 32000 && index>31000) {
		pointer = index;
	}
	else {
		return;
	}
	for (int i = 0; i < 9; i++) {
		results[pointer*9 + i+1] = result[i];
	}
	return 0;
}

__device__ void factorial(long * number) {
	int n = (int) *number;
	if (n == 3) {
		n = 3;
	}
	while (n-1 > 0) {
		n--;
		*number *= n;
	}
	return;
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

__device__ int getIth(int *c, int index) {
	int count = -1;
	int firstPositiveElement =-1;

	int i = -1;
	do {
		i++;
		if (c[i] != -1) {
			count++;
		}
		
	} while (i < 8 && count < index);
	for (; i < 9 && count < index; i++) {
		if (c[i] != -1) {
			count++;
		}
	}
	
	int k = c[i];
	c[i] = -1;
	return k;
}

/*
give y and x calculate c such that c*y!<x, for the largest c possible
*/
__device__ int largestC(int y, int x,int * factorial) {
	int c = 0;
	while (x >= (c+1)*factorial[y]) {
		c++;
	}
	return c;
}


__device__ void getPermutation(int *permu, int index, int* factorial) {
	int *temp = new int[9];
	int *ordering = new int[9];
	
	for (int i = 0; i < 9; i++) {
		temp[i] = i;
	}
	for (int i = 0; i < 9; i++) {
		int t = largestC(8 - i, index, factorial);
		index -= t * factorial[8 - i];
		int val=getIth(temp, t);
		ordering[i] = val;
	}
	//order the actual array into the correct order
	
	int *reordered = new int[9];

	for (int i = 0; i < 9; i++) {
		permu[i] = i;
	}
	/*
	for (int i = 0; i < 9; i++) {
		int k = ordering[i];
		int j = permu[k];
		reordered[i] = j;
	}
	for (int i = 0; i < 9; i++) {
		permu[i] = reordered[i];
	}*/
	delete [] reordered;
	delete [] temp;
	delete [] ordering;
	return;
}

__device__ void initialize(int c[]) {
	c[0] = 1;
	for (int i = 1; i < 9; i++) {
		c[i] = c[i - 1] * i;
	}
	return;
}

__global__ void addKernel(int *c, int * factorial, int* results)
{
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	int *ccopy = new int[9];
	for (int i = 0; i < 9; i++) {
		ccopy[i] = c[i];
	}

	if (index > 362880) {//ignore cases which are outside of the 9! range
		ccopy[0] = 0;
		delete[] ccopy;
		return;
	}
	else {
		getPermutation(ccopy, index, factorial);
		bool correct = magicSquare(ccopy);

		//if (correct) {
			/*int * temp = new int[9];
			for (int i = 0; i < 9; i++) {
				temp[i] = i;
			}*/
			myPushBack(ccopy, results,index);
			//delete[] temp;
		//}
		delete[] ccopy;
		
		return;
	}
}


int main()
{
	int c[9] = { 0,1,2,3,4,5,6,7,8 };
	// Add vectors in parallel.
	reportMemStatus();
	cudaError_t cudaStatus = addWithCuda(c);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}

		printf("{%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
			c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8]);

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		//testing
		/*
		init(20, 9, choose(20, 9));
		while (hasNext()) {
			int *t= next();
			printf("{%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
				t[0], t[1 ], t[2 ], t[3 ], t[4 ], t[5 ], t[6 ], t[7 ], t[8 ]);
		}*/
		return 0;
	}

static void reportMemStatus() {

	// show memory usage of GPU
	size_t free_byte;
	size_t total_byte;
	size_t malloc_byte;
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaSuccess != cudaStatus) {
		printf("Error: cudaMemGetInfo fails, %s \n",
			cudaGetErrorString(cudaStatus));
		return;
	}
	cudaError_t cuda_status;
	
	cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

	if (cudaSuccess != cuda_status) {
		printf("Error: cudaMemGetInfo fails, %s \n",
			cudaGetErrorString(cuda_status));
		return;
	}
	cuda_status=cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1000 * 500);
	if (cuda_status != cudaSuccess) {
		printf("fdskjfhdskajfdsafdsafs");
	}
	
	cuda_status = cudaDeviceGetLimit(&malloc_byte, cudaLimitMallocHeapSize);
	if (cudaSuccess != cuda_status) {
		printf("Error: cudaDeviceGetLimit fails, %s \n",
			cudaGetErrorString(cuda_status));
		return;
	}

	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB, malloc limit = %f MB\n",
		used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0,
		total_db / 1024.0 / 1024.0, malloc_byte / 1024.0 / 1024.0);
}


cudaError_t addWithCuda(int *c)
{
	
	int t =  BLOCKS;
	t *= THREADS;
	int *d_results =0;
	


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

	cudaStatus = cudaDeviceSetLimit(cudaLimitMallocHeapSize,1024*1000*500);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda limit malloc failed!");
		goto Error;
	}

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c,9*sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&d_results, (1+1000 * 9) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaMemset(d_results, 0,(1 + 1000 * 9) * sizeof(int));

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
	cudaStatus = cudaMemcpy(dev_c, c, 9*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<709, 512>>>(dev_c,d_factorial,d_results);

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
    cudaStatus = cudaMemcpy(c, dev_c, 9 * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!last");
        goto Error;
    }
	int *results = new int[(1 + 1000 * 9)];
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(results, d_results, (1 + 1000 * 9) * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!last");
		goto Error;
	}
	results[0] = 289;
	int nbresults = results[0];
	for (int i = 0; i < nbresults; i++) {
		printf("magic square {%d,%d,%d,%d,%d,%d,%d,%d,%d}\n",
			 results[1 + i * 9], results[2], results[3 + i * 9], results[4 + i * 9], results[5 + i * 9], results[6 + i * 9], results[7 + i * 9], results[8 + i * 9],results[9+i*9]);
	}

	
Error:
    cudaFree(dev_c);
	cudaFree(d_factorial);
	cudaFree(d_results);
    
    return cudaStatus;
}
