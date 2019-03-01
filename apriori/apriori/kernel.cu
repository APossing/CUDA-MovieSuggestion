
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include "CUDABackground.h"
#include "FileReader.h"
#include <atomic>
#include <iostream>
#include <pplinterface.h>
#include "FileReader2.h"
#include "UserTableReader.h"
using namespace std;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t findFrequents(int**main, unsigned int *counts, int mainSize, int countsSize);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void findFrequents(int**mainArray, unsigned int *counts)
{
	int i = threadIdx.x;
	//int end = mainArray[i][0] + 1;
	printf("%d:\t%d\n", threadIdx.x, mainArray[i][1]);
	int * startAddy = mainArray[i];
	for(int j = 1; j <= 1; j++)
	{
		//atomicAdd(counts + mainArray[i][j], 1);
	}
}

short ** createArr(int size)
{
	short ** arr = (short**)malloc(sizeof(short*) * size);
	for(int i = 0; i < size; i++)
	{
		arr[i] = (short*)malloc(sizeof(short) * size + 1);
	}
	return arr;
}

short ** createBlankUserMatrix(UserTableReader r, int columnMax)
{
	short ** arr = (short**)malloc(sizeof(short*) * columnMax+1);

	for (int i = 0; i < r.users.size(); i++)
	{
		arr[i] = (short*)malloc(sizeof(short) * columnMax + 1);
	}
	return arr;
}

bool ** createBlankUserDidReviewMatrix(UserTableReader r, int columnMax)
{
	bool ** arr = (bool**)malloc(sizeof(bool*) * columnMax + 1);

	for (int i = 0; i < r.users.size(); i++)
	{
		arr[i] = (bool*)malloc(sizeof(bool) * columnMax + 1);
	}
	return arr;
}

void populateUserReviewMatrix(short **userReviewMatrix, bool **originalReviewMatrix, UserTableReader r, MovieReader m)
{
	auto vec = r.users;
	for (auto it = vec.begin(); it != vec.end(); ++it)
	{
		for (auto sit = (*it).ratedMovies.begin(); sit != (*it).ratedMovies.end(); ++sit)
		{
			userReviewMatrix[(*it).userID][m.movieIDMapper[(*sit).movieID]] = (*sit).rating;
		}
	}
}

void doAlgo()
{
	MovieReader m = MovieReader("movie.csv");
	UserTableReader r = UserTableReader("ratings.csv");
	short ** movieMatrix = createArr(m.movieCount);
	short ** userReviewMatrix = createBlankUserMatrix(r, m.movieCount);
	bool ** originalReviewMatrix = createBlankUserDidReviewMatrix(r, m.movieCount);
	populateUserReviewMatrix(userReviewMatrix, originalReviewMatrix, r, m);
	short ** d_movieMatrix;
	short * devptr;
	size_t pitch;
	cudaError_t cudaStatus;


	cudaStatus = cudaMalloc((void**)&d_movieMatrix, m.movieCount * sizeof(short*));
	cudaMallocPitch(&devptr, &pitch, (m.movieCount + 1) * sizeof(short), (m.movieCount + 1));
	cudaMemcpy2D(d_movieMatrix, pitch, movieMatrix, (m.movieCount + 1) * sizeof(short), (m.movieCount + 1) * sizeof(short), m.movieCount + 1, cudaMemcpyHostToDevice);
 
	//short** temp_d_ptrs = (short **)malloc(sizeof(short*) * mainSize);
	for (int i = 0; i < m.movieCount; i++)
	{
		cudaMalloc((void**)&temp_d_ptrs[i], sizeof(int)* (main[i][0] + 1)); // allocate for 1 int in each int pointer
		//cudaMemcpy(temp, main[i], sizeof(int) * getsize, cudaMemcpyHostToDevice); // copy data
		//cudaMemcpy(devMain + i, &temp, sizeof(int*), cudaMemcpyHostToDevice);
	}


Error:
	cudaFree();
	cudaFree();

	return cudaStatus;



}

cudaError_t mallocCuda()
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&devMain, mainSize * sizeof(int*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	int** temp_d_ptrs = (int **)malloc(sizeof(int*) * mainSize);
	for (int i = 0; i < m; i++)
	{
		cudaMalloc((void**)&temp_d_ptrs[i], sizeof(int)* (main[i][0] + 1)); // allocate for 1 int in each int pointer
		//cudaMemcpy(temp, main[i], sizeof(int) * getsize, cudaMemcpyHostToDevice); // copy data
		//cudaMemcpy(devMain + i, &temp, sizeof(int*), cudaMemcpyHostToDevice);
	}


Error:
	cudaFree();
	cudaFree();

	return cudaStatus;

}


int main()
{

	CUDABackground cuda = CUDABackground();
	doAlgo();
	FileReader f = FileReader("data.txt");






















	int cudaCores = cuda.calculateCores();
	unsigned int * frequency = (unsigned int *)malloc(sizeof(unsigned int) * f.maxNumber);
	for (int i = 0; i < f.maxNumber; i++)
	{
		frequency[i] = 0;
	}
	for (int i = 0; i < 50; i++)
	{
		printf("\n%d: ", i);
		for (int j = 0; j < f.master[i][0];j++)
			printf("%d,", f.master[i][j]);
	}
	cout<<frequency[0];
	1 + 1;
	cudaError_t cudaStatus = findFrequents(f.master, frequency, f.count, f.maxNumber);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	int test2 = frequency[0];




    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t findFrequents(int* main[], unsigned int *counts, int mainSize, int countsSize)
{
	int **devMain = 0;
	unsigned int * devCounts = 0;
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&devMain, mainSize * sizeof(int*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	int** temp_d_ptrs = (int **)malloc(sizeof(int*) * mainSize);
	for (int i = 0; i < mainSize; i++)
	{
		cudaMalloc((void**)&temp_d_ptrs[i], sizeof(int)* (main[i][0] + 1)); // allocate for 1 int in each int pointer
		//cudaMemcpy(temp, main[i], sizeof(int) * getsize, cudaMemcpyHostToDevice); // copy data
		//cudaMemcpy(devMain + i, &temp, sizeof(int*), cudaMemcpyHostToDevice);
	}


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&devMain, mainSize * sizeof(int*));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	cudaStatus = cudaMalloc((void**)&devCounts, countsSize * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(devCounts, counts, countsSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	findFrequents << <1, 500 >> > (devMain, devCounts);


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
	cudaStatus = cudaMemcpy(counts, devCounts, countsSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(devMain);
	cudaFree(counts);

	return cudaStatus;

}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
