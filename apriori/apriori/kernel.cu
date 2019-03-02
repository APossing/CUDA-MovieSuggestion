
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include "CUDABackground.h"
#include <atomic>
#include <iostream>
#include <pplinterface.h>
#include "FileReader2.h"
#include "UserTableReader.h"
using namespace std;
/*
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
}*/

__global__ void test(float**mainArray)
{
	int i = threadIdx.x / 50;
	int j = threadIdx.x % 50;
	printf("%d:\t%f\n", threadIdx.x, mainArray[1][threadIdx.x]);

}

float ** createArr(int size)
{
	float ** arr = (float**)malloc(sizeof(float*) * size);
	for(int i = 0; i < size+1; i++)
	{
		arr[i] = (float*)malloc(sizeof(float) * size + 1);
	}
	return arr;
}

float ** createBlankUserMatrix(UserTableReader r, int columnMax)
{
	float ** arr = (float**)malloc(sizeof(float*) * columnMax+1);

	for (int i = 0; i < r.users.size() +1; i++)
	{
		arr[i] = (float*)malloc(sizeof(float) * columnMax + 1);
		for (int j = 0; j < r.users.size() + 1; j++)
		{
			arr[i][j] = 0.0;
		}
	}
	return arr;
}

bool ** createBlankUserDidReviewMatrix(UserTableReader r, int columnMax)
{
	bool ** arr = (bool**)malloc(sizeof(bool*) * columnMax + 1);

	for (int i = 0; i < r.users.size() +1; i++)
	{
		arr[i] = (bool*)malloc(sizeof(bool) * columnMax + 1);
		for (int j = 0; j < r.users.size() + 1; j++)
		{
			arr[i][j] = false;
		}
	}
	return arr;
}

void populateUserReviewMatrix(float **userReviewMatrix, bool **originalReviewMatrix, UserTableReader r, MovieReader m)
{
	auto vec = r.users;
	for (auto it = vec.begin(); it != vec.end(); ++it)
	{
		for (auto sit = (*it).ratedMovies.begin(); sit != (*it).ratedMovies.end(); ++sit)
		{
			userReviewMatrix[(*it).userID][m.movieIDMapper[(*sit).movieID]] = (*sit).rating;
			originalReviewMatrix[(*it).userID][m.movieIDMapper[(*sit).movieID]] = true;
		}
	}
}

cudaError_t doAlgo()
{
	//int cudaCores = cuda.calculateCores();
	MovieReader m = MovieReader("movie.csv");
	UserTableReader r = UserTableReader("ratings.csv");
	float ** movieMatrix = createArr(m.movieCount);
	float ** userReviewMatrix = createBlankUserMatrix(r, m.movieCount);
	bool ** originalReviewMatrix = createBlankUserDidReviewMatrix(r, m.movieCount);
	populateUserReviewMatrix(userReviewMatrix, originalReviewMatrix, r, m);

	float ** d_movieMatrix;
	float ** d_userReviewMatrix;
	bool ** d_didReviewMatrix;

	float * devptrMovie;
	float * devptrUser;
	bool * devptrUserDidReview;

	size_t pitchMovie;
	size_t pitchUserReview;
	size_t pitchDidReview;

	cudaError_t cudaStatus;
	int movieMatrixColumns = m.movieCount + 1;
	int userReviewColumns = m.movieCount + 1;
	int userReviewRows = r.users.size() + 1;
	/*
	float** temp_d_ptrs = (float **)malloc(sizeof(float*) * userReviewRows);
	for (int i = 0; i < userReviewColumns+1; i++)
	{
		cudaMalloc((void**)&temp_d_ptrs[i], sizeof(float)* (userReviewColumns)); // allocate for 1 int in each int pointer
		cudaMemcpy(temp, main[i], sizeof(int) * getsize, cudaMemcpyHostToDevice); // copy data
		//cudaMemcpy(devMain + i, &temp, sizeof(int*), cudaMemcpyHostToDevice);
	}*/

	/*
	cudaStatus = cudaMalloc((void**)&d_movieMatrix, movieMatrixColumns * sizeof(float*));
	cudaMallocPitch(&devptrMovie, &pitchMovie, movieMatrixColumns * sizeof(float), movieMatrixColumns);
	cudaMemcpy2D(d_movieMatrix, pitchMovie, movieMatrix, (m.movieCount + 1) * sizeof(float), (m.movieCount + 1) * sizeof(float), m.movieCount + 1, cudaMemcpyHostToDevice);
	*/
	cudaStatus = cudaMalloc((void***)&d_userReviewMatrix, userReviewRows * sizeof(float*));
	for (int i = 0; i < userReviewRows; i++)
	{
		float * temp;
		cudaMalloc((void**) &(temp), sizeof(float)*userReviewColumns);
		cudaMemcpy(temp, userReviewMatrix[i], sizeof(float) * userReviewColumns, cudaMemcpyHostToDevice);
		cudaMemcpy(d_userReviewMatrix + i, &temp, sizeof(float*), cudaMemcpyHostToDevice);
	}

	cudaStatus = cudaMalloc((void***)&d_movieMatrix, movieMatrixColumns * sizeof(float*));
	for (int i = 0; i < movieMatrixColumns; i++)
	{
		float * temp;
		cudaMalloc((void**) &(temp), sizeof(float)*movieMatrixColumns);
		cudaMemcpy(temp, movieMatrix[i], sizeof(float) * movieMatrixColumns, cudaMemcpyHostToDevice);
		cudaMemcpy(d_movieMatrix + i, &temp, sizeof(float*), cudaMemcpyHostToDevice);
	}


	/*cudaMallocPitch(&devptrUser, &pitchUserReview, userReviewColumns * sizeof(float), userReviewRows);
	cudaMemcpy2D(d_userReviewMatrix, pitchUserReview, userReviewMatrix, userReviewColumns * sizeof(float), userReviewColumns * sizeof(float), userReviewRows, cudaMemcpyHostToDevice);
	*/
	test << <1, userReviewRows >> > (d_userReviewMatrix, );


Error:
	//cudaFree();
	//cudaFree();
	return cudaStatus;
}



int main()
{

	CUDABackground cuda = CUDABackground();
	doAlgo();
}















/*
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
*/