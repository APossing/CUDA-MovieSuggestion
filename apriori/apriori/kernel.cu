
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

__global__ void computeAverageType2(float**mainArray, unsigned short *mainArrayColumns, unsigned short *mainArrayRows)
{
	short column = blockIdx.x * blockDim.x + threadIdx.x + 1;
	float cur;
	if (column < *mainArrayColumns)
	{
		double total = 0;
		unsigned short count = 0;
		for (short i = 1; i < *mainArrayRows; i++)
		{
			cur = mainArray[i][column];
			if (cur >= 0 && cur <= 5)
			{
				total += cur;
				count++;
			}
		}
		mainArray[0][column] = total / count;
		//printf("%d, %d, %f, %d, %f\n", row, 0, total, count, mainArray[row][0]);
	}
}

__global__ void computeSimularMoviesType2(float**userArray, unsigned short *userArrayRows, float**movieArray, unsigned short *movieArrayColumns)
{
	short movie1 = blockDim.x * blockIdx.x + threadIdx.x + 1;
	short movie2 = blockDim.y * blockIdx.y + threadIdx.y + 1;
	if (movie1 < *movieArrayColumns && movie2 < *movieArrayColumns && movie1 >= movie2)
	{
		//printf("%d,%d\n", movie1, movie2);

		double top = 0;
		float topLeft = 0;
		float topRight = 0;
		double bottomLeft = 0;
		double bottomRight = 0;

		for (short i = 1; i < *userArrayRows; i++)	//for every user
		{
			topLeft = userArray[i][movie1];			//get user rating for movie 1
			if (topLeft < 0 || topLeft > 5)			//if its not filled out by user, set to 0
				topLeft = 0;
			else
			{
				topLeft -= userArray[0][movie1];	//subtracting the average for that movie
			}

			topRight = userArray[i][movie2]; 		//get user rating for movie 2
			if (topRight < 0 || topRight > 5)		//if its not filled out by user, set to 0
				topRight = 0;
			else
			{
				topRight -= userArray[0][movie2]; //subtracting the average for that movie
			}							

			top += topRight * topLeft;				//compute this one and add to sum

			bottomLeft += topLeft * topLeft;		//A^2 and add to A's sum
			bottomRight += topRight * topRight;		//B^2 and add to B's sum
		}

		if (movie1 < 11 && movie2 < 11)
			printf("(%d,%d): \t%lf, %lf, %lf %lf\n", movie1, movie2, sqrt(bottomLeft), sqrt(bottomRight), top, top / (sqrt(bottomLeft) * sqrt(bottomRight)));

		movieArray[movie1][movie2] = movieArray[movie2][movie1] = top / (sqrt(bottomLeft) * sqrt(bottomRight));
		if (movie1 < 11 && movie2 < 11)
			printf("\t%d,%d: \t%lf, %lf, %lf %lf\n", movie1, movie2, bottomLeft, bottomRight, top, movieArray[movie2][movie1]);

	}

}




__global__ void computeAverage(float**mainArray, unsigned short *mainArrayColumns, unsigned short *mainArrayRows)
{
	short row = blockIdx.x * blockDim.x + threadIdx.x+1;
	float cur;
	if (row < *mainArrayRows)
	{
		double total = 0;
		unsigned short count = 0;
		for (short i = 1; i < *mainArrayColumns; i++)
		{
			cur = mainArray[row][i];
			if (cur >= 0 && cur <= 5)
			{
				total += mainArray[row][i];
				count++;
			}
		}
		mainArray[row][0] = total / count;
		//printf("%d, %d, %f, %d, %f\n", row, 0, total, count, mainArray[row][0]);
	}
}

__global__ void computeRecommendedMovies(float**userArray, unsigned short *userArrayColumns, unsigned short *userArrayRows, float**movieArray, bool **didSelect)
{
	short movie = blockDim.x * blockIdx.x + threadIdx.x + 1;
	short user = blockDim.y * blockIdx.y + threadIdx.y + 1;
	float tempSim;
	short selected = 0;
	float top5[6];
	short top5Index[6];
	if (movie < *userArrayColumns && user < *userArrayRows)
	{
		for (int i = 1; i < *userArrayColumns; i++)
		{
			if (didSelect[movie][i])
			{
				tempSim = movieArray[movie][i];
				if (selected < 5)
				{
					top5[selected] = tempSim;
					top5Index[selected] = i;
					selected++;
				}
				else
				{
					top5[0] = tempSim;
					top5Index[0] = i;
					float temp;
					short temp2;
					for (int i2 = 0; i2 <= 5; i2++)
					{
						for (int j = 0; j < 5; j++)
						{
							if (top5[j] > top5[j + 1])
							{
								temp = top5[j];
								temp2 = top5Index[j];
								top5[j] = top5[j + 1];
								top5Index[j] = top5Index[j + 1];
								top5[j + 1] = temp;
								top5Index[j + 1] = temp2;
							}
							else if (top5[j] == top5[j + 1] && top5Index[j] > top5Index[j+1])
							{
								temp = top5[j];
								temp2 = top5Index[j];
								top5[j] = top5[j + 1];
								top5Index[j] = top5Index[j + 1];
								top5[j + 1] = temp;
								top5Index[j + 1] = temp2;
							}
						}
					}
				}
			}
		}
		double sum;
		for (int i = 1; i <=5; i++)
			sum+= top5[i] * userArray[user][top5Index[i]];
		userArray[user][movie] = sum / selected;
	}

}


__global__ void computeSimularMovies(float**userArray, unsigned short *userArrayRows, float**movieArray, unsigned short *movieArrayColumns)
{
	short movie1 = blockDim.x * blockIdx.x + threadIdx.x + 1;
	short movie2 = blockDim.y * blockIdx.y + threadIdx.y + 1;
	if (movie1 < *movieArrayColumns && movie2 < *movieArrayColumns && movie1 >= movie2)
	{
		//printf("%d,%d\n", movie1, movie2);
		
		double top = 0;
		float topLeft = 0;
		float topRight = 0;
		double bottomLeft = 0;
		double bottomRight = 0;

		for (short i = 1; i < *userArrayRows; i++)	//for every user
		{
			topLeft = userArray[i][movie1];			//get user rating for movie 1
			if (topLeft < 0 || topLeft > 5)			//if its not filled out by user, set to 0
				topLeft = 0;
			else
			{
				topLeft -= userArray[i][0]; //subtracting the average for that user
				if (movie1 == 2 && movie2 == 1)
					printf("\ntopLeft: %lf", topLeft);
			}				

			topRight = userArray[i][movie2]; 		//get user rating for movie 2
			if (topRight < 0 || topRight > 5)		//if its not filled out by user, set to 0
				topRight = 0;	
			else
			{
				topRight -= userArray[i][0]; //subtracting the average for that user
				if (movie1 == 2 && movie2 == 1)
					printf("\ntopRight: %lf", topRight);
			}							//subtracting the average for that user

			top += topRight * topLeft;				//compute this one and add to sum

			bottomLeft += topLeft * topLeft;		//A^2 and add to A's sum
			bottomRight += topRight * topRight;		//B^2 and add to B's sum
		}

		if (movie1 < 11 && movie2 < 11)
			printf("(%d,%d): \t%lf, %lf, %lf %lf\n", movie1, movie2, sqrt(bottomLeft), sqrt(bottomRight), top, top / (sqrt(bottomLeft) * sqrt(bottomRight)));

		movieArray[movie1][movie2] = movieArray[movie2][movie1] = top / (sqrt(bottomLeft) * sqrt(bottomRight));
		if (movie1 < 11 && movie2 < 11)
			printf("\t%d,%d: \t%lf, %lf, %lf %lf\n", movie1, movie2, bottomLeft, bottomRight, top, movieArray[movie2][movie1]);

	}

}

float ** createArr(int size)
{
	float ** arr = (float**)malloc(sizeof(float*) * size);
	for(int i = 0; i < size+1; i++)
	{
		arr[i] = (float*)malloc(sizeof(float) * size + 1);
		for (int j = 0; j < size + 1; j++)
		{
			arr[i][j] = 0;
		}
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
			arr[i][j] = 6.0;
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
	printf("%f, %f, %lf, %lf\n%lf, %lf, %lf, %lf\n", movieMatrix[1][1], movieMatrix[1][2], movieMatrix[1][3], movieMatrix[1][4], movieMatrix[2][1], movieMatrix[2][2], movieMatrix[2][3], movieMatrix[2][4]);


	float ** d_movieMatrix;
	float ** d_userReviewMatrix;
	bool ** d_didReviewMatrix;

	cudaError_t cudaStatus;
	int movieMatrixColumns = m.movieCount + 1;
	int userReviewColumns = m.movieCount + 1;
	int userReviewRows = r.users.size() + 1;

	unsigned short * d_userReviewMatrixColumns;
	cudaMalloc((void**)&d_userReviewMatrixColumns, sizeof(unsigned short) * 1);
	cudaStatus = cudaMemcpy(d_userReviewMatrixColumns,&userReviewColumns,sizeof(unsigned short), cudaMemcpyHostToDevice);

	unsigned short * d_userReviewMatrixRows;
	cudaMalloc((void**)&d_userReviewMatrixRows, sizeof(unsigned short) * 1);
	cudaStatus = cudaMemcpy(d_userReviewMatrixRows, &userReviewRows, sizeof(unsigned short), cudaMemcpyHostToDevice);

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

	cudaStatus = cudaMalloc((void***)&d_didReviewMatrix, movieMatrixColumns * sizeof(bool*));
	for (int i = 0; i < userReviewRows; i++)
	{
		bool * temp;
		cudaMalloc((void**) &(temp), sizeof(bool)*userReviewColumns);
		cudaMemcpy(temp, originalReviewMatrix[i], sizeof(bool) * userReviewColumns, cudaMemcpyHostToDevice);
		cudaMemcpy(d_didReviewMatrix + i, &temp, sizeof(bool*), cudaMemcpyHostToDevice);
	}


	/*cudaMallocPitch(&devptrUser, &pitchUserReview, userReviewColumns * sizeof(float), userReviewRows);
	cudaMemcpy2D(d_userReviewMatrix, pitchUserReview, userReviewMatrix, userReviewColumns * sizeof(float), userReviewColumns * sizeof(float), userReviewRows, cudaMemcpyHostToDevice);
	*/

	int blockX = ceil(userReviewRows / 256.0);
	int blockY = ceil(userReviewRows / 16.0);
	int blockXType2 = ceil(userReviewColumns / 256);



	computeAverageType2 << <blockXType2, 256 >> > (d_userReviewMatrix, d_userReviewMatrixColumns, d_userReviewMatrixRows);
	printf("SUCCESS");
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	blockX = ceil(movieMatrixColumns / 16.0);
	blockY = ceil(movieMatrixColumns / 16.0);


	computeSimularMoviesType2<<<dim3(blockX, blockY), dim3(16,16) >>>(d_userReviewMatrix, d_userReviewMatrixRows, d_movieMatrix, d_userReviewMatrixColumns);
	cudaStatus = cudaGetLastError();
	if (cudaSuccess != cudaGetLastError())
		printf("Error!\n");
	cudaStatus = cudaGetLastError();
	cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	blockX = ceil(movieMatrixColumns / 16.0);
	blockY = ceil(userReviewRows / 16.0);
	computeRecommendedMovies<<<dim3(blockX, blockY), dim3(16, 16) >>>(d_userReviewMatrix, d_userReviewMatrixColumns, d_userReviewMatrixRows, d_movieMatrix, d_didReviewMatrix);
	cudaStatus = cudaGetLastError();
	if (cudaSuccess != cudaGetLastError())
		printf("Error!\n");
	printf("before");
	cudaStatus = cudaGetLastError();
	cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	printf("past");
	printf("%f, %f, %lf, %lf\n%lf, %lf, %lf, %lf\n", movieMatrix[1][1], movieMatrix[1][2], movieMatrix[1][3], movieMatrix[1][4], movieMatrix[2][1], movieMatrix[2][2], movieMatrix[2][3], movieMatrix[2][4]);
	/*
	for (int i = 0; i < movieMatrixColumns; i++)
	{
		float * temp;
		cudaMemcpy(movieMatrix[i], temp, sizeof(float) * movieMatrixColumns, cudaMemcpyDeviceToHost);
		cudaMemcpy(&temp, d_movieMatrix + i, sizeof(float*), cudaMemcpyDeviceToHost);
	}*/


	for (int i = 0; i < movieMatrixColumns; i++)
	{
		float temp[10000];
		temp[1] = 0.5;
		printf("%f\n", temp[1]);
		cudaError_t errr1 = cudaMemcpy(movieMatrix, temp, sizeof(float) * movieMatrixColumns, cudaMemcpyDeviceToHost);



		//cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

		//cudaMemcpy(temp, movieMatrix[i], sizeof(float) * movieMatrixColumns, cudaMemcpyHostToDevice);
		//cudaMemcpy(d_movieMatrix + i, &temp, sizeof(float*), cudaMemcpyHostToDevice);
		//cudaError_t errr = cudaMemcpy(movieMatrix[i], d_movieMatrix+i, sizeof(float) * movieMatrixColumns, cudaMemcpyDeviceToHost);
		printf("%f\n", temp[1]);
		//printf("h_array: %d, %s\n", movieMatrix[i][1], errr);
		printf("h_array: %s\n", errr1);
	}
	printf("%f, %f, %lf, %lf\n%lf, %lf, %lf, %lf", movieMatrix[1][1], movieMatrix[1][2], movieMatrix[1][3], movieMatrix[1][4], movieMatrix[2][1], movieMatrix[2][2], movieMatrix[2][3], movieMatrix[2][4]);


Error:
	for (int i = 0; i < movieMatrixColumns; i++)
	{
		cudaFree(d_movieMatrix+i);
	}
	for (int i = 0; i < userReviewRows; i++)
	{
		cudaFree(d_userReviewMatrix + i);
		cudaFree(d_didReviewMatrix + i);
	}
	cudaFree(d_didReviewMatrix);
	cudaFree(d_movieMatrix);
	cudaFree(d_userReviewMatrix);
	cudaFree(d_userReviewMatrixColumns);
	cudaFree(d_userReviewMatrixRows);

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