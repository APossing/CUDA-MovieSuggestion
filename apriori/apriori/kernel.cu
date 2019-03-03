
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
#include <chrono>
using namespace std;

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


__global__ void loadTop5(float**userArray, unsigned short *userArrayRows, unsigned short *userArrayColumns, unsigned short *top5UserArray, unsigned short *top5UserArrayColumns, bool **didSelect)
{
	short row = blockIdx.x * blockDim.x + threadIdx.x + 1;
	float biggest;
	short biggestIndex;
	if (row < *userArrayRows)
	{
		for (short i = 0; i < *top5UserArrayColumns; i++)
		{
			for (short j = 1; j < *userArrayColumns; j++)
			{
				if (!didSelect[row][j])
				{
					if (userArray[row][j] > biggest)
					{
						biggest = userArray[row][j];
						biggestIndex = j;
					}
				}
			}
			if (biggestIndex == 0)
				return;
			top5UserArray[row * (*top5UserArrayColumns) + i] = biggestIndex;
			didSelect[row][biggestIndex] = true;
			biggestIndex = 0;
			biggest = 0;
		}
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
		if (movie1 == 1867 && movie2 == 1)
			printf("");
		//if (movie1 < 11 && movie2 < 11)
			//printf("(%d,%d):\t%f, %f, %f %f\n", movie1, movie2, sqrt(bottomLeft), sqrt(bottomRight), top, top / (sqrt(bottomLeft) * sqrt(bottomRight)));
		float temp = top / (sqrt(bottomLeft) * sqrt(bottomRight));
		movieArray[movie1][movie2] = temp;
		movieArray[movie2][movie1] = temp;
	}

}

//void quicksort(float)

__global__ void computeRecommendedMovies(float**userArray, unsigned short *userArrayColumns, unsigned short *userArrayRows, float**movieArray, bool **didSelect)
{
	short movie = blockDim.x * blockIdx.x + threadIdx.x + 1;
	short user = blockDim.y * blockIdx.y + threadIdx.y + 1;
	float tempSim;
	short selected = 0;
	float top5[6];
	short top5Index[6];
	if (movie < *userArrayColumns && user < *userArrayRows && !didSelect[user][movie])
	{
		for (int i = 1; i < *userArrayColumns; i++)
		{
			if (didSelect[user][i])
			{
				tempSim = movieArray[movie][i];
				if (selected < 5)
				{
					top5[5-selected] = tempSim;
					top5Index[5-selected] = i;
					selected++;
				}
				else
				{
					top5[0] = tempSim;
					top5Index[0] = i;
					float temp;
					short temp2;

					//bubble sort......
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
		if (movie < 10 && user == 1)
			printf("\n");
		for (int i = 1; i <=selected; i++)
			sum+= top5[i] * movieArray[movie][top5Index[i]];
		if (movie < 10 && user == 1)
			printf("(user,movie,sum,selected,sum/selected)->(%d,%d,%f,%d,%f)\n", user, movie, sum, selected, sum / selected);
		userArray[user][movie] = sum / selected;
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
	printf("----------------------StartedCode-----------------------\n");
	auto t1 = std::chrono::high_resolution_clock::now();
	//int cudaCores = cuda.calculateCores();
	MovieReader m = MovieReader("movie.csv");
	UserTableReader r = UserTableReader("ratings.csv");

	auto t2 = std::chrono::high_resolution_clock::now();
	printf("-------Filing Reading completed in %d milliseconds------\n\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
	printf("--------------StartedMatrixBuilding---------------------\n");
	auto t3 = std::chrono::high_resolution_clock::now();


	float ** movieMatrix = createArr(m.movieCount);
	float ** userReviewMatrix = createBlankUserMatrix(r, m.movieCount);
	bool ** originalReviewMatrix = createBlankUserDidReviewMatrix(r, m.movieCount);
	unsigned short * recomendedMoviesMatrix = (unsigned short*)malloc(sizeof(unsigned short)* (r.users.size() + 1) * 5);
	populateUserReviewMatrix(userReviewMatrix, originalReviewMatrix, r, m);


	auto t4 = std::chrono::high_resolution_clock::now();
	printf("-------matrix created completed in %d milliseconds------\n\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count());
	printf("-----------------Started Cuda data copy-----------------\n");
	auto t5 = std::chrono::high_resolution_clock::now();


	float ** d_movieMatrix;
	float ** d_userReviewMatrix;
	bool ** d_didReviewMatrix;
	unsigned short * d_recomendedMoviesMatrix;
	string str2;
	cudaError_t cudaStatus;
	int movieMatrixColumns = m.movieCount + 1;
	int userReviewColumns = m.movieCount + 1;
	int userReviewRows = r.users.size() + 1;
	unsigned short recomendedMoviesMatrixColumns = 5;
	unsigned short recomendedMoviesMatrixRows = (r.users.size() + 1);


	cudaMalloc((void**)&d_recomendedMoviesMatrix, sizeof(unsigned short) * recomendedMoviesMatrixRows * recomendedMoviesMatrixColumns);
	cudaStatus = cudaMemcpy(d_recomendedMoviesMatrix, recomendedMoviesMatrix, sizeof(unsigned short)* recomendedMoviesMatrixRows * recomendedMoviesMatrixColumns, cudaMemcpyHostToDevice);


	unsigned short * d_recMoviesColumns;
	cudaMalloc((void**)&d_recMoviesColumns, sizeof(unsigned short));
	cudaStatus = cudaMemcpy(d_recMoviesColumns, &recomendedMoviesMatrixColumns, sizeof(unsigned short), cudaMemcpyHostToDevice);


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



	int blockX = ceil(userReviewRows / 256.0);
	int blockY = ceil(userReviewRows / 16.0);
	int blockXType2 = ceil(userReviewColumns / 256);

	auto t6 = std::chrono::high_resolution_clock::now();
	printf("-------cuda data copy completed in %d milliseconds------\n\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count());
	printf("--------Started Compute Averages for movies-------------\n");
	auto t7 = std::chrono::high_resolution_clock::now();

	computeAverageType2 << <blockXType2, 256 >> > (d_userReviewMatrix, d_userReviewMatrixColumns, d_userReviewMatrixRows);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	auto t8 = std::chrono::high_resolution_clock::now();
	printf("-------Compute Averages for movies completed in %d milliseconds------\n\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t7).count());
	printf("--------Started compute simularMovies-------------\n");
	auto t9 = std::chrono::high_resolution_clock::now();

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
	blockX = ceil(movieMatrixColumns / 8.0);
	blockY = ceil(userReviewRows / 8.0);

	t8 = std::chrono::high_resolution_clock::now();
	printf("Compute simular movies completed in %d milliseconds\n\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t9).count());
	printf("------Started compute recommended movies-----------\n");
	t9 = std::chrono::high_resolution_clock::now();


	computeRecommendedMovies<<<dim3(blockX, blockY), dim3(8, 8) >>>(d_userReviewMatrix, d_userReviewMatrixColumns, d_userReviewMatrixRows, d_movieMatrix, d_didReviewMatrix);
	cudaStatus = cudaGetLastError();
	if (cudaSuccess != cudaGetLastError())
		printf("Error!\n");

	cudaStatus = cudaGetLastError();
	cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	t8 = std::chrono::high_resolution_clock::now();
	printf("Compute recommended movies completed in %d milliseconds\n", std::chrono::duration_cast<std::chrono::milliseconds>(t8 - t9).count());
	blockX = ceil(userReviewRows / 8.0);
	cudaError_t cuda3 = cudaGetLastError();
	str2 = cudaGetErrorString(cuda3);

	loadTop5 << <blockX, 16 >> > (d_userReviewMatrix, d_userReviewMatrixRows, d_userReviewMatrixColumns, d_recomendedMoviesMatrix, d_recMoviesColumns, d_didReviewMatrix);
	cudaError_t cuda2 = cudaGetLastError();
	str2 = cudaGetErrorString(cuda2);
	cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(recomendedMoviesMatrix, d_recomendedMoviesMatrix, sizeof(unsigned short)* recomendedMoviesMatrixRows * recomendedMoviesMatrixColumns, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda MemCpy failed!!\n", cudaStatus);
		goto Error;
	}

	printf("%d, %d, %d, %d", recomendedMoviesMatrix[5], recomendedMoviesMatrix[6], recomendedMoviesMatrix[7], recomendedMoviesMatrix[8]);
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