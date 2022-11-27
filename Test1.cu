#include<time.h>
#include<string.h>
#include<math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>
using namespace std;

//Transform 3D coordinates into 1D coordinate
int getIndex(int x, int y, int z, int height, int width) {
	return x + y * height + z * height * width;
}

__device__ int arrayIndex(int x, int y, int z, int height, int width) {
	return x + y * height + z * height * width;
}

__global__ void calcTn(float dx, float dy, float dz, float dt, float alpha, int xLen, int yLen, int zLen, float* T, float* Tn)
{
	for (int i = 1; i < xLen - 1; i++) {
		for (int j = 1; j < yLen - 1; j++) {
			for (int k = 0; k < zLen; k++) {
				int i_j_k = arrayIndex(i, j, k, yLen, xLen);
				int bi_j_k = arrayIndex(i - 1, j, k, yLen, xLen);
				int fi_j_k = arrayIndex(i + 1, j, k, yLen, xLen);
				int i_bj_k = arrayIndex(i, j - 1, k, yLen, xLen);
				int i_fj_k = arrayIndex(i, j + 1, k, yLen, xLen);
				int i_j_bk = arrayIndex(i, j, k - 1, yLen, xLen);
				int i_j_fk = arrayIndex(i, j, k + 1, yLen, xLen);

				Tn[i_j_k] = T[i_j_k] + (dt * (alpha / (dx * dx) * (T[fi_j_k] - 2 * T[i_j_k] + T[bi_j_k]))) + (dt * (alpha / (dy * dy) * (T[i_fj_k] - 2 * T[i_j_k] + T[i_bj_k]))) + (dt * (alpha / (dz * dz) * (T[i_j_fk] - 2 * T[i_j_k] + T[i_j_bk])));
			}
		}
		
	}
	
}

int main()
{

	clock_t init, final;
	init = clock();

	int xNodes = 128;
	int yNodes = 128;
	int zNodes = 32;
	int N = xNodes * yNodes * zNodes;

	float dx, dy, dz, dt;
	dx = 0.1;
	dy = 0.1;
	dz = 0.1;
	dt = 0.0001;

	float cond, cp, rho, alpha;
	cond = 400.0f; // (W/(mK))
	cp = 385.0f; // J/(kgK)
	rho = 8940.0f; // kg/m^3
	alpha = cond * cp / rho;
	int Tini = 20.0f;

	//Initializing variable sized temperature arrays
	std::vector<float> T(N, Tini);
	std::vector<float> Tn(N, Tini);

	//Initializing lower wall to be T = 100
	for (int i = 0; i < xNodes; i++)
	{
		for (int j = 0; j < yNodes; j++)
		{
			int ikx = getIndex(i, j, 0, yNodes, xNodes);
			T[ikx] = 100.0f;
			Tn[ikx] = 100.0f;
		}
	}

	//Convert from std::vector to float to get working with CUDA
	float* Q = &T[0];
	float* Qn = &Tn[0];

	float* d_T, * d_Tn, * oTn;

	oTn = (float*)malloc(N * sizeof(float));

	cudaMalloc(&d_T, N * sizeof(float));
	cudaMalloc(&d_Tn, N * sizeof(float));

	checkCudaErrors(cudaMemcpy(d_T, Q, N * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Tn, Qn, N * sizeof(float), cudaMemcpyHostToDevice));

	calcTn << <1, 1024 >> > (dx, dy, dz, dt, alpha, xNodes, yNodes, zNodes, d_T, d_Tn);

	cudaMemcpy(oTn, d_Tn, N * sizeof(float), cudaMemcpyDeviceToHost);

	//Final output
	final = clock() - init;
	cout << "Calculation Time : " << (double)final / ((double)CLOCKS_PER_SEC) << " seconds" << endl;

	int arrayInt = getIndex(50, 50, 1, yNodes, xNodes);
	cout << "T[50][50][1] = " << oTn[arrayInt] << endl;
	cin.ignore();
}