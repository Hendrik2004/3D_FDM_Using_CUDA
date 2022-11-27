#include<time.h>
#include<string.h>
#include<math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iomanip>
#include "device_launch_parameters.h"

using namespace std;
struct float4_t
{
	double x, y, z, t;
};
__global__ void calcTprima(float4_t* temp, float4_t* tempprima, double *error_acum, int count, int ancho, int alto, int profundo)
{
	if (count <= 1)
		return;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (temp[idx].x == 0 || temp[idx].y == 0 || temp[idx].z == 0 || temp[idx].x == (ancho + 1) || temp[idx].y == (ancho + 1) || temp[idx].z == (profundo + 1)) {
		tempprima[idx].t = temp[idx].t;
	}
	else {
		tempprima[idx].t = (temp[idx + ancho + 2].t + temp[idx - ancho - 2].t + temp[idx - 1].t + temp[idx + 1].t + temp[idx - (ancho + 2) * (alto + 2)].t + temp[idx + (ancho + 2) * (alto + 2)].t) / 6;
		error_acum[0] = fabs(temp[idx].t - tempprima[idx].t);
		temp[idx].t = tempprima[idx].t;
	}
}
int main()
{
	clock_t start, end, end_2,start_2;
	start = clock(); // Recording the starting clock tick.
	double T0, T1, T2, T3, T4, T5, T6;
	const int alto = 20;
	const int ancho = 20;
	const int profundo = 50;
	int i, j, k;
	double startTime = std::clock();
	double runTime_GPU = 0;

	//clock_t cpu_start, cpu_end;
	//sum_arrays_cpu(h_a, h_b, cpu_result, size);
	//cpu_end = clock();
	double error = 0.00000000000001;
	T0 = 25;
	T1 = 150;
	T2 = 35;
	T3 = 55;
	T4 = 40;
	T5 = 65;
	T6 = 75;
	cudaError_t cudaStatus;
	const int count = (ancho + 2) * (alto + 2) * (profundo + 2);

	float4_t* h_temp = new float4_t[count];
	float4_t* d_temp;
	float4_t* h_tempprima = new float4_t[count];
	float4_t* d_tempprima;
	double h_error_acum = 0;
	double *d_error_acum = 0;
	FILE* fp;

	cudaMalloc(&d_temp, count * sizeof(float4_t));
	if (d_temp == NULL) {
		std::cout << "Failed to alloc mem in GPU" << std::endl;
		delete[] h_temp;
		return -1;
	}
	cudaMalloc(&d_tempprima, count * sizeof(float4_t));
	if (d_temp == NULL) {
		std::cout << "Failed to alloc mem in GPU" << std::endl;
		delete[] h_temp;
		delete[] h_tempprima;
		return -1;
	}
	cudaMalloc(&d_error_acum, sizeof(double));
	if (d_error_acum == NULL) {
		std::cout << "Failed to alloc mem in GPU" << std::endl;

		delete[] h_temp;
		delete[] h_tempprima;
		return -1;
	}

	// init host memory



	int z = 0;
	clock_t start_1, end_1;
	start_1 = clock(); // Recording the starting clock tick.

	while (z < count) {
	
	for (i = 0; i < profundo + 2; i++) // X
	{
		for (j = 0; j < ancho + 2; j++) //Y
		{
			for (k = 0; k < alto + 2; k++)// Z
			{
				if (k == 0) {
					h_temp[z].x = k;
					h_temp[z].y = j;
					h_temp[z].z = i;
					h_temp[z].t = T1;
					h_temp[z].t = T1;
					h_tempprima[z].x = k;
					h_tempprima[z].y = j;
					h_tempprima[z].z = i;
					h_tempprima[z].t = T1;
					z = z + 1;
				}
				else if (k > 0 && j == 0) {
					h_temp[z].x = k;
					h_temp[z].y = j;
					h_temp[z].z = i;
					h_temp[z].t = T2;
					h_tempprima[z].x = k;
					h_tempprima[z].y = j;
					h_tempprima[z].z = i;
					h_tempprima[z].t = T2;
					z = z + 1;
				}
				else if (k > 0 && j == (ancho + 1)) {
					h_temp[z].x = k;
					h_temp[z].y = j;
					h_temp[z].z = i;
					h_temp[z].t = T3;
					h_tempprima[z].x = k;
					h_tempprima[z].y = j;
					h_tempprima[z].z = i;
					h_tempprima[z].t = T3;
					z = z + 1;
				}
				else if (k == (alto + 1))
				{
					h_temp[z].x = k;
					h_temp[z].y = j;
					h_temp[z].z = i;
					h_temp[z].t = T4;
					h_tempprima[z].x = k;
					h_tempprima[z].y = j;
					h_tempprima[z].z = i;
					h_tempprima[z].t = T4;
					z = z + 1;
				}
				else if (i == 0)
				{
					h_temp[z].x = k;
					h_temp[z].y = j;
					h_temp[z].z = i;
					h_temp[z].t = T5;
					h_tempprima[z].x = k;
					h_tempprima[z].y = j;
					h_tempprima[z].z = i;
					h_tempprima[z].t = T5;
					z = z + 1;
				}
				else if (k == (profundo + 1))
				{
					h_temp[z].x = k;
					h_temp[z].y = j;
					h_temp[z].z = i;
					h_temp[z].t = T6;
					h_tempprima[z].x = k;
					h_tempprima[z].y = j;
					h_tempprima[z].z = i;
					h_tempprima[z].t = T6;
					z = z + 1;
				}
				else {
					h_temp[z].x = k;
					h_temp[z].y = j;
					h_temp[z].z = i;
					h_temp[z].t = T0;
					h_tempprima[z].x = k;
					h_tempprima[z].y = j;
					h_tempprima[z].z = i;
					h_tempprima[z].t = T0;
					z = z + 1;
				}
			}
		}
	}
	}
	// Recording the end clock tick.
	end_1 = clock();
	// Calculating time required to complete Matrix Assignation
	double time_taken_1 = double(end_1 - start_1) / double(CLOCKS_PER_SEC);
	cout << "Time taken for Matrix assignation : " << fixed << time_taken_1 << setprecision(5);
	cout << " sec " << endl;

	int iteracion = 0;
	start_2 = clock();
	while (h_error_acum == 0) {
		iteracion = iteracion + 1;
		// move data from host to device
		cudaMemcpy(d_temp, h_temp, count * sizeof(float4_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_tempprima, h_tempprima, count * sizeof(float4_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_error_acum, &h_error_acum, sizeof(double), cudaMemcpyHostToDevice);

		dim3 numBlocks(count / 32 + 1);
		dim3 numThreads(32);
		double startTime_GPU = std::clock();
		calcTprima << <numBlocks, numThreads >> > (d_temp, d_tempprima, d_error_acum, count, ancho, alto, profundo);


		double endTime_GPU = std::clock();
		runTime_GPU = runTime_GPU+endTime_GPU- startTime_GPU;
		//Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "\naddKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			//goto Error;
		}

		// move result from device to host
		cudaMemcpy(h_temp, d_temp, count * sizeof(float4_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_tempprima, d_tempprima, count * sizeof(float4_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_error_acum, d_error_acum, sizeof(double), cudaMemcpyDeviceToHost);
		
		if (h_error_acum > error) h_error_acum = 0;

	}
	end_2 = clock();
	// Calculating time required to complete Matrix Assignation
	double time_taken_2 = double(end_2 - start_2) / double(CLOCKS_PER_SEC);
	cout << "Time taken to Matrix calculation was : " << fixed << time_taken_2 << setprecision(5);
	cout << " sec " << endl;

	clock_t start_3, end_3;
	start_3 = clock();
	//double endTime = std::clock();
	//double runTime = endTime - startTime;
	//cout << "The total ejecution time was : " << runTime/1000<<" Seconds\n"<<"The GPU time was: "<<runTime_GPU/1000<<" Seconds\n";

	fp = fopen("Test_Data3D_test_CUDA.xls", "w");
	fprintf(fp, "\"X\", \"Y\", \"Z\", \"T\"\n");

	for (i = 0; i < count; i++)
	{
		fprintf(fp, "%f,%f,%f,%f \n", h_temp[i].x, h_temp[i].y, h_temp[i].z, h_temp[i].t);

	}
	fclose(fp);
	//cout << h_error_acum << " in " << count << " iteracions" << endl;
	cudaFree(d_temp);
	cudaFree(d_tempprima);
	cudaFree(d_error_acum);
	delete[] h_temp;
	delete[] h_tempprima;

	end_3 = clock();
	// Calculating time required to complete Matrix Assignation
	double time_taken_3 = double(end_3 - start_3) / double(CLOCKS_PER_SEC);
	cout << "Time taken to Matrix calculation was : " << fixed << time_taken_3 << setprecision(5);
	cout << " sec " << endl;

	cout << "Final error value: " << fixed << h_error_acum << setprecision(8) << " in " << count << " iterations" << endl;
	end = clock();
	double total_time = double(end - start) / double(CLOCKS_PER_SEC);
	cout << "Total time taken by the entire code : " << fixed << total_time << setprecision(5);
	cout << " sec " << endl;
	return 0;
}