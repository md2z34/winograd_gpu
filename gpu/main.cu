
#include <stdio.h>
#include <stdlib.h>
#include "debug.h"
#include "xprop_winograd.h"

int main()
{
	FLOAT *I; //[32][4][4][32]; 
	int lenI = 32 * 4 * 4 * 32;
	FLOAT *F; //[32][3][3][32];
	int lenF = 32 * 3 * 3 * 32;
	FLOAT *Ow; // [32][4][4][32];
	int lenOw = 32 * 4 * 4 * 32;
	int padding[2];
	// Init input values
	I = (FLOAT *)malloc(sizeof(FLOAT)*lenI);
	if (NULL == I) {
		return (-1);
	}
	F = (FLOAT *)malloc(sizeof(FLOAT)*lenF);
	if (NULL == F) {
		return (-1);
	}
	Ow = (FLOAT *)malloc(sizeof(FLOAT)*lenOw);
	if (NULL == Ow) {
		return (-1);
	}

	padding[0] = 1; padding[1] = 1;
	int k = 0;
	for (int a = 0; a < lenI; ++a) {
		//I[a] = (FLOAT)k++;
		I[a] = 1.0;
	}
	for (int a = 0; a < lenF; ++a) {
		//F[a] = (FLOAT)k++;
		F[a] = 1.0;
	}
#ifdef DEBUG
	FILE *F_file = fopen("F_gpu.txt","w");
	if (NULL == F_file) {
		printf("Error opening file!\n");
		exit(1);
	}
	for (int a = 0; a < lenF; ++a) {
		fprintf(F_file,"%f;", F[a]);
	}
	fclose(F_file);
#endif // DEBUG
	
	// Invoke Winograd convolution kernel.
	cudaError_t cudaStatus = xprop_winograd(I, lenI, F, lenF, Ow, lenOw, padding);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "xprop_winograd failed!");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	FILE *Ow_file = fopen("Ow_gpu.txt","w");
	if (NULL == Ow_file) {
		printf("Error opening file!\n");
		exit(1);
	}
	for (int a = 0; a < lenOw; ++a) {
		fprintf(Ow_file, "%f;", Ow[a]);
	}
	fclose(Ow_file);
	free(I);
	free(F);
	free(Ow);
	return 0;
}