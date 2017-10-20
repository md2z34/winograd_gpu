
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#define DEBUG
#define FLOAT float 
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__device__ __host__ void trans_F_2x2_3x3(FLOAT Fw[4][4], FLOAT F[3][3])
{
	// Minimal version only
	FLOAT T0[4][3];
	for (int ii = 0; ii < 3; ++ii) {
		T0[0][ii] = F[0][ii];
		T0[1][ii] = (F[0][ii] + F[2][ii] + F[1][ii]) * 0.5f;
		T0[2][ii] = (F[0][ii] + F[2][ii] - F[1][ii]) * 0.5f;
		T0[3][ii] = F[2][ii];
	}

	for (int ii = 0; ii < 4; ++ii) {
		Fw[ii][0] = T0[ii][0];
		Fw[ii][1] = (T0[ii][0] + T0[ii][2] + T0[ii][1]) * 0.5f;
		Fw[ii][2] = (T0[ii][0] + T0[ii][2] - T0[ii][1]) * 0.5f;
		Fw[ii][3] = T0[ii][2];
	}
}

__device__ __host__ void trans_I_2x2_3x3(FLOAT Iw[4][4], FLOAT I[4][4])
{
	// Minimal version only
	FLOAT T0[4][4];
	for (int ii = 0; ii < 4; ++ii) {
		T0[0][ii] = I[0][ii] - I[2][ii];
		T0[1][ii] = I[1][ii] + I[2][ii];
		T0[2][ii] = I[2][ii] - I[1][ii];
		T0[3][ii] = I[1][ii] - I[3][ii];
	}

	for (int ii = 0; ii < 4; ++ii) {
		Iw[ii][0] = T0[ii][0] - T0[ii][2];
		Iw[ii][1] = T0[ii][1] + T0[ii][2];
		Iw[ii][2] = T0[ii][2] - T0[ii][1];
		Iw[ii][3] = T0[ii][1] - T0[ii][3];
	}
}

__device__ __host__ void trans_O_2x2_3x3(FLOAT Mw[2][2], FLOAT M[4][4])
{
	// Minimal version only
	FLOAT T0[2][4];
	for (int ii = 0; ii < 4; ++ii) {
		T0[0][ii] = M[0][ii] + M[1][ii] + M[2][ii];
		T0[1][ii] = M[1][ii] - M[2][ii] - M[3][ii];
	}

	for (int ii = 0; ii < 2; ++ii) {
		Mw[ii][0] = T0[ii][0] + T0[ii][1] + T0[ii][2];
		Mw[ii][1] = T0[ii][1] - T0[ii][2] - T0[ii][3];
	}
}

__device__ __host__ void image_slice(int x, int X, int B, int D, int pad, int *start, int *stop, int pad_out[2]) {
	(*start) = x * B - pad;
	(*stop) = (*start) + D;
	pad_out[0] = 0; pad_out[1] = 0;
	if ((*start) < 0) {
		pad_out[0] = -(*start);
		(*start) = 0;
	}
	if ((*stop) - 1 >= X) {
		pad_out[1] = (*stop) - X;
	}
}

__device__ __host__ int ceil_div(int x, int y) {
	// For positive numbers only
	return x / y + (x%y != 0);
}


cudaError_t xprop_winograd(FLOAT *I, int lenI, FLOAT *F, int lenF, FLOAT *O, int lenO, int padding[2]);

__global__ void Fw_kernel(FLOAT *F_d, int lenF, FLOAT *Fw_d, int lenFw, int D, int C, int K)
{

	FLOAT tmp4x4[4][4];
	FLOAT tmp3x3[3][3];

	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int k = blockDim.y * blockIdx.y + threadIdx.y;
	for (int ii = 0; ii < 3; ++ii) {
		for (int jj = 0; jj < 3; ++jj) {
			tmp3x3[ii][jj] = F_d[c * 3 * 3 * 32 + ii * 3 * 32 + jj * 32 + k];
		}
	}

	trans_F_2x2_3x3(tmp4x4, tmp3x3);

	for (int ii = 0; ii < 4; ++ii) {
		for (int jj = 0; jj < 4; ++jj) {
			Fw_d[ii*D*C*K + jj*C*K + c*K + k] = tmp4x4[ii][jj];
		}
	}
}

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

#ifdef DEBUG
	FILE *Ow_file = fopen("Ow_gpu.txt","w");
	if (NULL == Ow_file) {
		printf("Error opening file!\n");
		exit(1);
	}
	for (int a = 0; a < lenOw; ++a) {
		fprintf(Ow_file, "%f;", Ow[a]);
	}
	fclose(Ow_file);
#endif // DEBUG
	free(I);
	free(F);
	free(Ow);
	return 0;
}

// Copy data to device memory, call the kernel, and copy the result back to host memory
cudaError_t xprop_winograd(FLOAT *I, int lenI, FLOAT *F, int lenF, FLOAT *O, int lenO, int padding[2])
{
	dim3 grid, block;
	block.x = 32;
	grid.x = 1;
	block.y = 16;
	grid.y = 2;

	FLOAT *I_d;
	FLOAT *F_d;
	FLOAT *Ow_d;
	// I shape
	int C = 32, Y = 4, X = 4, N = 32;
	// O shape
	int K = 32, P = 4, Q = 4;

	int B = 2;
	int D = B + 2;

	int Yw = ceil_div(P, B);
	int Xw = ceil_div(Q, B);

	FLOAT *Fw_d, *Fw;
	int lenFw = D*D*C*K;
	FLOAT *Iw_d;
	int lenIw = D*D*C*Yw*Xw*N;
	FLOAT *Mw_d;
	int lenMw = D*D*K*Yw*Xw*N;
	FLOAT *sliceI_d;
	int lenSliceI = C*Y*X*N;
	
	cudaError_t cudaStatus;

    // Choose which GPU to run on
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers
    cudaStatus = cudaMalloc((void**)&I_d, lenI * sizeof(FLOAT));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&F_d, lenF * sizeof(FLOAT));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&Ow_d, lenO * sizeof(FLOAT));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// Supplemetary matrices required only in device memory
	cudaStatus = cudaMalloc((void**)&Fw_d, lenFw * sizeof(FLOAT));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&Iw_d, lenIw * sizeof(FLOAT));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&Mw_d, lenIw * sizeof(FLOAT));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&sliceI_d, lenSliceI * sizeof(FLOAT));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	Fw = (FLOAT *)malloc(lenFw*sizeof(FLOAT));
	if (NULL == Fw) {
		printf("Error in allocation of F\n");
		exit(1);
	}
	// Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(I_d, I, lenI * sizeof(FLOAT), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(F_d, F, lenF * sizeof(FLOAT), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	// 	memset(O, 0, sizeof(FLOAT) * lenO);
	cudaStatus = cudaMemset(Ow_d, 0, lenO * sizeof(FLOAT));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
    // Launch kernels on the GPU with one thread for each element.
	Fw_kernel <<<grid, block >>> (F_d, lenF, Fw_d, lenFw, D, C, K);

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
    //cudaStatus = cudaMemcpy(O, Ow_d, lenO * sizeof(FLOAT), cudaMemcpyDeviceToHost);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaMemcpy failed!");
    //    goto Error;
    //}
	cudaStatus = cudaMemcpy(Fw, Fw_d, lenFw * sizeof(FLOAT), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
#ifdef DEBUG
	FILE *Fw_file = fopen("Fw_gpu.txt", "w");
	if (NULL == Fw_file) {
		printf("Error opening file!\n");
		exit(1);
	}
	for (int a = 0; a < lenFw; ++a) {
		fprintf(Fw_file, "%f;", Fw[a]);
	}
	fclose(Fw_file);
#endif // DEBUG


Error:
    cudaFree(F_d);
    cudaFree(I_d);
    cudaFree(Ow_d);
	cudaFree(Fw_d);
	free(Fw);
	cudaFree(Iw_d);
	cudaFree(Mw_d);
	cudaFree(sliceI_d);

    return cudaStatus;
}
