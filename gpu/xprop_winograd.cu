#include "xprop_winograd.h"

cudaError_t xprop_winograd(FLOAT *I, int lenI, FLOAT *F, int lenF, FLOAT *O, int lenO, int padding[2])
{
	dim3 grid, block;
	block.x = BLK_X;
	grid.x = GRD_X;
	block.y = BLK_Y;
	grid.y = GRD_Y;

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
	FLOAT *Iw_d, *Iw;
	int lenIw = D*D*C*Yw*Xw*N;
	FLOAT *Mw_d, *Mw;
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

	Fw = (FLOAT *)malloc(lenFw * sizeof(FLOAT));
	if (NULL == Fw) {
		printf("Error in allocation of Fw\n");
		exit(1);
	}
	Iw = (FLOAT *)malloc(lenIw * sizeof(FLOAT));
	if (NULL == Iw) {
		printf("Error in allocation of Iw\n");
		exit(1);
	}
	Mw = (FLOAT *)malloc(lenMw * sizeof(FLOAT));
	if (NULL == Mw) {
		printf("Error in allocation of Mw\n");
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
		fprintf(stderr, "cudaMemset failed!");
		goto Error;
	}

	Fw_call(F_d, lenF, Fw_d, lenFw, D, C, K);

	for (int y = 0; y < Yw; ++y) {
		int start_y, stop_y, pad_y[2];
		image_slice(y, Y, B, D, padding[0], &start_y, &stop_y, pad_y);
		for (int x = 0; x < Xw; ++x) {
			int start_x, stop_x, pad_x[2];
			image_slice(x, X, B, D, padding[1], &start_x, &stop_x, pad_x);
			// 	memset(sliceI, 0, sizeof(FLOAT)*C*Y*X*N);
			cudaStatus = cudaMemset(sliceI_d, 0, lenSliceI * sizeof(FLOAT));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed!");
				goto Error;
			}

			sliceI_call(sliceI_d, I_d, pad_x, pad_y,
				start_x, start_y, stop_x, stop_y,
				Y, X, C, N);

			Iw_call(sliceI_d, Iw_d, x, y, Xw, Yw, D, Y, X, C, N);
		}
	}

	FLOAT *mat1T_d;
	int lenMat1T = K*C;
	cudaStatus = cudaMalloc((void**)&mat1T_d, lenMat1T * sizeof(FLOAT));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	FLOAT *mat2_d;
	int lenMat2 = C*Yw*Xw*N;
	cudaStatus = cudaMalloc((void**)&mat2_d, lenMat2 * sizeof(FLOAT));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	FLOAT *matout_d;
	int lenMatout = C*Yw*Xw*N;
	cudaStatus = cudaMalloc((void**)&matout_d, lenMatout * sizeof(FLOAT));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	for (int s = 0; s < D; ++s) {
		for (int t = 0; t < D; ++t) {
			cpy_mat12_call(Fw_d, Iw_d, mat1T_d, mat2_d, s, t, Xw, Yw, D, C, N, K);

			// 	memset(matout, 0, sizeof(FLOAT)*C*Yw*Xw*N);
			cudaStatus = cudaMemset(matout_d, 0, lenMatout * sizeof(FLOAT));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemset failed!");
				goto Error;
			}

			MatMul_call(mat1T_d, mat2_d, matout_d, K, C, C, Yw*Xw*N, C, Yw*Xw*N,
				Xw, Yw, K, N, C);

			cpy_to_Mw_call(Mw_d, matout_d, s, t, Xw, Yw, D, N, K);
		}
	}

	// Iterate over the convovled result in the pointwise space and apply inverse transform
	for (int y = 0; y < Yw; ++y) {
		int p = y*B;
		int plen = ((p + 1) < P) ? 2 : 1;
		for (int x = 0; x < Xw; ++x) {
			int q = x*B;
			int qlen = ((q + 1) < Q) ? 2 : 1;

			Ow_call(Ow_d, Mw_d, plen, qlen, Xw, Yw, p, q, x, y, D, N, K);
		}
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(O, Ow_d, lenO * sizeof(FLOAT), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

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

	cudaStatus = cudaMemcpy(Iw, Iw_d, lenIw * sizeof(FLOAT), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
#ifdef DEBUG
	FILE *Iw_file = fopen("Iw_gpu.txt", "w");
	if (NULL == Iw_file) {
		printf("Error opening file!\n");
		exit(1);
	}
	for (int a = 0; a < lenIw; ++a) {
		fprintf(Fw_file, "%f;", Iw[a]);
	}
	fclose(Iw_file);
#endif // DEBUG

	cudaStatus = cudaMemcpy(Mw, Mw_d, lenMw * sizeof(FLOAT), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
#ifdef DEBUG
	FILE *Mw_file = fopen("Mw_gpu.txt", "w");
	if (NULL == Mw_file) {
		printf("Error opening file!\n");
		exit(1);
	}
	for (int a = 0; a < lenMw; ++a) {
		fprintf(Mw_file, "%f;", Mw[a]);
	}
	fclose(Mw_file);
#endif // DEBUG


Error:
	cudaFree(mat1T_d);
	cudaFree(mat2_d);
	cudaFree(matout_d);
	cudaFree(F_d);
	cudaFree(I_d);
	cudaFree(Ow_d);
	cudaFree(Fw_d);
	free(Fw);
	free(Iw);
	free(Mw);
	cudaFree(Iw_d);
	cudaFree(Mw_d);
	cudaFree(sliceI_d);

	return cudaStatus;
}
