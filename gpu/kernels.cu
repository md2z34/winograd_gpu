#include "kernels.h"
#include "debug.h"
#include <stdio.h>
#include <stdlib.h>
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

	if ((c < C) && (k < K)) {
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
}

__global__ void sliceI_kernel(FLOAT *sliceI_d, FLOAT *I_d, int pad_x0, int pad_x1, int pad_y0, int pad_y1,
	int start_x, int start_y, int stop_x, int stop_y,
	int Y, int X, int C, int N)
{
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int n = blockDim.y * blockIdx.y + threadIdx.y;
	int pad_x[2], pad_y[2];
	pad_x[0] = pad_x0; pad_x[1] = pad_x1;
	pad_y[0] = pad_y0; pad_y[1] = pad_y1;
	if ((c < C) && (n < N)) {

		for (int yy = start_y; yy < MIN(stop_y, Y); ++yy) {
			for (int xx = start_x; xx < MIN(stop_x, X); ++xx) {
				if ((pad_x[0] > 0) && (pad_y[0] > 0)) {
					sliceI_d[c*Y*X*N + (yy + pad_y[0])*X*N + (xx + pad_x[0])*N + n] = I_d[c * 4 * 4 * 32 + yy * 4 * 32 + xx * 32 + n];
				}
				else if ((pad_x[1] > 0) && (pad_y[0] > 0)) {
					sliceI_d[c*Y*X*N + (yy + pad_y[0])*X*N + (xx - pad_x[1])*N + n] = I_d[c * 4 * 4 * 32 + yy * 4 * 32 + xx * 32 + n];
				}
				else if ((pad_x[0] > 0) && (pad_y[1] > 0)) {
					sliceI_d[c*Y*X*N + (yy - pad_y[1])*X*N + (xx + pad_x[0])*N + n] = I_d[c * 4 * 4 * 32 + yy * 4 * 32 + xx * 32 + n];
				}
				else if ((pad_x[1] > 0) && (pad_y[1] > 0)) {
					sliceI_d[c*Y*X*N + (yy - pad_y[1])*X*N + (xx - pad_x[1])*N + n] = I_d[c * 4 * 4 * 32 + yy * 4 * 32 + xx * 32 + n];
				}
				else {
					sliceI_d[c*Y*X*N + (yy)*X*N + (xx)*N + n] = I_d[c * 4 * 4 * 32 + yy * 4 * 32 + xx * 32 + n];
				}
			}
		}
	}
}

__global__ void Iw_kernel(FLOAT *sliceI_d, FLOAT *Iw_d,
	int x, int y, int Xw, int Yw,
	int D, int Y, int X, int C, int N)
{
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int n = blockDim.y * blockIdx.y + threadIdx.y;
	FLOAT in[4][4];
	FLOAT out[4][4];
	if ((c < C) && (n < N)) {
		for (int yy = 0; yy < Y; ++yy) {
			for (int xx = 0; xx < X; ++xx) {
				in[yy][xx] = sliceI_d[c*Y*X*N + (yy)*X*N + (xx)*N + n];
			}
		}
		trans_I_2x2_3x3(out, in);
		// Iw[:,:,c,y,x,n] = out
		// std::cout << "Iw:" << std::endl;
		for (int dx = 0; dx < D; ++dx) {
			for (int dy = 0; dy < D; ++dy) {
				Iw_d[dx*D*C*Yw*Xw*N + dy*C*Yw*Xw*N + c*Yw*Xw*N + y*Xw*N + x*N + n] = out[dx][dy];
			}
		}
	}
}

__global__ void MatMul_kernel(FLOAT *A, FLOAT *B, FLOAT *C,
	int ar, int ac, int br, int bc, int cr, int cc)
{
	FLOAT val = 0;
	__shared__ FLOAT Ash[TILE_WIDTH][TILE_WIDTH];
	__shared__ FLOAT Bsh[TILE_WIDTH][TILE_WIDTH];
	int col = TILE_WIDTH*blockIdx.x + threadIdx.x;
	int row = TILE_WIDTH*blockIdx.y + threadIdx.y;

	for (int k = 0; k < ac / TILE_WIDTH; ++k) {
		Ash[threadIdx.y][threadIdx.x] = A[row*ac + k*TILE_WIDTH + threadIdx.x];
		Bsh[threadIdx.y][threadIdx.x] = B[(k*TILE_WIDTH + threadIdx.y)*bc + col];
		__syncthreads();
		for (int n = 0; n < TILE_WIDTH; ++n) {
			val += Ash[threadIdx.y][n] * Bsh[n][threadIdx.x];
		}
		__syncthreads();
	}
	if (row < cr && col < cc)
		C[(blockIdx.y*blockDim.y + threadIdx.y)*cc + blockIdx.x*blockDim.x + threadIdx.x] = val;
}

__global__ void cpy_mat12_kernel(FLOAT *Fw_d, FLOAT *Iw_d,
	FLOAT *mat1T_d, FLOAT *mat2_d,
	int s, int t, int Xw, int Yw,
	int D, int C, int N, int K)
{
	int c = blockDim.x * blockIdx.x + threadIdx.x;
	int n = blockDim.y * blockIdx.y + threadIdx.y;
	int k = n;
	if ((c < C) && (n < C)) {
		// Fill in mat1T = Fw[D][D][C][K]
		mat1T_d[k*C + c] = Fw_d[s*D*C*K + t*C*K + c*K + k];
		// Fill in mat2 = Iw[D][D][C][Yw][Xw][N]
		for (int yw = 0; yw < Yw; ++yw) {
			for (int xw = 0; xw < Xw; ++xw) {
				mat2_d[c*Yw*Xw*N + yw*Xw*N + xw*N + n] = Iw_d[s*D*C*Yw*Xw*N + t*C*Yw*Xw*N + c*Yw*Xw*N + yw*Xw*N + xw*N + n];
			}
		}
	}
}

__global__ void cpy_to_Mw_kernel(FLOAT *Mw_d, FLOAT *matout_d,
	int s, int t, int Xw, int Yw,
	int D, int N, int K)
{
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	int n = blockDim.y * blockIdx.y + threadIdx.y;
	if ((k < K) && (n < N)) {
		for (int yw = 0; yw < Yw; ++yw) {
			for (int xw = 0; xw < Xw; ++xw) {
				Mw_d[s*D*K*Yw*Xw*N + t*K*Yw*Xw*N + k*Yw*Xw*N + yw*Xw*N + xw*N + n] =
					matout_d[k*Yw*Xw*N + yw*Xw*N + xw*N + n];
			}
		}
	}
}

__global__ void Ow_kernel(FLOAT *O_d, FLOAT *Mw_d,
	int plen, int qlen, int Xw, int Yw,
	int p, int q,
	int x, int y,
	int D, int N, int K)
{
	FLOAT tmp4x4[4][4];
	FLOAT tmp2x2[2][2];

	int k = blockDim.x * blockIdx.x + threadIdx.x;
	int n = blockDim.y * blockIdx.y + threadIdx.y;
	if ((k < K) && (n < N)) {
		// Mw[:, : , k, y, x, n] -> tmp4x4
		for (int ii = 0; ii < D; ++ii) {
			for (int jj = 0; jj < D; ++jj) {
				tmp4x4[ii][jj] = Mw_d[ii*D*K*Yw*Xw*N + jj*K*Yw*Xw*N + k*Yw*Xw*N + y*Xw*N + x*N + n];
			}
		}
		trans_O_2x2_3x3(tmp2x2, tmp4x4);
		// tmp2x2 -> O[k,p:p + plen,q:q + qlen,n]
		for (int ii = 0; ii < plen; ++ii) {
			for (int jj = 0; jj < qlen; ++jj) {
				if (((p + ii) >= 0) && ((p + ii) < 4) && ((q + jj) >= 0) && ((q + jj) < 4)) {
					O_d[k * 4 * 4 * 32 + (p + ii) * 4 * 32 + (q + jj) * 32 + n] = tmp2x2[ii][jj];
				}
			}
		}
	}
}

void Fw_call(FLOAT *F_d, int lenF, FLOAT *Fw_d, int lenFw, int D, int C, int K) {
	dim3 grid, block;
	block.x = BLK_X;
	grid.x = GRD_X;
	block.y = BLK_Y;
	grid.y = GRD_Y;
	cudaError_t cudaStatus;

	Fw_kernel <<<grid, block >>> (F_d, lenF, Fw_d, lenFw, D, C, K);
	
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Fw_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
		exit(1);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Fw_kernel!\n", cudaStatus);
		//goto Error;
		exit(1);
	}
}

void sliceI_call(FLOAT *sliceI_d, FLOAT *I_d, int pad_x[2], int pad_y[2],
	int start_x, int start_y, int stop_x, int stop_y,
	int Y, int X, int C, int N) {
	dim3 grid, block;
	block.x = BLK_X;
	grid.x = GRD_X;
	block.y = BLK_Y;
	grid.y = GRD_Y;
	cudaError_t cudaStatus;

	sliceI_kernel <<<grid, block >>>(sliceI_d, I_d, pad_x[0], pad_x[1], pad_y[0], pad_y[1], 
		start_x, start_y, stop_x, stop_y, 
		Y, X, C, N);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "sliceI_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
		exit(1);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sliceI_kernel!\n", cudaStatus);
		//goto Error;
		exit(1);
	}
}

void Iw_call(FLOAT *sliceI_d, FLOAT *Iw_d,
	int x, int y, int Xw, int Yw,
	int D, int Y, int X, int C, int N) {
	dim3 grid, block;
	block.x = BLK_X;
	grid.x = GRD_X;
	block.y = BLK_Y;
	grid.y = GRD_Y;
	cudaError_t cudaStatus;

	Iw_kernel <<<grid, block >>>(sliceI_d, Iw_d, x, y, Xw, Yw, D, Y, X, C, N);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Iw_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
		exit(1);
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Iw_kernel!\n", cudaStatus);
		//goto Error;
		exit(1);
	}
}

void cpy_mat12_call(FLOAT *Fw_d, FLOAT *Iw_d,
	FLOAT *mat1T_d, FLOAT *mat2_d,
	int s, int t, int Xw, int Yw,
	int D, int C, int N, int K) {

	dim3 grid, block;
	block.x = BLK_X;
	grid.x = GRD_X;
	block.y = BLK_Y;
	grid.y = GRD_Y;
	cudaError_t cudaStatus;

	cpy_mat12_kernel <<<grid, block >>>(Fw_d, Iw_d, mat1T_d, mat2_d, s, t, Xw, Yw, D, C, N, K);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cpy_mat12_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
		exit(1);
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching cpy_mat12_kernel!\n", cudaStatus);
		//goto Error;
		exit(1);
	}
}

void MatMul_call(FLOAT *mat1T_d, FLOAT *mat2_d, FLOAT *matout_d,
	int ar, int ac, int br, int bc, int cr, int cc,
	int Xw, int Yw, int K, int N, int C)
{
	dim3 grid_mm, block_mm;
	block_mm.x = TILE_WIDTH;
	grid_mm.x = Yw*Xw*N / TILE_WIDTH;
	block_mm.y = TILE_WIDTH;
	grid_mm.y = K / TILE_WIDTH;
	cudaError_t cudaStatus;

	MatMul_kernel <<<grid_mm, block_mm >>>(mat1T_d, mat2_d, matout_d, K, C, C, Yw*Xw*N, C, Yw*Xw*N);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "MatMul_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
		exit(1);
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching MatMul_kernel!\n", cudaStatus);
		//goto Error;
		exit(1);
	}
}

void cpy_to_Mw_call(FLOAT *Mw_d, FLOAT *matout_d,
	int s, int t, int Xw, int Yw,
	int D, int N, int K) {

	dim3 grid, block;
	block.x = BLK_X;
	grid.x = GRD_X;
	block.y = BLK_Y;
	grid.y = GRD_Y;
	cudaError_t cudaStatus;

	cpy_to_Mw_kernel <<<grid, block >>>(Mw_d, matout_d, s, t, Xw, Yw, D, N, K);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cpy_to_Mw_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
		exit(1);
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching cpy_to_Mw_kernel!\n", cudaStatus);
		//goto Error;
		exit(1);
	}
}

void Ow_call(FLOAT *Ow_d, FLOAT *Mw_d,
	int plen, int qlen, int Xw, int Yw,
	int p, int q,
	int x, int y,
	int D, int N, int K) {

	dim3 grid, block;
	block.x = BLK_X;
	grid.x = GRD_X;
	block.y = BLK_Y;
	grid.y = GRD_Y;
	cudaError_t cudaStatus;

	Ow_kernel <<<grid, block >>>(Ow_d, Mw_d, plen, qlen, Xw, Yw, p, q, x, y, D, N, K);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "O_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//goto Error;
		exit(1);
	}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching O_kernel!\n", cudaStatus);
		//goto Error;
		exit(1);
	}
}