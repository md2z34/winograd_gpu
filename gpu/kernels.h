#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "debug.h"

#define BLK_X 32
#define BLK_Y 16
#define GRD_X 1
#define GRD_Y 2

#define TILE_WIDTH 8	// for matrix multiplication

__device__ __host__ int ceil_div(int x, int y);
__device__ __host__ void image_slice(int x, int X, int B, int D, int pad, int *start, int *stop, int pad_out[2]);
void Fw_call(FLOAT *F_d, int lenF, FLOAT *Fw_d, int lenFw, int D, int C, int K);
void sliceI_call(FLOAT *sliceI_d, FLOAT *I_d, int pad_x[2], int pad_y[2],
	int start_x, int start_y, int stop_x, int stop_y,
	int Y, int X, int C, int N);
void Iw_call(FLOAT *sliceI_d, FLOAT *Iw_d,
	int x, int y, int Xw, int Yw,
	int D, int Y, int X, int C, int N);
void MatMul_call(FLOAT *mat1T_d, FLOAT *mat2_d, FLOAT *matout_d,
	int ar, int ac, int br, int bc, int cr, int cc,
	int Xw, int Yw, int K, int N, int C);
void cpy_mat12_call(FLOAT *Fw_d, FLOAT *Iw_d,
	FLOAT *mat1T_d, FLOAT *mat2_d,
	int s, int t, int Xw, int Yw,
	int D, int C, int N, int K);
void cpy_to_Mw_call(FLOAT *Mw_d, FLOAT *matout_d,
	int s, int t, int Xw, int Yw,
	int D, int N, int K);
FLOAT Ow_call(FLOAT *O_d, FLOAT *Mw_d,
	int plen, int qlen, int Xw, int Yw,
	int p, int q,
	int x, int y,
	int D, int N, int K);
