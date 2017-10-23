#include <stdio.h>
#include <stdlib.h>
#include "debug.h"
#include "kernels.h"

cudaError_t xprop_winograd(FLOAT *I, int lenI, FLOAT *F, int lenF, FLOAT *O, int lenO, int padding[2]);