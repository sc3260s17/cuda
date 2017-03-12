// A naive matrix multiplication program without tiling
// Uses a single block

#include "stdio.h"
#include "stdlib.h"

// this is a naive code that only uses a single block
// since the GTX 480 is limited to 32*32 = 1024 threads
// per block, the max value of SIZE is 32.  Anything
// that exceeds 32 will not yield correct values
#define SIZE 32

// kernel definition
__global__ void MatrixMulKernel(float * A,float * B,float * C,int len)
{

    int col = threadIdx.x;
    int row = threadIdx.y;

    float sum=0.0, Aelement, Belement;
    int i;
    for (i = 0; i < len ; i++) {
        Aelement = A[ row*len + i ];
        Belement = B[ i*len + col ];
        sum += Aelement * Belement;
    }

    C[ row*len + col ] = sum;

}

int main(int argc, char ** argv) 
{

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   float h_A[SIZE*SIZE],h_B[SIZE*SIZE],h_C[SIZE*SIZE];
   float * d_A, * d_B, * d_C;

   // initialize host matrices with arbitrary data
   int i;
   for (i=0;i<SIZE*SIZE;i++) {
      h_A[i] = (float)i;
      h_B[i] = (float)SIZE * (float)SIZE - (float)i - 1.00;
      h_C[i] = 0.0;
   }

   // allocate space on device
   size_t size = SIZE*SIZE*sizeof(float);
   cudaMalloc(&d_A,size);
   cudaMalloc(&d_B,size);
   cudaMalloc(&d_C,size);

   //copy data to device
   cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
   cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
   cudaMemcpy(d_C,h_C,size,cudaMemcpyHostToDevice);

   dim3 blocksPerGrid(1); // 1 x 1 x 1
   dim3 threadsPerBlock(SIZE,SIZE); // SIZE x SIZE x 1

   cudaEventRecord(start);
   // invoke the kernel here
   MatrixMulKernel<<< blocksPerGrid, threadsPerBlock >>>(d_A,d_B,d_C,SIZE);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("kernel time (ms) : %7.5f\n",milliseconds);

   // copy results back to host
   cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);

   // output results
   /*for (i=0;i<SIZE*SIZE;i++) {
      printf("i: %d h_C[i]: %f\n",i,h_C[i]);
   }*/

   // Free up device memory
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   return 0;

}