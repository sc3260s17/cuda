/*  Vector addition on the GPU: C = A + B  */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 1024
#define BLOCKSIZE 16

// Device function (i.e. kernel)
__global__ void VecAdd(float * A, float * B, float * C, int N)
{

   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if ( i < N ) {
      C[i] = A[i] + B[i];
   }

}

// CPU version of the vector addition function
void vecAddCPU(float * A, float * B, float * C, int N)
{

   int i;
   for (i=0; i<N; i++)
   {
      C[i] = A[i] + B[i];
   }

}

// Function compares two 1d arrays
void compareVecs( float * vec1, float * vec2, int N )
{

   int i;
   int vecsEqual = 1;
   for (i=0; i<N; i++)
   {
      if ( abs (vec1[i] - vec2[i]) > 0.00001 )
      {
         printf("vectors not equal! i: %d  vec1[i]: %f  vec2[i]: %f\n",i,vec1[i],vec2[i]);
         vecsEqual = 0;
      }
   }
   if ( vecsEqual ) printf("GPU vector addition agrees with CPU version!\n");

}

/* Host function for filling vector (1d array) with 
   random numbers between -20.0 and 20.0 */
void fillOutVector( float * vec, int vec_length )
{

   time_t t;
   srand((unsigned) time(&t)); // initialize random number generator
   int i;
   for (i=0; i<vec_length; i++)
   {
      vec[i] = ( (float)rand() / (float)(RAND_MAX) ) * 40.0;
      vec[i] -= 20.0;
   }

}

// Host function for printing a vector (1d array)
void printVector( float * vec, int vec_length )
{

   int i;
   for (i=0; i<vec_length; i++) {
      printf("i: %d vec[i]: %f\n",i,vec[i]);
   }

}

void printDeviceSpecs()
{

   int devID;
   cudaDeviceProp props;
   cudaGetDevice(&devID);
   cudaGetDeviceProperties(&props, devID);
   printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major,props.minor);

}

void vecMultiplyGPU(float * product_device_array)
{
   // initialize timer events
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   // allocate space for host arrays
   size_t vec_bytes = SIZE * sizeof(float);
   float * h_A = (float *)malloc( vec_bytes );
   float * h_B = (float *)malloc( vec_bytes );
   float * h_C = (float *)malloc( vec_bytes );
   // initialize arrays
   fillOutVector( h_A, SIZE );
   fillOutVector( h_B, SIZE );
   printDeviceSpecs();

   // allocate space for A and B on the device and copy data into space
   float * d_A, * d_B;
   cudaMalloc(&d_A, vec_bytes);
   cudaMalloc(&d_B, vec_bytes);
   cudaMemcpy(d_A, h_A, vec_bytes, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, vec_bytes, cudaMemcpyHostToDevice);

   // launch kernel and get timing info
   dim3 threadsPerBlock(BLOCKSIZE);
   dim3 blocksPerGrid( (SIZE + BLOCKSIZE - 1) / BLOCKSIZE );
   cudaEventRecord(start);
   VecAdd<<< blocksPerGrid, threadsPerBlock >>>(d_A, d_B, product_device_array, SIZE);
   cudaEventRecord(stop);
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("kernel time (ms) : %7.5f\n",milliseconds);
   cudaMemcpy(h_C, product_device_array, vec_bytes, cudaMemcpyDeviceToHost);

   // verify you got the correct result
   float * gold_C = (float *)malloc( vec_bytes );
   vecAddCPU( h_A, h_B, gold_C, SIZE );
   compareVecs( gold_C, h_C, SIZE );

   // clean up
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   cudaFree(d_A);
   cudaFree(d_B);
   free(h_A);
   free(h_B);
   free(h_C);
   free(gold_C);
}

void allocateDeviceSpace(float ** array)
{

   size_t vec_bytes = SIZE * sizeof(float);
   cudaError_t rc; // return code from cuda functions
   rc = cudaMalloc(array, vec_bytes);
   if ( rc ) printf("Error from cudaMalloc: %s\n",cudaGetErrorString(rc));

}

// program execution begins here
int main( int argc, char ** argv )
{

   int nDevices;
   cudaGetDeviceCount(&nDevices);
   printf("Found %d GPUs!\n",nDevices);

   float * dev_C[nDevices];
   int i;
   for ( i=0; i<nDevices; i++ )
   {
      cudaSetDevice(i);
      allocateDeviceSpace(&dev_C[i]);
      vecMultiplyGPU(dev_C[i]);
   }

   return 0;

}