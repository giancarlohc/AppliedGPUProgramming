
#include <stdio.h>
#include <sys/time.h>

#define DataType double

double cpuSecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  int c = blockIdx.x * blockDim.x + threadIdx.x; // column
  int r = blockIdx.y * blockDim.y + threadIdx.y; // row
  if((c<numBColumns)&&(r<numARows)){
    DataType counter = 0.0;
    for (int i = 0; i < numAColumns; ++i) {
            counter += A[r * numAColumns + i] * B[i * numBColumns + c];
    }
    C[r * numBColumns+c] = counter;
  }

}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args

  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = atoi(argv[3]);
  numBColumns = atoi(argv[4]);
  numCRows = atoi(argv[1]);
  numCColumns = atoi(argv[4]);


  printf("Input matrix dim is (%d x %d) (%d x %d) with an output of (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  if(numAColumns != numBRows){
    printf("matrix A columns does not match matrix B, ERROR \n");
    return 0;
  }

  //@@ Insert code below to allocate Host memory for input and output

  hostA = (DataType *)malloc(numARows*numAColumns*sizeof(DataType));
  hostB = (DataType *)malloc(numBRows*numBColumns*sizeof(DataType));
  hostC = (DataType *)malloc(numARows*numBColumns*sizeof(DataType));
  resultRef = (DataType *)malloc(numARows*numBColumns*sizeof(DataType));

  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  for (int i = 0; i < numARows; i++) {
    for (int j = 0; j < numAColumns; j++) {
            hostA[i * numAColumns + j] = rand() / (DataType)RAND_MAX;
    }
  }
  for (int i = 0; i < numBRows; i++) {
      for (int j = 0; j < numBColumns; j++) {
          hostB[i * numBColumns + j] = rand() / (DataType)RAND_MAX;
      }
  }
  // Matrix Multiplication
  for (int i = 0; i < numCRows; i++) {
      for (int j = 0; j < numCColumns; j++) {
          resultRef[i * numCColumns + j] = 0;
          for (int k = 0; k < numAColumns; k++) {
              resultRef[i * numCColumns + j] += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
          }
      }
  }
  double iStart = cpuSecond();

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void**)&deviceA, numARows*numAColumns*sizeof(DataType));
  cudaMalloc((void**)&deviceB, numBRows*numBColumns*sizeof(DataType));
  cudaMalloc((void**)&deviceC, numCRows*numCColumns*sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  double iCopy = cpuSecond();
  cudaMemcpy(deviceA, hostA, numARows*numAColumns*sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows*numBColumns*sizeof(DataType), cudaMemcpyHostToDevice);
  double copyTo = cpuSecond() - iCopy;

  //@@ Initialize the grid and block dimensions here
  dim3 TPB(16,16); //@@256 total
  dim3 BLK((numCColumns+TPB.x-1)/TPB.x,(numCRows+TPB.y-1)/TPB.y);

  //@@ Launch the GPU Kernel here
  double iCompute = cpuSecond();
  gemm<<<BLK,TPB>>>(deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns);
  cudaDeviceSynchronize();
  double compTime = cpuSecond() - iCompute;

  //@@ Copy the GPU memory back to the CPU here
  double iCopyFrom = cpuSecond();
  cudaMemcpy(hostC, deviceC,numCRows*numCColumns*sizeof(DataType),cudaMemcpyDeviceToHost);
  double copyFrom = cpuSecond() - iCopyFrom;

  double iElaps = cpuSecond() - iStart;
  printf("Elapsed Time: %f seconds\n Copy To Time: %f\n Copy From Time: %f\n Compute Time: %f\n", iElaps, copyTo,copyFrom,compTime);
  //@@ Insert code below to compare the output with the reference
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
        if (fabs(hostC[i * numCColumns + j] - resultRef[i * numCColumns + j])>.01) {
            printf("Mismatch at index (%d, %d): Actual %.10f, Computed %.10f\n", i, j, resultRef[i * numCColumns + j], hostC[i * numCColumns + j]);
        } 
        else {
           // printf("SUCCESS: %.5f\n", hostC[i * numCColumns + j]);
        }
    }
  }

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);
  free(iCompute);
  free(iCopy);
  free(iCopyFrom);
  free(iElaps);
  free(compTime);
  free(iStart);
  free(copyFrom);
  free(copyTo);

  return 0;
}
