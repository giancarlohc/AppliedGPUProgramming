#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <iostream>
#include <cuda.h>
#include <cstdlib>
#include <ctime>
#define DataType double
#define n_streams 4


cudaStream_t streams[n_streams];

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= len) return;
  out[idx] = in1[idx] + in2[idx];
  __syncthreads();
  return;
}

//@@ Insert code to implement timer start
std::time_t timer_start(){
  return std::time(nullptr);
}

//@@ Insert code to implement timer stop
std::time_t stop_timer(std::time_t start_time){
  return std::difftime(start_time, std::time(nullptr));
}

int main(int argc, char **argv) {
  
  int inputLength;
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  std::time_t start_time;


  //for random floating point numbers
  int highest = 1e6;
  int upper = 10e3;

  //@@ Insert code below to read in inputLength from args
  if(argc>1){
    inputLength = std::stoi(argv[1]);
  }
  else{
    std::cout << "Please provide input length";
    return -1;
  }

  

  printf("The input length is %d\n", inputLength);
  int S_seg = 131072;

  //@@ Insert code below to allocate Host memory for input and output
  //saving error messages and the memory pointer
  hostInput1 = (double*)  std::malloc(sizeof(DataType) * inputLength);
  hostInput2 = (double*)  std::malloc(sizeof(DataType) * inputLength);
  hostOutput = (double*)  std::malloc(sizeof(DataType) * inputLength);
  resultRef  = (double*)  std::malloc(sizeof(DataType) * inputLength);


  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  std::srand(std::time(nullptr));
  for(int i = 0; i<inputLength; i++)
  {
    hostInput1[i] = upper * (double)(std::rand() % highest) / highest;
    hostInput2[i] = upper *(double)(std::rand() % highest) / highest;
    hostOutput[i] = hostInput1[i] + hostInput2[i];
  }
  
  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, sizeof(DataType)*inputLength);
  cudaMalloc(&deviceInput2, sizeof(DataType)*inputLength);
  cudaMalloc(&deviceOutput, sizeof(DataType)*inputLength);


  

  //@@ Initialize the 1D grid and block dimensions here
  int TPB = 64;
  int cursor = 0;
  int segment = S_seg;
  for(int i = 0; i<((inputLength/S_seg)+1); i++){
    cursor = std::min(i * S_seg, inputLength);
    segment = std::min(S_seg, inputLength - cursor);
    //@@ Insert code to below to Copy memory to the GPU here
    cudaMemcpyAsync(deviceInput1+cursor, hostInput1+cursor,  sizeof(DataType)*segment, cudaMemcpyHostToDevice, streams[i%n_streams]);
    cudaMemcpyAsync(deviceInput2+cursor, hostInput2+cursor,  sizeof(DataType)*segment, cudaMemcpyHostToDevice, streams[i%n_streams]);
    //@@ Launch the GPU Kernel here
    vecAdd<<<(S_seg + TPB - 1)/TPB, TPB, 0, streams[i%n_streams]>>>(deviceInput1+cursor, deviceInput2+cursor, deviceOutput+cursor, segment);
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpyAsync(resultRef+cursor, deviceOutput+cursor , sizeof(DataType)*segment, cudaMemcpyDeviceToHost, streams[i%n_streams]);
  }
  



  //@@ Insert code below to compare the output with the reference
  for(int i = 0; i < inputLength; i++){
        
    if(resultRef[i] != hostOutput[i]){

        std::cout << "Wrong computation!";
        return -1;
    }

  }
  std::cout << "Correct computation!";


  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);
  //@@ Free the CPU memory here
  std::free(hostInput1);
  std::free(hostInput2);
  std::free(hostOutput);

  return 0;
}
