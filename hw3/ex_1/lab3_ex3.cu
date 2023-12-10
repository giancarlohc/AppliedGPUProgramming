
#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <string>
#include <iostream>
#include <cuda.h>
#include <cstdlib>
#include <ctime>

#define NUM_BINS 4096
#define TPB 1024


// //using shared memory
// __global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
//                                  const unsigned int num_elements,
//                                  const unsigned int num_bins) {

// //@@ Insert code below to compute histogram of input using shared memory and atomics
//    __shared__ unsigned int temp_bins[NUM_BINS];
//   const int idx = threadIdx.x + blockDim.x * blockIdx.x;
//   temp_bins[threadIdx.x] = 0;
//   __syncthreads();
  
//     if (idx < num_elements) {
//         atomicAdd(temp_bins + input[idx], 1);
//     }
//     __syncthreads();

//    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
//         atomicAdd(&bins[i], temp_bins[i]);
//     }
  
//     __syncthreads();
// }


// using global memory
__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 const unsigned int num_elements,
                                 const unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if(idx<num_elements){
    atomicAdd(bins+input[idx], 1);
  __syncthreads();
}
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
  const int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if(idx < num_bins){
    if(bins[idx] > 127){bins[idx] = 127;}
  }
  __syncthreads();
}


int main(int argc, char **argv) {

  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args
  if(argc>1){
    inputLength = std::stoi(argv[1]);
  }
  else{
    //std::cout << "Please provide input length";
    return -1;
  }
  printf("The input length is %d\n", inputLength);

  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int*) calloc(inputLength , sizeof(int));
  hostBins = (unsigned int*)  calloc(NUM_BINS ,sizeof(int));
  resultRef = (unsigned int*) calloc(NUM_BINS , sizeof(int));

  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  std::srand(std::time(nullptr));
  for(int i = 0; i<(inputLength); i++)
  {
    hostInput[i] = ((int)(std::rand() % NUM_BINS));

  }

  //@@ Insert code below to create reference result in CPU
  for(int i = 0; i<(inputLength); i++)
  {
    if(hostBins[hostInput[i]]< 127) {
      hostBins[hostInput[i]]++;
    }

  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, sizeof(unsigned int)*(inputLength));
  cudaMalloc(&deviceBins, sizeof(unsigned int)*(NUM_BINS));
  cudaMemset(deviceBins, 0, sizeof(unsigned int)*(NUM_BINS));


  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput,  sizeof(unsigned int)*(inputLength), cudaMemcpyHostToDevice);


  //@@ Insert code to initialize GPU results


  //@@ Initialize the grid and block dimensions here

  //@@ Launch the GPU Kernel here
  //launching with shared memory
  // histogram_kernel<<<(inputLength+TPB-1)/TPB, TPB>>>(deviceInput, deviceBins, inputLength, NUM_BINS);
  
  //launching without shared memory
  histogram_kernel<<<NUM_BINS, TPB>>>(deviceInput, deviceBins, inputLength, NUM_BINS);


  //@@ Initialize the second grid and block dimensions here


  //@@ Launch the second GPU Kernel here
  convert_kernel<<<(NUM_BINS + TPB - 1)/TPB, TPB>>>(deviceBins, NUM_BINS);


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(resultRef, deviceBins , sizeof(int)*(NUM_BINS), cudaMemcpyDeviceToHost);


  //@@ Insert code below to compare the output with the reference
  bool result = true;
  
  for(int i = 0; i < NUM_BINS; i++){
    //std::cout << resultRef[i] << " : " <<hostBins[i] << " at " << i << "\n";

    if(resultRef[i] != hostBins[i]){
      result = false;
    }
  }

  if(result == false){
    std::cout << "Wrong computation!\n";

  }
  else{
    std::cout << "Correct computation!";

  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  std::free(hostInput);
  std::free(hostBins);
  std::free(resultRef);

  return result;
}

