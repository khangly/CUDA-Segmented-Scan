#include <cstdio>
#include <cuda.h>

#include <math.h>
#include <chrono>
#include <cooperative_groups.h>

#include "util.hpp"
#include "serial.hpp"

#define CUDA_CHECK(ans)                                                                  \
    {                                                                                    \
        gpuAssert((ans), __FILE__, __LINE__);                                            \
    }
   inline void
gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort)
            exit(code);
    }
}


__global__ void naive_scan(float* x, float* y, size_t n) {
  size_t tid = threadIdx.x + blockDim.x*blockIdx.x;

  if (tid < n) {
    y[tid] =  x[tid];
  }

  __syncthreads();

  size_t bound = (size_t) (log2f(n) + 0.5);

  for (size_t d = 1; d <= bound; d++) {
    size_t d2 = powf(2, d-1);  
    if (tid >= d2) {
      size_t offset = (size_t) (powf(2, d-1)); 
      y[tid] = y[tid] + y[tid - offset];
    }
    __syncthreads();

  }
}


__global__ void prescan(float *g_idata, float *g_odata, int n) {
   int np2 = powf(2, ceil(log2f(n))); // next power-of-2 >= n

   extern __shared__ float temp[];
  
  int thid = blockIdx.x*blockDim.x + threadIdx.x;
  int offset = 1;

  __syncthreads();

  // load input into shared memory
  temp[2*thid] = (2*thid < n) ? g_idata[2*thid] : 0;
  temp[2*thid+1] = (2*thid+1 < n) ? g_idata[2*thid+1] : 0; 

  // build sum in place up the tree
  for (int d = np2>>1; d > 0; d >>= 1) {
    __syncthreads();
    if (thid < d)    { 
      int ai = offset*(2*thid+1)-1;  // k + 2^d - 1
      int bi = offset*(2*thid+2)-1;  // k + 2^d+1 - 1
      if (bi < np2 && ai < np2) {
	temp[bi] += temp[ai];
      }
    }
    offset *= 2;
  }
  __syncthreads();

  // clear the last element
  if (thid == 0) {
    temp[np2 - 1] = 0;  
  } 

  // traverse down tree, swapping elements to build scan
  for (int d = 1; d < np2; d *= 2)    {
    offset >>= 1;
    __syncthreads();

    if (thid < d) { 
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;
      if (ai < np2 && bi < np2) {
        float t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
  }
  __syncthreads();

  // write results to device memory
  if (2*thid<n) {
    g_odata[2*thid] = temp[2*thid] + g_idata[2*thid];
  }
  if (2*thid+1<n) {
    g_odata[2*thid+1] = temp[2*thid+1] + g_idata[2*thid+1];
  }
 

}

// Requires: All arrays are of size n, which must be a power of 2
// Effect: Sets g_odata to be the segmented scan of g_idata with the flags flag_orig
//         Does not modify g_idata, or flag_orig
__global__ void segscan(float *g_idata, float *g_odata,
			float *flag_orig, int n){

  int thid = blockIdx.x*blockDim.x + threadIdx.x;
  int offset = 1;

  extern __shared__ float shmem[];
  float * temp = &shmem[0];
  float * flags = &shmem[n];
  
  __syncthreads();

  // load input into shared memory
  temp[2*thid] = (2*thid < n) ? g_idata[2*thid] : 0;
  temp[2*thid+1] = (2*thid+1 < n) ? g_idata[2*thid+1] : 0; 
  flags[2*thid] = (2*thid < n) ? flag_orig[2*thid] : 0;
  flags[2*thid+1] = (2*thid+1 < n) ? flag_orig[2*thid+1] : 0; 

  // build sum in place up the tree
  for (int d = n>>1; d > 0; d >>= 1) {
    __syncthreads();
    if (thid < d)    { 
      int ai = offset*(2*thid+1)-1;  // k + 2^d - 1
      int bi = offset*(2*thid+2)-1;  // k + 2^(d+1) - 1
      if (bi < n && ai < n) {
	if (flags[bi] == 0.0) {
	  temp[bi] += temp[ai];
	}
	flags[bi] = (flags[bi]==1.0 || flags[ai] == 1.0) ? 1.0: 0.0;
      }
    }
    offset *= 2;
  }
  
  // clear the last element
  if (thid == 0) {
    temp[n - 1] = 0;  
  } 

  // traverse down tree & build scan
  for (int d = 1; d < n; d *= 2)    {
    offset >>= 1;
    __syncthreads();
    if (thid < d) { 
      int ai = offset*(2*thid+1)-1;  // k + 2^d  - 1
      int bi = offset*(2*thid+2)-1;  // k + 2^d+1 - 1
      float t = temp[ai];
      temp[ai] = temp[bi];
      if (flag_orig[ai+1] == 1.0) {
	temp[bi] = 0;
      } else if (flags[ai] == 1.0) {
	temp[bi] = t;
      } else {
	temp[bi] += t;
      }
      flags[ai] = 0;
    }
  }

  __syncthreads();
  
  // write results to device memory
  if (2*thid < n) {
    g_odata[2*thid] = temp[2*thid] + g_idata[2*thid];
  }
  if (2*thid+1 < n) {
    g_odata[2*thid+1] = temp[2*thid+1] + g_idata[2*thid+1]; 
  }

  __syncthreads();

}


float run_scan(std::vector<float> inv, std::vector<float> outv) {
  return 0.0;
}

int main(int argc, char** argv) {

  if (argc < 2) {
    fprintf(stderr, "scan [size of array]\n");
    return 1;
  }
  
  size_t n = std::atoi(argv[1]);
  int np2 = powf(2, ceil(log2f(n))); // next power-of-2 >= n
  
  assert(n <= 2048);

  // 1 thread for every 2 elements (round up)
  size_t blocksize = std::min<size_t>(np2/2, 1024);
  size_t nblocks = (np2/2) / blocksize;

  std::vector<float> x = gen_vec(n);  // input
 
  auto begin = std::chrono::high_resolution_clock::now();
  std::vector<float> rv_cpu = cpu_scan(x);
  auto end = std::chrono::high_resolution_clock::now();
  double duration_cpu = std::chrono::duration<double>(end - begin).count();

  float* d_x;
  float* d_y;

  CUDA_CHECK(cudaMalloc(&d_x, sizeof(float)*np2));
  CUDA_CHECK(cudaMalloc(&d_y, sizeof(float)*np2));


  CUDA_CHECK(cudaMemset(d_y, 0, sizeof(float)*np2));
  
  std::vector<float> rv(n);  // gpu result
  /*
  // prescan
  CUDA_CHECK(cudaMemset(d_x, 0, sizeof(float)*np2));
  CUDA_CHECK(cudaMemcpy(d_x, x.data(), sizeof(float)*n,
			cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_y, 0, sizeof(float)*np2));
  CUDA_CHECK(cudaDeviceSynchronize());


  printf("calling prescan<<<%d,%d,%d>>> of %d elements.\n",
	 nblocks, blocksize, np2, n);

  begin = std::chrono::high_resolution_clock::now();
  prescan<<<nblocks, blocksize, np2*sizeof(float)>>>(d_x, d_y, n);

  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  double duration_gpu = std::chrono::duration<double>(end - begin).count();

  cudaMemcpy(rv.data(), d_y, sizeof(float)*n,
			cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  checkResult(x, rv, rv_cpu, duration_gpu, duration_cpu);
  */
  
  // segscan
  std::vector<float> flags(n,0.0);
  
  for (int i = 1; i < n; i*=2) {
      flags[i-1] = 1.0;
  }

  rv_cpu = cpu_segscan(x, flags);
  
  float* d_f;
  CUDA_CHECK(cudaMalloc(&d_f, sizeof(float)*np2));
  CUDA_CHECK(cudaMemset(d_f, 0, sizeof(float)*np2));
  CUDA_CHECK(cudaMemcpy(d_f, flags.data(), sizeof(float)*n,
			cudaMemcpyHostToDevice));

  cudaMemset(d_x, 0.0, sizeof(float)*np2);
  cudaMemcpy(d_x, x.data(), sizeof(float)*n, cudaMemcpyHostToDevice);
  cudaMemset(d_y, 0.0, sizeof(float)*np2);
  
  cudaDeviceSynchronize();

  printf("calling segscan<<<%d,%d>>> of %d elements.\n",
	 nblocks, blocksize, n);
  begin = std::chrono::high_resolution_clock::now();
  segscan<<<nblocks, blocksize, 2*n*sizeof(float)>>>(d_x, d_y, d_f, n);

  cudaDeviceSynchronize();
  end = std::chrono::high_resolution_clock::now();
  float duration_gpu = std::chrono::duration<double>(end - begin).count();

  cudaMemcpy(rv.data(), d_y, sizeof(float)*n, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("Flags are: ");
  print(flags);
  printf("\n");
  checkResult(x, rv, rv_cpu, duration_gpu, duration_cpu);
  
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_f);

  return 0;
}
