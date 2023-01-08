#include <chrono>
#include <iostream>

#include <cuda_runtime.h>

#include "matmul_tvm.h"
// #include "matmul_tvm_manual.h"

int main() {

    int M = 1024;
    int K = 1024;
    int N = 1024;

    int REPEAT_TIMES = 1000;

    cudaError_t err;

    // generate data on host
    float *h_A = (float *)malloc(M * K * sizeof(float));
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
    }

    float *h_B = (float *)malloc(K * N * sizeof(float));
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *h_C = (float *)malloc(M * N * sizeof(float));
    float *answer_C = (float *)malloc(M * N * sizeof(float));

    // allocate memory on device
    float *d_A = nullptr;
  err = cudaMalloc((void**)&d_A, M * K * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  
  float *d_B = nullptr;
  err = cudaMalloc((void**)&d_B, K * N * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  float *d_C = nullptr;
  err = cudaMalloc((void**)&d_C, M * N * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  std::chrono::nanoseconds total_time(0);
  for (int i = 0; i < REPEAT_TIMES; ++i) {
    // printf("Running %d / %d times\n", i, REPEAT_TIMES);
    err = cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr,
              "Failed to copy vector A from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr,
              "Failed to copy vector B from host to device (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    auto start = std::chrono::high_resolution_clock::now();
    // auto scheduler
    main_kernel0<<<32, 256>>>(d_A, d_B, d_C);
    // manul tunning 
    // main_kernel0<<<{16, 16, 1}, 64>>>(d_A, d_B, d_C);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    total_time += duration;

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to launch matmul kernel (error code %s)!\n",
            cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      fprintf(stderr,
              "Failed to copy vector C from device to host (error code %s)!\n",
              cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }
  }

  std::cout << "Run kernel " << REPEAT_TIMES << " times taken " << total_time.count() << " ns" << std::endl;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      answer_C[i * N + j] = 0.0f;
      for (int k = 0; k < K; ++k) {
        answer_C[i * N + j] += (h_A[i * K + k] * h_B[k * N + j]);
      }
    }
  }

  bool correct = true;
  for (int i = 0; i < M * N; ++i) {
    if (fabs(answer_C[i] - h_C[i]) > 1e-4) {
      printf("Result verification failure at element %d, answer_C = %f, h_C = %f\n", i, answer_C[i], h_C[i]);
      correct = false;
      break;
    }
  }

  if (correct) {
    printf("Precision is accurate\n");
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);
  free(answer_C);
}
