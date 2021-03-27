/**
 * Nbody Cuda
 * @author Juntao Chen
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BlockWidth 16
#define BlockSize 256
#define SOFTENING 1e-9f

typedef struct{ float x, y, z, vx, vy, vz; } Body;

// 初始化值为
void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__ void bodyForce(Body *p, float dt, int n) {

  // 线程要处理的行号
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  // 判断边界，行号要小于n
  if (row < n) {
    
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
    
    // 循环网格维度x次
    for (int j = 0; j < gridDim.x; ++j) {
      
      // 定义共享变量存取block需要的body值
      __shared__ Body locN[BlockWidth];
      
      // 从全局内存中取出值
      Body loc1 = p[j * blockDim.x + threadIdx.x];
      
      // 放入共享内存中
      locN[threadIdx.x].x = loc1.x;
      locN[threadIdx.x].y = loc1.y;
      locN[threadIdx.x].z = loc1.z;
      
      // 确保所有的值都被取出才能进行下一步计算，避免因时差导致取值错误
      __syncthreads();
      
      for (int k = 0; k < BlockWidth; ++k) {
        float dx = locN[k].x - p[row].x;
        float dy = locN[k].y - p[row].y;
        float dz = locN[k].z - p[row].z;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = sqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      }
      __syncthreads();
    }
    p[row].vx += dt*Fx; 
    p[row].vy += dt*Fy; 
    p[row].vz += dt*Fz;
  } 
}

int main(const int argc, const char** argv) {

  int nBodies = 2<<11;
  int salt = 0;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  if (argc > 2) salt = atoi(argv[2]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations
  
  // 声明变量，在主机中存放值
  int bytes = nBodies * sizeof(Body);
  float *buf = (float *)malloc(bytes);
  Body *p = (Body*)buf;
  // 声明变量，在device存放值
  float *devBuf;
  cudaMalloc(&devBuf, bytes);
  Body *devp = (Body*)devBuf;

  // 初始化变量
  randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

  // 计算得到块的个数
  int nBlocks = (nBodies + BlockSize -1) / BlockSize;

  for (int iter = 0; iter < nIters; iter++) {
    // 将值存入设备中    
    cudaMemcpy(devp, buf, bytes, cudaMemcpyHostToDevice);
    // 并行计算
    bodyForce<<<nBlocks, BLOCK_SIZE>>>(devp, dt, nBodies);
    // 将值从设备中取出来
    cudaMencpy(buf, devp, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }  
  }
  // 释放变量空间
  free(buf);
  cudafree(devBuf);
}