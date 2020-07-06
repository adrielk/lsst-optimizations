//Author: Adriel Kim
//6-27-2020
//Updated 7-5-2020
/*
Desc: Basic 2D matrix operations - element-wise addition, subtraction, multiplication, and division.

To do:
- Use vector instead of array?
- Be able to test for varying sizes of images. (For now we manually define with constant N)
- Add timer to compare CPU and GPU implementations
- Double check if all memory is freed
- Optimize by eliminating redundant calculations
- Test code on department servers
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <stdlib.h>
#include <stdio.h>

//define imin(a,b)  (a<b?a:b)//example of ternary operator in c++
//4176,2048
#define R 4176
#define C 2048
#define N (R*C)//# of elements in matrices
const int threadsPerBlock = 256;//threads in a block. A chunk that shares the same shared memory.
const int blocksPerGrid = 256;//imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);//this will be our output array size for sumKernel.

using namespace std;

cudaError_t matrixOperation(double* c, const double* a, const double* b, unsigned int arrSize, int operation);
//any advantages with mapping directly to strucutre of matrix? We're just representing 2D matrix using 1D array...
//it would be difficult to do the above since we want the operations to occur over abitrarily large matrices
//this can definitely be optimzied by elminating redundant calculations
__global__ void matrixAddKernel(double *c, const double *a, const double *b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] + b[tid];
        //adds total number of running threads to tid, the current index.
        tid += blockDim.x * gridDim.x;
    }
}
__global__ void matrixSubtractKernel(double* c, const double* a, const double* b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] - b[tid];
        //adds total number of running threads to tid, the current index.
        tid += blockDim.x * gridDim.x;
    }
}
__global__ void matrixMultiplyKernel(double* c, const double* a, const double* b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void matrixDivideKernel(double* c, const double* a, const double* b) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        c[tid] = (a[tid]/b[tid]);
        tid += blockDim.x * gridDim.x;
    }
}

void printMatrix(double* arr, int dims) {
    for (int i = 0;i < dims; i++) {
        for (int k = 0;k < dims; k++) {
            cout << (arr[k + i * dims])<<" ";
        }
        cout << endl;
    }
}

void fillWithFitImage(string fileName, double* arr){
    string line;
    ifstream fitsFile;
    fitsFile.open("./FitTextFiles/"+fileName);
    int ind = 0;
    if(fitsFile.is_open()){
        getline(fitsFile, line);//gets rid of initial part, which is just shape
    }

    if(fitsFile.is_open()){
        while(getline(fitsFile, line) && ind < N){
            char cstr[line.size()+1];
            strcpy(cstr, line.c_str());
            double num = atof(cstr);
            //add num to arr
            arr[ind] = num;
            ind++;
        }
        fitsFile.close()
    }
    else{
        cout<<"Error opening file"<<endl;
    }

}

int main()
{
    const int rows = R;
    const int cols = C;
    int inc = 0;

    //Must allocate host memory first before calling kernel.
    double* outputs = (double*)malloc(N * sizeof(double));
    double* doubleMatrix = (double*)malloc(N * sizeof(double));
    double* doubleMatrix2 = (double*)malloc(N * sizeof(double));
    int operation = 0;
    
    cout << "Enter which operation (1 = add, 2 = subtract, 3 = multiply, 4 = divide)" << endl;
    cin >> operation;
    //populated 2D array with data
    fillWithFitImage("imgraw1.txt", doubleMatrix);
    fillWithFitImage("img1.txt", doubleMatrix2);
    /*
    for (int i = 0;i<rows;i++) {
        for (int k = 0;k < cols;k++) {
            doubleMatrix[k+i*rows] = inc;
            doubleMatrix2[k+i*rows] = inc;
            outputs[k + i * rows] = 0;
           // cout << (doubleMatrix[i][k]);
            inc++;
        }
        //cout << endl;
    }*/
    cout << "Matrix 1:" << endl;
//    printMatrix(doubleMatrix, rows);
    cout << "Matrix 2:" << endl;
  //  printMatrix(doubleMatrix2, rows);
    cudaError_t cudaStatus = matrixOperation(outputs, doubleMatrix, doubleMatrix2, N, operation);
    cout << "Resulting Matrix:" << endl;
    printMatrix(outputs, rows);

    return 0;
}
cudaError_t matrixOperation(double* c, const double* a, const double* b, unsigned int arrSize, int operation) {
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    //Allocate GPU buffers for three vectors (two input, one output)
    cudaStatus = cudaMalloc((void**)&dev_c, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! 1");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! 2");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, N * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! 3");
        goto Error;
    }

    //Copy input vectors from host memory to GPU buffers
    cudaStatus = cudaMemcpy(dev_a, a, sizeof(double) * N, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! 1");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, sizeof(double) * N, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! 2");
        goto Error;
    }
    switch (operation) {
        case 1:
            matrixAddKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b);
            break;
        case 2:
            matrixSubtractKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b);
            break;
        case 3:
            matrixMultiplyKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b);
            break;
        case 4:
            matrixDivideKernel << <blocksPerGrid, threadsPerBlock >> > (dev_c, dev_a, dev_b);
            break;

    }
    //copies result to host so we can use it.
    cudaStatus = cudaMemcpy(c, dev_c, sizeof(double) * N, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! 3");
        goto Error;
    }

    Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;


}

