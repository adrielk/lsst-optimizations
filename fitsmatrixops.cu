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
#include <sys/time.h>
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
const int threadsPerBlock = 1024;//threads in a block. A chunk that shares the same shared memory.
const int blocksPerGrid = 8352;//imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);//this will be our output array size for sumKernel.

using namespace std;

cudaError_t matrixOperation(double* c, const double* a, const double* b, unsigned int arrSize, int operation, float* kernel_runtime, float* GPU_transfer_time);
void CPUMatrixOperation(double* c, const double* a, const double* b, unsigned int arrSize, int operation);
long long start_timer();
long long stop_timer(long long start_time, const char *name);

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
//---------------------------------------------------------------------------------
void CPUMatrixAdd(double* c, const double* a, const double* b, unsigned int arrSize){
    for(int i = 0; i < arrSize; i++){
        c[i] = a[i] + b[i];
    }
}
void CPUMatrixSubtract(double* c, const double* a, const double* b, unsigned int arrSize){
    for(int i = 0; i < arrSize; i++){
        c[i] = a[i] - b[i];
    }
}

void CPUMatrixMultiply(double* c, const double* a, const double* b, unsigned int arrSize){
    for(int i = 0; i < arrSize; i++){
        c[i] = a[i] * b[i];
    }
}

void CPUMatrixDivide(double* c, const double* a, const double* b, unsigned int arrSize){
    for(int i = 0; i < arrSize; i++){
        c[i] = a[i] / b[i];
    }
}
//----------------------------------------------------------------------------------
void printMatrix(double* arr) {
    for (int i = 0;i < R; i++) {
        for (int k = 0;k < C; k++) {
            cout << (arr[k + i * R])<<" ";
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
        fitsFile.close();
    }
    else{
        cout<<"Error opening file"<<endl;
    }

}

bool checkEquality(double* arr1, double* arr2, int arrSize){
    for(int i = 0;i < arrSize;i++){
        if (arr1[i]!=arr2[i]){
            return false;
        }
    }
    return true;

}

int main()
{
    //const int rows = R;
    //const int cols = C;
    //int inc = 0;

    //Must allocate host memory first before calling kernel.
    double* outputs = (double*)malloc(N * sizeof(double));
    double* doubleMatrix = (double*)malloc(N * sizeof(double));
    double* doubleMatrix2 = (double*)malloc(N * sizeof(double));
    double* CPUoutputs = (double*)malloc(N * sizeof(double));

    int operation = 0;
    
    cout << "Enter which operation (1 = add, 2 = subtract, 3 = multiply, 4 = divide)" << endl;
    cin >> operation;
    //populated 2D array with data
    cout<<"Populating image data"<<endl;
    fillWithFitImage("imgraw1.txt", doubleMatrix);
    fillWithFitImage("img1.txt", doubleMatrix2);

    cout<<"GPU Start!"<<endl;

    float* GPU_kernel_time = (float*)malloc(sizeof(float));
    float* GPU_transfer_time = (float*)malloc(sizeof(float));
    long long gpu_start_time = start_timer();
    cudaError_t cudaStatus = matrixOperation(outputs, doubleMatrix, doubleMatrix2, N, operation,GPU_kernel_time, GPU_transfer_time);
    long long totalGPUTime = stop_timer(gpu_start_time, "Total GPU time");//is slowness in the kernel? find out. get kerne ltime
    
    cout << "GPU DONE!" << endl;
    cout<<"Total GPU Time: "<<totalGPUTime<<" microseconds"<<endl;
    cout<<"Kernel GPU Time: "<<*GPU_kernel_time<<" microseconds"<<endl;
    cout<<"Kernel Memory transfer time:" <<*GPU_transfer_time<<" microseconds"<<endl;
    //printMatrix(outputs);

    

    cout << "CPU Start!" << endl;

    long long cpu_start_time = start_timer();
    CPUMatrixOperation(CPUoutputs, doubleMatrix, doubleMatrix2, N, operation);
    long long totalCPUTime = stop_timer(cpu_start_time, "Total CPU time");

    cout << "CPU DONE!" << endl;
    cout<<"Total CPU Time: "<<totalCPUTime<<" microseconds"<<endl;

    //printMatrix(CPUoutputs);
    bool equal = checkEquality(outputs, CPUoutputs, N);
    if(equal == true)
        cout<<"CPU and GPU outputs are equal"<<endl;
    else
        cout<<"CPU and GPU outputs are NOT equal"<<endl;

    float ratio;
    ratio = (float)(totalCPUTime/(*GPU_kernel_time));
    cout<<"Speed up based on GPU Kernel Time: "<<ratio<<endl;

    free(outputs);
    free(doubleMatrix);
    free(doubleMatrix2);
    free(CPUoutputs);
    free(GPU_kernel_time);

    return 0;
}
cudaError_t matrixOperation(double* c, const double* a, const double* b, unsigned int arrSize, int operation, float* kernel_runtime, float* GPU_transfer_time) {
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    float kernel_time = 0;
    float transfer_time = 0;
    cudaError_t cudaStatus;


    long long GPU_memory;
    GPU_memory = start_timer();

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
    transfer_time += stop_timer(GPU_memory, "GPU memory transfer time:");
    *GPU_transfer_time = transfer_time;

    long long kernel_start_time;
    kernel_start_time = start_timer();

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
    kernel_time += stop_timer(kernel_start_time, "Kernel execution time:");
    *kernel_runtime = kernel_time;
    //cout<<"Kernel time: "<<kernel_time<<endl;

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
/*
c is output, a and b are input arrays to perform operation, arrSize is size of array, operation is operation type
*/
void CPUMatrixOperation(double* c, const double* a, const double* b, unsigned int arrSize, int operation){
    switch (operation) {
        case 1:
            CPUMatrixAdd(c, a, b, arrSize);
            break;
        case 2:
            CPUMatrixSubtract(c, a, b, arrSize);
            break;
        case 3:
            CPUMatrixMultiply(c, a, b, arrSize);
            break;  
        case 4:
            CPUMatrixDivide(c, a, b, arrSize);
            break;

    }

}

// Returns the current time in microseconds
long long start_timer() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

// converts a long long ns value to float seconds
float usToSec(long long time) {
    return ((float)time)/(1000000);
}

// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, const char *name) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
    float elapsed = usToSec(end_time - start_time);
    printf("%s: %.5f sec\n", name, elapsed);
    return end_time - start_time;
}

