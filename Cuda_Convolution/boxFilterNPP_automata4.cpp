/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

//Adriel's To do:
/*
    1. Pack the convolution into a neat function with paramters: filename, kernel, image, etc..
    2. Find a way to bind this to python. Can discuss this. I can only find C++ and python binders? 
    3. What is oAnchor???(Seems to just offset the image sightly, be wary)
    4. Play with ROI
    5. TIME THIS CODE! (DONE)
    6. ISSUE: Kernel is Npp32f not Npp32f (we need it to be a float!) (DONE)
    7. ISSUE: There is a problem with your bfKernel, i suppose it might need some normalization?? It just makes everything black...
*/


#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#  pragma warning(disable:4819)
#endif



#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>

//Additional includes
#include <sys/time.h>
#include <bits/stdc++.h>
#include <stdlib.h>
#include <typeinfo>
#include <stdlib.h>
#include <math.h>
#include <set>

#include "matrix_ops.cuh"

long long start_timer();
long long stop_timer(long long start_time, const char* name);
void fillKernelArray(std::string kernelName, Npp32f* kernelArr, int kernelSize);


inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer   = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion/1000, (driverVersion%100)/10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion/1000, (runtimeVersion%100)/10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}
//NORMALIZE THE KERNEL (That's what makes it visible to us, although I'm not sure if this is done in the LSST codebase! (or maybe that do it differently))
void fillKernelArray(std::string kernelName, Npp32f* kernelArr, int kernelSize) {
    std::fstream file;
    std::string word, t, q, filename;
    // filename of the file
    filename = kernelName;

    // opening file
    file.open(filename.c_str());
    double sum = 0;

    // extracting words from the file
    if (file.is_open()) {
        for (int i = 0; i < kernelSize;i++) {
            file >> word;
            int n = word.length();
            char char_array[n + 1];
            strcpy(char_array, word.c_str());
            char* pEnd;
            double wordDouble = strtod(char_array, &pEnd);
            kernelArr[i] = wordDouble;
            sum += wordDouble;
            /*if (i == 144)//testing out identity kernel.
                kernelArr[i] = 1;//wordDouble;
            else
                kernelArr[i] = 0;*/
        }

    }

    for (int i = 0;i < kernelSize;i++) {
        kernelArr[i] = kernelArr[i] / sum;
    }

    file.close();
}

//discard rows just does not take into account a certain number of rows at the end (where most error seems to be acumulated?)
long long meanSquaredImages(double* img1, double* img2, int imgSize) {
    long long diff = 0;
    for (int i = 0;i < imgSize;i++) {
        long long pix1 = img1[i];
        long long pix2 = img2[i];
        long long pixDiff = pix1 - pix2;
        diff += pixDiff*pixDiff;
    }
    long long mse = diff / imgSize;
    return mse;
}

long long avgPixelValue(double* img, int imgSize) {
    long long sum = 0;
    for (int i = 0;i < imgSize;i++) {
        sum += img[i];
    }
    long long avg = sum / imgSize;
    return avg;
}

double zeroCount(double* img, int imgSize) {
    int count = 0;
    for (int i = 0;i < imgSize;i++) {
        if (img[i] <= 0) {
            count++;
        }
    }
    return count;
}


//FOR TESTING PURPOSES:
void fillWithRandomNumbers(double* arr, int arrSize) {
    for (int i = 0; i < arrSize;i++) {
        arr[i] = rand() % 100 + 1;
    }

}



/*void fillImageArray(npp::ImageCPU_32f_C1 imageCPU){//std::string imageFileName, npp::ImageCPU_32f_C1* imageCPU, int imageSize) {
    for (int i = 0;i < 512 * 512;i++) {//sorry ,it's easier to hardcode, this is just a hack...
        imageCPU.data()[i] = 100; //rand() % 100 + 1;//produces a seg fault...(8-31-2020)
        //::cout << oHostDst.data()[i] << std::endl;
    }
}*/


int main(int argc, char *argv[])
{
    printf("%s Starting...\n\n", argv[0]);

    try
    {
        std::string sFilename;
        char* filePath;

        cudaDeviceInit(argc, (const char**)argv);

        if (printfNPPinfo(argc, argv) == false)
        {
            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char**)argv, "input"))
        {
            getCmdLineArgumentString(argc, (const char**)argv, "input", &filePath);
        }
        else
        {
            filePath = sdkFindFilePath("Lena.pgm", argv[0]);
        }

        if (filePath)
        {
            sFilename = filePath;
        }
        else
        {
            sFilename = "Lena.pgm";
        }

        // if we specify the filename at the command line, then we only test sFilename[0].
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout << "boxFilterNPP opened: <" << sFilename.data() << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "boxFilterNPP unable to open: <" << sFilename.data() << ">" << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0)
        {
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind('.');

        if (dot != std::string::npos)
        {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_boxFilter.pgm";

        if (checkCmdLineFlag(argc, (const char**)argv, "output"))
        {
            char* outputFilePath;
            getCmdLineArgumentString(argc, (const char**)argv, "output", &outputFilePath);
            sResultFilename = outputFilePath;
        }
        long long start_time = start_timer();
        int imgDimX = 2048;
        int imgDimY = 4176;//2048;
        int imgSize =  imgDimX * imgDimY;
        //Code from stack overflow - begin
        std::string fileExtension = ".pgm";
        std::string dirFilename = "fitstest";
        std::string saveFilename = dirFilename + "_8bitBFconvolved";

        //npp::ImageCPU_8u_C1 oHostSrc8bit;//8 bit data loaded
        npp::ImageCPU_32f_C1 oHostSrc(imgDimX,imgDimY);//same 8 bit data, but loaded as 32bit from a textfile (this needs size specified)
        double* originalImg = new double[imgSize];
        double* compareImg= new double[imgSize];//this is the image we will want to compare to verify our results. just the first iteration I assume.
        double* textConvolved= new double[imgSize]; //this is just the result in array form for data type consistency

        //long long convolveTime = 0;
        //std::cout << "oHostSrc" << std::endl;


       // npp::loadImage(dirFilename + fileExtension, oHostSrc8bit);
        unsigned int srcWidth = oHostSrc.width();
        unsigned int srcHeight = oHostSrc.height();

        //fillImageArray(oHostSrc); Maybe get working later....
        

        /*Filling the array with image pixel values from textfile BEGIN CODE:*/
        //Might wanna put this in a function, but t'was a headache alas...
        std::fstream file;
        std::string word, t, q, filename, compareFilename;
        
        
        // filename of the file
        filename = "inputImgOG.txt";//a lena numpy array retrieved the same way as from lsst.
        compareFilename = "outputImgOG.txt";

        // opening file
        file.open(filename.c_str());
        double sum = 0;

        // extracting words from the file
        if (file.is_open()) {
            for (int i = 0; i < imgSize;i++) {
                file >> word;
                int n = word.length();
                char char_array[n + 1];
                strcpy(char_array, word.c_str());
                char* pEnd;
                double wordDouble = strtod(char_array, &pEnd);
                oHostSrc.data()[i] = wordDouble;
                originalImg[i] = wordDouble;
                //sum += wordDouble;
            }

        }
        file.close();
        /*
        for (int i = 0;i < imgSize;i++) {//sorry ,it's easier to hardcode, this is just a hack...
            //std::cout <<"Result: "<<oHostDst.data()[i] << std::endl;
            std::cout << "Original: " << originalImg[i] << std::endl;
        }*/

        
        file.open(compareFilename.c_str());
        if (file.is_open()) {
            for (int i = 0;i < imgSize;i++) {
                file >> word;
                int n = word.length();
                char char_array[n + 1];
                strcpy(char_array, word.c_str());
                char* pEnd;
                double wordDouble = strtod(char_array, &pEnd);
                compareImg[i] = wordDouble;
            }
        }
        std::cout << "Input image to be compared loaded onto double array" << std::endl;
        

        /*
        for (int i = 0;i < imgSize;i++) {//sorry ,it's easier to hardcode, this is just a hack...
          //std::cout <<"Result: "<<oHostDst.data()[i] << std::endl;
            std::cout << "Compare: " <<compareImg[i] << std::endl;
        }*/
        /*END OF TEXTFILE CODE*/

        /*
        std::cout << "INPUT IMAGE DATA (oHostSrc):" << std::endl;
        for (int i = 0;i < imgSize;i++) {//sorry ,it's easier to hardcode, this is just a hack...
            //oHostSrc.data()[i] = 100; //rand() % 100 + 1;//produces a seg fault...(8-31-2020)
            std::cout << oHostSrc.data()[i] << std::endl;
        }
        */
        long long convTimer = start_timer();

        npp::ImageNPP_32f_C1 oDeviceSrc(oHostSrc);//512, 512);//Works with hardcoded ints though.... (oHostSrc.width(), oHostSrc.height());//program is breaking here.....
        //npp::ImageNPP_8u_C1 oDeviceSrc8bit(oHostSrc8bit);//for loaded image

        NppiSize kernelSize = { 17,17 };//{ 3, 3 }; // dimensions of convolution kernel (filter)
        //9/7/2020 - Later you'll want to put nan padding, but this shall do for now...(NOTE: You also have to change oDeviceDst size)****
        //note, we're using 8bit value for widht, but it shouldnt matter...
        NppiSize oSizeROI = { imgDimY, imgDimX };//srcWidth - kernelSize.width + 1, srcHeight - kernelSize.height + 1 };//{ 510,510};//oHostSrc.width() - kernelSize.width + 1, oHostSrc.height() - kernelSize.height + 1 };//what is with the kernel offset of ROI? How does this deal with the edges? Avoiding them?
        
        //For the 8-bit loaded image:
        NppiSize oSizeROI8bit = { imgDimY, imgDimX };//srcWidth - kernelSize.width + 1, srcHeight - kernelSize.height + 1 };//what is with the kernel offset of ROI? How does this deal with the edges? Avoiding them?
        //npp::ImageNPP_8u_C1 oDeviceDst8bit(oSizeROI8bit.width, oSizeROI8bit.height); // allocate device image of appropriately reduced size
        //npp::ImageCPU_8u_C1 oHostDst8bit(oDeviceDst8bit.size());

        npp::ImageNPP_32f_C1 oDeviceDst(oSizeROI.width, oSizeROI.height);//imgDimX,imgDimY);//oSizeROI.width, oSizeROI.height); // allocate device image of appropriately reduced size
        npp::ImageCPU_32f_C1 oHostDst(oDeviceDst.size());
        NppiPoint oAnchor = { kernelSize.width / 2, kernelSize.height / 2 }; //**SEE DOCUMENTATION ON WHAT oAnchor is??? (Does this perfectly center it? Im just copying example code)
        NppStatus eStatusNPP;

        std::cout << "kernelSize.width: " << kernelSize.width / 2 << "/kernelSize.height: " << kernelSize.height / 2 << std::endl;
        int hostKSize = kernelSize.width * kernelSize.height;
        //Npp32f hostKernel[hostKSize];//= { 0, -1, 0, -1, 5, -1, 0, -1, 0 };//{ 0,0,0,0,1,0,0,0,0 };//Identity kernel to test alignment//{ 0, -1, 0, -1, 5, -1, 0, -1, 0 };//this is emboss//{ -1, 0, 1, -1, 0, 1, -1, 0, 1 }; // convolving with this should do edge detection
        Npp32f hostKernel[hostKSize];
        std::cout << "Host kernel size: " << hostKSize << std::endl;
        fillKernelArray("bfKernel.txt", hostKernel, hostKSize);

        /*
        for (int i = 0;i < 289;i++) {
            hostKernel[i] = 1/9; //a blur kernel...?
        }*/
        
        /*for (int i = 0; i < hostKSize;i++) {
            std::cout << hostKernel[i] << std::endl;
        }*/

        std::cout << "Loaded PGM Image Data First row vs Image Data from Text File: " << hostKSize << std::endl;

        /*for (int i = 0;i < 512; i++) {//instead we individually define each pixel. 255 for testing purposes (should be all white). 262144 is the number of pixels = (512*512)
          double c8bit = oHostSrc8bit.data()[i];
          double ctxt = oHostSrc.data()[i];
          std::cout <<"8-bit data: "<< c8bit << std::endl;
          std::cout << "32-bit data:" << ctxt <<"\n"<< std::endl;
        }*/
        


        Npp32f* deviceKernel;
        size_t deviceKernelPitch;
        //cudaMallocPitch((void**)&deviceKernel, &deviceKernelPitch, kernelSize.width * sizeof(Npp32f), kernelSize.height * sizeof(Npp32f));
        /*cudaMemcpy2D(deviceKernel, deviceKernelPitch, hostKernel,
            sizeof(Npp32f) * kernelSize.width, // sPitch
            sizeof(Npp32f) * kernelSize.width, // width
            kernelSize.height, // height
            cudaMemcpyHostToDevice);*/
        cudaMalloc((void**)&deviceKernel, kernelSize.width * kernelSize.height * sizeof(Npp32f));//is Npp32f 32 bit? We may be 64 bit in the future!
        cudaMemcpy(deviceKernel, hostKernel, kernelSize.width * kernelSize.height * sizeof(Npp32f), cudaMemcpyHostToDevice);
        Npp32f divisor = 1; // no scaling

        std::cout << "Calculated size: " << kernelSize.width * kernelSize.height * sizeof(Npp32f) << std::endl;
        std::cout << "Device kernel size: " << sizeof(deviceKernel) << std::endl;
        std::cout << "hostKernel size: " << sizeof(hostKernel) << std::endl;

        //eStatusNPP = nppiFilter32f_8u_C1R(oDeviceSrc.data());
        int devPitch = oDeviceSrc.pitch();
        int dstPitch = oDeviceDst.pitch();

        //int devPitch8bit = oDeviceSrc8bit.pitch();
        //int dstPitch8bit = oDeviceDst8bit.pitch();

        //std::cout <<"Pitch:" <<oDeviceSrc.pitch() << std::endl;
        //How pitch is calculated: how many bytes in a row? Calcualte by getting bytes in a pixel * image width
        
        //std::cout << "Source image: " << oDeviceSrc.data() << std::endl;
        std::cout << "Source image Line Step (bytes) " << devPitch << std::endl;
        //std::cout << "Destination Image: " << oDeviceDst.data() << std::endl;
        std::cout << "Destination Image line step (bytes): " << dstPitch << std::endl;
        //std::cout << "ROI: " << oSizeROI << std::endl;
        //std::cout << "Device Kernel: " << deviceKernel << std::endl;
        //std::cout << "Kernel Size: " << kernelSize << std::endl;
        //std::cout << "X and Y offsets of kernel origin frame: " << oAnchor << std::endl;

        /*
        std::cout << "DEVICE SRC DATA (Data before the convolution): " << std::endl;//this is technically the host, but there is no way to print device without transferring it to host first
        for (int i = 0;i < 512 * 512;i++) {//sorry ,it's easier to hardcode, this is just a hack...
            std::cout << oHostSrc.data()[i] << std::endl;
        }*/
        std::cout << "Convolution Step For 32-bit: " << std::endl;
        
	    for(int i = 0 ;i<10;i++){
	    std::cout<<"Iteration: "<<i<<std::endl;
	    eStatusNPP = nppiFilter_32f_C1R(oDeviceSrc.data(), devPitch, oDeviceDst.data(),
                dstPitch, oSizeROI, deviceKernel, kernelSize, oAnchor);

        cudaDeviceSynchronize();
	    }

        /*
        std::cout << "Convolution Step For 32-bit: " << std::endl;
        eStatusNPP = nppiFilter32f_8u_C1R(oDeviceSrc8bit.data(), devPitch8bit, oDeviceDst8bit.data(),
            dstPitch8bit, oSizeROI8bit, deviceKernel, kernelSize, oAnchor);*/

        /*eStatusNPP = nppiFilter_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(),
            oDeviceDst.data(), oDeviceDst.pitch(),
            oSizeROI, deviceKernel, kernelSize, oAnchor, divisor);*/


        //oDeviceDst is like the device array. The resulting image after the convolution, still in device memory!

        CudaWrapper::MatrixGradientDevice(oDeviceSrc.data(), oDeviceSrc.data());//is .data() a pointer to an array? (YES!)



        std::cout << "NppiFilter error status " << eStatusNPP << std::endl; // prints 0 (no errors) //-6 is NPP_SIZE_ERROR (ROI Height or ROI width are negative)
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch()); // memcpy to host
        //oDeviceDst8bit.copyTo(oHostDst8bit.data(), oHostDst8bit.pitch());
        
        //saveImage(saveFilename+fileExtension, oHostDst8bit);

        /*Npp8u* hostDstData = oHostDst.data();
        std::cout << "Host destination data type:  " << typeid(hostDstData).name() << std::endl; //unsigned char array of course...
        for (int i = 0;i < hostDstData.size();i++) {
            std::cout << hostDstData[i] << std::endl;
        }*/
        
        std::cout << "First 10 OUTPUTS: " << std::endl;

        for (int i = 0;i < 10;i++) {
            std::cout << i << ":" << oHostDst.data()[i] << std::endl;
        }
        
        int badappleCount = 0;
      

        std::set<int> badRows;
        std::cout << "RESULTING IMAGE DATA (oHostDst):"<<std::endl;
        for (int i = 0;i < imgSize;i++) {//just prints first row of values
            //std::cout <<"Result: "<<oHostDst.data()[i] << std::endl;
            //double p8bit = oHostDst8bit.data()[i];
            double p32bit = oHostDst.data()[i];
            
            textConvolved[i] = oHostDst.data()[i];
            //bit8Convolved[i] = oHostDst8bit.data()[i];
            /*
            if (abs(p8bit - p32bit) > 10) {

                std::cout << "8-bit convolved data: " << p8bit << std::endl;
                std::cout << "32-bit convolved data:" << p32bit << "\n" << std::endl;
                int row = i / (imgDimX);
                int col = i % imgDimX - 1;
                std::cout << "Row: " << row << " / " << "Column: " << col << std::endl;
                badRows.insert(row);

                badappleCount++;
            }*/
            //convolvedImg[i] = oHostDst.data()[i]; (This is just for loading into array for MSE, 
            //std::cout << "Result: " <<  convolvedImg[i] << std::endl;
        }


        //bad apples are outlying values that screw everything up. Could be edge pixels.
        //std::cout << "Bad apple count: " << badappleCount << std::endl;
        //std::cout << "Bad rows contains:";
        /*for (std::set<int>::iterator it = badRows.begin(); it != badRows.end(); ++it)
            std::cout << ' ' << *it;
            */

        int discardRows = imgDimX * 30;//discard 30 rows
        int inspectRows = imgDimX * 472;
        long long totalConvTime = stop_timer(convTimer, "NPP convolution time:");

        
        long long mse = meanSquaredImages(textConvolved, compareImg, inspectRows);
        long long avgConvolvedPixel = avgPixelValue(compareImg, inspectRows);
        long long percentage = sqrt(mse) / avgConvolvedPixel *100;
        double convZeroCount = zeroCount(compareImg, imgSize);

        std::cout << "avgConvolvedPixel: " << avgConvolvedPixel << std::endl;
        std::cout << "Zero count: " << convZeroCount << std::endl;
        //double mseWithOriginal = meanSquaredImages(originalImg, compareImg, imgSize);
        std::cout << "Mean squared error: " << mse << std::endl;
        std::cout << "Root MSE: " << sqrt(mse) << std::endl;
        std::cout << "Root MSE as a percentage of avg destination pixel: " << percentage << std::endl;
        //std::cout << "MSE between original input image and compareImg: " << mseWithOriginal << std::endl;
        //std::cout << "Root MSE between original input image and compareImg: " << sqrt(mseWithOriginal) << std::endl;
        

        //end code from SO
        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDst.data());
    
        long long totalTime = stop_timer(start_time, "Total NPP convolution time:");

        /*
        int R = 4176;
        int C = 2048;
        int N = R * C;


        std::cout << "TESTING IMPORTED MATRIX OPS" << std::endl;

        double* doubleMatrix = (double*)malloc(N * sizeof(double));
        double* doubleMatrix2 = (double*)malloc(N * sizeof(double));
        double* outputs = (double*)malloc(N * sizeof(double));

        fillWithRandomNumbers(doubleMatrix, N);
        fillWithRandomNumbers(doubleMatrix2, N);

        //CudaWrapper::MatrixDiff(doubleMatrix, outputs, 1);
        CudaWrapper::MatrixAdd(doubleMatrix, doubleMatrix2, outputs);

        std::cout << "First 10 INputs: " << std::endl;
        for (int i = 0;i < 10;i++) {
            std::cout << i << ":" << doubleMatrix[i]<<", "<<doubleMatrix2[i]<< std::endl;
        }
        std::cout << "First 10 OUTPUTS: " << std::endl;
        for (int i = 0; i < 10;i++) {
            std::cout << i << ":" << outputs[i] <<std::endl;
        }
        */






        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}

// Returns the current time in microseconds
long long start_timer() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000 + tv.tv_usec;
}

// converts a long long ns value to float seconds
float usToSec(long long time) {
    return ((float)time) / (1000000);
}

// Prints the time elapsed since the specified time
long long stop_timer(long long start_time, const char* name) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
    float elapsed = usToSec(end_time - start_time);
    printf("%s: %.5f sec\n", name, elapsed);
    return end_time - start_time;
}
