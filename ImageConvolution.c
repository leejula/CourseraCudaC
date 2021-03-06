	#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

#define OutTile 12
#define BlkW  (OutTile + Mask_width-1)

//here blkw 16
//@@ INSERT CODE HERE
__global__ void convolution_2D(float *P, float *N, int height, int width,int channels,
							   const float * __restrict__ M)
{
	int tx= threadIdx.x;
	int ty= threadIdx.y;
	int row_o = blockIdx.y * OutTile + ty;
	int col_o = blockIdx.x * OutTile + tx; 
	int row_i = row_o - 2;
	int col_i = col_o - 2;
	
	__shared__ float Ns[BlkW][BlkW][3];
	int count=0;
	
	//loading dat
		if((row_i>=0) && (row_i< height) && (col_i >=0) && (col_i< width)) 
	{
		Ns[ty][tx][0]= N[(row_i*width+ col_i)*3+0];	
		Ns[ty][tx][1]= N[(row_i*width+ col_i)*3+1];	
		Ns[ty][tx][2]= N[(row_i*width+ col_i)*3+2];	
			/*if(tx==0 && ty==0)
				printf("count= %f %f %f\n",Ns[ty][tx][0],Ns[ty][tx][1],Ns[ty][tx][2]);
				*/
	}
	
	else
	{	Ns[ty][tx][0]=0.0;
	 	Ns[ty][tx][1]=0.0;
		Ns[ty][tx][2]=0.0;
	 	if(blockIdx.x==0 && blockIdx.y==0)
	 	printf("zero- ty and tx is %d, %d \n", ty,tx);
	}
	__syncthreads();
	
	//compute
	float output[3]={0.0,0.0,0.0};
	if (ty< OutTile && tx<OutTile)
	{
		for(int i=0;i<Mask_width;i++)
		{
			for(int j=0;j<Mask_width;j++)
			{
				for(int k=0;k<3;k++)
				{
					output[k]+= M[i*5+j]*Ns[i+ty][j+tx][k];
				}
			}
		}
		if(row_o< height && col_o< width)
	{
		//if(output[0]<0)
		//	output[
		P[(row_o*width+ col_o)*3+0]= output[0];
		P[(row_o*width+ col_o)*3+1]= output[1];
		P[(row_o*width+ col_o)*3+2]=output[2];
		if(blockIdx.x==0 && blockIdx.y==0 && tx==12 && ty==0)
				printf("result= %f %f %f\n",output[0],output[1],output[2]);
	}
	}
	
	
	
	//__syncthreads();
}

__device__ float clamp(float x, float start, float end)
{
	return min(max(x, start), end);
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	dim3 dimBlock(BlkW,BlkW,1);
	dim3 dimGrid( 1+ (imageWidth-1)/OutTile, 1+ (imageHeight-1)/OutTile,1);
	
	convolution_2D<<<dimGrid,dimBlock>>>(deviceOutputImageData,deviceInputImageData,
										 imageHeight,imageWidth,imageChannels,deviceMaskData);
    wbTime_stop(Compute, "Doing the computation on the GPU");

	wbLog(ERROR,"err:",cudaGetErrorString(cudaGetLastError()));

    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
