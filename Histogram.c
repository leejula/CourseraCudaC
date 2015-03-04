// Histogram Equalization

#include    <wb.h>
#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)


#define HISTOGRAM_LENGTH 64.0

//@@ insert code here

__global__ void cast(float *imageInput, unsigned char *castImage,unsigned char * grayImage,int width, int height)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int col= blockIdx.x*blockDim.x+threadIdx.x;
	int row= blockIdx.y*blockDim.y+threadIdx.y;
	int r,g,b;
	if((col>0 && col<width) && (row>0 && row<height))
	{
		//cast image
		castImage[i*3] = (unsigned char) (255 * imageInput[i*3]);
		castImage[(i*3)+1] = (unsigned char) (255 * imageInput[(i*3)+1]);
		castImage[(i*3)+2] = (unsigned char) (255 * imageInput[(i*3)+2]);
		//convert to grayscale
		r=castImage[i*3]; 
		g=castImage[(i*3)+1];
		b=castImage[(i*3)+2];
		grayImage[i] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);	
	}
	if(i==10)
	{
	printf("old 1,2,3= %d,%d,%d new 1,2,3=%d,%d,%d\t",imageInput[(i*3)],imageInput[(i*3)+1],imageInput[(i*3)+2],castImage[(i*3)],
		   castImage[(i*3)+1],castImage[(i*3)+2]);
	//printf("image and cast loc %d %d %d", i*3,(i*3)+1,(i*3)+2);
	}
	
}




int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;
float *deviceInputImageData;
float *deviceOutputImageData;
unsigned char *castImage;
unsigned char *grayImage;

    //@@ Insert more code here

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
	wbLog(TRACE,"Width=",imageWidth);
    imageHeight = wbImage_getHeight(inputImage);
	wbLog(TRACE,"Height",imageWidth);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");
	hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
	
	
//@@ insert code here
	//kernel cast image
    wbCheck(cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float)));
    wbCheck(cudaMalloc((void **) &castImage, imageWidth * imageHeight * imageChannels * sizeof(float)));
	wbCheck(cudaMalloc((void **) &grayImage, imageWidth * imageHeight * sizeof(float)));
	wbCheck(cudaMemcpy(deviceInputImageData,hostInputImageData,imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice));
	dim3 dimGrid(ceil(imageWidth/HISTOGRAM_LENGTH),ceil(imageHeight/HISTOGRAM_LENGTH),1);
	dim3 dimBlock( HISTOGRAM_LENGTH,HISTOGRAM_LENGTH,1);
	
	wbLog(TRACE,"grid", ceil(imageWidth/HISTOGRAM_LENGTH),"*",ceil(imageHeight/HISTOGRAM_LENGTH));
	cast<<<dimGrid,dimBlock>>> (deviceInputImageData, castImage,grayImage,imageWidth , imageHeight);
	wbCheck(cudaMemcpy(hostOutputImageData,castImage,imageWidth * imageHeight * imageChannels * sizeof(float),
			   cudaMemcpyDeviceToHost));
	cudaFree(deviceInputImageData);
	//kernel grey image
	cast<<<dimGrid,dimBlock>>> (deviceInputImageData, castImage,grayImage,imageWidth, imageHeight);
		
			
	wbLog(ERROR,"err:",cudaGetErrorString(cudaGetLastError()));
    wbSolution(args, outputImage);

    //@@ insert code here

    return 0;
}

