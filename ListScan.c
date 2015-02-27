// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

    
__global__ void scan(float * input, float * output, int len, float *sum) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
	
	__shared__ float XY[2*BLOCK_SIZE];
	int t=threadIdx.x;
	int start=blockIdx.x*blockDim.x*2;
	if(start+t < len)
		XY[t]= input[start+t];
	else
		XY[t]=0.0;
	
	if(start+BLOCK_SIZE+t<len)
		XY[BLOCK_SIZE+t]=input[start+BLOCK_SIZE+t];
	else
		XY[BLOCK_SIZE+t]=0.0;
	
	__syncthreads();
	
	for(int stride=1;stride<=BLOCK_SIZE;stride*=2)
	{
		int index=(threadIdx.x+1)*stride*2-1;
		if(index<2*BLOCK_SIZE)
			XY[index]+=XY[index-stride];
		__syncthreads();
	}
	
	
	for(int stride=BLOCK_SIZE/2;stride>0;stride/=2)
	{
		__syncthreads();
		int index = (threadIdx.x+1)*stride*2-1;
		if(index+stride < 2*BLOCK_SIZE)
			XY[index+stride]+=XY[index];
	}
	__syncthreads();
	
	int i= start + t;
	if(i<len)
	{
		output[i]=XY[t];
				//printf("%f",output[blockIdx.x*blockDim.x+t]);
	}
	if(start+BLOCK_SIZE+t<len)
	{
		output[start+BLOCK_SIZE+t]=XY[BLOCK_SIZE+t];
		//printf("%d=%f \t",start+BLOCK_SIZE+t,output[start+BLOCK_SIZE+t]);
	}
	
	__syncthreads();
	
	//write to global sum array for more than one block problems
	 if (t==0) 
        //sum[blockIdx.x] = output[(blockIdx.x+1)*1024-1];
		sum[blockIdx.x] =XY[2*BLOCK_SIZE -1];
	__syncthreads();
	
	if(blockIdx.x*blockDim.x+threadIdx.x==0)
	{
		for(int i=0;i<(((len-1)/(BLOCK_SIZE*2))+1);++i)
		{
			printf("\t%d=%f",i,sum[i]);
		}
		
	}
	//if(sum[len/2]!=0)
	/*float sumOutput[10];float junk[10];
	sectionScan(sum,sumOutput,((len-1)/(BLOCK_SIZE*2))+1,junk);
	if(blockIdx.x*blockDim.x+threadIdx.x==0)
	{
		for(int i=0;i<(((len-1)/(BLOCK_SIZE*2))+1);++i)
		{
			printf("\t%d=%f",i,sumOutput[i]);
		}
		
	}
*/	
		int value = 0;
		for (int l =0; l < blockIdx.x;l++)
		{
			value += sum[l];
		}
		__syncthreads();
		i = (blockIdx.x*2*BLOCK_SIZE + threadIdx.x);
		if (i < len) 
			output[i] += value;
		i = (blockIdx.x*2*BLOCK_SIZE + threadIdx.x + BLOCK_SIZE);
		if (i < len) 
   			output[i] += value;
		__syncthreads();	
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
	float * sum;//to hold iterative sum for more than one block
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
	wbCheck(cudaMalloc((void**)&sum, ceil(numElements/1024.0)*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
	dim3 dimBlock(512,1,1);
	dim3 dimGrid(ceil(numElements/1024.0),1,1);
	wbLog(TRACE,"Num of blocks needed=",ceil(numElements/1024.0));
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
	scan<<<dimGrid, dimBlock>>> (deviceInput, deviceOutput,numElements,sum);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}

