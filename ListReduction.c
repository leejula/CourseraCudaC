// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

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

__global__ void reductionKernel(float * input, float * output, int len) {
    //@@ Load a segment of the input vector into shared memory
	__shared__ float partialSum[BLOCK_SIZE*2];
	int t= threadIdx.x;
	int start= blockIdx.x* blockDim.x*2;
	
	if(start+t < len)
		partialSum[t]= input[start+t];
	else
		partialSum[t]=0.0;
	
	if(start+BLOCK_SIZE+t<len)
		partialSum[BLOCK_SIZE+t]=input[start+BLOCK_SIZE+t];
	else
		partialSum[BLOCK_SIZE+t]=0.0;
	
	__syncthreads();
    //@@ Traverse the reduction tree
	
	for(int stride= BLOCK_SIZE; stride>0; stride/=2)
	{
		__syncthreads();
		if(t<stride)
		partialSum[t]+= partialSum[t+stride];
		if(blockIdx.x==0 && t==0)
			printf("%f\t",partialSum[t]);
	}
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
	if(t==0)
	{output[blockIdx.x]=partialSum[0];
	//printf("%f-ouput", output[blockIdx.x]);
	}
	__syncthreads();
	
	if(blockIdx.x==0 && t==0)
	{
		int numBlocks = len / (BLOCK_SIZE<<1);
    		if (len % (BLOCK_SIZE<<1)) {
        		numBlocks++;
    		}
		for(int i=1;i<numBlocks;++i)
			output[0]+=output[i];
	}
	
}

int main(int argc, char ** argv) {
    int ii;
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    //<< denotes shift left. << of 512 gives 1024
	numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
    hostOutput = (float*) malloc(numOutputElements * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
	cudaError_t err=cudaMalloc((void **)&deviceInput, numInputElements * sizeof(float));
	 if (err != cudaSuccess) {                                            
            wbLog(ERROR, "Failed to run malloc device Input ");           
        } 
	err=cudaMalloc((void **)&deviceOutput, numOutputElements * sizeof(float));
	 if (err != cudaSuccess) {                                             
            wbLog(ERROR, "Failed to run malloc device Output ");           
        } 

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
	
	cudaMemcpy(deviceInput,hostInput,numInputElements * sizeof(float), cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
	//@@ Initialize the grid and block dimensions here
	dim3 dimBlock(512,1,1);
	dim3 dimGrid(numOutputElements,1,1);
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
	reductionKernel<<<dimGrid,dimBlock>>> (deviceInput, deviceOutput, numInputElements);
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	err=cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) {                                             
            wbLog(ERROR, "Failed to copy Output ");           
        } 
    wbTime_stop(Copy, "Copying output memory to the CPU");
	wbLog(ERROR,"err:",cudaGetErrorString(cudaGetLastError()));

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
	wbLog(TRACE,hostOutput[0]);
    /*for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }
*/
    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
	cudaFree(deviceInput);
	cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}

