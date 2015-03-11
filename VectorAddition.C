#include	<wb.h>
#define SegSize 256

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)


__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
	int i= threadIdx.x+ blockIdx.x* blockDim.x;
	if(i<len)
		out[i]=in1[i]+in2[i];
	printf("i=%d val=%f",i,out[i]);
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    float * deviceInput1Stream0;
    float * deviceInput2Stream0;
    float * deviceOutputStream0;
	
	float * deviceInput1Stream1;
    float * deviceInput2Stream1;
    float * deviceOutputStream1;
	
	float * deviceInput1Stream2;
    float * deviceInput2Stream2;
    float * deviceOutputStream2;
	
	float * deviceInput1Stream3;
    float * deviceInput2Stream3;
    float * deviceOutputStream3;
	
cudaStream_t stream0,stream1,stream2,stream3;
cudaStreamCreate(&stream0);
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaStreamCreate(&stream3);

//Allocating pinned memory for host source and destination
wbCheck(cudaHostAlloc((void **)&hostInput1, inputLength* sizeof(float),cudaHostAllocDefault));	
wbCheck(cudaHostAlloc((void **)&hostInput2, inputLength* sizeof(float),cudaHostAllocDefault));	
wbCheck(cudaHostAlloc((void **)&hostOutput, inputLength* sizeof(float),cudaHostAllocDefault));	
    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");
	wbLog(TRACE, "The input length is ", inputLength);
	
	//Allocating device mem
	cudaMalloc((void **) &deviceInput1Stream0, sizeof(float)*SegSize);
	cudaMalloc((void **) &deviceInput2Stream0, sizeof(float)*SegSize);
	cudaMalloc((void **) &deviceOutputStream0, sizeof(float)*SegSize);
	
	cudaMalloc((void **) &deviceInput1Stream1, sizeof(float)*SegSize);
	cudaMalloc((void **) &deviceInput2Stream1, sizeof(float)*SegSize);
	cudaMalloc((void **) &deviceOutputStream1, sizeof(float)*SegSize);
	
	cudaMalloc((void **) &deviceInput1Stream2, sizeof(float)*SegSize);
	cudaMalloc((void **) &deviceInput2Stream2, sizeof(float)*SegSize);
	cudaMalloc((void **) &deviceOutputStream2, sizeof(float)*SegSize);
	
	cudaMalloc((void **) &deviceInput1Stream3, sizeof(float)*SegSize);
	cudaMalloc((void **) &deviceInput2Stream3, sizeof(float)*SegSize);
	cudaMalloc((void **) &deviceOutputStream3, sizeof(float)*SegSize);
	
for(int i=0;i<inputLength;i+=SegSize*2)
{
	wbCheck(cudaMemcpyAsync(deviceInput1Stream0,hostInput1+i, SegSize*sizeof(float),cudaMemcpyHostToDevice,stream0));
	wbCheck(cudaMemcpyAsync(deviceInput2Stream0,hostInput2+i, SegSize*sizeof(float),cudaMemcpyHostToDevice,stream0));
	//if(hostInput1+i+SegSize <len)
	wbCheck(cudaMemcpyAsync(deviceInput1Stream1,hostInput1+i+SegSize, SegSize*sizeof(float),cudaMemcpyHostToDevice,stream1));
	wbCheck(cudaMemcpyAsync(deviceInput2Stream1,hostInput2+i+SegSize, SegSize*sizeof(float),cudaMemcpyHostToDevice,stream1));
	
	wbCheck(cudaMemcpyAsync(deviceInput1Stream2,hostInput1+i+(SegSize*2), SegSize*sizeof(float),cudaMemcpyHostToDevice,stream2));
	wbCheck(cudaMemcpyAsync(deviceInput2Stream2,hostInput2+i+(SegSize*2), SegSize*sizeof(float),cudaMemcpyHostToDevice,stream2));
	
	wbCheck(cudaMemcpyAsync(deviceInput1Stream3,hostInput1+i+(SegSize*3), SegSize*sizeof(float),cudaMemcpyHostToDevice,stream3));
	wbCheck(cudaMemcpyAsync(deviceInput2Stream3,hostInput2+i+(SegSize*3), SegSize*sizeof(float),cudaMemcpyHostToDevice,stream3));
	
	vecAdd<<<SegSize/256,256,0,stream0>>> (deviceInput1Stream0,deviceInput2Stream0,deviceOutputStream0,SegSize);
	wbLog(ERROR,"errStream0:",cudaGetErrorString(cudaGetLastError()));
	vecAdd<<<SegSize/256,256,0,stream1>>> (deviceInput1Stream1,deviceInput2Stream1,deviceOutputStream1,SegSize);
	wbLog(ERROR,"errStream1:",cudaGetErrorString(cudaGetLastError()));
	
	vecAdd<<<SegSize/256,256,0,stream2>>> (deviceInput1Stream2,deviceInput2Stream2,deviceOutputStream2,SegSize);
	wbLog(ERROR,"errStream2:",cudaGetErrorString(cudaGetLastError()));
	vecAdd<<<SegSize/256,256,0,stream3>>> (deviceInput1Stream3,deviceInput2Stream3,deviceOutputStream3,SegSize);
	wbLog(ERROR,"errStream3:",cudaGetErrorString(cudaGetLastError()));


	
	wbCheck(cudaMemcpyAsync(hostOutput+i,deviceOutputStream0,SegSize*sizeof(float),cudaMemcpyDeviceToHost,stream0));
	
	wbLog(TRACE,"*",hostOutput[i]);
	wbCheck(cudaMemcpyAsync(hostOutput+i+SegSize,deviceOutputStream1,SegSize*sizeof(float),cudaMemcpyDeviceToHost,stream1));
	wbCheck(cudaMemcpyAsync(hostOutput+i+(SegSize*2),deviceOutputStream2,SegSize*sizeof(float),cudaMemcpyDeviceToHost,stream2));
	wbCheck(cudaMemcpyAsync(hostOutput+i+(SegSize*3),deviceOutputStream3,SegSize*sizeof(float),cudaMemcpyDeviceToHost,stream3));
	
	wbCheck(cudaStreamSynchronize(stream0));
	wbCheck(cudaStreamSynchronize(stream1));
	wbCheck(cudaStreamSynchronize(stream2));
	wbCheck(cudaStreamSynchronize(stream3));
}
    wbSolution(args, hostOutput, inputLength);
	
	cudaFree(deviceInput1Stream0);
	cudaFree(deviceInput2Stream0);
	cudaFree(deviceOutputStream0);
	cudaFree(deviceInput1Stream1);
	cudaFree(deviceInput2Stream1);
	cudaFree(deviceOutputStream1);
	cudaFree(deviceInput1Stream2);
	cudaFree(deviceInput2Stream2);
	cudaFree(deviceOutputStream2);
	cudaFree(deviceInput1Stream3);
	cudaFree(deviceInput2Stream3);
	cudaFree(deviceOutputStream3);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

