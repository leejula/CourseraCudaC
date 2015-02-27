#include <wb.h>
#define TILE_WIDTH 16


// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows,
                                     int numAColumns, int numBRows,
                                     int numBColumns, int numCRows,
                                     int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
	__shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
	int Row=threadIdx.y+ blockIdx.y * blockDim.y;
	int Col=threadIdx.x+ blockIdx.x * blockDim.x;
	int bx=blockIdx.x; int by= blockIdx.y;
	int tx= threadIdx.x; int ty=threadIdx.y;
	float Cvalue=0;
	for( int t= 0; t< (numAColumns-1)/TILE_WIDTH+1 ; t++)
	{
		if(Row<numARows && t*TILE_WIDTH+tx<numAColumns)
			ds_A[ty][tx] =  A[Row* numAColumns+ t* TILE_WIDTH + tx];
		else
			ds_A[ty][tx]=0.0;
		if(t*TILE_WIDTH+ty < numBRows && Col <numBColumns)
			ds_B[ty][tx]= B[(t*TILE_WIDTH+ty)*numBColumns + Col];
		else
			ds_B[ty][tx]==0.0;
		
		__syncthreads();
		for( int i=0;i<TILE_WIDTH;++i)
		{
			Cvalue+= ds_A[ty][i]*ds_B[i][tx];
			
		}
	
	__syncthreads();
	
	}
	if(Row<numCRows && Col< numCColumns)
		C[Row* numCColumns +Col]= Cvalue;
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
	if(numAColumns!=numBRows)
	{
		wbLog(TRACE, "The dimensions of A & B are inappropriate");
		exit(0);
	}
	else
	{
		numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
		hostC = ( float * )malloc(numCRows * numCColumns * sizeof(float));
		
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
	cudaError_t err= cudaMalloc((void **) &deviceA, numARows*numAColumns* sizeof(float));
	if (err != cudaSuccess) {                                                  
      wbLog(ERROR, "Failed to run stmt - malloc A");                              
      exit(0);                                                               
    }
	err= cudaMalloc((void **) &deviceB, numBRows*numBColumns* sizeof(float));
	if (err != cudaSuccess) {                                                  
      wbLog(ERROR, "Failed to run stmt - malloc B");                              
      exit(0);                                                               
    }
	err= cudaMalloc((void **) &deviceC, numCRows*numCColumns* sizeof(float));
	if (err != cudaSuccess) {                                                  
      wbLog(ERROR, "Failed to run stmt - malloc C");                              
      exit(0);                                                               
    }
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
	err =cudaMemcpy(deviceA, hostA, numARows*numAColumns* sizeof(float),cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {                                                  
      wbLog(ERROR, "Failed to run stmt - mem cpy of A");                              
      exit(0);                                                               
    }
	err= cudaMemcpy(deviceB, hostB, numBRows*numBColumns* sizeof(float),cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {                                                  
      wbLog(ERROR, "Failed to run stmt - mem cpy of B");                              
      exit(0);                                                               
    }
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
	dim3 DimGrid ( ceil(numCColumns/16.0), ceil(numCRows/16.0),1);
	dim3 DimBlock (16,16,1);
		
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
	matrixMultiplyShared<<<DimGrid,DimBlock>>> (deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);
  cudaDeviceSynchronize();
		wbLog(ERROR,"err:",cudaGetErrorString(cudaGetLastError()));
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
	err=cudaMemcpy(hostC, deviceC, numCRows*numCColumns* sizeof(float),cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {                                                  
      wbLog(ERROR, "Failed to run stmt - mem cpy of C");                              
      exit(0);                                                               
    }
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);
	}
  return 0;
}
