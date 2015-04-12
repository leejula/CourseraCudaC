#include <wb.h> 
void vecAdd(float *input1, float *input2,float *output,int len)
{
	#pragma acc parallel loop copyin (input1[0:len]) copyin (input2[0:len]) copyout (output[0:len]) 
	
	for(int i=0;i<len;i++)
	{
		
		output[i]=input1[i]+input2[i];
	}
}
int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
 // float *deviceInput1;
 // float *deviceInput2;
 // float *deviceOutput;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");
	
  wbLog(TRACE, "The input length is ", inputLength);
	
	vecAdd(hostInput1,hostInput2,hostOutput,inputLength);

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
