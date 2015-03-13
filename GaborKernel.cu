#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"



#define HEIGHT 960
#define WIDTH 1280

#define MASK_WIDTH 53
#define MASK_RADIUS (MASK_WIDTH/2)
#define BLOCK_SIZE 32

// gabor filter.
__global__ void convolution_kernel(float * input, float * output, int height, 
	int width, const float * __restrict__ Mask)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float accum = 0.0;

	if (row < height && col < width)
	{
		for (int y = -MASK_RADIUS; y <= MASK_RADIUS;++y)
		{
			for (int x = -MASK_RADIUS; x <= MASK_RADIUS; ++x)
			{
				int xoffset = col + x;
				int yoffset = row + y;
				if (xoffset >= 0 && xoffset < WIDTH && yoffset >= 0 && yoffset < HEIGHT)
					accum += input[yoffset*WIDTH + xoffset]*Mask[(y+MASK_RADIUS)*MASK_WIDTH+x+MASK_RADIUS];
			}
		
		}
		output[row*WIDTH + col] = accum;
	}


}

//	»ñÈ¡½Ø¶ÏÏàÎ»²î
__global__ void get_phase(float * ref, float * sce, const float * __restrict__ maskReal,
	const float * __restrict__ maskImage, float * phase)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float ref_real = 0.0;
	float ref_image = 0.0;
	float sce_real = 0.0;
	float sce_image = 0.0;

	if (row < HEIGHT && col < WIDTH)
	{
		for (int y = -MASK_RADIUS; y <= MASK_RADIUS;++y)
		{
			for (int x = -MASK_RADIUS; x <= MASK_RADIUS; ++x)
			{
				int xoffset = col + x;
				int yoffset = row + y;
				if (xoffset >= 0 && xoffset < WIDTH && yoffset >= 0 && yoffset < HEIGHT)
				{
					ref_real += ref[yoffset*WIDTH + xoffset]*maskReal[(y+MASK_RADIUS)*MASK_WIDTH+x+MASK_RADIUS];
					ref_image += ref[yoffset*WIDTH + xoffset]*maskImage[(y+MASK_RADIUS)*MASK_WIDTH+x+MASK_RADIUS];
					sce_real += sce[yoffset*WIDTH + xoffset]*maskReal[(y+MASK_RADIUS)*MASK_WIDTH+x+MASK_RADIUS];
					sce_image += sce[yoffset*WIDTH + xoffset]*maskImage[(y+MASK_RADIUS)*MASK_WIDTH+x+MASK_RADIUS];
				}
					
			}
		
		}
		// sce - ref
		phase[row*WIDTH + col] = atan2(sce_image, sce_real) - atan2(ref_image, ref_real);
	}

}

//	´ÓÎÄ¼þÖÐµ¼ÈëÍ¼ÏñÊý¾Ý
void import_data(const char * inputFile, float * hostInput,
	int height, int width)
{
	FILE *input = fopen(inputFile, "r");

	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			fscanf(input, "%f ",&hostInput[i*width+j]);
		
		}
	
	}

	fclose(input);
}

int main()
{
	float * hostRef;	// ²Î¿¼Æ½Ãæ
	float * hostSce;	// ³¡¾°
	float * hostPhase;	// ½Ø¶ÏÏàÎ»
	float * hostMaskReal;// GaborºËÊµ²¿
	float * hostMaskImage;// GaborºËÐé²¿


	float * deviceRef;
	float * deviceSce;
	float * devicePhase;
	float * deviceMaskReal;
	float * deviceMaskImage;

	float time_elapsed = 0;	//¼ÇÂ¼GPU¼ÆËãµÄÊ±¼ä
	cudaError_t err = cudaSuccess; //GPU ×´Ì¬(¼ÇÂ¼ÄÚ´æ·ÖÅäÊÇ·ñ³É¹¦µÈµÈ)
	cudaEvent_t start, stop;

	//@@ allocate memory on CPU
	hostRef = (float *)malloc(HEIGHT*WIDTH*sizeof(float));
	hostSce = (float *)malloc(HEIGHT*WIDTH*sizeof(float));
	hostPhase = (float *)malloc(HEIGHT*WIDTH*sizeof(float));
	hostMaskReal = (float *)malloc(MASK_WIDTH*MASK_WIDTH*sizeof(float));
	hostMaskImage = (float *)malloc(MASK_WIDTH*MASK_WIDTH*sizeof(float));


	//@@ load image data and Gabor mask. 
	import_data("ref.txt", hostRef, HEIGHT, WIDTH);
	import_data("sce.txt", hostSce, HEIGHT, WIDTH);
	import_data("gabor_kernel_re.txt", hostMaskReal, MASK_WIDTH, MASK_WIDTH);
	import_data("gabor_kernel_im.txt", hostMaskImage, MASK_WIDTH, MASK_WIDTH);

	
	//@@ allocate memory on gpu
	err = cudaMalloc((void **)&deviceRef, HEIGHT*WIDTH*sizeof(float));
	if(err!=cudaSuccess)
    {
		perror("the cudaMalloc on GPU is failed");
		return 1;
    }
	err = cudaMalloc((void **)&deviceSce, HEIGHT*WIDTH*sizeof(float));
	if(err!=cudaSuccess)
    {
		perror("the cudaMalloc on GPU is failed");
		return 1;
    }
	err = cudaMalloc((void **)&devicePhase, HEIGHT*WIDTH*sizeof(float));
	if(err!=cudaSuccess)
    {
		perror("the cudaMalloc on GPU is failed");
		return 1;
    }
	err = cudaMalloc((void **)&deviceMaskReal, MASK_WIDTH*MASK_WIDTH*sizeof(float));
	if(err!=cudaSuccess)
    {
		perror("the cudaMalloc on GPU is failed");
		return 1;
    }
	err = cudaMalloc((void **)&deviceMaskImage, MASK_WIDTH*MASK_WIDTH*sizeof(float));
	if(err!=cudaSuccess)
    {
		perror("the cudaMalloc on GPU is failed");
		return 1;
    }


	//@@ copy data to device
	cudaMemcpy(deviceRef, hostRef, HEIGHT*WIDTH*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceSce, hostSce, HEIGHT*WIDTH*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskReal, hostMaskReal, MASK_WIDTH*MASK_WIDTH*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceMaskImage, hostMaskImage, MASK_WIDTH*MASK_WIDTH*sizeof(float), cudaMemcpyHostToDevice);


	cudaEventCreate(&start);    //´´½¨Event
    cudaEventCreate(&stop);


	//@@ launch the kernel

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 dimGrid((WIDTH-1)/BLOCK_SIZE+1, (HEIGHT-1)/BLOCK_SIZE+1,1);

	cudaEventRecord(start, 0);
	/*convolution_kernel<<<dimGrid, dimBlock>>>(deviceInputImageData, deviceOutputImageData, HEIGHT, 
		WIDTH, deviceMaskData);*/
	get_phase<<<dimGrid, dimBlock>>>(deviceRef, deviceSce, deviceMaskReal,
		deviceMaskImage, devicePhase);

	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);

	cudaEventSynchronize(start);    //Waits for an event to complete.
	cudaEventSynchronize(stop);    //Waits for an event to complete.RecordÖ®Ç°µÄÈÎÎñ
    cudaEventElapsedTime(&time_elapsed,start,stop);    //¼ÆËãÊ±¼ä²î


	 printf("Ö´ÐÐÊ±¼ä£º%f(ms)\n",time_elapsed);


	 cudaMemcpy(hostPhase, devicePhase, HEIGHT*WIDTH*sizeof(float), cudaMemcpyDeviceToHost);

	 
	 FILE *outputImage = fopen("phase.txt", "w");

	 for (int i = 0; i < HEIGHT; ++i)
	 {
		 for (int j = 0; j < WIDTH; ++j)
		 {
			 fprintf(outputImage, j < WIDTH -1 ? "%f ":"%f\n", hostPhase[i*WIDTH + j]);
		 }
	 }

	 fclose(outputImage);

	

	cudaEventDestroy(start);    //destory the event
    cudaEventDestroy(stop);

	// free the memory on CPU and GPU.
	cudaFree(deviceRef);
	cudaFree(deviceSce);
	cudaFree(devicePhase);
	cudaFree(deviceMaskReal);
	cudaFree(deviceMaskImage);

	free(hostRef);
	free(hostSce);
	free(hostPhase);
	free(hostMaskReal);
	free(hostMaskImage);

	getchar();

	return 0;

}
