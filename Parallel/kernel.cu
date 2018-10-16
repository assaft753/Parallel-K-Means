#include "Cuda_Parallel.h"
#include "Parallel.h"
#include <stdio.h>

__global__ void kernel_points_position_calculation(Point* points,int jump,int myid,double t)
{
	int thread_index = threadIdx.x;
	int block_index = blockIdx.x;
	int offset = (jump*block_index) + thread_index;
	
	Point* pos = points + offset;

	printf("id [%d] in cuda point: %d\n", myid, pos->id);
	Axis current_position = pos->axis_location;
	Axis current_velocity = pos->axis_velocity;

	pos->axis_location.x = current_position.x + t*current_velocity.x;
	pos->axis_location.y = current_position.y + t*current_velocity.y;
	pos->axis_location.z = current_position.z + t*current_velocity.z;

	//printf("in cuda thread:[%d] block:[%d] id:[%d] %d\n", thread_index, block_index, myid, pos->id);
}

__global__ void kernel_group_points(int* cluster_of_points, Point* points, int jump, Axis* clusters_center_axis, int cluster_amount)
{
	int thread_index = threadIdx.x;
	int block_index = blockIdx.x;
	int offset = (jump*block_index) + thread_index;

	Point* point_pos = points + offset;

	int* int_pos = cluster_of_points + offset;


	int index = -1;
	double value = -1, min_value = -1;
	for (int i = 0; i < cluster_amount; i++)
	{
		value = cuda_point_2_point_distance(point_pos->axis_location, clusters_center_axis[i]);

		if (min_value == -1)
		{
			min_value = value;
			index = i;
		}

		else if (min_value > value)
		{
			min_value = value;
			index = i;
		}
	}

	*int_pos = index;
	
}

__global__ void kernel_test(int myid)
{
	//printf("in cuda myid:[%d] thread:[%d] block:[%d]\n",myid,threadIdx.x,blockIdx.x);
}



int cuda_kernel_points_calculation(Point * points, Point * leftover_points, int thread_per_block, int req_blocks, int leftover_threads, int leftover_block, double t)
{
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	cudaError_t cudaStatus;
	if (req_blocks > 0)
	{
		kernel_points_position_calculation << < req_blocks, thread_per_block >> > (points, thread_per_block,myid,t);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			return 0;
		}
	}

	if (leftover_block > 0)
	{
		kernel_points_position_calculation << < leftover_block, leftover_threads >> > (leftover_points,0,myid,t);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			return 0;
		}
	}
	return 1;
}

int cuda_kernel_group_points_to_clusters(int* cuda_cluster_of_points, int* cuda_cluster_of_points_leftover, Point* cuda_points, Point* cuda_points_leftover, Axis* cuda_clusters_center_axis, int cluster_amount, int thread_per_block, int req_blocks, int leftover_threads, int leftover_block)
{
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	cudaError_t cudaStatus;
	if (req_blocks > 0)
	{
		kernel_group_points << < req_blocks, thread_per_block >> > (cuda_cluster_of_points, cuda_points, thread_per_block, cuda_clusters_center_axis, cluster_amount);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			return 0;
		}
	}

	if (leftover_block > 0)
	{
		kernel_group_points << < leftover_block, leftover_threads >> > (cuda_cluster_of_points_leftover, cuda_points_leftover, 0, cuda_clusters_center_axis, cluster_amount);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			return 0;
		}
	}
	return 1;
}

double cuda_point_2_point_distance(Axis p1, Axis p2)
{
	double sub_x, sub_y, sub_z;
	double pow_x, pow_y, pow_z;
	double result;

	sub_x = fabs(p1.x - p2.x);
	sub_y = fabs(p1.y - p2.y);
	sub_z = fabs(p1.z - p2.z);

	pow_x = pow(sub_x, 2);
	pow_y = pow(sub_y, 2);
	pow_z = pow(sub_z, 2);

	result = sqrt(pow_x + pow_y + pow_z);

	return result;
}

void test(int myid)
{
	kernel_test << <2, 1024 >> > (myid);
}





/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}*/
