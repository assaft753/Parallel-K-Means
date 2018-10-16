#pragma once
#include "Parallel.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/*int cuda_init();
int cuda_reset();
void cuda_init_properties(cudaDeviceProp* device_prop);
void cuda_free_memory(void* pointer);

int cuda_start_points_location_calculation(Point* points, int amount, cudaDeviceProp device_prop, Point** cuda_points, int* amount_cuda_points, Point** cuda_points_leftover, int* amount_cuda_points_leftover);

int cuda_blocks_calculation(int * thread_per_block, int * req_blocks, int * leftover_threads, int * leftover_block, cudaDeviceProp device_prop, int amount);

int cuda_get_thread_per_block(cudaDeviceProp device_prop);
int cuda_get_amount_of_blocks(cudaDeviceProp device_prop);

int cuda_malloc_elements(void** elements_arr, int amount, void** elements_leftover_arr, int amount_leftover, int size_of_element);
int cuda_malloc_memory(void** elements_arr, int amount, int size_of_element);

int cuda_copy_elements(void* dst_elements_arr, void* src_elements_arr, int amount, void* dst_elements_leftover_arr, void* src_elements_leftover_arr, int amount_leftover, int size_of_element, cudaMemcpyKind direction);
int cuda_copy_memory(void* dst_memory, void* src_memory, int amount, int size_of_element, cudaMemcpyKind direction);

int cuda_kernel_points_calculation(Point * points, Point * leftover_points, int thread_per_block, int req_blocks, int leftover_threads, int leftover_block);*/