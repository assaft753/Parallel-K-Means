#pragma once
#include "Cuda_Parallel.h"
#include "mpi.h"
#include <math.h>

#define CALCULATE_POINTS_FLAG 1
#define GROUP_POINTS_FLAG 3
#define MASTER 0
struct Axis
{
	double x = 1;
	double y = 0;
	double z = 0;
};

struct Point
{
	Axis axis_location;
	Axis axis_velocity;
	int id = 0;
};

struct Points
{
	Point* points = 0;
	int size = 0;
};

struct Cluster
{
	Point* cluster_points = 0;
	int size = 0;
	Axis center;
};

struct Cluster_Parallel
{
	int size = 0;
	Point* cluster_points = 0;
};

struct Clusters
{
	int size = 0;
	Cluster* clusters = 0;
};

void test(int myid);
void test_cuda(int myid);
void test_cuda_omp(int myid);





void init_structs(MPI_Datatype* axis_type, MPI_Datatype* point_type);
void init_axis_struct(MPI_Datatype* axis_type);
void init_point_struct(MPI_Datatype* point_type, MPI_Datatype axis_type);
Cluster * init_data(Points * points, Clusters * clusters, int * limit, double * qm, int * t, int * points_amount, double * dt);
void init_processes(int* points_amount);

void broadcast_flag(int* flag);

void broadcast_time(double* t);

void broadcast_value(void* element, int amount, MPI_Datatype type);

void pack_elements(int* position, char* buffer, int buffer_size, void* elements, int elements_amount, MPI_Datatype type);

void send_elements(void* elements_buffer, int amount, MPI_Datatype type, int dest, int flag);
void recieve_elements(void* elments_buffer, int amount, MPI_Datatype type, int src, int flag);

int slave_calculate_points_positions(MPI_Datatype point_type, int amount, cudaDeviceProp device_prop);
int master_calculate_points_positions(Points* points, double t, int num_of_proc, MPI_Datatype point_type, cudaDeviceProp device_prop);

int master_group_points_to_clusters(Points* points, Cluster* new_cluster, int cluster_amount, MPI_Datatype axis_type, int num_of_proc, MPI_Datatype point_type, cudaDeviceProp device_prop);
int slave_group_points_to_clusters(MPI_Datatype axis_type, MPI_Datatype point_type, cudaDeviceProp device_prop);

Cluster_Parallel* group_points_to_clusters(Axis* center_axis_arr, int cluster_amount, Point* points, int amount, cudaDeviceProp device_prop);

double cuda_point_2_point_distance(Axis p1, Axis p2);

//int cuda_find_min_distance_cluster(Point point, Cluster_Parallel* cluster_parallel, int cluster_amount);

int check_amounts_leftover(int* amount_each_element, int total_amount, int num_of_proc, int* leftover);
void realloc_lefover_points(int leftover, Point** points_arr, int amount_with_lefover, Points* points);
void copy_lefover_points(int leftover, Point* points_recv_buffer, Point* points_arr, Points* points, int amount_with_lefover);


Point* scater_points(MPI_Datatype point_type, int amount_each_element, Point* points_arr);
void gather_points(MPI_Datatype point_type, int amount_each_element, Point* points_arr, Point* points_recv_buffer);

void* scater_elements(void* elements_arr, int amount_each_element, int sizeof_element, MPI_Datatype element_type);
void gather_elements(void* elements_recv_buffer, void* elements_arr, int amount_each_element, MPI_Datatype element_type);

void free_point_array(Point* points);
void print_points(Point* points, int amount, int myid);

int parallel_calculate_points_location(Point* points_arr, int amount, cudaDeviceProp device_prop, double t);
void calculate_point_position(Point* point, double t);



int cuda_init();
int cuda_reset();
void cuda_init_properties(cudaDeviceProp* device_prop);

int cuda_start_points_location_calculation(Point* points, int amount, cudaDeviceProp device_prop, Point** cuda_points, int* amount_cuda_points, Point** cuda_points_leftover, int* amount_cuda_points_leftover, double t);
int cuda_end_points_location_calculation(Point* points, Point* cuda_points, int amount_cuda_points, Point* cuda_points_leftover, int amount_cuda_points_leftover);

int cuda_blocks_calculation(int * thread_per_block, int * req_blocks, int * leftover_threads, int * leftover_block, cudaDeviceProp device_prop, int amount);

int cuda_get_thread_per_block(cudaDeviceProp device_prop);
int cuda_get_amount_of_blocks(cudaDeviceProp device_prop);

int cuda_malloc_elements(void** elements_arr, int amount, void** elements_leftover_arr, int amount_leftover, int size_of_element);
int cuda_malloc_memory(void** elements_arr, int amount, int size_of_element);

int cuda_copy_elements(void* dst_elements_arr, void* src_elements_arr, int amount, void* dst_elements_leftover_arr, void* src_elements_leftover_arr, int amount_leftover, int size_of_element, cudaMemcpyKind direction);
int cuda_copy_memory(void* dst_memory, void* src_memory, int amount, int size_of_element, cudaMemcpyKind direction);

int cuda_kernel_points_calculation(Point * points, Point * leftover_points, int thread_per_block, int req_blocks, int leftover_threads, int leftover_block, double t);
int cuda_kernel_group_points_to_clusters(int* cuda_cluster_of_points, int* cuda_cluster_of_points_leftover, Point* cuda_points, Point* cuda_points_leftover, Axis* cuda_clusters_center_axis, int cluster_amount, int thread_per_block, int req_blocks, int leftover_threads, int leftover_block);