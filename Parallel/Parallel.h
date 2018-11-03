#pragma once
#include "Cuda_Parallel.h"
#include "mpi.h"
#include <math.h>

#define CALCULATE_POINTS_FLAG 1
#define GROUP_POINTS_FLAG 3
#define CHECK_TRANSFER_POINTS 4
#define EVALUATE_QM 5
#define MASTER 0
struct Axis
{
	double x = 0;
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

struct Transfer_Point
{
	int cluster_id;
	int point_id;
};

struct Qm_Point
{
	int cluster_id;
	Axis point_loc;
};

struct Max_point_Diameter
{
	int cluster_id;
	double max_diameter;
};

void test(int myid);
void test_cuda(int myid);
void test_cuda_omp(int myid);





void init_structs(MPI_Datatype* axis_type, MPI_Datatype* point_type, MPI_Datatype* transfer_points_type, MPI_Datatype* qm_point_type);
void init_axis_struct(MPI_Datatype* axis_type);
void init_transfer_points_struct(MPI_Datatype* transfer_points_type);
void init_point_struct(MPI_Datatype* point_type, MPI_Datatype axis_type);
void init_qm_point_struct(MPI_Datatype* qm_point_type, MPI_Datatype axis_type);
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

int* cuda_group_points_to_clusters(Point* points, int points_amount, Axis* clusters_center_axis, int cluster_amount, cudaDeviceProp device_prop);
Cluster_Parallel* init_cluster_parallel(int cluster_amount, int points_amount);
void unpack_elements(int* position, char* buffer, int buffer_size, void* elements, int elements_amount, MPI_Datatype type);
void unpack_clusters_points_from_slaves(Cluster* new_cluster, int cluster_amount, MPI_Datatype point_type, int src);

void recieve_points_from_slaves(Cluster* new_cluster, int clusters_amount, int num_of_proc, MPI_Datatype point_type);

void shrink_clusters_points(Cluster* new_cluster, int clusters_amount);

double cuda_point_2_point_distance(Axis p1, Axis p2);
int cuda_check_exists_points_transfer(int* points_id, int points_id_amount, int transfer_point_id);

//int cuda_find_min_distance_cluster(Point point, Cluster_Parallel* cluster_parallel, int cluster_amount);

int check_amounts_leftover(int* amount_each_element, int total_amount, int num_of_proc, int* leftover);
void realloc_lefover_points(int leftover, Point** points_arr, int amount_with_lefover, Points* points);
void realloc_lefover_transfer_point(int leftover, Transfer_Point** transfer_points_arr, int amount_with_lefover, Transfer_Point* transfer_points, int transfer_points_amount);
void copy_lefover_points(int leftover, Point* points_recv_buffer, Point* points_arr, Points* points, int amount_with_lefover);


Point* scater_points(MPI_Datatype point_type, int amount_each_element, Point* points_arr);
void gather_points(MPI_Datatype point_type, int amount_each_element, Point* points_arr, Point* points_recv_buffer);

void* scater_elements(void* elements_arr, int amount_each_element, int sizeof_element, MPI_Datatype element_type);
void gather_elements(void* elements_recv_buffer, void* elements_arr, int amount_each_element, MPI_Datatype element_type);

void free_point_array(Point* points);
void print_points(Point* points, int amount, int myid);

int parallel_calculate_points_location(Point* points_arr, int amount, cudaDeviceProp device_prop, double t);
void calculate_point_position(Point* point, double t);

void calculate_cluster_center(Cluster* new_cluster, Clusters* clusters, int amount);
Axis axis_avg(Point* points, int amount);

int master_check_points_transfer(Cluster * original_cluster, Cluster * new_cluster, int cluster_amount, int points_amount, int num_of_proc, MPI_Datatype transfer_points_type ,cudaDeviceProp device_prop);
int slave_check_points_transfer(MPI_Datatype transfer_points_type, cudaDeviceProp device_prop);
void prepare_for_pack_elements(Cluster* clusters, int** cluster_points_amount_arr, int** cluster_points_offset_arr, int cluster_amount);
void init_cluster_points_offset(int* cluster_points_amount_arr, int* cluster_points_offset_arr, int cluster_amount);
int pre_check_points_transfer(Cluster * original_cluster, Cluster * new_cluster, int cluster_amount);
int check_points_transfer(int** points_id, int cluster_amount, int* cluster_points_amount, Transfer_Point* transfer_points, int amount_each_element, cudaDeviceProp device_prop);

int check_exists_points_transfer(int* points_id, int points_id_amount, int transfer_point_id);
int cuda_start_check_points_transfer(int** points_id, int cluster_amount, int* cluster_points_amount, int** cuda_points_id_arr_pointers, int** cuda_cluster_points_amount, Transfer_Point* transfer_points, int** result, int*** cuda_points_id, Transfer_Point** cuda_transfer_points, Transfer_Point** cuda_transfer_points_leftover, int thread_per_block, int req_block, int leftover_threads, int leftover_block);
int cuda_end_check_points_transfer(int** cuda_points_id, int cluster_amount, int** cuda_points_id_arr_pointers, Transfer_Point* cuda_transfer_points, Transfer_Point* cuda_transfer_points_leftover, int* cuda_result_pointer, int* cuda_cluster_points_amount, int* result_of_cuda);

double master_evaluate_qm(Cluster* clusters, int cluster_amount, int point_amount, int num_of_proc, MPI_Datatype axis_type, MPI_Datatype qm_point_type);
int slave_evaluate_qm(MPI_Datatype axis_type, MPI_Datatype qm_point_type);

void prepare_for_pack_elements(Cluster* clusters, int** cluster_points_amount_arr, int** cluster_points_offset_arr, int cluster_amount);
void malloc_cluster_points(int* cluster_points_amount, void*** cluster_points_element, int cluster_amount, int size_of_pointer, int size_of_element);
char* malloc_packed_buffer(int amount, int size_of, int* buffer_size);
void realloc_lefover_qm_points(int leftover, Qm_Point** qm_points_arr, int amount_with_leftover, Qm_Point* qm_points, int qm_points_amount);

int** malloc_points_id(int cluster_amount, int* cluster_points_amount_arr);
void free_points_id(int** points_id, int cluster_amount);

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
int cuda_kernel_check_transfer_points(int* result, int** cuda_points_id, int* cuda_cluster_points_amount, int cluster_amount, Transfer_Point* cuda_transfer_points, Transfer_Point* cuda_transfer_points_leftover, int thread_per_block, int req_block, int leftover_threads, int leftover_block);