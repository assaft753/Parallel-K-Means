#include "Parallel.h"
#include "Cuda_Parallel.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int my_id;
int main(int argc, char **argv)
{
	int myid, num_of_proc;
	int points_amount;
	int cuda_error;
	
	MPI_Status status;

	MPI_Datatype axis_type;
	MPI_Datatype point_type;

	cudaDeviceProp device_prop;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &num_of_proc);

	my_id = myid;

	cuda_error = cuda_init();

	if (num_of_proc < 3 || cuda_error == 0)
	{
		printf("run this K-Means on 3 or more proccess or check your cuda card\n");
		fflush(stdout);
		MPI_Abort(MPI_COMM_WORLD, 0);
	}

	else
	{
		init_structs(&axis_type, &point_type);

		cuda_init_properties(&device_prop);

		if (myid == MASTER)
		{
			Points points;
			Clusters clusters;
			Cluster* new_cluster;
			int limit, t, i, n, max_n, good = 0;
			double qm, dt, current_t, q;

			new_cluster = init_data(&points, &clusters, &limit, &qm, &t, &points_amount, &dt);
			init_processes(&points_amount);

			max_n = t / dt;


			//test_cuda(myid);
			//master_calculate_points_positions(&points, 0.1, num_of_proc, point_type, device_prop);
			master_group_points_to_clusters(&points, new_cluster, clusters.size, axis_type,num_of_proc,point_type,device_prop);
		}

		else
		{
			int flag = 1;
			init_processes(&points_amount);

			while (flag != 0)
			{
				printf("myid [%d] listenning\n", my_id);
				fflush(stdout);
				broadcast_flag(&flag);

				switch (flag)
				{
				
				case CALCULATE_POINTS_FLAG:
					slave_calculate_points_positions(point_type, points_amount / num_of_proc, device_prop);
					break;
				
				case 2:
					//test_cuda_omp(myid);
					break;

				case GROUP_POINTS_FLAG:
					slave_group_points_to_clusters(axis_type,point_type,device_prop);

				default:
					break;
				}
			}
		}
	}
	
	cuda_reset();
	MPI_Finalize();
	return 0;
}

void test_cuda(int myid)
{
	int flag = 2;
	broadcast_flag(&flag);
	test_cuda_omp(myid);
}

void test_cuda_omp(int myid)
{
	cudaError_t cudaStatus;
	
	#pragma omp parallel for
	for (int i = 0; i < 10; i++)
	{
		printf("core %d myid %d\n", omp_get_thread_num(),myid);
		fflush(stdout);
		test(myid);
	}
	cudaStatus = cudaDeviceSynchronize();
	
	if (cudaStatus != cudaSuccess)
	{
		printf("id [%d] error in cuda\n",myid);
		fflush(stdout);
	}
}


void init_structs(MPI_Datatype* axis_type, MPI_Datatype* point_type)
{
	init_axis_struct(axis_type);
	init_point_struct(point_type, *axis_type);
}

void init_axis_struct(MPI_Datatype* axis_type)
{
	Axis axis;

	MPI_Datatype type[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int blocklen[3] = { 1, 1, 1 };
	MPI_Aint disp[3];

	disp[0] = (char *)&axis.x - (char *)&axis;
	disp[1] = (char *)&axis.y - (char *)&axis;
	disp[2] = (char *)&axis.z - (char *)&axis;

	MPI_Type_create_struct(3, blocklen, disp, type, axis_type);
	MPI_Type_commit(axis_type);
}

void init_point_struct(MPI_Datatype* point_type, MPI_Datatype axis_type)
{
	Point point;

	MPI_Datatype type[3] = { axis_type, axis_type, MPI_INT };
	int blocklen[3] = { 1, 1, 1 };
	MPI_Aint disp[3];

	disp[0] = (char *)&point.axis_location - (char *)&point;
	disp[1] = (char *)&point.axis_velocity - (char *)&point;
	disp[2] = (char *)&point.id - (char *)&point;

	MPI_Type_create_struct(3, blocklen, disp, type, point_type);
	MPI_Type_commit(point_type);
}


void cuda_init_properties(cudaDeviceProp * device_prop)
{
	cudaGetDeviceProperties(device_prop, 0);
}

Cluster * init_data(Points * points, Clusters * clusters, int * limit, double * qm, int * t, int * points_amount, double * dt)
{
	int clusters_amount;
	double pos_x, pos_y, pos_z;
	double vel_x, vel_y, vel_z;
	Cluster* new_cluster = 0;
	FILE *fp;
	fp = fopen("C:\\Users\\Assaf Tayouri\\Documents\\Visual Studio 2015\\Projects\\K-Means\\Parallel\\abc.txt", "r");
	fscanf(fp, "%d %d %d %lf %d %lf\n", points_amount, &clusters_amount, t, dt, limit, qm);

	points->size = *points_amount;
	points->points = (Point*)calloc(*points_amount, sizeof(Point));

	clusters->size = clusters_amount;
	clusters->clusters = (Cluster*)calloc(clusters_amount, sizeof(Cluster));

	new_cluster = (Cluster*)calloc(clusters_amount, sizeof(Cluster));

	for (int i = 0; i < *points_amount; i++)//parallel with critical
	{
		fscanf(fp, "%lf %lf %lf %lf %lf %lf\n", &pos_x, &pos_y, &pos_z, &vel_x, &vel_y, &vel_z);

		points->points[i].id = i;

		points->points[i].axis_location.x = pos_x;
		points->points[i].axis_location.y = pos_y;
		points->points[i].axis_location.z = pos_z;

		points->points[i].axis_velocity.x = vel_x;
		points->points[i].axis_velocity.y = vel_y;
		points->points[i].axis_velocity.z = vel_z;

		if (i < clusters_amount)
		{
			clusters->clusters[i].center.x = pos_x;
			clusters->clusters[i].center.y = pos_y;
			clusters->clusters[i].center.z = pos_z;

			new_cluster[i].center.x = pos_x;
			new_cluster[i].center.y = pos_y;
			new_cluster[i].center.z = pos_z;
		}
	}

	fclose(fp);
	return new_cluster;
}

void init_processes(int* points_amount)
{
	MPI_Bcast(points_amount, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
}

void broadcast_flag(int* flag)
{
	MPI_Bcast(flag, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
}


int master_calculate_points_positions(Points* points, double t, int num_of_proc, MPI_Datatype point_type, cudaDeviceProp device_prop)
{
	int leftover;
	int amount_each_element;
	int amount_with_lefover;
	int good;
	Point* points_arr;
	Point* points_recv_buffer;

	int flag = CALCULATE_POINTS_FLAG;

	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	broadcast_flag(&flag);

	amount_with_lefover = check_amounts_leftover(&amount_each_element, points->size, num_of_proc, &leftover);

	broadcast_time(&t);
	points_arr = scater_points(point_type, amount_each_element, points->points);

	realloc_lefover_points(leftover, &points_arr, amount_with_lefover, points);
	/*if (leftover > 0)
	{

	points_arr =(Point*) realloc(points_arr, sizeof(Point)*amount_with_lefover);//realloc freed

	#pragma omp parallel for shared(points, points_arr)
	for (int i = leftover; i > 0; i--)//omp
	{
	points_arr[amount_with_lefover - i] = points->points[points->size - i];
	}
	}*/

	good = parallel_calculate_points_location(points_arr, amount_with_lefover, device_prop, t);

	if (good == 0)
	{
		return 0;
	}

	points_recv_buffer = (Point*)calloc(points->size, sizeof(Point));//realloc shouldnt free

	gather_points(point_type, amount_each_element, points_arr, points_recv_buffer);

	copy_lefover_points(leftover, points_recv_buffer, points_arr, points, amount_with_lefover);

	/*if (leftover > 0)
	{
	#pragma omp parallel for shared(points_recv_buffer, points_arr)
	for (int i = leftover; i > 0; i--)//omp
	{
	points_recv_buffer[points->size -i] = points_arr[amount_with_lefover - i];
	}
	}*/

	print_points(points_recv_buffer, points->size, myid);

	free_point_array(points->points);
	free_point_array(points_arr);
	points->points = points_recv_buffer;
}

int slave_calculate_points_positions(MPI_Datatype point_type, int amount, cudaDeviceProp device_prop)
{
	double t;
	int good;
	Point* points_arr;

	broadcast_time(&t);
	points_arr = scater_points(point_type, amount, 0);

	good = parallel_calculate_points_location(points_arr, amount, device_prop, t);

	if (good == 0)
	{
		return 0;
	}

	gather_points(point_type, amount, points_arr, 0);

	free_point_array(points_arr);
}

int check_amounts_leftover(int* amount_each_element, int total_amount, int num_of_proc, int* leftover)
{
	*amount_each_element = total_amount / num_of_proc;
	*leftover = total_amount % num_of_proc;

	return *leftover + *amount_each_element;
}

void broadcast_time(double* t)
{
	MPI_Bcast(t, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
}

Point* scater_points(MPI_Datatype point_type, int amount_each_element, Point* points_arr)
{
	Point* points_buffer = (Point*)scater_elements(points_arr, amount_each_element, sizeof(Point), point_type);
	return points_buffer;
}

void gather_points(MPI_Datatype point_type, int amount_each_element, Point* points_arr, Point* points_recv_buffer)
{
	gather_elements(points_recv_buffer, points_arr, amount_each_element, point_type);
}

void realloc_lefover_points(int leftover, Point** points_arr, int amount_with_lefover, Points* points)
{
	if (leftover > 0)
	{

		*points_arr = (Point*)realloc(*points_arr, sizeof(Point)*amount_with_lefover);//realloc freed

#pragma omp parallel for shared(points, points_arr)
		for (int i = leftover; i > 0; i--)//omp
		{
			*points_arr[amount_with_lefover - i] = points->points[points->size - i];
		}
	}
}

int parallel_calculate_points_location(Point* points_arr, int amount, cudaDeviceProp device_prop, double t)
{
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	Point* cuda_points;
	Point* cuda_points_leftover;

	int cuda_points_amount;
	int cuda_points_leftover_amount;
	int good;
	int amount_omp = amount / 2;
	int amount_cuda = amount_omp + (amount % 2);

	Point* omp_offset = points_arr + amount_cuda;

	good = cuda_start_points_location_calculation(points_arr, amount_cuda, device_prop, &cuda_points, &cuda_points_amount, &cuda_points_leftover, &cuda_points_leftover_amount, t);

#pragma omp parallel for shared(omp_offset)
	for (int i = 0; i < amount_omp; i++)
	{
		calculate_point_position(omp_offset + i, t);
		printf("id [%d] in omp in core:%d point: %d\n", myid, omp_get_thread_num(), omp_offset[i].id);
		fflush(stdout);
	}

	if (good == 0)
	{
		return 0;
	}
	good = cuda_end_points_location_calculation(points_arr, cuda_points, cuda_points_amount, cuda_points_leftover, cuda_points_leftover_amount);

	if (good == 0)
	{
		return 0;
	}

}

int cuda_start_points_location_calculation(Point* points, int amount, cudaDeviceProp device_prop, Point** cuda_points, int* amount_cuda_points, Point** cuda_points_leftover, int* amount_cuda_points_leftover, double t)
{

	int thread_per_block;
	int req_blocks;
	int leftover_threads;
	int leftover_block;

	int good;

	good = cuda_blocks_calculation(&thread_per_block, &req_blocks, &leftover_threads, &leftover_block, device_prop, amount);

	if (good == 0)
	{
		printf("exit 1 [%d]", my_id);
		return 0;
	}

	*amount_cuda_points = thread_per_block * req_blocks;
	*amount_cuda_points_leftover = leftover_threads * leftover_block;

	good = cuda_malloc_elements((void**)cuda_points, *amount_cuda_points, (void**)cuda_points_leftover, *amount_cuda_points_leftover, sizeof(Point));

	if (good == 0)
	{
		printf("exit 2 [%d]", my_id);
		return 0;
	}

	Point* leftover_points_start = (points + *(amount_cuda_points));
	Point* points_start = (points);

	good = cuda_copy_elements((void*)(*cuda_points), (void*)points_start, *amount_cuda_points, (void*)(*cuda_points_leftover), (void*)leftover_points_start, *amount_cuda_points_leftover, sizeof(Point), cudaMemcpyHostToDevice);

	if (good == 0)
	{
		printf("exit 3 [%d]", my_id);
		return 0;
	}

	good = cuda_kernel_points_calculation(*cuda_points, *cuda_points_leftover, thread_per_block, req_blocks, leftover_threads, leftover_block, t);

	if (good == 0)
	{
		printf("exit 4 [%d]", my_id);
		return 0;
	}
}

int cuda_end_points_location_calculation(Point * points, Point * cuda_points, int  amount_cuda_points, Point * cuda_points_leftover, int  amount_cuda_points_leftover)
{
	int good;
	cudaError_t cudaStatus;

	cudaStatus = cudaDeviceSynchronize();// error check
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}

	Point* leftover_points_offset = points + amount_cuda_points;
	good = cuda_copy_elements(points, cuda_points, amount_cuda_points, leftover_points_offset, cuda_points_leftover, amount_cuda_points_leftover, sizeof(Point), cudaMemcpyDeviceToHost);

	if (good == 0)
	{
		return 0;
	}

	return 1;
}

void copy_lefover_points(int leftover, Point* points_recv_buffer, Point* points_arr, Points* points, int amount_with_lefover)
{
	if (leftover > 0)
	{
#pragma omp parallel for shared(points_recv_buffer, points_arr)
		for (int i = leftover; i > 0; i--)//omp
		{
			points_recv_buffer[points->size - i] = points_arr[amount_with_lefover - i];
		}
	}
}

void free_point_array(Point* points)
{
	free(points);
}

void calculate_point_position(Point* point, double t)
{
	Axis current_position = point->axis_location;
	Axis current_velocity = point->axis_velocity;

	point->axis_location.x = current_position.x + t*current_velocity.x;
	point->axis_location.y = current_position.y + t*current_velocity.y;
	point->axis_location.z = current_position.z + t*current_velocity.z;
}


int master_group_points_to_clusters(Points* points, Cluster* new_cluster, int cluster_amount, MPI_Datatype axis_type,int num_of_proc,MPI_Datatype point_type, cudaDeviceProp device_prop)//
{
	int buffer_size = sizeof(Axis)*cluster_amount + sizeof(int);
	char* buffer = (char*)calloc(buffer_size, sizeof(char));//freed
	int position = 0;
	int amount_with_lefover;
	int amount_each_element;
	int leftover;
	int flag = GROUP_POINTS_FLAG;
	Point* points_arr;//freed
	Cluster_Parallel* cluster_parallel;//freed

	Axis* clusters_center = (Axis*)calloc(cluster_amount, sizeof(Axis));//freed

	for (int i = 0; i < cluster_amount; i++)//omp
	{
		clusters_center[i] = new_cluster[i].center;
	}

	pack_elements(&position, buffer, buffer_size, &cluster_amount, 1, MPI_INT);
	pack_elements(&position, buffer, buffer_size, clusters_center, cluster_amount, axis_type);

	broadcast_flag(&flag);

	broadcast_value(&buffer_size, 1, MPI_INT);
	broadcast_value(buffer, buffer_size, MPI_PACKED);

	amount_with_lefover = check_amounts_leftover(&amount_each_element, points->size, num_of_proc, &leftover);

	broadcast_value(&amount_each_element, 1, MPI_INT);

	points_arr = scater_points(point_type, amount_each_element, points->points);

	realloc_lefover_points(leftover, &points_arr, amount_with_lefover, points);

	cluster_parallel = group_points_to_clusters(clusters_center, cluster_amount, points_arr, amount_with_lefover,device_prop);

	if (cluster_parallel == 0)
	{
		return 0;
	}

	for (int i = 0; i < cluster_amount; i++)//omp
	{
		if (new_cluster[i].size > 0)
		{
			free(new_cluster[i].cluster_points);
		}

		new_cluster[i].cluster_points = (Point*)calloc(points->size, sizeof(Point));
		new_cluster[i].size = 0;

		int size = cluster_parallel[i].size;
		for (int j = 0; j < size; j++)//copy points by value to original current cluster in new_cluster array
		{
			new_cluster[i].cluster_points[j] = cluster_parallel[i].cluster_points[j];
			new_cluster[i].size++;
		}

		free(cluster_parallel[i].cluster_points);
	}

	recieve_points_from_slaves(new_cluster, cluster_amount,num_of_proc,point_type);
	shrink_clusters_points(new_cluster, cluster_amount);

	free(clusters_center);
	free(cluster_parallel);
	free(buffer);
	free(points_arr);

	return 1;
	
}

int slave_group_points_to_clusters(MPI_Datatype axis_type, MPI_Datatype point_type, cudaDeviceProp device_prop)// put in h file //
{
	int buffer_size;
	int position = 0;
	int max_points_amount;
	int cluster_amount;
	int amount_each_element;
	char* buffer;//freed
	Axis* clusters_center;//freed
	Point* points_arr;//freed
	Cluster_Parallel* cluster_parallel;//freed

	broadcast_value(&buffer_size, 1, MPI_INT);

	buffer = (char*)calloc(buffer_size, sizeof(char));

	broadcast_value(buffer, buffer_size, MPI_PACKED);

	broadcast_value(&amount_each_element,1,MPI_INT);

	unpack_elements(&position, buffer, buffer_size, &cluster_amount, 1, MPI_INT);
	
	clusters_center = (Axis*)calloc(cluster_amount, sizeof(Axis));
	unpack_elements(&position, buffer, buffer_size, clusters_center, cluster_amount, axis_type);

	points_arr = scater_points(point_type, amount_each_element, 0);

	cluster_parallel = group_points_to_clusters(clusters_center, cluster_amount, points_arr, amount_each_element, device_prop);

	send_elements(&my_id, 1, MPI_INT, MASTER, MPI_ANY_TAG);
	
	int position;
	int points_amount;
	int size_buffer;
	char* p_buffer;//freed
	for (int i = 0; i < cluster_amount; i++)//omp
	{
		position = 0;
		points_amount = cluster_parallel[i].size;
		size_buffer = points_amount * sizeof(Point) + sizeof(int);

		send_elements(&size_buffer, 1, MPI_INT, MASTER, i);

		p_buffer = (char*)calloc(size_buffer, sizeof(char));//critical

		pack_elements(&position, p_buffer, size_buffer, &points_amount, 1, MPI_INT);
		pack_elements(&position, p_buffer, size_buffer, cluster_parallel[i].cluster_points, points_amount, point_type);

		send_elements(p_buffer, size_buffer, MPI_PACKED, MASTER, i);

		free(cluster_parallel[i].cluster_points);
		free(p_buffer);
	}

	free(buffer);
	free(cluster_parallel);
	free(clusters_center);
	free(points_arr);





	/*for (int i = 0; i < cluster_amount; i++)
	{
		printf("myid :[%d] cluster center x:%lf y:%lf z:%lf\n", my_id, clusters_center[i].x, clusters_center[i].y, clusters_center[i].z);
		fflush(stdout);
	}

	printf("myid :[%d] buffer_size %d cluster_amount %d max_points_amount %d \n", my_id, buffer_size, cluster_amount, max_points_amount);
	fflush(stdout);*/

	return 1;
}

Cluster_Parallel* group_points_to_clusters(Axis* center_axis_arr, int cluster_amount,Point* points,int amount, cudaDeviceProp device_prop)
{
	int* cluster_of_points;
	Cluster_Parallel* cluster_parallel;

	cluster_parallel = init_cluster_parallel(cluster_amount, amount);

	cluster_of_points = cuda_group_points_to_clusters(points, amount,center_axis_arr, cluster_amount, device_prop);

	if (cluster_of_points == 0)
	{
		return 0;
	}

	for (int i = 0; i < amount; i++)//cant omp
	{
		int cluster_index = cluster_of_points[i];
		cluster_parallel[cluster_index].size++;
		cluster_parallel[cluster_index].cluster_points[cluster_parallel[cluster_index].size - 1] = points[i];
	}

	free(cluster_of_points);

	return cluster_parallel;
}

Cluster_Parallel* init_cluster_parallel(int cluster_amount, int points_amount)
{
	Cluster_Parallel* cluster_parallel_arr = (Cluster_Parallel*)calloc(cluster_amount, sizeof(Cluster_Parallel));

	for (int i = 0; i < cluster_amount; i++)//omp
	{
		cluster_parallel_arr[i].size = 0;
		cluster_parallel_arr[i].cluster_points = (Point*)calloc(points_amount, sizeof(Point));//critical ?
	}
}

int* cuda_group_points_to_clusters(Point* points,int points_amount,Axis* clusters_center_axis, int cluster_amount, cudaDeviceProp device_prop)
{
	int* cuda_cluster_of_points;//freed
	int* cuda_cluster_of_points_leftover;//freed
	int* cluster_of_points;//shouldnt free
	int* cluster_of_points_leftover_offset;//shouldnt free
	Point* cuda_points;//freed
	Point* cuda_points_leftover;//freed
	Axis* cuda_clusters_center_axis;//freed
	cudaError_t cudaStatus;
	int thread_per_block;
	int req_blocks;
	int leftover_threads;
	int leftover_block;
	int amount_cuda_points;
	int amount_cuda_points_leftover;
	int good;
	
	cuda_blocks_calculation(&thread_per_block, &req_blocks, &leftover_threads, &leftover_block, device_prop, points_amount);

	amount_cuda_points = thread_per_block * req_blocks;
	amount_cuda_points_leftover = leftover_threads * leftover_block;

	good = cuda_malloc_elements((void**)&cuda_cluster_of_points, amount_cuda_points, (void**)&cuda_cluster_of_points_leftover, amount_cuda_points_leftover, sizeof(int));
	if (good == 0)
	{
		return 0;
	}

	good = cuda_malloc_elements((void**)&cuda_points, amount_cuda_points, (void**)&cuda_points_leftover, amount_cuda_points_leftover, sizeof(Point));
	if (good == 0)
	{
		return 0;
	}

	good = cuda_malloc_elements((void**)&cuda_clusters_center_axis, cluster_amount, 0, 0, sizeof(Axis));
	if (good == 0)
	{
		return 0;
	}

	Point* leftover_points_offset = points + amount_cuda_points;
	good = cuda_copy_elements(cuda_points, points, amount_cuda_points, cuda_points_leftover, leftover_points_offset, amount_cuda_points_leftover, sizeof(Point), cudaMemcpyHostToDevice);
	if (good == 0)
	{
		return 0;
	}

	good = cuda_copy_elements(cuda_clusters_center_axis, clusters_center_axis, cluster_amount, 0, 0, 0, sizeof(Axis), cudaMemcpyHostToDevice);
	if (good == 0)
	{
		return 0;
	}

	cuda_kernel_group_points_to_clusters(cuda_cluster_of_points, cuda_cluster_of_points_leftover, cuda_points, cuda_points_leftover, cuda_clusters_center_axis, cluster_amount, thread_per_block, req_blocks, leftover_threads, leftover_block);//in cu file
	
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}
	
	cluster_of_points = (int*)calloc(amount_cuda_points + amount_cuda_points_leftover, sizeof(int));
	
	cluster_of_points_leftover_offset = cluster_of_points + amount_cuda_points;
	
	good = cuda_copy_elements(cluster_of_points, cuda_cluster_of_points, amount_cuda_points, cluster_of_points_leftover_offset, cuda_cluster_of_points_leftover, amount_cuda_points_leftover, sizeof(int), cudaMemcpyDeviceToHost);
	if (good == 0)
	{
		return 0;
	}

	cudaFree(cuda_cluster_of_points);
	cudaFree(cuda_cluster_of_points_leftover);
	cudaFree(cuda_points);
	cudaFree(cuda_points_leftover);
	cudaFree(cuda_clusters_center_axis);

	return cluster_of_points;
}

void recieve_points_from_slaves(Cluster* new_cluster,int clusters_amount, int num_of_proc, MPI_Datatype point_type)
{
	int proc = num_of_proc - 1;
	int current_proc;

	while (proc > 0)//maybe omp
	{
		recieve_elements(&current_proc, 1, MPI_INT, MPI_ANY_SOURCE,MPI_ANY_TAG);

		unpack_clusters_points_from_slaves(new_cluster, clusters_amount, point_type, current_proc);
	}
}

void unpack_clusters_points_from_slaves(Cluster* new_cluster, int cluster_amount,MPI_Datatype point_type,int src)
{
	int position;
	int points_amount;
	int buffer_size;
	char* buffer;//freed
	Point* points;
	for (int i = 0; i < cluster_amount; i++)//omp
	{
		position = 0;
		recieve_elements(&buffer_size, 1, MPI_INT, src,i);

		buffer = (char*)calloc(buffer_size, sizeof(char));

		recieve_elements(&buffer, buffer_size, MPI_PACKED, src,i);

		unpack_elements(&position, buffer, buffer_size, &points_amount, 1, MPI_INT);
		
		if (points_amount != 0)
		{
			int start_offset = new_cluster[i].size;

			Point* points_offset = new_cluster[i].cluster_points + start_offset;

			unpack_elements(&position, buffer, buffer_size, points_offset, points_amount, point_type);
			
			new_cluster[i].size += points_amount;

		}

		free(buffer);
	}
}

void shrink_clusters_points(Cluster* new_cluster, int clusters_amount)
{
	for (int i = 0; i < clusters_amount; i++)//omp
	{
		new_cluster[i].cluster_points = (Point*)realloc(new_cluster[i].cluster_points, sizeof(Point)*new_cluster[i].size);
	}
}


void print_points(Point* points, int amount,int myid)
{
	for (int i = 0; i < amount; i++)
	{
		printf("from [%d] axis: x:%lf %lf %lf %lf %lf %lf %d\n", myid, points[i].axis_location.x, points[i].axis_location.y, points[i].axis_location.z, points[i].axis_velocity.x, points[i].axis_velocity.y, points[i].axis_velocity.z, points[i].id);
		fflush(stdout);
	}
}

void broadcast_value(void* element, int amount, MPI_Datatype type)
{
	MPI_Bcast(element, amount, type, MASTER, MPI_COMM_WORLD);
}

void pack_elements(int* position, char* buffer, int buffer_size, void* elements, int elements_amount, MPI_Datatype type)
{
	MPI_Pack(elements, elements_amount, type, buffer, buffer_size, position, MPI_COMM_WORLD);
}

void unpack_elements(int* position, char* buffer, int buffer_size, void* elements, int elements_amount, MPI_Datatype type)
{

	MPI_Unpack(buffer, buffer_size, position, elements, elements_amount, type, MPI_COMM_WORLD);

}

void send_elements(void* elements_buffer, int amount, MPI_Datatype type, int dest,int flag)
{
	MPI_Send(elements_buffer, amount, type, dest, flag, MPI_COMM_WORLD);
}

void recieve_elements(void* elments_buffer, int amount, MPI_Datatype type, int src,int flag)
{
	MPI_Status status;
	MPI_Recv(elments_buffer, amount, type, src, flag, MPI_COMM_WORLD, &status);
}

void* scater_elements(void* elements_arr, int amount_each_element, int sizeof_element, MPI_Datatype element_type)
{
	void* elements_buffer = calloc(amount_each_element, sizeof_element);//malloc shouldnt free
	MPI_Scatter(elements_arr, amount_each_element, element_type, elements_buffer, amount_each_element, element_type, MASTER, MPI_COMM_WORLD);
	return elements_buffer;
}

void gather_elements(void* elements_recv_buffer, void* elements_arr, int amount_each_element, MPI_Datatype element_type)
{
	MPI_Gather(elements_arr, amount_each_element, element_type, elements_recv_buffer, amount_each_element, element_type, MASTER, MPI_COMM_WORLD);
}




int cuda_init()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		return 0;
	}
	return 1;
}

int cuda_reset()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 0;
	}
	return 1;
}

int cuda_malloc_elements(void** elements_arr, int amount, void** elements_leftover_arr, int amount_leftover, int size_of_element)
{
	int flag;

	if (amount > 0)
	{
		flag = cuda_malloc_memory(elements_arr, amount, size_of_element);
		if (flag == 0)
		{
			return 0;
		}
	}

	if (amount_leftover > 0)
	{
		flag = cuda_malloc_memory(elements_leftover_arr, amount_leftover, size_of_element);
		if (flag == 0)
		{
			return 0;
		}
	}
	return 1;
}

int cuda_malloc_memory(void ** elements_arr, int amount, int size_of_element)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(elements_arr, amount * size_of_element);
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}
	return 1;
}

int cuda_copy_elements(void* dst_elements_arr, void* src_elements_arr, int amount, void* dst_elements_leftover_arr, void* src_elements_leftover_arr, int amount_leftover, int size_of_element, cudaMemcpyKind direction)
{
	int flag;
	int myid;
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	if (amount > 0)
	{
		flag = cuda_copy_memory(dst_elements_arr, src_elements_arr, amount, size_of_element, direction);
		if (flag == 0)
		{
			printf("exit cpy1 [%d]", myid);
			return 0;
		}
	}

	if (amount_leftover > 0)
	{
		flag = cuda_copy_memory(dst_elements_leftover_arr, src_elements_leftover_arr, amount_leftover, size_of_element, direction);
		if (flag == 0)
		{
			printf("exit cpy2 [%d]", myid);
			return 0;
		}
	}
	return 1;
}

int cuda_copy_memory(void* dst_memory, void* src_memory, int amount, int size_of_element, cudaMemcpyKind direction)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dst_memory, src_memory,amount*size_of_element,direction);
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}
	return 1;
}

int cuda_blocks_calculation(int * thread_per_block, int * req_blocks, int * leftover_threads, int * leftover_block, cudaDeviceProp device_prop,int amount)
{
	*thread_per_block =cuda_get_thread_per_block(device_prop);
	int amount_of_blocks = cuda_get_amount_of_blocks(device_prop);

	*req_blocks = amount / *thread_per_block;
	*leftover_threads = amount % *thread_per_block;
	*leftover_block = *leftover_threads > 0;

	if (amount_of_blocks < *req_blocks)
	{
		return 0;
	}
	return 1;
}

int cuda_get_thread_per_block(cudaDeviceProp device_prop)
{
	return device_prop.maxThreadsPerBlock;
}

int cuda_get_amount_of_blocks(cudaDeviceProp device_prop)
{
	return device_prop.maxGridSize[0];
}
