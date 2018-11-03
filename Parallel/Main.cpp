#include "Parallel.h"
#include "Cuda_Parallel.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void print(Clusters* clusters, Cluster* cluster, int d, Points points)
{
	printf("\nfrom %d:\n #################################\n", d);
	fflush(stdout);
	printf("Status of Points:[\n");
	fflush(stdout);
	for (int k = 0; k < points.size; k++)
	{
		printf("id = %d x = %lf y = %lf z = %lf\n", points.points[k].id, points.points[k].axis_location.x, points.points[k].axis_location.y, points.points[k].axis_location.z);
	}
	printf("]\n");
	printf("start main clusters*******************************\n");
	for (int i = 0; i < clusters->size; i++)
	{
		Cluster c = clusters->clusters[i];
		Point* p = c.cluster_points;
		printf("cluster number (%d) with center x = %lf y = %lf z = %lf\n", i, c.center.x, c.center.y, c.center.z);
		printf("++++++++++++++++++++++++++++++++++\n");
		for (int j = 0; j < c.size; j++)
		{
			printf("id = %d x = %lf y = %lf z = %lf\n", p[j].id, p[j].axis_location.x, p[j].axis_location.y, p[j].axis_location.z);
		}
		printf("\n++++++++++++++++++++++++++++++++++\n");
	}
	printf("end main clusters*******************************\n\n");

	printf("start new clusters*******************************\n");
	for (int i = 0; i < clusters->size; i++)
	{
		Cluster c = cluster[i];
		Point* p = c.cluster_points;
		printf("cluster number (%d) with center x = %lf y = %lf z = %lf\n", i, c.center.x, c.center.y, c.center.z);
		printf("++++++++++++++++++++++++++++++++++\n");
		for (int j = 0; j < c.size; j++)
		{
			printf("id = %d x = %lf y = %lf z = %lf\n", p[j].id, p[j].axis_location.x, p[j].axis_location.y, p[j].axis_location.z);
		}
		printf("\n++++++++++++++++++++++++++++++++++\n");
	}
	printf("end new clusters*******************************\n\n");

	printf("###################################################\n");
	fflush(stdout);
}
double t1, t2;
int my_id;
void start()
{
		t1 = MPI_Wtime();
}

void finish() {
		t2 = MPI_Wtime();
		printf("time %1.4f id %d\n", t2 - t1, my_id);
		fflush(stdout);
}

int main(int argc, char **argv)
{
	int myid, num_of_proc;
	int points_amount;
	int cuda_error;
	int good;
	
	MPI_Status status;

	MPI_Datatype axis_type;
	MPI_Datatype point_type;
	MPI_Datatype transfer_points_type;
	MPI_Datatype qm_point_type;

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
		init_structs(&axis_type, &point_type, &transfer_points_type,&qm_point_type);

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
			start();
			good = master_calculate_points_positions(&points, 0, num_of_proc, point_type, device_prop);//error check
			printf("get here2 %d\n", good);
			fflush(stdout);
			good = master_group_points_to_clusters(&points, new_cluster, clusters.size, axis_type, num_of_proc, point_type, device_prop); //error check
			printf("get here2 %d\n", good);
			fflush(stdout);

			calculate_cluster_center(new_cluster, &clusters, clusters.size);//can be done with mpi
			/*Point p[4];

			p[0] = new_cluster[0].cluster_points[1];
			p[1] = new_cluster[0].cluster_points[2];
			p[2] = new_cluster[0].cluster_points[0];
			p[3] = new_cluster[1].cluster_points[4];
			clusters.clusters[0].cluster_points = p;
			clusters.clusters[0].size = new_cluster[0].size;

			Point p1[6];
			p1[0] = new_cluster[1].cluster_points[0];
			p1[1] = new_cluster[1].cluster_points[1];
			p1[2] = new_cluster[1].cluster_points[2];
			p1[3] = new_cluster[1].cluster_points[3];
			p1[4] = new_cluster[0].cluster_points[3];
			p1[5] = new_cluster[1].cluster_points[5];

			clusters.clusters[1].cluster_points = p1;
			clusters.clusters[1].size = new_cluster[1].size;
			printf("%d!!!!!!!!!!!!!!!!!!!!!!!!\n", p1[0].id);
			fflush(stdout);

			Point p2[2];
			p2[0] = new_cluster[2].cluster_points[1];
			p2[1] = new_cluster[2].cluster_points[0];
			clusters.clusters[2].cluster_points = p2;
			clusters.clusters[2].size = new_cluster[2].size;
			
			print(&clusters, new_cluster, 5, points);
			fflush(stdout);*/
			good = master_check_points_transfer(clusters.clusters, new_cluster, clusters.size,points.size,num_of_proc,transfer_points_type, device_prop);// clusters.clusters first argument %% check if_transfered transfered = 1  || Not transfered = 0
			printf("get here22 %d\n", good);
			fflush(stdout);

			master_evaluate_qm(new_cluster, clusters.size, points.size,num_of_proc, axis_type, qm_point_type);
			printf("get here23 %d\n", good);
			fflush(stdout);
			finish();
			
			//print(&clusters, new_cluster, 5, points);
			//fflush(stdout);
		
			/*int thread_per_block;
			int req_blocks;
			int leftover_threads;
			int leftover_block;
			cuda_blocks_calculation(&thread_per_block, &req_blocks, &leftover_threads, &leftover_block, device_prop, 1000000);
			printf("thread_per_block %d req_blocks %d leftover_threads %d leftover_block %d\n", thread_per_block, req_blocks, leftover_threads, leftover_block);
			fflush(stdout);
			*/
		
		}

		else
		{
			int flag = 1;
			init_processes(&points_amount);

			while (flag != 0)
			{
				//printf("myid [%d] listenning\n", my_id);
				//fflush(stdout);
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
					break;
				
				case CHECK_TRANSFER_POINTS:
					slave_check_points_transfer(transfer_points_type,device_prop);
					break;
				
				case EVALUATE_QM:
					slave_evaluate_qm(axis_type,qm_point_type);

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
	for (int i = 0; i < 1; i++)
	{
		//printf("core %d myid %d\n", omp_get_thread_num(),myid);
		//fflush(stdout);
		test(myid);
	}
	cudaStatus = cudaDeviceSynchronize();
	
	if (cudaStatus != cudaSuccess)
	{
		//printf("id [%d] error in cuda\n",myid);
		//fflush(stdout);
	}
}


void init_structs(MPI_Datatype* axis_type, MPI_Datatype* point_type, MPI_Datatype* transfer_points_type, MPI_Datatype* qm_point_type)
{
	init_axis_struct(axis_type);
	init_point_struct(point_type, *axis_type);
	init_transfer_points_struct(transfer_points_type);
	init_qm_point_struct(qm_point_type,*axis_type);
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

void init_transfer_points_struct(MPI_Datatype * transfer_points_type)
{
	Transfer_Point transfer_point;

	MPI_Datatype type[2] = { MPI_INT, MPI_INT };
	int blocklen[2] = { 1, 1 };
	MPI_Aint disp[2];

	disp[0] = (char *)&transfer_point.cluster_id - (char *)&transfer_point;
	disp[1] = (char *)&transfer_point.point_id - (char *)&transfer_point;

	MPI_Type_create_struct(2, blocklen, disp, type, transfer_points_type);
	MPI_Type_commit(transfer_points_type);
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

void init_qm_point_struct(MPI_Datatype* qm_point_type, MPI_Datatype axis_type)
{
	Qm_Point qm_point;

	MPI_Datatype type[2] = { MPI_INT, axis_type };
	int blocklen[2] = { 1, 1 };
	MPI_Aint disp[2];

	disp[0] = (char *)&qm_point.cluster_id - (char *)&qm_point;
	disp[1] = (char *)&qm_point.point_loc - (char *)&qm_point;
	

	MPI_Type_create_struct(2, blocklen, disp, type, qm_point_type);
	MPI_Type_commit(qm_point_type);

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
	fp = fopen("C:\\Users\\Assaf Tayouri\\Documents\\Visual Studio 2015\\Projects\\K-Means\\abc.txt", "r");
	fscanf(fp, "%d %d %d %lf %d %lf\n", points_amount, &clusters_amount, t, dt, limit, qm);

	points->size = *points_amount;
	points->points = (Point*)calloc(*points_amount, sizeof(Point));

	clusters->size = clusters_amount;
	clusters->clusters = (Cluster*)calloc(clusters_amount, sizeof(Cluster));

	new_cluster = (Cluster*)calloc(clusters_amount, sizeof(Cluster));

	for (int i = 0; i < *points_amount; i++)//cant omp
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

	broadcast_flag(&flag);
	amount_with_lefover = check_amounts_leftover(&amount_each_element, points->size, num_of_proc, &leftover);
	broadcast_time(&t);
	points_arr = scater_points(point_type, amount_each_element, points->points);
	realloc_lefover_points(leftover, &points_arr, amount_with_lefover, points);
	good = parallel_calculate_points_location(points_arr, amount_with_lefover, device_prop, t);

	if (good == 0)
	{
		return 0;
	}

	points_recv_buffer = (Point*)calloc(points->size, sizeof(Point));//realloc shouldnt free
	gather_points(point_type, amount_each_element, points_arr, points_recv_buffer);
	copy_lefover_points(leftover, points_recv_buffer, points_arr, points, amount_with_lefover);
	free_point_array(points->points);
	free_point_array(points_arr);
	points->points = points_recv_buffer;

	return 1;
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
		for (int i = leftover; i > 0; i--)
		{
			(*points_arr)[amount_with_lefover - i] = points->points[points->size - i];
		}
	}
}

int parallel_calculate_points_location(Point* points_arr, int amount, cudaDeviceProp device_prop, double t)
{
	Point* cuda_points = 0;
	Point* cuda_points_leftover = 0;

	int cuda_points_amount;
	int cuda_points_leftover_amount;
	int good;
	int amount_omp = amount / 2;
	int amount_cuda = amount_omp + (amount % 2);

	Point* omp_offset = points_arr + amount_cuda;
	good = cuda_start_points_location_calculation(points_arr, amount_cuda, device_prop, &cuda_points, &cuda_points_amount, &cuda_points_leftover, &cuda_points_leftover_amount, t);
	printf("%d %d\n", cuda_points, cuda_points_leftover);
	fflush(stdout);
	if (good == 0)
	{
		return 0;
	}

#pragma omp parallel for shared(omp_offset)
	for (int i = 0; i < amount_omp; i++)
	{
		calculate_point_position(omp_offset + i, t);
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
		return 0;
	}

	*amount_cuda_points = thread_per_block * req_blocks;
	*amount_cuda_points_leftover = leftover_threads * leftover_block;
	
	good = cuda_malloc_elements((void**)cuda_points, *amount_cuda_points, (void**)cuda_points_leftover, *amount_cuda_points_leftover, sizeof(Point));
	if (good == 0)
	{
		return 0;
	}

	Point* leftover_points_start = (points + *(amount_cuda_points));
	Point* points_start = (points);

	good = cuda_copy_elements((void*)(*cuda_points), (void*)points_start, *amount_cuda_points, (void*)(*cuda_points_leftover), (void*)leftover_points_start, *amount_cuda_points_leftover, sizeof(Point), cudaMemcpyHostToDevice);

	if (good == 0)
	{
		return 0;
	}

	good = cuda_kernel_points_calculation(*cuda_points, *cuda_points_leftover, thread_per_block, req_blocks, leftover_threads, leftover_block, t);
	
	if (good == 0)
	{
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

	cudaStatus = cudaFree(cuda_points);
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}
	cudaStatus = cudaFree(cuda_points_leftover);
	if (cudaStatus != cudaSuccess)
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

#pragma omp parallel for shared(clusters_center,new_cluster)
	for (int i = 0; i < cluster_amount; i++)
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

#pragma omp parallel for shared(new_cluster)
	for (int i = 0; i < cluster_amount; i++)
	{
		if (new_cluster[i].size > 0)
		{
			free(new_cluster[i].cluster_points);
		}
#pragma omp critical
		{
			new_cluster[i].cluster_points = (Point*)calloc(points->size, sizeof(Point));
			new_cluster[i].size = 0;
		}

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

int slave_group_points_to_clusters(MPI_Datatype axis_type, MPI_Datatype point_type, cudaDeviceProp device_prop)// !!!!!!!!!!!!
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

	int total_size_buffer;
	char* p_buffer;//freed
	int* cluster_points_amount_arr;//freed
	int* cluster_points_offset_arr;//freed

	broadcast_value(&buffer_size, 1, MPI_INT);

	buffer = (char*)calloc(buffer_size, sizeof(char));

	broadcast_value(buffer, buffer_size, MPI_PACKED);

	broadcast_value(&amount_each_element,1,MPI_INT);

	unpack_elements(&position, buffer, buffer_size, &cluster_amount, 1, MPI_INT);
	
	clusters_center = (Axis*)calloc(cluster_amount, sizeof(Axis));
	unpack_elements(&position, buffer, buffer_size, clusters_center, cluster_amount, axis_type);

	points_arr = scater_points(point_type, amount_each_element, 0);

	cluster_parallel = group_points_to_clusters(clusters_center, cluster_amount, points_arr, amount_each_element, device_prop);

	send_elements(&my_id, 1, MPI_INT, MASTER, 0);
	
	total_size_buffer = sizeof(int)*cluster_amount*2;
	total_size_buffer += sizeof(Point)*amount_each_element;
	cluster_points_amount_arr = (int*)calloc(cluster_amount, sizeof(int));
	cluster_points_offset_arr = (int*)calloc(cluster_amount, sizeof(int));

#pragma omp parallel for shared(cluster_points_amount_arr,cluster_parallel)
	for (int i = 0; i < cluster_amount; i++)
	{
		cluster_points_amount_arr[i] = cluster_parallel[i].size;
	}

	init_cluster_points_offset(cluster_points_amount_arr, cluster_points_offset_arr, cluster_amount);

	p_buffer = (char*)calloc(total_size_buffer, sizeof(char));

	position = 0;

	pack_elements(&position, p_buffer, total_size_buffer, cluster_points_amount_arr, cluster_amount, MPI_INT);//
	pack_elements(&position, p_buffer, total_size_buffer, cluster_points_offset_arr, cluster_amount, MPI_INT);//

#pragma omp parallel for shared(cluster_parallel,cluster_points_offset_arr,p_buffer,total_size_buffer)
	for (int i = 0; i < cluster_amount; i++)
	{
		int points_amount = cluster_parallel[i].size;
		int current_position = position + cluster_points_offset_arr[i]*sizeof(Point);
		pack_elements(&current_position, p_buffer, total_size_buffer, cluster_parallel[i].cluster_points, points_amount, point_type);
		free(cluster_parallel[i].cluster_points);
	}

	send_elements(&total_size_buffer, 1, MPI_INT, MASTER, 0);
	send_elements(p_buffer, total_size_buffer, MPI_PACKED, MASTER, 0);

	free(buffer);
	free(cluster_parallel);
	free(clusters_center);
	free(points_arr);
	free(p_buffer);
	free(cluster_points_amount_arr);
	free(cluster_points_offset_arr);

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

	#pragma omp parallel for shared(cluster_parallel_arr,points_amount)
	for (int i = 0; i < cluster_amount; i++)
	{
		cluster_parallel_arr[i].size = 0;
	#pragma omp critical
		{
			cluster_parallel_arr[i].cluster_points = (Point*)calloc(points_amount, sizeof(Point));
		}
	}
	return cluster_parallel_arr;
}

int* cuda_group_points_to_clusters(Point* points,int points_amount,Axis* clusters_center_axis, int cluster_amount, cudaDeviceProp device_prop)
{
	int* cuda_cluster_of_points = 0;//freed
	int* cuda_cluster_of_points_leftover = 0;//freed
	int* cluster_of_points;//shouldnt free
	int* cluster_of_points_leftover_offset;//shouldnt free
	Point* cuda_points = 0;//freed
	Point* cuda_points_leftover = 0;//freed
	Axis* cuda_clusters_center_axis = 0;//freed
	cudaError_t cudaStatus;
	int thread_per_block;
	int req_blocks;
	int leftover_threads;
	int leftover_block;
	int amount_cuda_points;
	int amount_cuda_points_leftover;
	int good;
	
	good = cuda_blocks_calculation(&thread_per_block, &req_blocks, &leftover_threads, &leftover_block, device_prop, points_amount);
	if (good == 0)
	{
		return 0;
	}
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

	cudaStatus = cudaFree(cuda_cluster_of_points);
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}

	cudaStatus = cudaFree(cuda_cluster_of_points_leftover);
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}
	cudaStatus = cudaFree(cuda_points);
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}
	cudaStatus = cudaFree(cuda_points_leftover);
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}
	cudaStatus = cudaFree(cuda_clusters_center_axis);
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}

	return cluster_of_points;
}

void recieve_points_from_slaves(Cluster* new_cluster,int clusters_amount, int num_of_proc, MPI_Datatype point_type)
{
	int proc = num_of_proc - 1;
	int current_proc;

	while (proc > 0)
	{

		recieve_elements(&current_proc, 1, MPI_INT, MPI_ANY_SOURCE,MPI_ANY_TAG);
		unpack_clusters_points_from_slaves(new_cluster, clusters_amount, point_type, current_proc);
		proc--;
	}
}

void unpack_clusters_points_from_slaves(Cluster* new_cluster, int cluster_amount,MPI_Datatype point_type,int src)
{
	int position = 0;
	int points_amount;
	int buffer_size;
	int current_position;
	char* buffer;//freed
	int start_offset;
	Point* points_offset;
	int* cluster_points_amount_arr;//freed
	int* cluster_points_offset_arr;//freed
	
	recieve_elements(&buffer_size, 1, MPI_INT, src,0);

	buffer = (char*)calloc(buffer_size, sizeof(char));
	cluster_points_amount_arr = (int*)calloc(cluster_amount, sizeof(int));
	cluster_points_offset_arr = (int*)calloc(cluster_amount, sizeof(int));

	recieve_elements(buffer, buffer_size, MPI_PACKED, src, 0);

	unpack_elements(&position, buffer, buffer_size, cluster_points_amount_arr, cluster_amount, MPI_INT);

	unpack_elements(&position, buffer, buffer_size, cluster_points_offset_arr, cluster_amount, MPI_INT);

#pragma omp parallel for shared(position, buffer,buffer_size,cluster_points_offset_arr,cluster_points_amount_arr) private(points_amount,start_offset,points_offset,current_position)
	for (int i = 0; i < cluster_amount; i++)
	{
			points_amount = cluster_points_amount_arr[i];
			if (points_amount != 0)
			{
				start_offset = new_cluster[i].size;

				points_offset = new_cluster[i].cluster_points + start_offset;

				current_position = position + cluster_points_offset_arr[i]*sizeof(Point);

				unpack_elements(&current_position, buffer, buffer_size, points_offset, points_amount, point_type);

				new_cluster[i].size += points_amount;
			}
	}

	free(cluster_points_amount_arr);
	free(cluster_points_offset_arr);
	free(buffer);
}

void shrink_clusters_points(Cluster* new_cluster, int clusters_amount)
{
#pragma omp parallel for shared(new_cluster)//may produce error because realloc
	for (int i = 0; i < clusters_amount; i++)
	{
		new_cluster[i].cluster_points = (Point*)realloc(new_cluster[i].cluster_points, sizeof(Point)*new_cluster[i].size);
	}
}

void calculate_cluster_center(Cluster* new_cluster, Clusters* clusters, int amount)
{
#pragma omp parallel for shared(new_cluster,clusters)
	for (int i = 0; i < amount; i++)
	{
		//printf("%d\n", omp_get_thread_num());
		//fflush(stdout);
		if (new_cluster[i].size > 0)
		{
			new_cluster[i].center = axis_avg(new_cluster[i].cluster_points, new_cluster[i].size);
		}
		else
		{
			new_cluster[i].center = clusters->clusters[i].center;
		}
	}
}

Axis axis_avg(Point* points, int amount)
{
	double sum_x = 0, sum_y = 0, sum_z = 0;
	Axis center_axis;

	for (int i = 0; i < amount; i++)//maybe omp
	{
		Point* p = (points + i);
		sum_x += p->axis_location.x;
		sum_y += p->axis_location.y;
		sum_z += p->axis_location.z;
	}

	center_axis.x = sum_x / amount;
	center_axis.y = sum_y / amount;
	center_axis.z = sum_z / amount;
	return center_axis;
}

int master_check_points_transfer(Cluster * original_cluster, Cluster * new_cluster, int cluster_amount, int points_amount, int num_of_proc, MPI_Datatype transfer_points_type, cudaDeviceProp device_prop)
{
	int* original_cluster_points_amount_arr = 0;//freed
	int* original_cluster_points_offset_arr = 0;//freed
	int* new_cluster_points_amount_arr = 0;//freed
	int* new_cluster_points_offset_arr = 0;//freed
	char* buffer = 0;//freed
	Transfer_Point* transfer_points = 0;//freed
	Transfer_Point* transfer_points_per_element = 0;//freed
	int* is_transfered_arr = 0;//freed
	int** points_id = 0;//freed
	int buffer_size;
	int flag = CHECK_TRANSFER_POINTS;
	int amount_each_element;
	int amount_leftover_element;
	int amount_each_element_with_leftover;
	int is_transfered;

	
	
	if (pre_check_points_transfer(original_cluster, new_cluster, cluster_amount) > 0)
	{
		return 1;
	}

	transfer_points = (Transfer_Point*)calloc(points_amount, sizeof(Transfer_Point));
	
	is_transfered_arr = (int*)calloc(num_of_proc, sizeof(int));

	//original_cluster_points_amount_arr = (int*)calloc(cluster_amount, sizeof(int));//
	//original_cluster_points_offset_arr = (int*)calloc(cluster_amount, sizeof(int));//
	
	//new_cluster_points_amount_arr = (int*)calloc(cluster_amount, sizeof(int));//
	//new_cluster_points_offset_arr = (int*)calloc(cluster_amount, sizeof(int));//

	//buffer_size = points_amount * sizeof(int);//
	//buffer = (char*)calloc(buffer_size,sizeof(char));//

/*#pragma omp parallel for shared(original_cluster_points_amount_arr,new_cluster_points_amount_arr,original_cluster)
	for (int i = 0; i < cluster_amount; i++)
	{
		original_cluster_points_amount_arr[i] = original_cluster[i].size;
		new_cluster_points_amount_arr[i] = new_cluster[i].size;
	}*/

	buffer = malloc_packed_buffer(points_amount,sizeof(int),&buffer_size);

	prepare_for_pack_elements(original_cluster,&original_cluster_points_amount_arr, &original_cluster_points_offset_arr, cluster_amount);
	prepare_for_pack_elements(new_cluster,&new_cluster_points_amount_arr, &new_cluster_points_offset_arr, cluster_amount);
	
	points_id = malloc_points_id(cluster_amount, original_cluster_points_amount_arr);

#pragma omp parallel for shared(original_cluster_points_offset_arr,original_cluster,buffer,buffer_size)
	for (int i = 0; i < cluster_amount; i++)
	{
		int current_position = original_cluster_points_offset_arr[i]*sizeof(int);
		for (int j = 0; j < original_cluster[i].size; j++)
		{
			Point* point = (original_cluster[i].cluster_points + j);
			points_id[i][j] = point->id;
			pack_elements(&current_position, buffer, buffer_size, &(point->id), 1, MPI_INT);
		}
	}

	check_amounts_leftover(&amount_each_element, points_amount, num_of_proc, &amount_leftover_element);
	amount_each_element_with_leftover = amount_each_element + amount_leftover_element;

	//handle special elments cluster_index--point_id **make new struct 
#pragma omp parallel for shared(new_cluster_points_amount_arr,new_cluster_points_offset_arr,transfer_points,new_cluster)
	for (int i = 0; i < cluster_amount; i++)
	{
		int p_amount = new_cluster_points_amount_arr[i];
		int p_offset = new_cluster_points_offset_arr[i];

		for (int j = 0; j < p_amount; j++)
		{
			Point* p = (new_cluster[i].cluster_points +j);
			*(transfer_points + p_offset + j) = Transfer_Point{ i,p->id };
		}
	}

	/*for (int i = 0; i < points_amount; i++)//can delete
	{
		printf("cluster_id %d point_id %d\n", transfer_points[i].cluster_id, transfer_points[i].point_id);
		fflush(stdout);
	}*/

	broadcast_flag(&flag);
	broadcast_value(&cluster_amount,1,MPI_INT);
	broadcast_value(original_cluster_points_amount_arr, cluster_amount, MPI_INT);
	broadcast_value(original_cluster_points_offset_arr, cluster_amount, MPI_INT);
	broadcast_value(&buffer_size, 1, MPI_INT);
	broadcast_value(buffer, buffer_size, MPI_PACKED);
	broadcast_value(&amount_each_element, 1, MPI_INT);
	
	//scater special elments
	transfer_points_per_element = (Transfer_Point*) scater_elements(transfer_points, amount_each_element, sizeof(Transfer_Point), transfer_points_type);

	realloc_lefover_transfer_point(amount_leftover_element, &transfer_points_per_element, amount_each_element_with_leftover, transfer_points, points_amount);
	
	is_transfered = check_points_transfer(points_id,cluster_amount, original_cluster_points_amount_arr,transfer_points_per_element,amount_each_element_with_leftover,device_prop);
	
	if (is_transfered == -1)
	{
		return -1;
	}

	is_transfered = is_transfered != 0;

	//recieve answers from slaves
	gather_elements(is_transfered_arr, &is_transfered, 1, MPI_INT);

#pragma omp parallel for shared(is_transfered, is_transfered_arr)
	for (int i = 0; i < cluster_amount; i++)
	{
		if (is_transfered_arr[i] == 1)
		{
			is_transfered = 1;
		}
	}
	
	free(original_cluster_points_amount_arr);
	free(original_cluster_points_offset_arr);
	free(new_cluster_points_amount_arr);
	free(new_cluster_points_offset_arr);
	free(buffer);
	free(transfer_points);
	free(transfer_points_per_element);
	free(is_transfered_arr);
	free_points_id(points_id, cluster_amount);


	return is_transfered;
}
int slave_check_points_transfer(MPI_Datatype transfer_points_type,cudaDeviceProp device_prop)
{
	int cluster_amount;
	int buffer_size;
	int amount_each_element;
	int is_transfered;
	int* cluster_points_amount_arr;//freed
	int* cluster_points_offset_arr;//freed
	int** points_id;//freed
	char* buffer;//freed
	Transfer_Point* transfer_points_per_element;//freed

	broadcast_value(&cluster_amount, 1, MPI_INT);

	cluster_points_amount_arr = (int*)calloc(cluster_amount, sizeof(int));
	cluster_points_offset_arr = (int*)calloc(cluster_amount, sizeof(int));
	broadcast_value(cluster_points_amount_arr, cluster_amount, MPI_INT);
	broadcast_value(cluster_points_offset_arr, cluster_amount, MPI_INT);
	
	/*for (int i = 0; i < cluster_amount; i++)//can delete
	{
	printf("cluster_amount %d cluster_offset %d\n", cluster_points_amount_arr[i], cluster_points_offset_arr[i]);
	fflush(stdout);
	}*/

	broadcast_value(&buffer_size, 1, MPI_INT);

	buffer = (char*)calloc(buffer_size, sizeof(char));
	broadcast_value(buffer, buffer_size, MPI_PACKED);
	
	broadcast_value(&amount_each_element, 1, MPI_INT);

	transfer_points_per_element = (Transfer_Point*) scater_elements(0, amount_each_element, sizeof(Transfer_Point), transfer_points_type);
	
	points_id = malloc_points_id(cluster_amount, cluster_points_amount_arr);
	

#pragma omp parallel for shared(cluster_points_offset_arr,cluster_points_amount_arr,buffer,buffer_size)
	for (int i = 0; i < cluster_amount; i++)//might be merge with upper for loop
	{
		int current_position = cluster_points_offset_arr[i]*sizeof(int);
		int points_amount = cluster_points_amount_arr[i];
		unpack_elements(&current_position, buffer, buffer_size, points_id[i], points_amount, MPI_INT);
	}

	/*for (int i = 0; i < cluster_amount; i++)//can delete
	{
		for (int j = 0; j < cluster_points_amount_arr[i]; j++)
		{
			printf("point_id %d my_id %d cluster %d\n", points_id[i][j], my_id,i);
			fflush(stdout);
		}
	}*/

	

	is_transfered = check_points_transfer(points_id,cluster_amount, cluster_points_amount_arr, transfer_points_per_element,amount_each_element,device_prop);

	if (is_transfered == -1)
	{
		return 0;
	}

	is_transfered = is_transfered != 0;

	gather_elements(0, &is_transfered, 1, MPI_INT);

	free(cluster_points_amount_arr);
	free(cluster_points_offset_arr);
	free(buffer);
	free(transfer_points_per_element);
	free_points_id(points_id, cluster_amount);

	return 1;
}

int pre_check_points_transfer(Cluster * original_cluster, Cluster * new_cluster, int cluster_amount)
{
	int good = 0;
#pragma omp parallel for shared(original_cluster,new_cluster) reduction (+:good)
	for (int i = 0; i < cluster_amount; i++)
	{
		if (original_cluster[i].size != new_cluster[i].size)
		{
			good = 1;
		}
		else
		{
			good = 0;
		}
	}
	
	return good;
}
void prepare_for_pack_elements(Cluster* clusters,int** cluster_points_amount_arr, int** cluster_points_offset_arr, int cluster_amount)
{
	*cluster_points_amount_arr = (int*)calloc(cluster_amount, sizeof(int));//
	*cluster_points_offset_arr = (int*)calloc(cluster_amount, sizeof(int));//

#pragma omp parallel for shared(clusters,cluster_points_amount_arr)
	for (int i = 0; i < cluster_amount; i++)
	{
		cluster_points_amount_arr[0][i] = clusters[i].size;//
	}

	init_cluster_points_offset(*cluster_points_amount_arr, *cluster_points_offset_arr, cluster_amount);

}

void init_cluster_points_offset(int* cluster_points_amount_arr, int* cluster_points_offset_arr, int cluster_amount)
{
	for (int i = 0; i < cluster_amount; i++)//cant omp
	{
		if (i == 0)
		{
			cluster_points_offset_arr[i] = 0;///
		}
		else
		{
			for (int j = 0; j < i; j++)
			{
				cluster_points_offset_arr[i] += cluster_points_amount_arr[j];///
			}
		}
	}
}

char* malloc_packed_buffer(int amount, int size_of,int* buffer_size) {
	char* buffer;
	*buffer_size = amount * size_of;
	buffer = (char*)calloc(*buffer_size, sizeof(char));
	return buffer;
}

int** malloc_points_id(int cluster_amount,int* cluster_points_amount_arr)
{
	int** points_id;
	points_id = (int**)calloc(cluster_amount, sizeof(int*));

	for (int i = 0; i < cluster_amount; i++)//cant omp because calloc
	{
		points_id[i] = (int*)calloc(cluster_points_amount_arr[i], sizeof(int));
	}

	return points_id;
}

void free_points_id(int** points_id,int cluster_amount)
{
	for (int i = 0; i < cluster_amount; i++)
	{
		free(points_id[i]);
	}

	free(points_id);
}

int check_points_transfer(int** points_id, int cluster_amount, int* cluster_points_amount, Transfer_Point* transfer_points, int amount, cudaDeviceProp device_prop)
{
	int thread_per_block;
	int req_block;
	int leftover_threads;
	int leftover_block;
	int* cuda_result_pointer = 0;//freed
	int result_of_cuda;
	int result_of_omp = 0;
	int amount_omp = amount / 2;
	int amount_cuda = amount_omp + (amount % 2);
	int good;
	int** cuda_points_id = 0;//freed
	int* cuda_cluster_points_amount = 0;//freed
	int** cuda_points_id_arr_pointers = 0;//freed
	Transfer_Point* cuda_transfer_points = 0;//freed
	Transfer_Point* cuda_transfer_points_leftover = 0;//freed
	Transfer_Point* omp_offset = transfer_points + amount_cuda;//shouldnt free

	cuda_points_id_arr_pointers = (int**)calloc(cluster_amount, sizeof(int*));

	good = cuda_blocks_calculation(&thread_per_block, &req_block, &leftover_threads, &leftover_block, device_prop, amount_cuda);
	if (good == 0)
	{
		return -1;
	}

	good = cuda_start_check_points_transfer(points_id, cluster_amount, cluster_points_amount, &cuda_cluster_points_amount, cuda_points_id_arr_pointers, transfer_points, &cuda_result_pointer, &cuda_points_id, &cuda_transfer_points, &cuda_transfer_points_leftover, thread_per_block, req_block, leftover_threads, leftover_block);
	if (good == 0)
	{
		return -1;
	}

	//omp loop
	result_of_omp = 0;
#pragma omp parallel for shared(omp_offset,points_id,cluster_points_amount) reduction (+:result_of_omp)
	for (int i = 0; i < amount_omp; i++)
	{
		Transfer_Point* current_transfer_point = omp_offset+i;
		int cluster_id = current_transfer_point->cluster_id;
		int point_id = current_transfer_point->point_id;
		result_of_omp = !(check_exists_points_transfer(points_id[cluster_id], cluster_points_amount[cluster_id], point_id) == 1);
		if (result_of_omp == 1)
		{
			printf("in mop with point %d\n", point_id);
			fflush(stdout);
		}
		printf("in omp id of point %d id of cluster %d id %d\n", point_id, cluster_id, my_id);
		fflush(stdout);
	}

	printf("result of omp %d id %d\n", result_of_omp,my_id);
	fflush(stdout);

	good = cuda_end_check_points_transfer(cuda_points_id, cluster_amount, cuda_points_id_arr_pointers, cuda_transfer_points, cuda_transfer_points_leftover, cuda_result_pointer,cuda_cluster_points_amount, &result_of_cuda);
	
	printf("result of cuda %d id %d\n", result_of_cuda, my_id);
	fflush(stdout);

	if (good == 0)
	{
		return -1;
	}

	free(cuda_points_id_arr_pointers);
	
	return result_of_cuda + result_of_omp;


}

void realloc_lefover_transfer_point(int leftover, Transfer_Point** transfer_points_arr, int amount_with_lefover, Transfer_Point* transfer_points,int transfer_points_amount)
{
	if (leftover > 0)
	{
		*transfer_points_arr = (Transfer_Point*)realloc(*transfer_points_arr, sizeof(Transfer_Point)*amount_with_lefover);//freed
#pragma omp parallel for shared(transfer_points, transfer_points_arr)
		for (int i = leftover; i > 0; i--)
		{
			(*transfer_points_arr)[amount_with_lefover - i] = transfer_points[transfer_points_amount - i];
		}
	}
}

int cuda_start_check_points_transfer(int** points_id, int cluster_amount, int* cluster_points_amount,int** cuda_points_id_arr_pointers,int** cuda_cluster_points_amount, Transfer_Point* transfer_points,int** result, int*** cuda_points_id, Transfer_Point** cuda_transfer_points, Transfer_Point** cuda_transfer_points_leftover, int thread_per_block, int req_block, int leftover_threads, int leftover_block)
{
	Transfer_Point* transfer_points_offset_leftover = 0;
	int good;
	int res = 0;
	int transfer_points_amount_leftover;

	/*good = cuda_malloc_elements((void**)cuda_points_id, cluster_amount, 0, 0, sizeof(int*));
	if (good == 0)
	{
		return good;
	}*/

	good = cuda_malloc_elements((void**)result, 1, 0, 0, sizeof(int));
	if (good == 0)
	{
		return good;
	}

	good = cuda_copy_elements(*result, &res, 1, 0, 0, 0, sizeof(int), cudaMemcpyHostToDevice);
	if (good == 0)
	{
		return good;
	}

	cuda_copy_matrix(cluster_points_amount, cluster_amount, (void***)cuda_points_id, (void**)points_id, sizeof(int*), sizeof(int), (void**)cuda_points_id_arr_pointers);
	
	/*for (int i = 0; i < cluster_amount; i++)//maybe omp
	{
		int amount = cluster_points_amount[i];
		int* cuda_points_id_arr;
		good = cuda_malloc_elements((void**)&cuda_points_id_arr, amount, 0, 0, sizeof(int));
		cuda_points_id_arr_pointers[i] = cuda_points_id_arr;
		if (good == 0)
		{
			return good;
		}

		good = cuda_copy_elements((void*)((*cuda_points_id) + i), &cuda_points_id_arr, 1, 0, 0, 0, sizeof(int*), cudaMemcpyHostToDevice);
		if (good == 0)
		{
			return good;
		}

		good = cuda_copy_elements(cuda_points_id_arr, points_id[i], amount, 0, 0, 0, sizeof(int), cudaMemcpyHostToDevice);
		if (good == 0)
		{
			return good;
		}
	}*/
	
	good = cuda_malloc_elements((void**)cuda_transfer_points, req_block*thread_per_block,(void**)cuda_transfer_points_leftover, leftover_block*leftover_threads, sizeof(Transfer_Point));
	if (good == 0)
	{
		return good;
	}

	good = cuda_malloc_elements((void**)cuda_cluster_points_amount, cluster_amount, 0, 0, sizeof(int));
	if (good == 0)
	{
		return good;
	}

	good = cuda_copy_elements(*cuda_cluster_points_amount, cluster_points_amount, cluster_amount, 0, 0, 0, sizeof(int), cudaMemcpyHostToDevice);
	if (good == 0)
	{
		return good;
	}

	transfer_points_offset_leftover = transfer_points + req_block*thread_per_block;
	transfer_points_amount_leftover = leftover_block*leftover_threads;

	good = cuda_copy_elements(*cuda_transfer_points, transfer_points, req_block*thread_per_block, *cuda_transfer_points_leftover, transfer_points_offset_leftover, transfer_points_amount_leftover, sizeof(Transfer_Point), cudaMemcpyHostToDevice);
	if (good == 0)
	{
		return good;
	}

	good = cuda_kernel_check_transfer_points(*result, *cuda_points_id,*cuda_cluster_points_amount,cluster_amount, *cuda_transfer_points, *cuda_transfer_points_leftover, thread_per_block, req_block, leftover_threads, leftover_block);
	if (good == 0)
	{
		return good;
	}

	return 1;
}

int cuda_end_check_points_transfer(int** cuda_points_id, int cluster_amount, int** cuda_points_id_arr_pointers, Transfer_Point* cuda_transfer_points, Transfer_Point* cuda_transfer_points_leftover, int* cuda_result_pointer,int* cuda_cluster_points_amount, int* result_of_cuda)
{
	int good;
	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();// error check
	
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}

	good = cuda_copy_elements(result_of_cuda, cuda_result_pointer, 1, 0, 0, 0, sizeof(int), cudaMemcpyDeviceToHost);
	if (good == 0)
	{
		return good;
	}

//#pragma omp parallel for shared (cuda_points_id,good)
//	for (int i = 0; i < cluster_amount; i++)
//	{
//		cudaStatus = cudaFree(cuda_points_id_arr_pointers[i]);
//		if (cudaStatus != cudaSuccess)
//		{
//			good = 0;
//		}
//	}
	

	if (good == 0)
	{
		return good;
	}

	

	cudaStatus = cudaFree(cuda_cluster_points_amount);
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}

	cudaStatus = cudaFree(cuda_points_id);
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}
	cudaStatus = cudaFree(cuda_transfer_points);
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}
	cudaStatus = cudaFree(cuda_transfer_points_leftover);
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}
	cudaStatus = cudaFree(cuda_result_pointer);
	if (cudaStatus != cudaSuccess)
	{
		return 0;
	}

	return 1;
}

int check_exists_points_transfer(int* points_id, int points_id_amount, int transfer_point_id)
{
	int good = 0;
	for (int i = 0; i < points_id_amount; i++)
	{
		if (points_id[i] == transfer_point_id)
		{
			return 1;
		}
	}
	return 0;
}

double master_evaluate_qm(Cluster* clusters, int cluster_amount,int point_amount,int num_of_proc,MPI_Datatype axis_type, MPI_Datatype qm_point_type)
{
	char* buffer;//should free
	int buffer_size;
	int* cluster_points_amount;//shouyld free
	int* cluster_points_offset;//should free
	int amount_each_proc;
	int amount_each_proc_leftover;
	int amount_each_proc_with_leftover;
	int flag = EVALUATE_QM;
	Qm_Point* qm_points;//should free
	Qm_Point* qm_points_arr;//should free

	prepare_for_pack_elements(clusters, &cluster_points_amount, &cluster_points_offset, cluster_amount);
	
	buffer = malloc_packed_buffer(point_amount, sizeof(Axis), &buffer_size);

	qm_points = (Qm_Point*)calloc(point_amount, sizeof(Qm_Point));
	
#pragma omp parallel for shared(clusters,cluster_points_offset,cluster_points_amount,buffer,buffer_size,axis_type)
	for (int i = 0; i < cluster_amount; i++)
	{
		int current_cluster_offset = cluster_points_offset[i];
		int current_cluster_offset_char = current_cluster_offset * sizeof(Axis);
		int points_amount = cluster_points_amount[i];
		
		
		for (int j = 0; j < points_amount; j++)
		{
			Axis point_axis = clusters[i].cluster_points[j].axis_location;
			
			Qm_Point* qm_point = (qm_points + current_cluster_offset + j);
			qm_point->cluster_id = i;
			qm_point->point_loc = point_axis;
			pack_elements(&current_cluster_offset_char, buffer, buffer_size, &point_axis, 1, axis_type);
		}
	}

	check_amounts_leftover(&amount_each_proc, point_amount, num_of_proc, &amount_each_proc_leftover);
	
	amount_each_proc_with_leftover = amount_each_proc + amount_each_proc_leftover;

	//broad flag
	broadcast_flag(&flag);
	//broad cluster_amount//
	broadcast_value(&cluster_amount, 1, MPI_INT);
	//broad cluster_points_amount//
	broadcast_value(cluster_points_amount, cluster_amount, MPI_INT);
	//broad cluster_points_offset//
	broadcast_value(cluster_points_offset, cluster_amount, MPI_INT);
	//broad buffer_size//
	broadcast_value(&buffer_size, 1, MPI_INT);
	//broad buffer//
	broadcast_value(buffer, buffer_size, MPI_PACKED);
	//broad amount_each_proc
	broadcast_value(&amount_each_proc, 1, MPI_INT);

	qm_points_arr = (Qm_Point*)scater_elements(qm_points, amount_each_proc, sizeof(Qm_Point), qm_point_type);

	realloc_lefover_qm_points(amount_each_proc_leftover, &qm_points_arr, amount_each_proc_with_leftover, qm_points, point_amount);
	
	/*for (int i = 0; i < amount_each_proc_with_leftover; i++)
	{
		printf("id %d cluster %d x %lf y %lf z %lf\n", my_id, qm_points_arr[i].cluster_id, qm_points_arr[i].point_loc.x, qm_points_arr[i].point_loc.y, qm_points_arr[i].point_loc.z);
		fflush(stdout);
	}*/




}

int slave_evaluate_qm(MPI_Datatype axis_type, MPI_Datatype qm_point_type)
{
	char* buffer;//should free
	int buffer_size;
	int cluster_amount;
	int amount_each_proc;
	int* cluster_points_amount;//shouyld free
	int* cluster_points_offset;//should free
	Axis** cluster_points_axis;//should free
	Qm_Point* qm_points_arr;//should free



	broadcast_value(&cluster_amount, 1, MPI_INT);

	cluster_points_amount = (int*)calloc(cluster_amount, sizeof(int));
	broadcast_value(cluster_points_amount,cluster_amount,MPI_INT);

	cluster_points_offset = (int*)calloc(cluster_amount, sizeof(int));
	broadcast_value(cluster_points_offset, cluster_amount, MPI_INT);

	broadcast_value(&buffer_size, 1, MPI_INT);

	buffer = (char*)calloc(buffer_size, sizeof(char));
	broadcast_value(buffer, buffer_size, MPI_PACKED);

	broadcast_value(&amount_each_proc, 1, MPI_INT);

	qm_points_arr = (Qm_Point*) scater_elements(0, amount_each_proc, sizeof(Qm_Point), qm_point_type);

	malloc_cluster_points(cluster_points_amount, (void***)&cluster_points_axis, cluster_amount, sizeof(Axis*), sizeof(Axis));
	
#pragma omp parallel for shared(cluster_points_offset,cluster_points_amount,buffer,buffer_size,cluster_points_axis,axis_type)
	for (int i = 0; i < cluster_amount; i++)
	{
		int current_cluster_offset = cluster_points_offset[i] * sizeof(Axis);
		int points_amount = cluster_points_amount[i];
		unpack_elements(&current_cluster_offset, buffer, buffer_size, cluster_points_axis[i], points_amount, axis_type);
	}

	

	return 1;

}

double* find_max_diameter(Axis** cluster_points_axis, Qm_Point* qm_points_arr, int* cluster_points_amount, int amount_each_proc, int cluster_amount,cudaDeviceProp device_prop)//put content in function
{
	Axis** cuda_cluster_points_axis = 0;//freed
	Axis** cuda_cluster_points_axis_pointer = 0;//freed
	Qm_Point* cuda_qm_points_arr = 0;//freed
	Qm_Point* cuda_qm_points_arr_leftover = 0;//freed
	Qm_Point* qm_points_lefover_offset;
	Max_point_Diameter* cuda_max_qm_points_arr = 0;//freed
	Max_point_Diameter* cuda_max_qm_points_arr_leftover = 0;//freed
	Max_point_Diameter* max_qm_points_arr = 0;//freed
	Max_point_Diameter* max_qm_points_offset = 0;
	double* max_cluster;//shouldnt free

	int thread_per_block;
	int req_block;
	int leftover_threads;
	int leftover_block;
	int good;
	int cuda_amount;
	int cuda_amount_leftover;

	good = cuda_blocks_calculation(&thread_per_block, &req_block, &leftover_threads, &leftover_block, device_prop, amount_each_proc);
	if (good == 0)
	{
		return 0;
	}

	cuda_amount = thread_per_block*req_block;
	cuda_amount_leftover = leftover_threads*leftover_block;
	qm_points_lefover_offset = qm_points_arr + cuda_amount;
	
	good = cuda_copy_matrix(cluster_points_amount, cluster_amount,(void***) &cuda_cluster_points_axis, (void**)cluster_points_axis, sizeof(Axis*), sizeof(Axis), (void**)cuda_cluster_points_axis_pointer);
	if (good == 0)
	{
		return 0;
	}

	good = cuda_malloc_elements((void**)&cuda_qm_points_arr, cuda_amount,(void**) &cuda_qm_points_arr_leftover, cuda_amount_leftover, sizeof(Qm_Point));
	if (good == 0)
	{
		return 0;
	}

	good = cuda_copy_elements(cuda_qm_points_arr, cluster_points_axis, cuda_amount, cuda_qm_points_arr_leftover, qm_points_lefover_offset, cuda_amount_leftover, sizeof(Qm_Point), cudaMemcpyHostToDevice);
	if (good == 0)
	{
		return 0;
	}

	good = cuda_malloc_elements((void**)&cuda_max_qm_points_arr, cuda_amount, (void**)&cuda_max_qm_points_arr_leftover, cuda_amount_leftover, sizeof(Max_point_Diameter));
	if (good == 0)
	{
		return 0;
	}

	//activate cuda calculation

	max_qm_points_arr = (Max_point_Diameter*)calloc(amount_each_proc, sizeof(Max_point_Diameter));
	max_qm_points_offset = max_qm_points_arr + cuda_amount;

	good = cuda_copy_elements(max_qm_points_arr, cuda_max_qm_points_arr, cuda_amount, max_qm_points_offset, cuda_max_qm_points_arr_leftover, cuda_amount_leftover, sizeof(Max_point_Diameter), cudaMemcpyDeviceToHost);
	if (good == 0)
	{
		return 0;
	}

	max_cluster = (double*)calloc(cluster_amount, sizeof(double));

#pragma omp parallel for shared(amount_each_proc,max_qm_points_arr,max_cluster)
	for (int i = 0; i < cluster_amount; i++)
	{
		for (int j = 0; j < amount_each_proc; j++)
		{
			if (max_qm_points_arr[j].cluster_id == i && max_qm_points_arr[j].max_diameter > max_cluster[i])
			{
				max_cluster[i] = max_qm_points_arr[j].max_diameter;
			}
		}
	}

	cuda_free_matrix((void**)cuda_cluster_points_axis_pointer,cluster_amount);
	cudaFree(cuda_cluster_points_axis_pointer);
	cudaFree(cuda_cluster_points_axis);
	cudaFree(cuda_qm_points_arr);
	cudaFree(cuda_qm_points_arr_leftover);
	cudaFree(cuda_max_qm_points_arr);
	cudaFree(cuda_max_qm_points_arr_leftover);
	free(max_qm_points_arr);

	return max_cluster;
}

void realloc_lefover_qm_points(int leftover, Qm_Point** qm_points_arr, int amount_with_leftover, Qm_Point* qm_points,int qm_points_amount)
{
	if (leftover > 0)
	{
		*qm_points_arr = (Qm_Point*)realloc(*qm_points_arr, sizeof(Qm_Point)*amount_with_leftover);//should free
#pragma omp parallel for shared(qm_points, qm_points_arr)
		for (int i = leftover; i > 0; i--)
		{
			(*qm_points_arr)[amount_with_leftover - i] = qm_points[qm_points_amount -i];
		}
	}
}

int cuda_copy_matrix(int* cluster_points_amount, int cluster_amount, void*** cuda_outer_arr,void** src_arr, int size_of_pointer, int size_of_element,void** cuda_arr_pointers)
{
	int good;

	good = cuda_malloc_elements((void**)cuda_outer_arr, cluster_amount, 0, 0, size_of_pointer);
	if (good == 0)
	{
		return good;
	}

	for (int i = 0; i < cluster_amount; i++)//maybe omp
	{
		int good;
		int amount = cluster_points_amount[i];
		void* cuda_elements = 0;
		good = cuda_malloc_elements(&cuda_elements, amount, 0, 0, size_of_element);
		cuda_arr_pointers[i] = cuda_elements;
		if (good == 0)
		{
			return good;
		}

		good = cuda_copy_elements((void*)((*cuda_outer_arr) + i), &cuda_elements, 1, 0, 0, 0, size_of_pointer, cudaMemcpyHostToDevice);
		if (good == 0)
		{
			return good;
		}

		good = cuda_copy_elements(cuda_elements, src_arr[i], amount, 0, 0, 0, size_of_element, cudaMemcpyHostToDevice);
		if (good == 0)
		{
			return good;
		}
	}
}

int cuda_free_matrix(void** cuda_arr_pointers,int cluster_amount)
{
#pragma omp parallel for shared (cuda_arr_pointers)
	for (int i = 0; i < cluster_amount; i++)
	{
		cudaError_t cudaStatus;
		cudaStatus = cudaFree(cuda_arr_pointers[i]);
		if (cudaStatus != cudaSuccess)
		{
			return 0;
		}
	}
}

void malloc_cluster_points(int* cluster_points_amount, void*** cluster_points_element,int cluster_amount,int size_of_pointer,int size_of_element)
{
	*cluster_points_element = (void**)calloc(cluster_amount, size_of_pointer);

	void** points_element = *cluster_points_element;

	for (int i = 0; i < cluster_amount; i++)
	{
		//int x = cluster_points_amount[i];
		points_element[i] = calloc(cluster_points_amount[i], size_of_element);
		//printf("ddddd22222 %d %d\n", my_id, points_element[i]);
		//fflush(stdout);
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
	char* dummy;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		return 0;
	}
	cuda_malloc_memory((void**)&dummy, 1, sizeof(char));
	cudaFree(dummy);
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
