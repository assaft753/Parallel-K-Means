#include "Serial.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main()
{
	Points points;
	Clusters clusters;
	Cluster* new_cluster;
	int limit, t, i, n, max_n, good = 0;
	double qm, dt,current_t, q;
	new_cluster = init_data(&points, &clusters, &limit, &qm, &t, &dt);
	max_n = t / dt;
	for (i = 0, n = 0; (i < limit || n <= max_n) && good == 0; i++, n++)
	{
		current_t = n*dt;
		calculate_points_positions(&points,current_t);
		
		group_points_to_clusters(&points, new_cluster,clusters.size);
		
		calculate_cluster_center(new_cluster,&clusters,clusters.size);
		
		good = check_points_transfer(clusters.clusters, new_cluster, clusters.size);
		
		copy_cluster(&clusters, &new_cluster);
		
		if (good == 1)
		{
			q = evaluate_qm(&clusters);
			if (q < qm)
			{
				print_cluster(current_t,q,&clusters);
			}
			else
			{
				good = 0;
			}
		}
	}
	if (good == 0)
	{
		printf("\n No clusters found\n");
	}

	return 0;
}


Cluster * init_data(Points * points, Clusters * clusters, int * limit, double * qm, int * t, double * dt)
{
	int points_amount, clusters_amount;
	int pos_x, pos_y, pos_z;
	int vel_x, vel_y,vel_z;
	Cluster* temp_cluster = 0;
	FILE *fp;
	fp = fopen("input.txt", "r");
	fscanf(fp, "%d %d %d %f %d %f\n",&points_amount,&clusters_amount,t,dt,limit,qm);
	
	points->size = points_amount;
	points->points = (Point*)malloc(sizeof(Point)*points_amount);
	
	clusters->size = clusters_amount;
	clusters->clusters = (Cluster*)malloc(sizeof(Cluster)*clusters_amount);
	
	temp_cluster = (Cluster*)malloc(sizeof(Cluster)*clusters_amount);

	for (int i = 0; i < points_amount; i++)
	{
		fscanf(fp, "%d %d %d %d %d %d\n", &pos_x, &pos_y, &pos_z, &vel_x, &vel_y, &vel_z);
		
		points->points[i].id = i;
		
		points->points[i].axis_location.x = pos_x;
		points->points[i].axis_location.y = pos_y;
		points->points[i].axis_location.z = pos_z;

		points->points[i].axis_velocity.x = vel_x;
		points->points[i].axis_velocity.y = vel_y;
		points->points[i].axis_velocity.z = vel_z;

		if (i < clusters_amount)
		{
			clusters->clusters[i].center.x = pos_x;//
			clusters->clusters[i].center.y = pos_y;//
			clusters->clusters[i].center.z = pos_z;//

			temp_cluster[i].center.x = pos_x;
			temp_cluster[i].center.y = pos_y;
			temp_cluster[i].center.z = pos_z;
		}
	}

	fclose(fp);
	return temp_cluster;
}

void calculate_points_positions(Points* points, double t)
{
	for (int i = 0; i < points->size; i++)
	{
		calculate_point_position(&points->points[i],t);
	}

}

void calculate_point_position(Point* point,double t)
{
	Axis current_position = point->axis_location;
	Axis current_velocity = point->axis_velocity;

	point->axis_location.x = current_position.x + t*current_velocity.x;
	point->axis_location.y = current_position.y + t*current_velocity.y;
	point->axis_location.z = current_position.z + t*current_velocity.z;
}

void group_points_to_clusters(Points* points, Cluster* new_cluster, int cluster_amount)
{
	int index;
	for (int i = 0; i < points->size; i++)
	{
		index = find_min_distance_cluster(points->points[i], new_cluster, cluster_amount);
		Point** cluster_points = new_cluster[index].cluster_points;
		if (new_cluster[index].size == 0)
		{
			cluster_points = (Point**)malloc(sizeof(Point*));
			new_cluster[index].size = 1;
		}
		else
		{
			new_cluster[index].size++;
			cluster_points = (Point**)realloc(cluster_points, sizeof(Point*)*new_cluster[index].size);
		}
		cluster_points[new_cluster[index].size - 1] = (points->points + 1);
	}
}

double point_2_point_distance(Axis p1, Axis p2)
{
	double sub_x, sub_y, sub_z;
	double pow_x, pow_y, pow_z;
	double result;
	
	sub_x = fabs(p1.x - p2.x);
	sub_y = fabs(p1.y - p2.y);
	sub_z = fabs(p1.z - p2.z);

	pow_x = pow(sub_x,2);
	pow_y = pow(sub_y, 2);
	pow_z = pow(sub_z, 2);

	result = sqrt(pow_x + pow_y + pow_z);

	return result;
}

int find_min_distance_cluster(Point point, Cluster* new_cluster, int cluster_amount)
{
	int index = -1,min_value = -1;
	double value = -1;
	for (int i = 0; i < cluster_amount; i++)
	{
		value = point_2_point_distance(point.axis_location, new_cluster->center);
		
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
	return index;
}

Axis axis_avg(Point** points, int amount)
{
	double sum_x = 0,sum_y = 0,sum_z = 0;
	Axis* center_axis = (Axis*)malloc(sizeof(Axis));//
	
	for (int i = 0; i < amount; i++)
	{
		Point* p = *(points+i);
		sum_x += p->axis_location.x;
		sum_y += p->axis_location.y;
		sum_z += p->axis_location.z;
	}

	center_axis->x = sum_x / amount;
	center_axis->y = sum_y / amount;
	center_axis->z = sum_z / amount;
	return *(center_axis);
}

void calculate_cluster_center(Cluster* new_cluster,Clusters* clusters, int amount)
{
	for (int i = 0; i < amount; i++)
	{
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

int check_points_transfer(Cluster* clusters1, Cluster* clusters2, int amount)
{
	int good;
	for (int i = 0; i < amount; i++)
	{
		good = check_transfer(clusters1[i].cluster_points, clusters1[i].size, clusters2[i].cluster_points, clusters2[i].size);
		if (good == 0)
		{
			break;
		}
	}
	return good;
}

int check_transfer(Point** ps1, int amount_ps1, Point** ps2, int amount_ps2)
{
	int good = 0;

	if (amount_ps1 != amount_ps2 && (amount_ps2 == 0 || amount_ps1 == 0))//////
	{
		return 0;
	}

	for (int i = 0; i < amount_ps1; i++)
	{
		Point p1 = *(ps1[i]);
		for (int j = 0; j < amount_ps2; j++)
		{
			Point p2 = *(ps2[j]);
			good += compare_points(p1, p2);
		}
		if (good == 0)
		{
			return 0;
		}
		good = 0;
	}
	return 1;
}

int compare_points(Point p1, Point p2)
{
	return p1.id == p2.id;
}

void copy_cluster(Clusters* clusters, Cluster** new_cluster)
{
	free_clusters(clusters);
	clusters->clusters = *new_cluster;
	*new_cluster = (Cluster*)malloc(sizeof(Cluster)*clusters->size);

}

void free_clusters(Clusters* clusters)
{
	for (int i = 0; i < clusters->size; i++)
	{
		Cluster* c = (clusters->clusters+i);
		free(c->cluster_points);
	}
	free(clusters->clusters);
}

double evaluate_qm(Clusters* clusters)
{
	double** mat = malloc_helper_distance_matrix(clusters->size);
	double cluster_max_distance,sum=0,q;
	int counter = 0;
	for (int i = 0; i < clusters->size; i++)
	{
		cluster_max_distance = find_diameter(clusters->clusters[i]);
		for (int j = 0; j < clusters->size; j++)
		{
			if (i != j)
			{
				double distance_from_clusters;
				counter++;
				if (i < j)
				{
					Axis c1 = clusters[i].clusters->center;
					Axis c2 = clusters[j].clusters->center;
					distance_from_clusters = point_2_point_distance(c1, c2);
					mat[i][j] = distance_from_clusters;
				}
				else
				{
					distance_from_clusters = mat[j][i];
				}
				sum += cluster_max_distance / distance_from_clusters;
			}
		}
	}
	free_helper_mat(mat, clusters->size);
	q = sum / counter;
	return q;
}

double** malloc_helper_distance_matrix(int clusters_amount)
{
	double** mat = (double**)malloc(sizeof(double*)*clusters_amount);
	for (int i = 0; i < clusters_amount; i++)
	{
		mat[i] = (double*)malloc(sizeof(double)*clusters_amount);
	}
	return mat;
}

double find_diameter(Cluster cluster)
{
	double max_value=-1,temp_value;
	for (int i = 0; i < cluster.size; i++)
	{
		Point* p1 = cluster.cluster_points[i];
		for (int j = 0; j < cluster.size; j++)
		{
			if (i != j)
			{
				Point* p2 = cluster.cluster_points[j];
				temp_value = point_2_point_distance(p1->axis_location, p2->axis_location);
				if (temp_value > max_value)
				{
					max_value = temp_value;
				}
			}
		}
	}
	return max_value;
}

void free_helper_mat(double** mat, int clusters_amount)
{
	for (int i = 0; i < clusters_amount; i++)
	{
		double* arr = *(mat + i);
		free(arr);
	}
	free(mat);
}

void print_cluster(double t, double q, Clusters* clusters)
{
	printf("\nFirst occurrence t = %f  with q = %f\n", t, q);
	printf("Centers of the clusters:\n");
	for (int i = 0; i < clusters->size; i++)
	{
		double x = clusters->clusters[i].center.x;
		double y = clusters->clusters[i].center.y;
		double z = clusters->clusters[i].center.z;

		printf("%f %f %f\n", x, y, z);
	}
	fflush(stdout);
}


