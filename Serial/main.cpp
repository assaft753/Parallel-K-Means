#include "Serial.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void print(Clusters* clusters, Cluster* cluster, int d, Points points);

int main()
{
	Points points;
	Clusters clusters;
	Cluster* new_cluster;
	int limit, t, i, n, max_n, good = 0;
	double qm, dt,current_t, q;
	new_cluster = init_data(&points, &clusters, &limit, &qm, &t, &dt);
	max_n = t / dt;
	print(&clusters, new_cluster, 0, points);//
	for (i = 0, n = 0; (i < limit || n <= max_n) && good == 0; i++, n++)
	{
		current_t = n*dt;
		calculate_points_positions(&points,current_t);//OK
		print(&clusters, new_cluster, 1,points);//
		
		group_points_to_clusters(&points, new_cluster,clusters.size);//OK
		print(&clusters, new_cluster, 2, points);//
		
		calculate_cluster_center(new_cluster,&clusters,clusters.size);//OK
		print(&clusters, new_cluster, 3, points);//
		
		good = check_points_transfer(clusters.clusters, new_cluster, clusters.size);//OK
		printf("\ngood = %d \n", good);//
		print(&clusters, new_cluster, 4, points);//
		
		copy_cluster(&clusters, &new_cluster);//OK
		print(&clusters, new_cluster, 5, points);//
		
		if (good == 1)
		{
			q = evaluate_qm(&clusters);//OK
			if (q < qm)
			{
				print_cluster(current_t,q,&clusters);//
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

	free_points(&points);
	free_clusters(&clusters);
	free(new_cluster);

	return 0;
}

Cluster * init_data(Points * points, Clusters * clusters, int * limit, double * qm, int * t, double * dt)
{
	int points_amount, clusters_amount;
	double pos_x, pos_y, pos_z;
	double vel_x, vel_y,vel_z;
	Cluster* new_cluster = 0;
	FILE *fp;
	fp = fopen("C:\\Users\\Assaf Tayouri\\Documents\\Visual Studio 2015\\Projects\\K-Means\\Debug\\abc.txt", "r");
	fscanf(fp, "%d %d %d %lf %d %lf\n",&points_amount,&clusters_amount,t,dt,limit,qm);
	
	points->size = points_amount;
	points->points = (Point*)calloc(points_amount, sizeof(Point));
	
	clusters->size = clusters_amount;
	clusters->clusters = (Cluster*)calloc(clusters_amount, sizeof(Cluster));
	
	new_cluster = (Cluster*)calloc(clusters_amount, sizeof(Cluster));

	for (int i = 0; i < points_amount; i++)
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
		index = find_min_distance_cluster(points->points[i], new_cluster, cluster_amount);//V
		Point** cluster_points = new_cluster[index].cluster_points;
		
		if (new_cluster[index].size == 0)//
		{//
			cluster_points = (Point**)malloc(sizeof(Point*));//
			new_cluster[index].size = 1;//
		}//

		else//
		{//
			new_cluster[index].size++;//
			cluster_points = (Point**)realloc(cluster_points, sizeof(Point*)*new_cluster[index].size);//
		}//

		//new_cluster[index].size++;
		cluster_points[new_cluster[index].size - 1] = (points->points + i);
		new_cluster[index].cluster_points = cluster_points;//
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
	int index = -1;
	double value = -1, min_value = -1;
	for (int i = 0; i < cluster_amount; i++)
	{
		value = point_2_point_distance(point.axis_location, new_cluster[i].center);
		
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
	Axis center_axis;
	
	for (int i = 0; i < amount; i++)
	{
		Point* p = *(points+i);
		sum_x += p->axis_location.x;
		sum_y += p->axis_location.y;
		sum_z += p->axis_location.z;
	}

	center_axis.x = sum_x / amount;
	center_axis.y = sum_y / amount;
	center_axis.z = sum_z / amount;
	return center_axis;
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
	int good;

	if (amount_ps1 != amount_ps2)
	{
		return 0;
	}

	for (int i = 0; i < amount_ps1; i++)
	{
		Point p1 = *(ps1[i]);
		good = 0;
		for (int j = 0; j < amount_ps2 && good == 0; j++)
		{
			Point p2 = *(ps2[j]);
			good += compare_points(p1, p2);
		}
		if (good == 0)
		{
			return 0;
		}
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
	*new_cluster =(Cluster*)calloc(clusters->size, sizeof(Cluster));
	
	for (int i = 0; i < clusters->size; i++)
	{
		(*new_cluster)[i].center = clusters->clusters[i].center;
	}
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
					Axis c1 = clusters->clusters[i].center;
					Axis c2 = clusters->clusters[j].center;
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
	int cluster_size = cluster.size;
	
	if (cluster_size <= 1)//
	{
		return 0;
	}

	/*else if (cluster_size == 1)//
	{
		//Axis temp_axis;
		//return point_2_point_distance(cluster.cluster_points[0]->axis_location, temp_axis);
	}*/

	for (int i = 0; i < cluster_size; i++)
	{
		Point* p1 = cluster.cluster_points[i];
		for (int j = 0; j < cluster_size; j++)
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

void free_points(Points* points)
{
	free(points->points);
}

void print(Clusters* clusters, Cluster* cluster,int d,Points points)
{
	printf("\nfrom %d:\n #################################\n", d);
	printf("Status of Points:[\n");
	for (int k = 0; k < points.size; k++)
	{
		printf("id = %d x = %lf y = %lf z = %lf\n", points.points[k].id, points.points[k].axis_location.x, points.points[k].axis_location.y, points.points[k].axis_location.z);
	}
	printf("]\n");
	printf("start main clusters*******************************\n");
	for (int i = 0; i < clusters->size; i++)
	{
		Cluster c = clusters->clusters[i];
		Point** p = c.cluster_points;
		printf("cluster number (%d) with center x = %lf y = %lf z = %lf\n", i, c.center.x, c.center.y, c.center.z);
		printf("++++++++++++++++++++++++++++++++++\n");
		for (int j = 0; j < c.size;j++)
		{
			printf("id = %d x = %lf y = %lf z = %lf\n", p[j]->id,p[j]->axis_location.x, p[j]->axis_location.y, p[j]->axis_location.z);
		}
		printf("\n++++++++++++++++++++++++++++++++++\n");
	}
	printf("end main clusters*******************************\n\n");

	printf("start new clusters*******************************\n");
	for (int i = 0; i < clusters->size; i++)
	{
		Cluster c = cluster[i];
		Point** p = c.cluster_points;
		printf("cluster number (%d) with center x = %lf y = %lf z = %lf\n", i, c.center.x, c.center.y, c.center.z);
		printf("++++++++++++++++++++++++++++++++++\n");
		for (int j = 0; j < c.size; j++)
		{
			printf("id = %d x = %lf y = %lf z = %lf\n", p[j]->id, p[j]->axis_location.x, p[j]->axis_location.y, p[j]->axis_location.z);
		}
		printf("\n++++++++++++++++++++++++++++++++++\n");
	}
	printf("end new clusters*******************************\n\n");

	printf("###################################################\n");
	fflush(stdout);
}


