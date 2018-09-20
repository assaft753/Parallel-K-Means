#pragma once
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
	Point** cluster_points = 0;
	int size = 0;
	Axis center;
};

struct Clusters
{
	int size = 0;
	Cluster* clusters;
};

Cluster* init_data(Points* points, Clusters* clusters, int* limit, double* qm, int* t, double* dt);

void calculate_points_positions(Points* points,int n,double dt);
void calculate_point_position(Point* point,double t);

void group_points_to_clusters(Points* points, Cluster* new_cluster,int cluster_amount);

int point_2_point_distance(Axis p1,Axis p2);
int find_min_distance_cluster(Point point, Cluster* new_cluster, int cluster_amount);

void calculate_cluster_center(Cluster* new_cluster, int amount);
Axis axis_avg(Point** points, int amount);

int check_points_transfer(Cluster* clusters1, Cluster* clusters2, int amount);
int check_transfer(Point** ps1, int amount_ps1, Point** ps2, int amount_ps2);
int compare_points(Point p1, Point p2);

