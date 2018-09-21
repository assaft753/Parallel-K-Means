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
	Cluster* clusters = 0;
};

Cluster* init_data(Points* points, Clusters* clusters, int* limit, double* qm, int* t, double* dt);

void calculate_points_positions(Points* points, double t);
void calculate_point_position(Point* point,double t);

void group_points_to_clusters(Points* points, Cluster* new_cluster,int cluster_amount);

double point_2_point_distance(Axis p1,Axis p2);
int find_min_distance_cluster(Point point, Cluster* new_cluster, int cluster_amount);

void calculate_cluster_center(Cluster* new_cluster, Clusters* clusters, int amount);
Axis axis_avg(Point** points, int amount);

int check_points_transfer(Cluster* clusters1, Cluster* clusters2, int amount);
int check_transfer(Point** ps1, int amount_ps1, Point** ps2, int amount_ps2);
int compare_points(Point p1, Point p2);

void copy_cluster(Clusters* clusters, Cluster** new_cluster);

void free_clusters(Clusters* clusters);

double evaluate_qm(Clusters* clusters);
double** malloc_helper_distance_matrix(int clusters_amount);
double find_diameter(Cluster cluster);
void free_helper_mat(double** mat, int clusters_amount);

void print_cluster(double t, double q, Clusters* clusters);

void free_points(Points* points);


