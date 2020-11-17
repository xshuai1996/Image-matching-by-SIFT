#ifndef HEAEDER_H
#endif HEAEDER_H


#include<string>
#include <opencv2/opencv.hpp>
#include<vector>
#include "opencv2/xfeatures2d.hpp"


using std::string;
using cv::Mat;
using std::vector;
using cv::KeyPoint;
using cv::DMatch;


struct position_count_struct {
	float* position;
	int cnt;
	int length_of_pos;
};


void image_matching(string dataset_path, string result_path);
vector<KeyPoint> detect_keypoints_by_SIFT(Mat gray_image, string save_path, string alias);
Mat read_image_RGB_as_gray(string path);
void display_image(Mat image, string alias);
vector<KeyPoint> find_largest_scale_keypoints(vector<KeyPoint> keypoints, Mat image, string save_path, string alias);
Mat generate_descriptor(Mat image, vector<KeyPoint> keypoints);
vector<DMatch> find_nearest_matches(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, Mat descriptor1, Mat descriptor2);
void draw_matches(Mat image1, vector<KeyPoint> keypoints1, Mat image2, vector<KeyPoint> keypoints2, vector <DMatch> matches, string path, string alias);
vector<DMatch> selective_match(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, Mat descriptor1, Mat descriptor2, float ratio_thresh);
vector<position_count_struct> K_means_for_descriptor(Mat descriptor);
int find_best_assignment(Mat one_row, vector<float*> centers);
vector<position_count_struct> reorder_position_count_struct_2(vector<position_count_struct> struct_1, vector<position_count_struct> struct_2);
int find_best_centroid(position_count_struct struct_1, vector<position_count_struct> vec_struct_2);
void write_centroids(vector<vector<position_count_struct>> centroids, string path);
void save_image(Mat image, string path, string alias);


