#include "Header.h"
#include <iostream>
#include <fstream>


using cv::imread;
using cv::COLOR_BGR2GRAY;
using cv::Ptr;
using cv::xfeatures2d::SIFT;
using cv::waitKey;
using cv::destroyAllWindows;
using cv::DescriptorMatcher;
using std::ofstream;
using std::endl;


Mat read_image_RGB_as_gray(string path) {
	Mat color_mat, gray_mat;
	color_mat = imread(path);
	cvtColor(color_mat, gray_mat, COLOR_BGR2GRAY);
	return gray_mat;
}


vector<KeyPoint> detect_keypoints_by_SIFT(Mat gray_image, string save_path, string alias) {
	Ptr<SIFT> f2d = SIFT::create();
	vector<KeyPoint> keypoints;
	f2d->detect(gray_image, keypoints);

	Mat img_keypoints;
	drawKeypoints(gray_image, keypoints, img_keypoints, cv::Scalar(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	display_image(img_keypoints, alias);
	save_image(img_keypoints, save_path, alias);
	return keypoints;
}


void display_image(Mat image, string alias) {
	imshow(alias, image);
	waitKey(0);
	destroyAllWindows();
}


void save_image(Mat image, string path, string alias) {
	imwrite(path + alias + ".jpg", image);
}


vector<KeyPoint> find_largest_scale_keypoints(vector<KeyPoint> keypoints, Mat image, string save_path, string alias) {
	vector<KeyPoint> ret;
	float largest_size = -10;
	for (int i = 0; i < keypoints.size(); i++) {
		if (keypoints[i].size > largest_size) {
			largest_size = keypoints[i].size;
		}
	}
	for (int i = 0; i < keypoints.size(); i++) {
		if (keypoints[i].size == largest_size) {
			ret.push_back(keypoints[i]);
		}
	}
	Mat img_keypoints;
	drawKeypoints(image, ret, img_keypoints);
	display_image(img_keypoints, alias);
	save_image(img_keypoints, save_path, alias);
	return ret;
}


Mat generate_descriptor(Mat image, vector<KeyPoint> keypoints) {
	Ptr<SIFT> f2d = SIFT::create();
	Mat descriptor;
	f2d->compute(image, keypoints, descriptor);
	return descriptor;
}


vector<DMatch> find_nearest_matches(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, Mat descriptor1, Mat descriptor2) {
	vector<DMatch> ret;
	float min_distance, distance;
	int keypoint_with_min_distance;
	for (int i = 0; i < keypoints1.size(); i++) {
		min_distance = pow(10, 8);
		for (int j = 0; j < keypoints2.size(); j++) {
			distance = 0;
			for (int k = 0; k < descriptor1.cols; k++) {
				distance += pow(descriptor1.at<float>(i, k) - descriptor2.at<float>(j, k), 2);
			}
			distance = pow(distance, 0.5);
			if (distance < min_distance) {
				min_distance = distance;
				keypoint_with_min_distance = j;
			}
		}
		DMatch match(i, keypoint_with_min_distance, min_distance);
		ret.push_back(match);
	}
	return ret;
}


void draw_matches(Mat image1, vector<KeyPoint> keypoints1, Mat image2, vector<KeyPoint> keypoints2, vector <DMatch> matches, string path, string alias) {
	Mat res;
	drawMatches(image1, keypoints1, image2, keypoints2, matches, res);
	display_image(res, alias);
	save_image(res, path, alias);
}


vector<DMatch> selective_match(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, Mat descriptor1, Mat descriptor2, float ratio_thresh) {
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	vector<vector<DMatch>> matches;
	matcher->knnMatch(descriptor1, descriptor2, matches, 2);

	//-- Filter matches using the Lowe's ratio test
	vector<DMatch> good_matches;
	for (int i = 0; i < matches.size(); i++) {
		if (matches[i][0].distance < ratio_thresh * matches[i][1].distance) {
			good_matches.push_back(matches[i][0]);
		}
	}
	return good_matches;
}


vector<position_count_struct> K_means_for_descriptor(Mat descriptor) {
	vector<position_count_struct> ret;

	// set initial cluster centers
	vector<float*> centers;
	for (int i = 0; i < 8; i++) {
		float* center = new float[descriptor.cols];
		for (int j = 0; j < descriptor.cols; j++) {
			center[j] = descriptor.at<float>(i, j);
		}
		centers.push_back(center);
		position_count_struct obj;
		obj.position = center;
		obj.cnt = 0;
		obj.length_of_pos = descriptor.cols;
		ret.push_back(obj);
	}
	// set initial assignment
	vector<int> assignment;
	for (int i = 0; i < descriptor.rows; i++) {
		assignment.push_back(-1);
	}

	// stop K-means when assignment and centers are no longer change
	int changes = -1, new_assignment = -1;
	while (changes != 0) {
		changes = 0;
		// fix center, find best assignment
		for (int i = 0; i < descriptor.rows; i++) {
			new_assignment = find_best_assignment(descriptor.row(i), centers);
			if (new_assignment != assignment[i]) {
				changes += 1;
				assignment[i] = new_assignment;
			}
		}

		// fix assignment, calculate new centers
		for (int i = 0; i < centers.size(); i++) {
			// scan assignment to find beloning data samples
			float cnt_belong = 0;
			for (int j = 0; j < descriptor.cols; j++) {
				centers[i][j] = 0;
			}

			for (int j = 0; j < descriptor.rows; j++) {
				if (assignment[j] == i) {
					cnt_belong += 1;
					for (int k = 0; k < descriptor.cols; k++) {
						centers[i][k] += descriptor.at<float>(j, k);
					}
				}
			}
			// get the average position
			for (int k = 0; k < descriptor.cols; k++) {
				centers[i][k] /= cnt_belong;
			}
			ret[i].cnt = cnt_belong;
		}
	}
	for (int i = 0; i < centers.size(); i++) {
		for (int j = 0; j < descriptor.rows; j++) {
			ret[i].position[j] = centers[i][j];
		}
	}
	return ret;
}


int find_best_assignment(Mat one_row, vector<float*> centers) {
	int ret = -1;
	float shortest_distance = 1000000000;
	float distance;
	for (int c = 0; c < centers.size(); c++) {
		distance = 0;
		for (int i = 0; i < one_row.cols; i++) {
			distance += pow(one_row.at<float>(i) - centers[c][i], 2);
		}
		if (distance < shortest_distance) {
			ret = c;
			shortest_distance = distance;
		}
	}
	return ret;
}


vector<position_count_struct> reorder_position_count_struct_2(vector<position_count_struct> struct_1, vector<position_count_struct> struct_2) {
	vector<position_count_struct> ret;

	for (int i = 0; i < struct_1.size(); i++) {
		ret.push_back(struct_2[find_best_centroid(struct_1[i], struct_2)]);
	}
	return ret;
}


int find_best_centroid(position_count_struct struct_1, vector<position_count_struct> vec_struct_2) {
	int ret = -1;
	float shortest_distance = 1000000000;
	float distance;
	for (int c = 0; c < vec_struct_2.size(); c++) {
		distance = 0;
		for (int i = 0; i < struct_1.length_of_pos; i++) {
			distance += pow(struct_1.position[i] - vec_struct_2[c].position[i], 2);
		}
		if (distance < shortest_distance) {
			ret = c;
			shortest_distance = distance;
		}
	}
	return ret;
}


void write_centroids(vector<vector<position_count_struct>> centroids, string path) {
	ofstream fout(path + "centroids.txt");
	for (int i = 0; i < centroids.size(); i++) {
		for (int j = 0; j < centroids[0].size(); j++) {
			fout << centroids[i][j].cnt << ",";
		}
		fout << endl;
	}
	fout.close();
}


