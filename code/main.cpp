#include "Header.h"


int main() {
    string dataset_path = "../dataset/";
    string result_path = "../results/";

    image_matching(dataset_path, result_path);

    return 0;
}


void image_matching(string dataset_path, string result_path) {
    // read files
    Mat husky_1 = read_image_RGB_as_gray(dataset_path + "Husky_1.jpg");
    Mat husky_2 = read_image_RGB_as_gray(dataset_path + "Husky_2.jpg");
    Mat husky_3 = read_image_RGB_as_gray(dataset_path + "Husky_3.jpg");
    Mat puppy_1 = read_image_RGB_as_gray(dataset_path + "Puppy_1.jpg");

    vector<KeyPoint> keypoints_1 = detect_keypoints_by_SIFT(husky_1, result_path, "Husky_1");
    vector<KeyPoint> keypoints_2 = detect_keypoints_by_SIFT(husky_2, result_path, "Husky_2");
    vector<KeyPoint> keypoints_3 = detect_keypoints_by_SIFT(husky_3, result_path, "Husky_3");
    vector<KeyPoint> keypoints_4 = detect_keypoints_by_SIFT(puppy_1, result_path, "puppy_1");

    vector<KeyPoint> largest_scale_keypoints_3 = find_largest_scale_keypoints(keypoints_3, husky_3, result_path, "largest_scale_Husky_3");

    Mat descriptor_1 = generate_descriptor(husky_1, keypoints_1);
    Mat descriptor_largest_scale_3 = generate_descriptor(husky_3, largest_scale_keypoints_3);
    vector<DMatch> best_match = find_nearest_matches(largest_scale_keypoints_3, keypoints_1, descriptor_largest_scale_3, descriptor_1);
    draw_matches(husky_3, largest_scale_keypoints_3, husky_1, keypoints_1, best_match, result_path, "best matches");

    Mat descriptor_2 = generate_descriptor(husky_2, keypoints_2);
    Mat descriptor_3 = generate_descriptor(husky_3, keypoints_3);
    Mat descriptor_4 = generate_descriptor(puppy_1, keypoints_4);

    vector<DMatch> good_matches_1_and_3 = selective_match(keypoints_1, keypoints_3, descriptor_1, descriptor_3, 0.8);
    draw_matches(husky_1, keypoints_1, husky_3, keypoints_3, good_matches_1_and_3, result_path, "matches 1 and 3");

    vector<DMatch> good_matches_2_and_3 = selective_match(keypoints_2, keypoints_3, descriptor_2, descriptor_3, 0.8);
    draw_matches(husky_2, keypoints_2, husky_3, keypoints_3, good_matches_2_and_3, result_path, "matches 2 and 3");

    vector<DMatch> good_matches_3_and_4 = selective_match(keypoints_3, keypoints_4, descriptor_3, descriptor_4, 0.8);
    draw_matches(husky_3, keypoints_3, puppy_1, keypoints_4, good_matches_3_and_4, result_path, "matches 3 and 4");

    vector<DMatch> good_matches_1_and_4 = selective_match(keypoints_1, keypoints_4, descriptor_1, descriptor_4, 0.8);
    draw_matches(husky_1, keypoints_1, puppy_1, keypoints_4, good_matches_1_and_4, result_path, "matches 1 and 4");

    vector<position_count_struct> pos_cnt_1 = K_means_for_descriptor(descriptor_1);
    vector<position_count_struct> pos_cnt_2 = K_means_for_descriptor(descriptor_2);
    vector<position_count_struct> pos_cnt_3 = K_means_for_descriptor(descriptor_3);
    vector<position_count_struct> pos_cnt_4 = K_means_for_descriptor(descriptor_4);

    vector<position_count_struct> reordered_pos_cnt_1 =
        reorder_position_count_struct_2(pos_cnt_3, pos_cnt_1);
    vector<position_count_struct> reordered_pos_cnt_2 =
        reorder_position_count_struct_2(pos_cnt_3, pos_cnt_2);
    vector<position_count_struct> reordered_pos_cnt_4 =
        reorder_position_count_struct_2(pos_cnt_3, pos_cnt_4);

    write_centroids({ reordered_pos_cnt_1 , reordered_pos_cnt_2 ,pos_cnt_3, reordered_pos_cnt_4 }, result_path);
}


