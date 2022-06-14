#include <iostream>
#include <vector>
#include <tuple>
#include "utility.h"
using namespace std;

// include meta data and functions to calculate frequency matrix
#define N_INSTANCE 324
#define N_TEST 137
#define N_FEATURE 49
#define N_CLASS 6
#define TRAIN_FEATURE_FILE_NAME "train_feature.csv"
#define TRAIN_RESULT_FILE_NAME "all_ordering_time_train_result.csv"
#define TEST_FEATURE_FILE_NAME "test_dimacs_feature.csv"
#define TEST_RESULT_FILE_NAME "all_ordering_time_test_dimacs_result.csv"
#define TEST_SCORE_FILE_NAME "all_ordering_zero_one_test_culb_result.csv"
#define PRED_FILE_NAME "predicate_graph_color.csv"
#define SBS_IND 0
#define N_ITER 1
#define N_PRED_START 10
#define N_PRED_END 250
#define INTERVAL 20
#define DIR "../../data_graph_color/"

vector<vector<float>> computeSingleFQ(vector<vector<float>>& data, vector<vector<float>>& result, vector<tuple<int, float>>& predicate);

vector<float> computeAllFQ(vector<vector<float>>& data, vector<vector<float>>& result);

vector<vector<vector<float>>> computePairFQ(vector<vector<float>>& data, vector<vector<float>>& result, vector<tuple<int, float>>& predicate);