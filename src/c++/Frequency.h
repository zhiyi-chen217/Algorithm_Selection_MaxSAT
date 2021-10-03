#include <iostream>
#include <vector>
#include <tuple>
#include "utility.h"
using namespace std;

// include meta data and functions to calculate frequency matrix
#define N_INSTANCE 297
#define N_FEATURE 28
#define N_CLASS 7
#define FEATURE_FILE_NAME "feature_reduction_lasso_unweighted.csv"
#define RESULT_FILE_NAME "result_unweighted_shuffled.csv"
#define PRED_FILE_NAME "predicate_reduction_lasso.csv"
#define SBS_IND 0
#define N_ITER 5
#define N_PRED_START 4
#define N_PRED_END 540
#define INTERVAL 3
#define DIR "../../data/"

vector<vector<float>> computeSingleFQ(vector<vector<float>>& data, vector<vector<float>>& result, vector<tuple<int, float>>& predicate);

vector<float> computeAllFQ(vector<vector<float>>& data, vector<vector<float>>& result);

vector<vector<vector<float>>> computePairFQ(vector<vector<float>>& data, vector<vector<float>>& result, vector<tuple<int, float>>& predicate);