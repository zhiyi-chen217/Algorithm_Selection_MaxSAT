#include <iostream>
#include <vector>
#include <tuple>
#include "utility.h"
using namespace std;

// include meta data and functions to calculate frequency matrix
#define N_INSTANCE 297
#define N_FEATURE 49
#define N_CLASS 7
#define FEATURE_FILE_NAME "feature_unweighted_shuffled.csv"
#define RESULT_FILE_NAME "result_unweighted_shuffled.csv"
#define PRED_FILE_NAME "predicate.csv"
#define SBS_IND 0
#define N_ITER 5
#define DIR "../../data/"

vector<vector<float>> computeSingleFQ(vector<vector<float>>& data, vector<vector<float>>& result, vector<tuple<int, float>>& predicate);

vector<float> computeAllFQ(vector<vector<float>>& data, vector<vector<float>>& result);

vector<vector<vector<float>>> computePairFQ(vector<vector<float>>& data, vector<vector<float>>& result, vector<tuple<int, float>>& predicate);