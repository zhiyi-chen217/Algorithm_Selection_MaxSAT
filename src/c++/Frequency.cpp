#include "Frequency.h"


vector<vector<float>> computeSingleFQ(vector<vector<float>>& data, vector<vector<float>>& result, vector<tuple<int, float>>& predicate) {
    int n_predicate = predicate.size();
    int n_instance = data.size();
    int n_class = N_CLASS;
    vector<vector<float>> FQ_single_2d(n_class, vector<float>(n_predicate, 0.0));
    for (int c = 0; c < n_class; c++) {
        vector<float>& FQ_single = FQ_single_2d[c];
        for (int ind = 0; ind < n_instance; ind++) {
            vector<float>& d = data[ind];
            vector<float>& r = result[ind];
            for (int i = 0 ; i < n_predicate; i++) {
                tuple<int, float>& t = predicate[i];
                if (d[get<0>(t)] > get<1>(t)) {
                    FQ_single[i] += r[c];
                }
            }
        }
    }
    return FQ_single_2d;
}

vector<float> computeAllFQ(vector<vector<float>>& data, vector<vector<float>>& result) {
    int n_class = N_CLASS;
    vector<float> FQ_ALL(n_class, 0.0);
    for (int c = 0; c < n_class; c++) {
        FQ_ALL[c] = sumCol(result, c);
    }
    return FQ_ALL;
}

vector<vector<vector<float>>> computePairFQ(vector<vector<float>>& data, vector<vector<float>>& result, vector<tuple<int, float>>& predicate) {
    int n_class = N_CLASS;
    int n_instance = data.size();
    int n_predicate = predicate.size();
    vector<vector<vector<float>>> FQ_pair_3d(n_class, vector<vector<float>>(n_predicate, vector<float>(n_predicate, 0.0)));
    for (int c = 0; c < n_class; c++) {
        vector<vector<float>>& FQ_pair = FQ_pair_3d[c];
        for (int ind = 0; ind < n_instance; ind++) {
            vector<float>& d = data[ind];
            vector<float>& r = result[ind];
            for (int i = 0; i < n_predicate; i++) {
                tuple<int, float>& t1 = predicate[i];
                for (int j = i; j < n_predicate; j++) {
                    tuple<int, float>& t2 = predicate[j];
                    if (d[get<0>(t1)] > get<1>(t1) && d[get<0>(t2)] > get<1>(t2)) {
                        FQ_pair[i][j] += r[c];
                    }
                }
            }
        }
    }
    return FQ_pair_3d;
}