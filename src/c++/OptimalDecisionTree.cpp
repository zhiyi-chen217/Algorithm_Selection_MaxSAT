#include "MurTree.h"
#include "Frequency.h"

using namespace std;
float computeGain(MurTree& tree, vector<vector<float>> data, vector<vector<float>> result) {
    // mata data
    int n_instance = data.size();
    float gain = 0;
    for (int i = 0; i < n_instance; i++) {
        int label = tree.classify(data[i]);
        gain += result[i][label];
    }
    gain = gain / n_instance;

    return gain;
}

float computeOracleGain(vector<vector<float>> result) {
    int n_instance = result.size();
    float gain = 0;
    for (int i = 0; i < n_instance; i++) {
        int max_ind = argmax(result[i]);
        gain += result[i][max_ind];
    }
    return gain / n_instance;
}

float computeSBSGain(vector<vector<float>> result) {
    int n_instance = result.size();
    float gain = 0;
    for (int i = 0; i < n_instance; i++) {
        gain += result[i][SBS_IND];
    }
    return gain / n_instance;
}

float findDepthTwo(PNode* root, vector<vector<float>> data, vector<vector<float>> result, vector<tuple<int, float>> predicate) {
    // mata data
    int n_class = N_CLASS;
    int n_instance = data.size();
    int n_predicate = predicate.size();

    // computing frequencies
    vector<vector<vector<float>>> FQ_pair = computePairFQ(data, result, predicate);
    vector<vector<float>> FQ_single = computeSingleFQ(data, result, predicate);
    vector<float> FQ_all = computeAllFQ(data, result);

    // left
    vector<float> best_left_subtree_gain(n_predicate, 0);
    vector<float> best_left_subtree_feature(n_predicate, 0);
    vector<float> best_left_subtree_left_label(n_predicate, 0);
    vector<float> best_left_subtree_right_label(n_predicate, 0);

    // right
    vector<float> best_right_subtree_gain(n_predicate, 0);
    vector<float> best_right_subtree_feature(n_predicate, 0);
    vector<float> best_right_subtree_left_label(n_predicate, 0);
    vector<float> best_right_subtree_right_label(n_predicate, 0);

    // try all predicates as root 
    // find corresponding best left and right subtrees 
    for (int i = 0; i < n_predicate; i++) {
        for (int j = 0; j < n_predicate; j++) {
            if (i == j) {
                continue;
            }
            vector<float> cs_left_left(n_class, 0.0);
            vector<float> cs_left_right(n_class, 0.0);
            vector<float> cs_right_left(n_class, 0.0);
            vector<float> cs_right_right(n_class, 0.0);

            for (int c = 0; c < n_class; c++) {
                float FQ_c = FQ_all[c];
                float FQ_c_i = FQ_single[c][i];
                float FQ_c_j = FQ_single[c][j];
                float FQ_c_i_j = 0;
                if (j > i) {
                    FQ_c_i_j = FQ_pair[c][i][j];
                }
                else {
                    FQ_c_i_j = FQ_pair[c][j][i];
                }
                cs_left_left[c] = FQ_c - FQ_c_i - FQ_c_j + FQ_c_i_j;
                cs_left_right[c] = FQ_c_j - FQ_c_i_j;
                cs_right_left[c] = FQ_c_i - FQ_c_i_j;
                cs_right_right[c] = FQ_c_i_j;
            }

            // find optimal class for left_left classification node
            int ind_left_left = argmax(cs_left_left);
            int label_left_left = ind_left_left;
            float gain_left_left = cs_left_left[ind_left_left];

            // find optimal class for left_right classification node
            int ind_left_right = argmax(cs_left_right);
            int label_left_right = ind_left_right;
            float gain_left_right = cs_left_right[ind_left_right];

            float gain_left = gain_left_left + gain_left_right;

            // find optimal class for right_left classification node
            int ind_right_left = argmax(cs_right_left);
            int label_right_left = ind_right_left;
            float gain_right_left = cs_right_left[ind_right_left];

            // find optimal class for right_right classification node
            int ind_right_right = argmax(cs_right_right);
            int label_right_right = ind_right_right;
            float gain_right_right = cs_right_right[ind_right_right];

            float gain_right = gain_right_left + gain_right_right;

            // check if the current left subtree gains more then the previous max
            if (gain_left > best_left_subtree_gain[i]) {
                best_left_subtree_gain[i] = gain_left;
                best_left_subtree_feature[i] = j;
                best_left_subtree_left_label[i] = label_left_left;
                best_left_subtree_right_label[i] = label_left_right;
            }

            // check if the current right subtree gains more then the previous max
            if (gain_right > best_right_subtree_gain[i]) {
                best_right_subtree_gain[i] = gain_right;
                best_right_subtree_feature[i] = j;
                best_right_subtree_left_label[i] = label_right_left;
                best_right_subtree_right_label[i] = label_right_right;
            }
        }
    }

    // find best predicate for root
    vector<float> total_gain(n_predicate, 0);
    for (int i = 0; i < n_predicate; i++) {
        total_gain[i] = best_left_subtree_gain[i] + best_right_subtree_gain[i];
    }

    // create root predicate node
    int ind_pred = argmax(total_gain);
    tuple<int, float> pred_root = predicate[ind_pred];
    root->setFeature(get<0>(pred_root));
    root->setThreshold(get<1>(pred_root));
    
    // create left predicate node
    tuple<int, float> pred_left = predicate[best_left_subtree_feature[ind_pred]];
    PNode* left = new PNode(get<0>(pred_left), get<1>(pred_left));

    // create right predicate node
    tuple<int, float> pred_right = predicate[best_right_subtree_feature[ind_pred]];
    PNode* right = new PNode(get<0>(pred_right), get<1>(pred_right));

    // create classification nodes
    CNode* left_left = new CNode(best_left_subtree_left_label[ind_pred]);
    CNode* left_right = new CNode(best_left_subtree_right_label[ind_pred]);
    CNode* right_left = new CNode(best_right_subtree_left_label[ind_pred]);
    CNode* right_right = new CNode(best_right_subtree_right_label[ind_pred]);

    // construct tree
    left->setLeft(left_left);
    left->setRight(left_right);
    right->setLeft(right_left);
    right->setRight(right_right);

    root->setLeft(left);
    root->setRight(right);

    return total_gain[ind_pred];
}

float findOptimalTree(PNode* root, vector<vector<float>> data, vector<vector<float>> result, vector<tuple<int, float>> predicate, int depth) {
    if (depth == 2) {
        return findDepthTwo(root, data, result, predicate);
    }

    // meta data
    int n_class = N_CLASS;
    int n_instance = data.size();
    int n_predicate = predicate.size();


    float max_gain = 0;
    PNode* min_left_subtree = new PNode(-1, -1);
    PNode* min_right_subtree = new PNode(-1, -1);
    // try each predicate as root
    for (int i = 0; i < n_predicate; i++) {
        PNode left_subtree(-1, -1);
        PNode right_subtree(-1, -1);
        tuple<int, float> p = predicate[i];
        int f = get<0>(p);
        float t = get<1>(p);
        vector<vector<float>> right_data;
        vector<vector<float>> left_data;
        vector<vector<float>> right_result;
        vector<vector<float>> left_result;

        // divide data
        for (int j = 0; j < n_instance; j++) {
            if (data[j][f] > t) {
                right_data.push_back(data[j]);
                right_result.push_back(result[j]);
            }
            else {
                left_data.push_back(data[j]);
                left_result.push_back(result[j]);
            }
        }
        float left_gain = findOptimalTree(&left_subtree, left_data, left_result, predicate, depth-1);
        float right_gain = findOptimalTree(&right_subtree, right_data, right_result, predicate, depth-1);
        float total_gain = left_gain + right_gain;

        if (total_gain > max_gain) {
            max_gain = total_gain;
            root->setFeature(f);
            root->setThreshold(t);
            copyPNodeToHeap(&left_subtree, min_left_subtree);
            copyPNodeToHeap(&right_subtree, min_right_subtree);
        }
    }
    root->setLeft(min_left_subtree);
    root->setRight(min_right_subtree);
    return max_gain;
}


int main() {
    fstream fin;
    string temp;
    cout << "start" << endl;
    // read feature data 
    fin.open(string(DIR) + string(FEATURE_FILE_NAME), ios::in);
    getline(fin, temp);
    map <int, string> feature_map = createMap(temp);
    vector<vector<float>> features = readCSV(&fin, N_INSTANCE, N_FEATURE);

    // read result data
    fin.open(string(DIR) + string(RESULT_FILE_NAME), ios::in);
    getline(fin, temp);
    map <int, string> class_map = createMap(temp);
    vector<vector<float>> results_solvers = readCSV(&fin, N_INSTANCE, N_CLASS);

    // read predicates
    vector<tuple<int, float>> predicates = readPred(PRED_FILE_NAME);

    // split train test
    int test_size = (int)(N_INSTANCE / N_ITER);
    int train_size = N_INSTANCE - test_size;
    vector<vector<float>> train_feature(train_size, vector<float>(N_FEATURE));
    vector<vector<float>> test_feature(test_size, vector<float>(N_FEATURE));
    
    vector<vector<float>> train_result(train_size, vector<float>(N_CLASS));
    vector<vector<float>> test_result(test_size, vector<float>(N_CLASS));

    // N-fold validation
    for (int i = 0; i < N_ITER; i++) {
        int start = i * test_size;
        int end = (i + 1) * test_size;

        // split data for current iteration
        splitData(features, train_feature, test_feature, start, end);
        splitData(results_solvers, train_result, test_result, start, end);

        PNode* root = new PNode(1, 2);
        float result = findOptimalTree(root, features, results_solvers, predicates, 2);

        MurTree tree(root);
        float gain = computeGain(tree, test_feature, test_result);
        float oracle = computeOracleGain(test_result);
        float SBS_gain = computeSBSGain(test_result);

        cout << "Average score of the prediction: " << gain << endl;
        cout << "Score of the single best solver: " << SBS_gain << endl;
        cout << "Score of the oracle: " << oracle << endl;

        deleteTree(root);
    }
}
