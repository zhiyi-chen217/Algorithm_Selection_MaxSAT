#include "utility.h"

using namespace std;

int argmax(vector<float> data) {
    if (data.size() == 0)
    {
        return 0;
    }

    int max_ind = 0;
    int max = data[0];
    for (int i = 0; i < data.size(); i++) {
        if (data[i] > max) {
            max = data[i];
            max_ind = i;
        }
    }
    return max_ind;
}

float sumCol(vector<vector<float>> data, int col) {
    float sum = 0;
    for (int i = 0; i < data.size(); i++) {
        sum += data[i][col];
    }
    return sum;
}



void copyPNodeToHeap(PNode* node, PNode* heap_node) {
    heap_node->setFeature(node->getFeature());
    heap_node->setThreshold(node->getThreshold());
    heap_node->setLeft(node->getLeft());
    heap_node->setRight(node->getRight());
}

void deleteTree (Node* root) {
    if (root->getType() == 'c') {
        delete root;
        return;
    }
    PNode* p_node = (PNode*) root;
    deleteTree(p_node->getLeft());
    deleteTree(p_node->getRight());
    delete root;
    return;
}

vector<tuple<int, float>> readPred(string filename) {
    vector<tuple<int, float>> result;
    fstream fin;
    fin.open(filename, ios::in);
    string row;
    while (getline(fin, row)) {
        stringstream row_stream(row);
        string f, t;
        getline(row_stream, f, ',');
        getline(row_stream, t, ',');
        result.push_back({stoi(f), stof(t)});
    }
    fin.close();
    return result;
}

vector<vector<float>> readCSV(fstream* fin, int n_instance, int n_feature) {
    vector<vector<float>> data(n_instance, vector<float>(n_feature));
    string row;
    for (int i = 0; i < n_instance; i++)
    {
        getline(*fin, row);
        stringstream row_stream(row);
        string feature;
        int c = 0;
        getline(row_stream, feature, ',');
        while (getline(row_stream, feature, ','))
        {
            data[i][c] = stof(feature);
            c++;
        }
    }
    fin->close();
    return data;
}

void splitData(vector<vector<float>>& data, vector<vector<float>>& train, vector<vector<float>>& test, int start, int end) {
    int n_instance = data.size();
    int test_size = end - start;
    // load first part training data
    for (int i = 0; i < start; i++) {
        train[i] = data[i];
    }
    // load testing data
    for (int i = start; i < end; i++) {
        test[i - start] = data[i];
    }
    // load second part training data
    for (int i = end; i < n_instance; i++) {
        train[i - test_size] = data[i];
    }
}

map<int, string> createMap(string l) {
    stringstream line_stream(l);
    string value;
    map<int, string> result;
    int key = 0;
    while (getline(line_stream, value, ','))
    {
        result[key] = value;
        key++;
    }
    return result;
}