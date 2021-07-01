#include <vector>
#include "Node.h"
#include <fstream>
#include <sstream>
#include <map>
#include <tuple>
using namespace std;

// helper functions
int argmax(vector<float> data);

float sumCol(vector<vector<float>> data, int col);

void copyPNodeToHeap(PNode* node, PNode* heap_node);

void deleteTree (Node* root);

vector<tuple<int, float>> readPred(string filename);

vector<vector<float>> readCSV(fstream* fin, int n_instance, int n_feature);


map<int, string> createMap(string l);


