#include "Node.h"
#include <iostream>
#include <string>

using namespace std;

class MurTree
{
private:
    Node* root;
public:
    MurTree(Node* root);
    Node* getRoot();
    int classify(vector<float> data);  
};



