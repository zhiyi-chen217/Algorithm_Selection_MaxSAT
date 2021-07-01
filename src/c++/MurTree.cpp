#include "MurTree.h"
using namespace std;


MurTree::MurTree(Node* root) {
    this->root = root;
}

Node* MurTree::getRoot() {
    return this->root;
}

int MurTree::classify(vector<float> data) {
    Node* cur_node = this->root;
    while (cur_node->getType() != 'c')
    {
        PNode* cur_p_node = (PNode*)cur_node;
        if (cur_p_node->evaluate(data))
        {
            cur_node = cur_p_node->getRight();
        }
        else 
        {
            cur_node = cur_p_node->getLeft();
        }
        
    }
    CNode* cur_c_node = (CNode*)cur_node;
    return cur_c_node->getLabel();
}

