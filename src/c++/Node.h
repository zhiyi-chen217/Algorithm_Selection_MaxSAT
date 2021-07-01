//
// Created by zhiyi on 6/29/21.
//

#ifndef C___NODE_H
#define C___NODE_H
#include <iostream>
#include <string>
#include <vector>
using namespace std;
class Node {
private:
    char type;

public:
    char getType();
    void setType(char type);
};

class PNode: public Node {
private:
    int feature;
    float threshold;
    Node* left_child;
    Node* right_child;

public:
    
    PNode(int feature, float threshold);
    
    // evaluate whether should go left of right
    bool evaluate(vector<float> data);

    // getters and setters
    int getFeature();
    void setFeature(int feature);

    float getThreshold();
    void setThreshold(float threshold);

    Node* getRight();
    void setRight(Node* right);

    Node* getLeft();
    void setLeft(Node* left);
};

class CNode: public Node {
private:
    int label;

public:
    CNode(int label);
    int getLabel();
    void setLabel(int label);

};


#endif //C___NODE_H
