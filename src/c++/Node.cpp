//
// Created by zhiyi on 6/29/21.
//

#include "Node.h"

// parent Node Class
char Node::getType() {
    return this->type;
}

void Node::setType(char type) {
    this->type = type;
}


// child PNode class

PNode::PNode(int feature, float threshold) {
    this->feature = feature;
    this->threshold = threshold;
    this->setType('p');
}

bool PNode::evaluate(vector<float> data) {
    float f = data[feature];
    return f > threshold;
}

int PNode::getFeature() {
    return this->feature;
}

void PNode::setFeature(int feature) {
    this->feature = feature;
}

float PNode::getThreshold() {
    return this->threshold;
}

void PNode::setThreshold(float threshold) {
    this->threshold = threshold;
}

void PNode::setRight(Node* right) {
    this->right_child = right;
}

void PNode::setLeft(Node* left) {
    this->left_child = left;
}

Node* PNode::getRight() {
    return this->right_child;
}

Node* PNode::getLeft() {
    return this->left_child;
}

// child CNode type

CNode::CNode(int label) {
    this->setType('c');
    this->label = label;
}

int CNode::getLabel() {
    return this->label;
}

void CNode::setLabel(int label) {
    this->label = label;
}

