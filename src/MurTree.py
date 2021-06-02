import pandas as pd
class PredicateNode:
    def __init__(self, feature, predicate, left=None, right=None):
        self.feature = feature
        self.predicate = predicate
        self.left_child = left
        self.right_child = right

    def evaluate(self, data):
        f = data[self.feature]
        return self.predicate(f)

    def setRight(self, right):
        self.right_child = right

    def setLeft(self, left):
        self.left_child = left

    def getRight(self):
        return self.right_child

    def getLeft(self):
        return self.left_child


class ClassificationNode:
    def __init__(self, label):
        self.label = label


class MurTree:
    def __init__(self, root):
        self.root = root

    def classify(self, data):

        curNode = self.root
        while type(curNode).__name__ == "PredicateNode":
            if curNode.evaluate(data):
                curNode = curNode.getRight()
            else:
                curNode = curNode.getLeft()

        return curNode.label

#
# root_node = PredicateNode("pixel263", lambda x: x > 100)
# left_node = PredicateNode("pixel179", lambda x: x > 100)
# right_node = PredicateNode("pixel233", lambda x: x > 100)
#
# root_node.setLeft(left_node)
# root_node.setRight(right_node)
#
# left_node.setLeft(ClassificationNode(1))
# left_node.setRight(ClassificationNode(3))
#
# right_node.setLeft(ClassificationNode(9))
# right_node.setRight(ClassificationNode(7))
#
# mur_tree = MurTree(root_node)
#
# data_set = pd.read_csv("../../train.csv").iloc[:50, :]
# result = []
# for ind in data_set.index:
#     data = data_set.loc[ind, :]
#     result.append(mur_tree.classify(data))
#
# count = 0
# for i in range(50):
#     if not data_set.iloc[i,:]['label'] == result[i]:
#         count += 1
