from graphviz import Digraph, Graph
class PredicateNode(object):
    def __init__(self, feature, predicate, threshold, left=None, right=None):
        self.feature = feature
        self.predicate = predicate
        self.threshold = threshold
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

    def __str__(self):
        return "feature: {}\nthreshold: {}".format(self.feature, self.threshold)



class ClassificationNode:
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return "label: {}".format(self.label)


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

    def drawTree(self):
        dot = Digraph(node_attr={'fontsize': '5', 'style': 'filled'})
        dot.attr(size='10,10')
        queue = [("", self.root)]
        edges = []
        while len(queue) > 0:
            (name, node) = queue.pop(0)
            dot.node(name, str(node))
            if not name == "":
                edges.append((name[:-1], name))
            print(node)
            if not type(node).__name__ == "ClassificationNode":
                left = node.getLeft()
                queue.append((name+"0", left))
                right = node.getRight()
                queue.append((name+"1", right))
        dot.edges(edges)
        dot.render('decision_tree', view=True)


#
# root_node = PredicateNode("pixel263", lambda x: x > 100, 100)
# left_node = PredicateNode("pixel179", lambda x: x > 100, 100)
# right_node = PredicateNode("pixel233", lambda x: x > 100, 100)
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
# mur_tree.drawTree()
