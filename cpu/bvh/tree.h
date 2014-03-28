
#ifndef __TREE_H__
#define __TREE_H__

#include <list>

class Node
{
public:
	Node();
	Node(int depth);
	~Node();
	
	void addChild(Node* child);
	void setParent(const Node* parent);
	
	int getID();
	int getDepth();
	std::list<Node*>& getChildList();
	
	static Node* buildTree(std::list<Node*> &leafList);
	static void printTree(Node* root);
	static void breadthWalk(Node* root, std::list<Node*>& queue);
	
protected:
	std::list<Node*> childList;
	const Node* parent;
	int id;
	int depth;
};

#endif
