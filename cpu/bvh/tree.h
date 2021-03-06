
#ifndef __TREE_H__
#define __TREE_H__

#include <list>

#include "kdop.h"

class NodePair;

class Node
{
public:
	
	Node(KDOP* kdop);
	Node(KDOP* kdop, int depth);
	
	~Node();
	
	void addChild(Node* child);
	void setParent(const Node* parent);
	
	int getID();
	int getDepth();
	KDOP* getKDOP();
	std::list<Node*>& getChildList();
	
	static Node* buildTree(std::list<Node*> &leafList);
	static void printTree(Node* root);
	static void breadthWalk(Node* root, std::list<Node*>& queue);
	
	static void updateTree(Node* root);
	static void collides(Node* one, Node* two, std::vector<NodePair>& pairs);
	
protected:
	//TODO Refactor to be STL vector instead of list
	std::list<Node*> childList;
	const Node* parent;
	int id;
	int depth;
	
	KDOP* kdop;
};

class NodePair
{
public:
	NodePair(Node* first, Node* second):one(first), two(second) {}
	
	Node* one;
	Node* two;
};

#endif
