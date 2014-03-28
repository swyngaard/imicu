
#include "tree.h"

#include <iostream>

Node::Node():depth(0)
{
	
}

Node::Node(int depth)
{
	this->depth = depth;
}

Node::~Node()
{
	//Recursively delete children
	while(!childList.empty())
	{
		delete childList.front();
		
		childList.pop_front();
	}
}

int Node::getID()
{
	return id;
}

int Node::getDepth()
{
	return depth;
}

std::list<Node*>& Node::getChildList()
{
	return childList;
}

void Node::addChild(Node* child)
{
	childList.push_back(child);
}

void Node::setParent(const Node* parent)
{
	this->parent = parent;
}

Node* Node::buildTree(std::list<Node*> &leafList)
{
	std::list<Node*> currentNodes(leafList);
	std::list<Node*> nextNodes;
	
	int depth = 1;
	
	while(currentNodes.size() > 4)
	{
		int quotient  = currentNodes.size()/4;
		int remainder = currentNodes.size()%4;
		
		//Add an internal node for every four current nodes
		for(int i = 0; i < quotient; i++)
		{
			Node* internalNode = new Node(depth);
			
			currentNodes.front()->setParent(internalNode);
			internalNode->addChild(currentNodes.front());
			currentNodes.pop_front();
			
			currentNodes.front()->setParent(internalNode);
			internalNode->addChild(currentNodes.front());
			currentNodes.pop_front();
			
			currentNodes.front()->setParent(internalNode);
			internalNode->addChild(currentNodes.front());
			currentNodes.pop_front();
			
			currentNodes.front()->setParent(internalNode);
			internalNode->addChild(currentNodes.front());
			currentNodes.pop_front();
			
			nextNodes.push_back(internalNode);
		}
		
		//Add an internal node for any of the remaining current nodes
		if(remainder > 0)
		{
			Node * internalNode = new Node(depth);
			
			for(int i = 0; i < remainder; i++)
			{
				currentNodes.front()->setParent(internalNode);
				internalNode->addChild(currentNodes.front());
				currentNodes.pop_front();
			}
			
			nextNodes.push_back(internalNode);
		}
		
		currentNodes = nextNodes;
		
		nextNodes.clear();
		
		depth++;
	}
	
	Node* root = new Node(depth);
	
	while(!currentNodes.empty())
	{
		root->addChild(currentNodes.front());
		currentNodes.pop_front();
	}
	
	//Set root's parent to NULL
	root->setParent(0);
	
	return root;
}

void Node::printTree(Node* root)
{
	std::list<Node*> walkQ;
	std::list<Node*> printQ;
	
	walkQ.push_back(root);
	
	while(!walkQ.empty())
	{
		printQ.push_back(walkQ.front());
		
		std::list<Node*> childList = walkQ.front()->getChildList();
		
		for(std::list<Node*>::const_iterator child = childList.begin(); child != childList.end(); child++)
		{
			walkQ.push_back(*child);
		}
		
		walkQ.pop_front();
	}
	
	//Print
	int depth = printQ.front()->getDepth();
	
	std::cout << std::endl;
	
	while(!printQ.empty())
	{
		int currentDepth = printQ.front()->getDepth();
		
		if(currentDepth != depth)
		{
			depth = currentDepth;
			
			std::cout << std::endl;
		}
		
		std::cout << "* " << std::ends;
		
		printQ.pop_front();
	}
	
	std::cout << std::endl;
}

void Node::breadthWalk(Node* root, std::list<Node*>& queue)
{
	std::list<Node*> walkQ;
	
	walkQ.push_back(root);
	
	while(!walkQ.empty())
	{
		queue.push_back(walkQ.front());
		
		std::list<Node*> childList = walkQ.front()->getChildList();
		
		for(std::list<Node*>::const_iterator child = childList.begin(); child != childList.end(); child++)
		{
			walkQ.push_back(*child);
		}
		
		walkQ.pop_front();
	}
}
