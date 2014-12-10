
#include "tree.h"

#include <iostream>

namespace pilar
{
	//Constructor that accepts KDOP pointer
	Node::Node(KDOP* kdop):depth(0),id(-1),strandID(-1)
	{
		this->kdop = kdop;
	}

	//Constructor that accepts KDOP and depth
	Node::Node(KDOP* kdop, int depth):id(-1),strandID(-1)
	{
		this->kdop = kdop;
		this->depth = depth;
	}
	
	//Constructor that accepts KDOP, depth, node ID and strand ID
	Node::Node(KDOP* kdop, int depth, int id, int strandID)
	{
		this->kdop = kdop;
		this->depth = depth;
		this->id = id;
		this->strandID = strandID;
		
		//Remember which particles are affected by this node
		this->particle1 = id;
		this->particle2 = id + 1;
	}
	
	Node::~Node()
	{
		//Recursively delete children
		while(!childList.empty())
		{
			delete childList.front();
			
			childList.pop_front();
		}
		
		//delete KDOP
		if(kdop != 0)
		{
			delete kdop;
			kdop = 0;
		}
	}

	int Node::getID()
	{
		return id;
	}
	
	int Node::getStrandID()
	{
		return strandID;
	}
	
	int Node::getParticleOneID()
	{
		return particle1;
	}
	
	int Node::getParticleTwoID()
	{
		return particle2;
	}

	int Node::getDepth()
	{
		return depth;
	}

	KDOP* Node::getKDOP()
	{
		return kdop;
	}

	std::list<Node*>& Node::getChildList()
	{
		return childList;
	}

	void Node::addChild(Node* child)
	{
		childList.push_back(child);
		
		//If empty KDOP set KDOP to child KDOP
		if(kdop == 0)
		{
			kdop = new KDOP(*child->getKDOP());
		}
		//else merge child KDOP with this KDOP
		else
		{
			kdop->merge(child->getKDOP());
		}
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
				Node* internalNode = new Node(0, depth);
				
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
				Node * internalNode = new Node(0, depth);
				
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
		
		Node* root = new Node(0, depth);
		
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

	//Recalculates KDOP of all internal nodes from leaf KDOPs
	void Node::updateTree(Node* root)
	{
		//Does the given node have children?
		if(root->getChildList().size() > 0)
		{
			std::list<Node*> childList = root->getChildList();
			
			std::list<Node*>::const_iterator child = childList.begin();
			
			for(; child != childList.end(); child++)
			{
				updateTree(*child);
			}
			
			child = childList.begin();
			
			//Set KDOP to KDOP of first child
			root->getKDOP()->setDistances((*child)->getKDOP());
			
			child++;
			
			//Merge the KDOP of the remaining children with this KDOP
			for(; child != childList.end(); child++)
			{
				root->getKDOP()->merge((*child)->getKDOP());
			}
		}
	}

	//Checks for collisions between leaf nodes of the given trees. The list of colliding nodes is saved in a vector.
	void Node::collides(Node* one, Node* two, std::vector<NodePair>& pairs)
	{
		//Check for collision
		if(one->getKDOP()->collides(two->getKDOP()))
		{
			//Do these nodes have children?
			if(one->getChildList().size() > 0 && two->getChildList().size() > 0)
			{
				std::list<Node*> oneChildren = one->getChildList();
				std::list<Node*> twoChildren = two->getChildList();
				
				//Recursively check each child node for collisions
				for(std::list<Node*>::const_iterator oneChild = oneChildren.begin(); oneChild != oneChildren.end(); oneChild++)
				{
					for(std::list<Node*>::const_iterator twoChild = twoChildren.begin(); twoChild != twoChildren.end(); twoChild++)
					{
						collides(*oneChild, *twoChild, pairs);
					}
				}
			}
			else
			{
				//If the given nodes have no children this is a leaf node collision. Save result as collision pair.
				pairs.push_back(NodePair(one, two));
			}
		}
	}

}
