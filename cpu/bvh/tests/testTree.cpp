
#include "testTree.h"

#include <vector>

void TreeTest::testBuildTree()
{
	//1 leaf node
	std::vector<int> expected1;
	expected1.push_back(1); //level 0
	expected1.push_back(1); //level 1
	
	buildTree(1, expected1);
	
	//3 leaf nodes
	std::vector<int> expected3;
	expected3.push_back(3); //level 0
	expected3.push_back(1); //level 1
	
	buildTree(3, expected3);
	
	//4 leaf nodes
	std::vector<int> expected4;
	expected4.push_back(4); //level 0
	expected4.push_back(1); //level 1
	
	buildTree(4, expected4);
	
	//5 leaf nodes
	std::vector<int> expected5;
	expected5.push_back(5); //level 0
	expected5.push_back(2); //level 1
	expected5.push_back(1); //level 2
	
	buildTree(5, expected5);
	
	//16 leaf nodes
	std::vector<int> expected16;
	expected16.push_back(16); //level 0
	expected16.push_back(4);  //level 1
	expected16.push_back(1);  //level 2
	
	buildTree(16, expected16);
	
	//17 leaf nodes
	std::vector<int> expected17;
	expected17.push_back(17); //level 0
	expected17.push_back(5);  //level 1
	expected17.push_back(2);  //level 2
	expected17.push_back(1);  //level 3
	
	buildTree(17, expected17);
	
}

void TreeTest::buildTree(int numLeaves, std::vector<int>& expected)
{
	std::list<Node*> leafNodes;
	
	for(int i = 0; i < numLeaves; i++)
		leafNodes.push_front(new Node);
	
	Node* root = Node::buildTree(leafNodes);
	
	std::list<Node*> queue;
	Node::breadthWalk(root, queue);
	
	int depth = -1;
	std::vector<int> actual;
	
	while(!queue.empty())
	{
		int currentDepth = queue.back()->getDepth();
		
		if(currentDepth != depth)
		{
			depth = currentDepth;
			
			actual.push_back(0);
		}
		
		actual[depth]++;
		
		queue.pop_back();
	}
	
	Node::printTree(root);
	
	delete root;
	
	for(int i = 0; i < expected.size(); i++)
		CPPUNIT_ASSERT_EQUAL(expected[i], actual[i]);
}
