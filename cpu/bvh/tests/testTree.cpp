
#include "testTree.h"

#include <vector>

void TreeTest::testBuildTree()
{
	//1 leaf node
	std::vector<int> expected1;
	expected1.push_back(1); //level 0
	expected1.push_back(1); //level 1
	
	checkTreeDepths(1, expected1);
	
	//3 leaf nodes
	std::vector<int> expected3;
	expected3.push_back(3); //level 0
	expected3.push_back(1); //level 1
	
	checkTreeDepths(3, expected3);
	
	//4 leaf nodes
	std::vector<int> expected4;
	expected4.push_back(4); //level 0
	expected4.push_back(1); //level 1
	
	checkTreeDepths(4, expected4);
	
	//5 leaf nodes
	std::vector<int> expected5;
	expected5.push_back(5); //level 0
	expected5.push_back(2); //level 1
	expected5.push_back(1); //level 2
	
	checkTreeDepths(5, expected5);
	
	//16 leaf nodes
	std::vector<int> expected16;
	expected16.push_back(16); //level 0
	expected16.push_back(4);  //level 1
	expected16.push_back(1);  //level 2
	
	checkTreeDepths(16, expected16);
	
	//17 leaf nodes
	std::vector<int> expected17;
	expected17.push_back(17); //level 0
	expected17.push_back(5);  //level 1
	expected17.push_back(2);  //level 2
	expected17.push_back(1);  //level 3
	
	checkTreeDepths(17, expected17);
	
}

void TreeTest::checkTreeDepths(int numLeaves, std::vector<int>& expected)
{
	std::list<Node*> leafNodes;
	
	//add leaf nodes with KDOPs
	for(int i = 0; i < numLeaves; i++)
	{
		//Create vertices of triangle and offset Y position by index
		std::vector<Vector3f> vertices;
		vertices.push_back(Vector3f( 5.5f,  2.3f - (i*5.0f), -7.9f));
		vertices.push_back(Vector3f(-7.1f,  8.8f - (i*5.0f),  0.3f));
		vertices.push_back(Vector3f( 1.5f, -6.4f - (i*5.0f),  4.0f));
		
		//Create KDOP from vertices
		KDOP* kdop = new KDOP(vertices);
		
		//Add list of leaf nodes to the tree
		leafNodes.push_front(new Node(kdop));
	}
	
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
	
	//DEBUG print tree node hierarchy
	//~ Node::printTree(root);
	
	delete root;
	
	for(int i = 0; i < expected.size(); i++)
		CPPUNIT_ASSERT_EQUAL(expected[i], actual[i]);
}

void TreeTest::testKDOPTree()
{
	//Generate 1 leaf nodes with KDOP from vertices
	std::vector<Vector3f> leaf_1_1;
	leaf_1_1.push_back(Vector3f( 5.5f,  2.3f, -7.9f));
	leaf_1_1.push_back(Vector3f(-7.1f,  8.8f,  0.3f));
	leaf_1_1.push_back(Vector3f( 1.5f, -6.4f,  4.0f));
	
	std::vector< std::vector<Vector3f> > inputVerts_1;
	inputVerts_1.push_back(leaf_1_1);
	
	//Expected output
	float expected_1_6 [2][6]  = {{ -7.1f,  -6.4f,  -7.9f,   5.5f,   8.8f,   4.0f},
								  { -7.1f,  -6.4f,  -7.9f,   5.5f,   8.8f,   4.0f}};
	float expected_1_14[2][14] = {{ -7.1f,  -6.4f,  -7.9f,  -0.9f, -15.6f,  -8.9f, -16.2f,   5.5f,   8.8f,   4.0f,   2.0f,  11.9f,  15.7f,  11.1f},
								  { -7.1f,  -6.4f,  -7.9f,  -0.9f, -15.6f,  -8.9f, -16.2f,   5.5f,   8.8f,   4.0f,   2.0f,  11.9f,  15.7f,  11.1f}};
	float expected_1_18[2][18] = {{ -7.1f,  -6.4f,  -7.9f,  -4.9f,  -6.8f,  -5.6f, -15.9f,  -7.4f, -10.4f,   5.5f,   8.8f,   4.0f,   7.8f,   5.5f,   9.1f,   7.9f,  13.4f,  10.2f},
								  { -7.1f,  -6.4f,  -7.9f,  -4.9f,  -6.8f,  -5.6f, -15.9f,  -7.4f, -10.4f,   5.5f,   8.8f,   4.0f,   7.8f,   5.5f,   9.1f,   7.9f,  13.4f,  10.2f}};
	float expected_1_26[2][26] = {{ -7.1f,  -6.4f,  -7.9f,  -0.9f, -15.6f,  -8.9f, -16.2f,  -4.9f,  -6.8f,  -5.6f, -15.9f,  -7.4f, -10.4f,   5.5f,   8.8f,   4.0f,   2.0f,  11.9f,  15.7f,  11.1f,   7.8f,   5.5f,   9.1f,   7.9f,  13.4f,  10.2f},
								  { -7.1f,  -6.4f,  -7.9f,  -0.9f, -15.6f,  -8.9f, -16.2f,  -4.9f,  -6.8f,  -5.6f, -15.9f,  -7.4f, -10.4f,   5.5f,   8.8f,   4.0f,   2.0f,  11.9f,  15.7f,  11.1f,   7.8f,   5.5f,   9.1f,   7.9f,  13.4f,  10.2f}};
	
	checkTreeDistances(inputVerts_1, expected_1_6, expected_1_14, expected_1_18, expected_1_26);
	
	//Generate 4 leaf nodes with KDOP from vertices
	std::vector<Vector3f> leaf_4_1;
	leaf_4_1.push_back(Vector3f(-2.6f, -3.2f, -4.5f));
	leaf_4_1.push_back(Vector3f(-9.7f, -7.1f, -9.0f));
	leaf_4_1.push_back(Vector3f(-9.8f, -9.1f, -6.4f));
	std::vector<Vector3f> leaf_4_2;
	leaf_4_2.push_back(Vector3f( 8.5f,  1.9f, -6.4f));
	leaf_4_2.push_back(Vector3f(-1.1f, -4.1f,  3.8f));
	leaf_4_2.push_back(Vector3f( 9.8f, -2.9f,  7.4f));
	std::vector<Vector3f> leaf_4_3;
	leaf_4_3.push_back(Vector3f(-3.7f, -7.5f, -2.5f));
	leaf_4_3.push_back(Vector3f( 6.4f, -5.6f,  6.0f));
	leaf_4_3.push_back(Vector3f(-8.1f,  0.1f, -9.4f));
	std::vector<Vector3f> leaf_4_4;
	leaf_4_4.push_back(Vector3f(-4.7f, -2.0f,  6.8f));
	leaf_4_4.push_back(Vector3f( 1.3f,  7.8f, -6.1f));
	leaf_4_4.push_back(Vector3f(-3.7f,  8.6f, -3.8f));
	
	std::vector< std::vector<Vector3f> > inputVerts_4;
	inputVerts_4.push_back(leaf_4_1);
	inputVerts_4.push_back(leaf_4_2);
	inputVerts_4.push_back(leaf_4_3);
	inputVerts_4.push_back(leaf_4_4);
	
	
	float expected_4_6 [5][6] = {{-9.8f,  -9.1f,  -9.4f,   9.8f,   8.6f,   7.4f},
								 {-9.8f,  -9.1f,  -9.0f,  -2.6f,  -3.2f,  -4.5f},
								 {-1.1f,  -4.1f,  -6.4f,   9.8f,   1.9f,   7.4f},
								 {-8.1f,  -7.5f,  -9.4f,   6.4f,   0.1f,   6.0f},
								 {-4.7f,  -2.0f,  -6.1f,   1.3f,   8.6f,   6.8f}};
	float expected_4_14[5][14] = {{ -9.8f,  -9.1f,  -9.4f, -25.8f, -17.6f, -13.5f,  -9.5f,   9.8f,   8.6f,   7.4f,  14.3f,  20.1f,  16.8f,  13.0f},
								  { -9.8f,  -9.1f,  -9.0f, -25.8f, -11.6f, -12.5f,   5.1f,  -2.6f,  -3.2f,  -4.5f, -10.3f,  -3.9f,  -1.3f,   6.4f},
								  { -1.1f,  -4.1f,  -6.4f,  -1.4f,   0.2f,  -9.0f,  -0.8f,   9.8f,   1.9f,   7.4f,  14.3f,  20.1f,  16.8f,  13.0f},
								  { -8.1f,  -7.5f,  -9.4f, -17.4f, -17.6f,  -8.7f,   1.2f,   6.4f,   0.1f,   6.0f,   6.8f,  18.0f,   1.4f,   6.3f},
								  { -4.7f,  -2.0f,  -6.1f,   0.1f, -16.1f, -13.5f,  -9.5f,   1.3f,   8.6f,   6.8f,   3.0f,   4.1f,  15.2f,  -0.4f}};
	float expected_4_18[5][18] = {{ -9.8f,  -9.1f,  -9.4f, -18.9f, -18.7f, -16.1f, -12.3f, -11.5f, -11.6f,   9.8f,   8.6f,   7.4f,  10.4f,  17.2f,   4.8f,  12.7f,  14.9f,  13.9f},
								  { -9.8f,  -9.1f,  -9.0f, -18.9f, -18.7f, -16.1f,  -2.6f,  -3.4f,  -2.7f,  -2.6f,  -3.2f,  -4.5f,  -5.8f,  -7.1f,  -7.7f,   0.6f,   1.9f,   1.9f},
								  { -1.1f,  -4.1f,  -6.4f,  -5.2f,   2.1f,  -4.5f,   3.0f,  -4.9f, -10.3f,   9.8f,   1.9f,   7.4f,  10.4f,  17.2f,   4.5f,  12.7f,  14.9f,   8.3f},
								  { -8.1f,  -7.5f,  -9.4f, -11.2f, -17.5f, -10.0f,  -8.2f,  -1.2f, -11.6f,   6.4f,   0.1f,   6.0f,   0.8f,  12.4f,   0.4f,  12.0f,   1.3f,   9.5f},
								  { -4.7f,  -2.0f,  -6.1f,  -6.7f,  -7.5f,   1.7f, -12.3f, -11.5f,  -8.8f,   1.3f,   8.6f,   6.8f,   9.1f,   2.1f,   4.8f,  -2.7f,   7.4f,  13.9f}};
	float expected_4_26[5][26] = {{ -9.8f,  -9.1f,  -9.4f, -25.8f, -17.6f, -13.5f,  -9.5f, -18.9f, -18.7f, -16.1f, -12.3f, -11.5f, -11.6f,   9.8f,   8.6f,   7.4f,  14.3f,  20.1f,  16.8f,  13.0f,  10.4f,  17.2f,   4.8f,  12.7f,  14.9f,  13.9f},
								  { -9.8f,  -9.1f,  -9.0f, -25.8f, -11.6f, -12.5f,   5.1f, -18.9f, -18.7f, -16.1f,  -2.6f,  -3.4f,  -2.7f,  -2.6f,  -3.2f,  -4.5f, -10.3f,  -3.9f,  -1.3f,   6.4f,  -5.8f,  -7.1f,  -7.7f,   0.6f,   1.9f,   1.9f},
								  { -1.1f,  -4.1f,  -6.4f,  -1.4f,   0.2f,  -9.0f,  -0.8f,  -5.2f,   2.1f,  -4.5f,   3.0f,  -4.9f, -10.3f,   9.8f,   1.9f,   7.4f,  14.3f,  20.1f,  16.8f,  13.0f,  10.4f,  17.2f,   4.5f,  12.7f,  14.9f,   8.3f},
								  { -8.1f,  -7.5f,  -9.4f, -17.4f, -17.6f,  -8.7f,   1.2f, -11.2f, -17.5f, -10.0f,  -8.2f,  -1.2f, -11.6f,   6.4f,   0.1f,   6.0f,   6.8f,  18.0f,   1.4f,   6.3f,   0.8f,  12.4f,   0.4f,  12.0f,   1.3f,   9.5f},
								  { -4.7f,  -2.0f,  -6.1f,   0.1f, -16.1f, -13.5f,  -9.5f,  -6.7f,  -7.5f,   1.7f, -12.3f, -11.5f,  -8.8f,   1.3f,   8.6f,   6.8f,   3.0f,   4.1f,  15.2f,  -0.4f,   9.1f,   2.1f,   4.8f,  -2.7f,   7.4f,  13.9f}};
	
	checkTreeDistances(inputVerts_4,
					   expected_4_6,
					   expected_4_14,
					   expected_4_18,
					   expected_4_26);
	
	//Generate 5 leaf nodes with KDOP from vertices
	std::vector<Vector3f> leaf_5_1;
	leaf_5_1.push_back(Vector3f( 5.9f,  8.3f,  2.1f));
	leaf_5_1.push_back(Vector3f( 7.7f, -0.8f, -4.9f));
	leaf_5_1.push_back(Vector3f(-3.5f,  4.3f, -8.6f));
	std::vector<Vector3f> leaf_5_2;
	leaf_5_2.push_back(Vector3f( 5.3f,  5.8f,  2.4f));
	leaf_5_2.push_back(Vector3f( 1.1f, -4.4f, -8.1f));
	leaf_5_2.push_back(Vector3f( 3.0f, -2.6f, -2.4f));
	std::vector<Vector3f> leaf_5_3;
	leaf_5_3.push_back(Vector3f( 3.1f,  1.8f,  6.7f));
	leaf_5_3.push_back(Vector3f( 0.1f,  9.6f,  8.8f));
	leaf_5_3.push_back(Vector3f( 2.6f, -7.5f, -5.0f));
	std::vector<Vector3f> leaf_5_4;
	leaf_5_4.push_back(Vector3f(-0.8f, -0.3f,  0.8f));
	leaf_5_4.push_back(Vector3f( 9.9f,  1.2f, -6.5f));
	leaf_5_4.push_back(Vector3f( 7.6f, -3.7f,  0.9f));
	std::vector<Vector3f> leaf_5_5;
	leaf_5_5.push_back(Vector3f(-9.3f, -2.7f,  9.1f));
	leaf_5_5.push_back(Vector3f( 1.8f, -5.0f,  6.6f));
	leaf_5_5.push_back(Vector3f(-5.5f, -1.8f,  6.3f));
	
	std::vector< std::vector<Vector3f> > inputVerts_5;
	inputVerts_5.push_back(leaf_5_1);
	inputVerts_5.push_back(leaf_5_2);
	inputVerts_5.push_back(leaf_5_3);
	inputVerts_5.push_back(leaf_5_4);
	inputVerts_5.push_back(leaf_5_5);
	
	float expected_5_6[8][6] = {{ -9.3f,  -7.5f,  -8.6f,   9.9f,   9.6f,   9.1f},
								{ -3.5f,  -7.5f,  -8.6f,   9.9f,   9.6f,   8.8f},
								{ -9.3f,  -5.0f,   6.3f,   1.8f,  -1.8f,   9.1f},
								{ -3.5f,  -0.8f,  -8.6f,   7.7f,   8.3f,   2.1f},
								{  1.1f,  -4.4f,  -8.1f,   5.3f,   5.8f,   2.4f},
								{  0.1f,  -7.5f,  -5.0f,   3.1f,   9.6f,   8.8f},
								{ -0.8f,  -3.7f,  -6.5f,   9.9f,   1.2f,   0.9f},
								{ -9.3f,  -5.0f,   6.3f,   1.8f,  -1.8f,   9.1f}};
	float expected_5_14[8][14] = {{ -9.3f,  -7.5f,  -8.6f, -11.4f, -16.4f, -21.1f, -18.3f,   9.9f,   9.6f,   9.1f,  18.5f,  13.4f,  17.6f,  15.2f},
								  { -3.5f,  -7.5f,  -8.6f, -11.4f, -16.4f,  -1.9f, -18.3f,   9.9f,   9.6f,   8.8f,  18.5f,  12.2f,  17.6f,  15.2f},
								  { -9.3f,  -5.0f,   6.3f,  -2.9f,   2.5f, -21.1f, -15.7f,   1.8f,  -1.8f,   9.1f,   3.4f,  13.4f,  -9.8f,   0.2f},
								  { -3.5f,  -0.8f,  -8.6f,  -7.8f, -16.4f,   9.4f,  -4.5f,   7.7f,   8.3f,   2.1f,  16.3f,   3.6f,  12.1f,  13.4f},
								  {  1.1f,  -4.4f,  -8.1f, -11.4f,  -2.6f,   2.8f,  -2.9f,   5.3f,   5.8f,   2.4f,  13.5f,   3.2f,   8.7f,  13.6f},
								  {  0.1f,  -7.5f,  -5.0f,  -9.9f,  -0.7f,  -1.8f, -18.3f,   3.1f,   9.6f,   8.8f,  18.5f,   8.0f,   0.9f,  15.1f},
								  { -0.8f,  -3.7f,  -6.5f,  -0.3f,   0.3f,  -1.9f,  -1.3f,   9.9f,   1.2f,   0.9f,   4.8f,  12.2f,  17.6f,  15.2f},
								  { -9.3f,  -5.0f,   6.3f,  -2.9f,   2.5f, -21.1f, -15.7f,   1.8f,  -1.8f,   9.1f,   3.4f,  13.4f,  -9.8f,   0.2f}};
	float expected_5_18[8][18] = {{ -9.3f,  -7.5f,  -8.6f, -12.0f, -12.1f, -12.5f,  -9.5f, -18.4f, -11.8f,   9.9f,   9.6f,   9.1f,  14.2f,   9.8f,  18.4f,  11.3f,  16.4f,  12.9f},
								  { -3.5f,  -7.5f,  -8.6f,  -4.9f, -12.1f, -12.5f,  -9.5f,  -8.7f,  -4.9f,   9.9f,   9.6f,   8.8f,  14.2f,   9.8f,  18.4f,  11.3f,  16.4f,  12.9f},
								  { -9.3f,  -5.0f,   6.3f, -12.0f,  -0.2f,   1.6f,  -6.6f, -18.4f, -11.8f,   1.8f,  -1.8f,   9.1f,  -3.2f,   8.4f,   6.4f,   6.8f,  -4.8f,  -8.1f},
								  { -3.5f,  -0.8f,  -8.6f,   0.8f, -12.1f,  -5.7f,  -7.8f,   3.8f,   4.1f,   7.7f,   8.3f,   2.1f,  14.2f,   8.0f,  10.4f,   8.5f,  12.6f,  12.9f},
								  {  1.1f,  -4.4f,  -8.1f,  -3.3f,  -7.0f, -12.5f,  -0.5f,   2.9f,  -0.2f,   5.3f,   5.8f,   2.4f,  11.1f,   7.7f,   8.2f,   5.6f,   9.2f,   3.7f},
								  {  0.1f,  -7.5f,  -5.0f,  -4.9f,  -2.4f, -12.5f,  -9.5f,  -8.7f,  -4.9f,   3.1f,   9.6f,   8.8f,   9.7f,   9.8f,  18.4f,  10.1f,   7.6f,   0.8f},
								  { -0.8f,  -3.7f,  -6.5f,  -1.1f,   0.0f,  -5.3f,  -0.5f,  -1.6f,  -4.6f,   9.9f,   1.2f,   0.9f,  11.1f,   8.5f,   0.5f,  11.3f,  16.4f,   7.7f},
								  { -9.3f,  -5.0f,   6.3f, -12.0f,  -0.2f,   1.6f,  -6.6f, -18.4f, -11.8f,   1.8f,  -1.8f,   9.1f,  -3.2f,   8.4f,   6.4f,   6.8f,  -4.8f,  -8.1f}};
	float expected_5_26[8][26] = {{ -9.3f,  -7.5f,  -8.6f, -11.4f, -16.4f, -21.1f, -18.3f, -12.0f, -12.1f, -12.5f,  -9.5f, -18.4f, -11.8f,   9.9f,   9.6f,   9.1f,  18.5f,  13.4f,  17.6f,  15.2f,  14.2f,   9.8f,  18.4f,  11.3f,  16.4f,  12.9f},
								  { -3.5f,  -7.5f,  -8.6f, -11.4f, -16.4f,  -1.9f, -18.3f,  -4.9f, -12.1f, -12.5f,  -9.5f,  -8.7f,  -4.9f,   9.9f,   9.6f,   8.8f,  18.5f,  12.2f,  17.6f,  15.2f,  14.2f,   9.8f,  18.4f,  11.3f,  16.4f,  12.9f},
								  { -9.3f,  -5.0f,   6.3f,  -2.9f,   2.5f, -21.1f, -15.7f, -12.0f,  -0.2f,   1.6f,  -6.6f, -18.4f, -11.8f,   1.8f,  -1.8f,   9.1f,   3.4f,  13.4f,  -9.8f,   0.2f,  -3.2f,   8.4f,   6.4f,   6.8f,  -4.8f,  -8.1f},
								  { -3.5f,  -0.8f,  -8.6f,  -7.8f, -16.4f,   9.4f,  -4.5f,   0.8f, -12.1f,  -5.7f,  -7.8f,   3.8f,   4.1f,   7.7f,   8.3f,   2.1f,  16.3f,   3.6f,  12.1f,  13.4f,  14.2f,   8.0f,  10.4f,   8.5f,  12.6f,  12.9f},
								  {  1.1f,  -4.4f,  -8.1f, -11.4f,  -2.6f,   2.8f,  -2.9f,  -3.3f,  -7.0f, -12.5f,  -0.5f,   2.9f,  -0.2f,   5.3f,   5.8f,   2.4f,  13.5f,   3.2f,   8.7f,  13.6f,  11.1f,   7.7f,   8.2f,   5.6f,   9.2f,   3.7f},
								  {  0.1f,  -7.5f,  -5.0f,  -9.9f,  -0.7f,  -1.8f, -18.3f,  -4.9f,  -2.4f, -12.5f,  -9.5f,  -8.7f,  -4.9f,   3.1f,   9.6f,   8.8f,  18.5f,   8.0f,   0.9f,  15.1f,   9.7f,   9.8f,  18.4f,  10.1f,   7.6f,   0.8f},
								  { -0.8f,  -3.7f,  -6.5f,  -0.3f,   0.3f,  -1.9f,  -1.3f,  -1.1f,   0.0f,  -5.3f,  -0.5f,  -1.6f,  -4.6f,   9.9f,   1.2f,   0.9f,   4.8f,  12.2f,  17.6f,  15.2f,  11.1f,   8.5f,   0.5f,  11.3f,  16.4f,   7.7f},
								  { -9.3f,  -5.0f,   6.3f,  -2.9f,   2.5f, -21.1f, -15.7f, -12.0f,  -0.2f,   1.6f,  -6.6f, -18.4f, -11.8f,   1.8f,  -1.8f,   9.1f,   3.4f,  13.4f,  -9.8f,   0.2f,  -3.2f,   8.4f,   6.4f,   6.8f,  -4.8f,  -8.1f}};
	
	checkTreeDistances(inputVerts_5, expected_5_6, expected_5_14, expected_5_18, expected_5_26);
	
	//Generate 16 leaf nodes with KDOP from vertices
	std::vector<Vector3f> leaf_16_1;
	leaf_16_1.push_back(Vector3f(-2.6f, -2.8f,  6.4f));
	leaf_16_1.push_back(Vector3f(-6.6f,  1.6f, -7.3f));
	leaf_16_1.push_back(Vector3f( 5.3f,  3.6f,  3.4f));
	
	std::vector<Vector3f> leaf_16_2;
	leaf_16_2.push_back(Vector3f(10.0f, -2.1f,  0.0f));
	leaf_16_2.push_back(Vector3f(-3.2f, -7.9f,  7.0f));
	leaf_16_2.push_back(Vector3f( 2.2f, -8.0f, -6.7f));
	
	std::vector<Vector3f> leaf_16_3;
	leaf_16_3.push_back(Vector3f( 3.0f, -3.5f,  8.9f));
	leaf_16_3.push_back(Vector3f( 7.0f, -0.4f,  7.6f));
	leaf_16_3.push_back(Vector3f( 2.2f,  2.1f, -9.3f));
	
	std::vector<Vector3f> leaf_16_4;
	leaf_16_4.push_back(Vector3f(-3.8f, -1.3f, -8.5f));
	leaf_16_4.push_back(Vector3f(-2.9f, -5.5f, -4.0f));
	leaf_16_4.push_back(Vector3f( 5.0f, -2.8f,  1.4f));
	
	std::vector<Vector3f> leaf_16_5;
	leaf_16_5.push_back(Vector3f(-2.8f,  5.2f, -3.3f));
	leaf_16_5.push_back(Vector3f( 5.5f,  0.1f,  3.1f));
	leaf_16_5.push_back(Vector3f( 1.1f,  0.3f,  8.9f));
	
	std::vector<Vector3f> leaf_16_6;
	leaf_16_6.push_back(Vector3f( 2.9f,  0.2f,  0.9f));
	leaf_16_6.push_back(Vector3f(-3.7f, -9.2f,  9.7f));
	leaf_16_6.push_back(Vector3f(-4.6f, -5.9f,  5.2f));
	
	std::vector<Vector3f> leaf_16_7;
	leaf_16_7.push_back(Vector3f(-5.4f, -8.9f, -7.8f));
	leaf_16_7.push_back(Vector3f(-8.2f,  3.7f, -1.5f));
	leaf_16_7.push_back(Vector3f(-8.1f, -9.6f,  3.7f));
	
	std::vector<Vector3f> leaf_16_8;
	leaf_16_8.push_back(Vector3f( 8.6f,  9.8f, -3.7f));
	leaf_16_8.push_back(Vector3f(-6.1f, -5.7f,  7.3f));
	leaf_16_8.push_back(Vector3f( 3.3f, -5.9f, -3.9f));
	
	std::vector<Vector3f> leaf_16_9;
	leaf_16_9.push_back(Vector3f(-1.0f,  9.6f,  7.2f));
	leaf_16_9.push_back(Vector3f( 5.2f, -1.9f,  7.7f));
	leaf_16_9.push_back(Vector3f(-6.6f,  9.7f,  3.3f));
	
	std::vector<Vector3f> leaf_16_10;
	leaf_16_10.push_back(Vector3f(-3.5f,  2.6f,  2.3f));
	leaf_16_10.push_back(Vector3f( 7.1f, -0.2f,  5.5f));
	leaf_16_10.push_back(Vector3f(-8.8f,  9.0f, -7.5f));
	
	std::vector<Vector3f> leaf_16_11;
	leaf_16_11.push_back(Vector3f(-0.3f, -5.0f, 10.0f));
	leaf_16_11.push_back(Vector3f(-5.6f,  5.1f, -1.5f));
	leaf_16_11.push_back(Vector3f( 7.4f, -7.3f, -4.6f));
	
	std::vector<Vector3f> leaf_16_12;
	leaf_16_12.push_back(Vector3f( 1.1f,  4.9f, -3.3f));
	leaf_16_12.push_back(Vector3f(-9.2f,  4.7f,  0.5f));
	leaf_16_12.push_back(Vector3f(-6.4f,  4.8f, -9.5f));
	
	std::vector<Vector3f> leaf_16_13;
	leaf_16_13.push_back(Vector3f(-7.9f,  9.4f,  9.1f));
	leaf_16_13.push_back(Vector3f( 0.2f, -3.8f,  6.0f));
	leaf_16_13.push_back(Vector3f( 3.3f,  8.9f,  9.2f));
	
	std::vector<Vector3f> leaf_16_14;
	leaf_16_14.push_back(Vector3f(-0.6f,  2.6f,  2.1f));
	leaf_16_14.push_back(Vector3f(-9.1f, -3.1f, -0.5f));
	leaf_16_14.push_back(Vector3f(-4.2f,  3.6f, -5.6f));
	
	std::vector<Vector3f> leaf_16_15;
	leaf_16_15.push_back(Vector3f(-2.1f,  5.5f, -3.8f));
	leaf_16_15.push_back(Vector3f(-1.1f,  7.5f, -0.4f));
	leaf_16_15.push_back(Vector3f( 1.0f, -4.3f, -7.2f));
	
	std::vector<Vector3f> leaf_16_16;
	leaf_16_16.push_back(Vector3f(-6.6f,  9.3f, -4.5f));
	leaf_16_16.push_back(Vector3f(-0.9f, -9.7f,  3.3f));
	leaf_16_16.push_back(Vector3f(-6.1f, -1.0f,  3.3f));
	
	std::vector< std::vector<Vector3f> > inputVerts_16;
	inputVerts_16.push_back(leaf_16_1);
	inputVerts_16.push_back(leaf_16_2);
	inputVerts_16.push_back(leaf_16_3);
	inputVerts_16.push_back(leaf_16_4);
	inputVerts_16.push_back(leaf_16_5);
	inputVerts_16.push_back(leaf_16_6);
	inputVerts_16.push_back(leaf_16_7);
	inputVerts_16.push_back(leaf_16_8);
	inputVerts_16.push_back(leaf_16_9);
	inputVerts_16.push_back(leaf_16_10);
	inputVerts_16.push_back(leaf_16_11);
	inputVerts_16.push_back(leaf_16_12);
	inputVerts_16.push_back(leaf_16_13);
	inputVerts_16.push_back(leaf_16_14);
	inputVerts_16.push_back(leaf_16_15);
	inputVerts_16.push_back(leaf_16_16);
	
	float expected_16_6[21][6 ] = {{ -9.2f,  -9.7f,  -9.5f,  10.0f,   9.8f,  10.0f},
								   { -6.6f,  -8.0f,  -9.3f,  10.0f,   3.6f,   8.9f},
								   { -8.2f,  -9.6f,  -7.8f,   8.6f,   9.8f,   9.7f},
								   { -9.2f,  -7.3f,  -9.5f,   7.4f,   9.7f,  10.0f},
								   { -9.1f,  -9.7f,  -7.2f,   3.3f,   9.4f,   9.2f},
								   { -6.6f,  -2.8f,  -7.3f,   5.3f,   3.6f,   6.4f},
								   { -3.2f,  -8.0f,  -6.7f,  10.0f,  -2.1f,   7.0f},
								   {  2.2f,  -3.5f,  -9.3f,   7.0f,   2.1f,   8.9f},
								   { -3.8f,  -5.5f,  -8.5f,   5.0f,  -1.3f,   1.4f},
								   { -2.8f,   0.1f,  -3.3f,   5.5f,   5.2f,   8.9f},
								   { -4.6f,  -9.2f,   0.9f,   2.9f,   0.2f,   9.7f},
								   { -8.2f,  -9.6f,  -7.8f,  -5.4f,   3.7f,   3.7f},
								   { -6.1f,  -5.9f,  -3.9f,   8.6f,   9.8f,   7.3f},
								   { -6.6f,  -1.9f,   3.3f,   5.2f,   9.7f,   7.7f},
								   { -8.8f,  -0.2f,  -7.5f,   7.1f,   9.0f,   5.5f},
								   { -5.6f,  -7.3f,  -4.6f,   7.4f,   5.1f,  10.0f},
								   { -9.2f,   4.7f,  -9.5f,   1.1f,   4.9f,   0.5f},
								   { -7.9f,  -3.8f,   6.0f,   3.3f,   9.4f,   9.2f},
								   { -9.1f,  -3.1f,  -5.6f,  -0.6f,   3.6f,   2.1f},
								   { -2.1f,  -4.3f,  -7.2f,   1.0f,   7.5f,  -0.4f},
								   { -6.6f,  -9.7f,  -4.5f,  -0.9f,   9.3f,   3.3f}};
	float expected_16_14[21][14] = {{ -9.2f,  -9.7f,  -9.5f, -22.1f, -25.3f, -22.6f, -26.4f,  10.0f,   9.8f,  10.0f,  21.4f,  15.4f,  22.1f,  19.3f},
								    { -6.6f,  -8.0f,  -9.3f, -13.6f, -15.5f, -18.1f,  -6.2f,  10.0f,   3.6f,   8.9f,  14.2f,  15.4f,  13.6f,  16.9f},
								    { -8.2f,  -9.6f,  -7.8f, -22.1f, -13.4f, -22.6f, -10.4f,   8.6f,   9.8f,   9.7f,  14.7f,  15.2f,  22.1f,  13.1f},
								    { -9.2f,  -7.3f,  -9.5f, -11.1f, -25.3f, -15.3f, -19.6f,   7.4f,   9.7f,  10.0f,  15.8f,  14.8f,   9.3f,  19.3f},
								    { -9.1f,  -9.7f,  -7.2f, -12.7f, -20.4f, -13.9f, -26.4f,   3.3f,   9.4f,   9.2f,  21.4f,  12.1f,   7.2f,  12.5f},
								    { -6.6f,  -2.8f,  -7.3f, -12.3f, -15.5f, -11.8f,  -6.2f,   5.3f,   3.6f,   6.4f,  12.3f,   6.6f,   5.5f,  -0.9f},
								    { -3.2f,  -8.0f,  -6.7f, -12.5f,   3.5f, -18.1f,  -2.3f,  10.0f,  -2.1f,   7.0f,   7.9f,  12.1f,   7.9f,  16.9f},
								    {  2.2f,  -3.5f,  -9.3f,  -5.0f,  -9.2f,  -9.4f,  -2.4f,   7.0f,   2.1f,   8.9f,  14.2f,  15.4f,  13.6f,   9.4f},
								    { -3.8f,  -5.5f,  -8.5f, -13.6f, -11.0f,  -4.4f,   6.0f,   5.0f,  -1.3f,   1.4f,   3.6f,   9.2f,   3.4f,   6.6f},
								    { -2.8f,   0.1f,  -3.3f,  -0.9f, -11.3f,  -7.5f,  -8.1f,   5.5f,   5.2f,   8.9f,  10.3f,   9.7f,   5.7f,   2.3f},
								    { -4.6f,  -9.2f,   0.9f,  -5.3f,   3.6f, -22.6f,  -4.2f,   2.9f,   0.2f,   9.7f,   4.0f,  15.2f,   2.2f,   1.8f},
								    { -8.2f,  -9.6f,  -7.8f, -22.1f, -13.4f, -21.4f, -10.4f,  -5.4f,   3.7f,   3.7f,  -6.0f,   5.2f,  -3.0f,  11.3f},
								    { -6.1f,  -5.9f,  -3.9f,  -6.5f,  -4.9f, -19.1f,  -7.7f,   8.6f,   9.8f,   7.3f,  14.7f,   6.9f,  22.1f,  13.1f},
								    { -6.6f,  -1.9f,   3.3f,   6.4f, -13.0f,  -4.4f, -19.6f,   5.2f,   9.7f,   7.7f,  15.8f,  14.8f,   1.4f,  -0.6f},
								    { -8.8f,  -0.2f,  -7.5f,  -7.3f, -25.3f,  -3.2f, -10.3f,   7.1f,   9.0f,   5.5f,  12.4f,  12.8f,   7.7f,   1.8f},
								    { -5.6f,  -7.3f,  -4.6f,  -4.5f, -12.2f, -15.3f,  -9.2f,   7.4f,   5.1f,  10.0f,   4.7f,  14.7f,   4.7f,  19.3f},
								    { -9.2f,   4.7f,  -9.5f, -11.1f, -20.7f,  -5.0f, -14.4f,   1.1f,   4.9f,   0.5f,   2.7f,  -7.1f,   9.3f,  -0.5f},
								    { -7.9f,  -3.8f,   6.0f,   2.4f,  -8.2f,  -9.6f, -26.4f,   3.3f,   9.4f,   9.2f,  21.4f,  10.0f,   3.0f,  -2.0f},
								    { -9.1f,  -3.1f,  -5.6f, -12.7f, -13.4f, -11.7f,  -5.5f,  -0.6f,   3.6f,   2.1f,   4.1f,  -1.1f,   5.0f,  -2.2f},
								    { -2.1f,  -4.3f,  -7.2f, -10.5f, -11.4f,   3.9f,  -8.2f,   1.0f,   7.5f,  -0.4f,   6.0f,  -1.9f,   7.2f,  12.5f},
								    { -6.6f,  -9.7f,  -4.5f,  -7.3f, -20.4f, -13.9f, -11.4f,  -0.9f,   9.3f,   3.3f,  -1.8f,  12.1f,   7.2f,   5.5f}};
	float expected_16_18[21][18] = {{ -9.2f,  -9.7f,  -9.5f, -17.7f, -16.3f, -16.7f, -17.8f, -17.0f, -18.9f,  10.0f,   9.8f,  10.0f,  18.4f,  14.6f,  18.5f,  14.7f,  12.3f,  16.5f},
								    { -6.6f,  -8.0f,  -9.3f, -11.1f, -13.9f, -14.7f,  -8.2f, -10.2f, -14.9f,  10.0f,   3.6f,   8.9f,   8.9f,  14.6f,   7.2f,  12.1f,  11.5f,  11.4f},
								    { -8.2f,  -9.6f,  -7.8f, -17.7f, -13.2f, -16.7f, -11.9f, -13.4f, -18.9f,   8.6f,   9.8f,   9.7f,  18.4f,  10.0f,   9.2f,   9.2f,  12.3f,  13.5f},
								    { -9.2f,  -7.3f,  -9.5f,  -5.3f, -16.3f, -11.9f, -17.8f, -10.3f, -15.0f,   7.4f,   9.7f,  10.0f,   8.6f,  12.9f,  16.8f,  14.7f,  12.0f,  16.5f},
								    { -9.1f,  -9.7f,  -7.2f, -12.2f, -11.1f, -11.5f, -17.3f, -17.0f, -13.0f,   3.3f,   9.4f,   9.2f,  12.2f,  12.5f,  18.5f,   8.8f,   8.2f,  13.8f},
								    { -6.6f,  -2.8f,  -7.3f,  -5.4f, -13.9f,  -5.7f,  -8.2f,  -9.0f,  -9.2f,   5.3f,   3.6f,   6.4f,   8.9f,   8.7f,   7.0f,   1.7f,   1.9f,   8.9f},
								    { -3.2f,  -8.0f,  -6.7f, -11.1f,  -4.5f, -14.7f,   4.7f, -10.2f, -14.9f,  10.0f,  -2.1f,   7.0f,   7.9f,  10.0f,  -0.9f,  12.1f,  10.0f,  -1.3f},
								    {  2.2f,  -3.5f,  -9.3f,  -0.5f,  -7.1f,  -7.2f,   0.1f,  -5.9f, -12.4f,   7.0f,   2.1f,   8.9f,   6.6f,  14.6f,   7.2f,   7.4f,  11.5f,  11.4f},
								    { -3.8f,  -5.5f,  -8.5f,  -8.4f, -12.3f,  -9.8f,  -2.5f,   1.1f,  -4.2f,   5.0f,  -1.3f,   1.4f,   2.2f,   6.4f,  -1.4f,   7.8f,   4.7f,   7.2f},
								    { -2.8f,   0.1f,  -3.3f,   1.4f,  -6.1f,   1.9f,  -8.0f,  -7.8f,  -8.6f,   5.5f,   5.2f,   8.9f,   5.6f,  10.0f,   9.2f,   5.4f,   2.4f,   8.5f},
								    { -4.6f,  -9.2f,   0.9f, -12.9f,   0.6f,  -0.7f,   1.3f, -13.4f, -18.9f,   2.9f,   0.2f,   9.7f,   3.1f,   6.0f,   1.1f,   5.5f,   2.0f,  -0.7f},
								    { -8.2f,  -9.6f,  -7.8f, -17.7f, -13.2f, -16.7f, -11.9f, -11.8f, -13.3f,  -5.4f,   3.7f,   3.7f,  -4.5f,  -4.4f,   2.2f,   3.5f,   2.4f,   5.2f},
								    { -6.1f,  -5.9f,  -3.9f, -11.8f,  -0.6f,  -9.8f,  -1.2f, -13.4f, -13.0f,   8.6f,   9.8f,   7.3f,  18.4f,   4.9f,   6.1f,   9.2f,  12.3f,  13.5f},
								    { -6.6f,  -1.9f,   3.3f,   3.1f,  -3.3f,   5.8f, -16.3f,  -9.9f,  -9.6f,   5.2f,   9.7f,   7.7f,   8.6f,  12.9f,  16.8f,   7.1f,  -2.5f,   6.4f},
								    { -8.8f,  -0.2f,  -7.5f,  -0.9f, -16.3f,   1.5f, -17.8f,  -5.8f,  -5.7f,   7.1f,   9.0f,   5.5f,   6.9f,  12.6f,   5.3f,   7.3f,   1.6f,  16.5f},
								    { -5.6f,  -7.3f,  -4.6f,  -5.3f,  -7.1f, -11.9f, -10.7f, -10.3f, -15.0f,   7.4f,   5.1f,  10.0f,   0.1f,   9.7f,   5.0f,  14.7f,  12.0f,   6.6f},
								    { -9.2f,   4.7f,  -9.5f,  -4.5f, -15.9f,  -4.7f, -13.9f,  -9.7f,   4.2f,   1.1f,   4.9f,   0.5f,   6.0f,  -2.2f,   5.2f,  -3.8f,   4.4f,  14.3f},
								    { -7.9f,  -3.8f,   6.0f,  -3.6f,   1.2f,   2.2f, -17.3f, -17.0f,  -9.8f,   3.3f,   9.4f,   9.2f,  12.2f,  12.5f,  18.5f,   4.0f,  -5.8f,   0.3f},
								    { -9.1f,  -3.1f,  -5.6f, -12.2f,  -9.8f,  -3.6f,  -7.8f,  -8.6f,  -2.6f,  -0.6f,   3.6f,   2.1f,   2.0f,   1.5f,   4.7f,  -3.2f,   1.4f,   9.2f},
								    { -2.1f,  -4.3f,  -7.2f,  -3.3f,  -6.2f, -11.5f,  -8.6f,  -0.7f,   2.9f,   1.0f,   7.5f,  -0.4f,   6.4f,  -1.5f,   7.1f,   5.3f,   8.2f,   9.3f},
								    { -6.6f,  -9.7f,  -4.5f, -10.6f, -11.1f,  -6.4f, -15.9f,  -9.4f, -13.0f,  -0.9f,   9.3f,   3.3f,   2.7f,   2.4f,   4.8f,   8.8f,  -2.1f,  13.8f}};
	float expected_16_26[21][26] = {{ -9.2f,  -9.7f,  -9.5f, -22.1f, -25.3f, -22.6f, -26.4f, -17.7f, -16.3f, -16.7f, -17.8f, -17.0f, -18.9f,  10.0f,   9.8f,  10.0f,  21.4f,  15.4f,  22.1f,  19.3f,  18.4f,  14.6f,  18.5f,  14.7f,  12.3f,  16.5f},
								    { -6.6f,  -8.0f,  -9.3f, -13.6f, -15.5f, -18.1f,  -6.2f, -11.1f, -13.9f, -14.7f,  -8.2f, -10.2f, -14.9f,  10.0f,   3.6f,   8.9f,  14.2f,  15.4f,  13.6f,  16.9f,   8.9f,  14.6f,   7.2f,  12.1f,  11.5f,  11.4f},
								    { -8.2f,  -9.6f,  -7.8f, -22.1f, -13.4f, -22.6f, -10.4f, -17.7f, -13.2f, -16.7f, -11.9f, -13.4f, -18.9f,   8.6f,   9.8f,   9.7f,  14.7f,  15.2f,  22.1f,  13.1f,  18.4f,  10.0f,   9.2f,   9.2f,  12.3f,  13.5f},
								    { -9.2f,  -7.3f,  -9.5f, -11.1f, -25.3f, -15.3f, -19.6f,  -5.3f, -16.3f, -11.9f, -17.8f, -10.3f, -15.0f,   7.4f,   9.7f,  10.0f,  15.8f,  14.8f,   9.3f,  19.3f,   8.6f,  12.9f,  16.8f,  14.7f,  12.0f,  16.5f},
								    { -9.1f,  -9.7f,  -7.2f, -12.7f, -20.4f, -13.9f, -26.4f, -12.2f, -11.1f, -11.5f, -17.3f, -17.0f, -13.0f,   3.3f,   9.4f,   9.2f,  21.4f,  12.1f,   7.2f,  12.5f,  12.2f,  12.5f,  18.5f,   8.8f,   8.2f,  13.8f},
								    { -6.6f,  -2.8f,  -7.3f, -12.3f, -15.5f, -11.8f,  -6.2f,  -5.4f, -13.9f,  -5.7f,  -8.2f,  -9.0f,  -9.2f,   5.3f,   3.6f,   6.4f,  12.3f,   6.6f,   5.5f,  -0.9f,   8.9f,   8.7f,   7.0f,   1.7f,   1.9f,   8.9f},
								    { -3.2f,  -8.0f,  -6.7f, -12.5f,   3.5f, -18.1f,  -2.3f, -11.1f,  -4.5f, -14.7f,   4.7f, -10.2f, -14.9f,  10.0f,  -2.1f,   7.0f,   7.9f,  12.1f,   7.9f,  16.9f,   7.9f,  10.0f,  -0.9f,  12.1f,  10.0f,  -1.3f},
								    {  2.2f,  -3.5f,  -9.3f,  -5.0f,  -9.2f,  -9.4f,  -2.4f,  -0.5f,  -7.1f,  -7.2f,   0.1f,  -5.9f, -12.4f,   7.0f,   2.1f,   8.9f,  14.2f,  15.4f,  13.6f,   9.4f,   6.6f,  14.6f,   7.2f,   7.4f,  11.5f,  11.4f},
								    { -3.8f,  -5.5f,  -8.5f, -13.6f, -11.0f,  -4.4f,   6.0f,  -8.4f, -12.3f,  -9.8f,  -2.5f,   1.1f,  -4.2f,   5.0f,  -1.3f,   1.4f,   3.6f,   9.2f,   3.4f,   6.6f,   2.2f,   6.4f,  -1.4f,   7.8f,   4.7f,   7.2f},
								    { -2.8f,   0.1f,  -3.3f,  -0.9f, -11.3f,  -7.5f,  -8.1f,   1.4f,  -6.1f,   1.9f,  -8.0f,  -7.8f,  -8.6f,   5.5f,   5.2f,   8.9f,  10.3f,   9.7f,   5.7f,   2.3f,   5.6f,  10.0f,   9.2f,   5.4f,   2.4f,   8.5f},
								    { -4.6f,  -9.2f,   0.9f,  -5.3f,   3.6f, -22.6f,  -4.2f, -12.9f,   0.6f,  -0.7f,   1.3f, -13.4f, -18.9f,   2.9f,   0.2f,   9.7f,   4.0f,  15.2f,   2.2f,   1.8f,   3.1f,   6.0f,   1.1f,   5.5f,   2.0f,  -0.7f},
								    { -8.2f,  -9.6f,  -7.8f, -22.1f, -13.4f, -21.4f, -10.4f, -17.7f, -13.2f, -16.7f, -11.9f, -11.8f, -13.3f,  -5.4f,   3.7f,   3.7f,  -6.0f,   5.2f,  -3.0f,  11.3f,  -4.5f,  -4.4f,   2.2f,   3.5f,   2.4f,   5.2f},
								    { -6.1f,  -5.9f,  -3.9f,  -6.5f,  -4.9f, -19.1f,  -7.7f, -11.8f,  -0.6f,  -9.8f,  -1.2f, -13.4f, -13.0f,   8.6f,   9.8f,   7.3f,  14.7f,   6.9f,  22.1f,  13.1f,  18.4f,   4.9f,   6.1f,   9.2f,  12.3f,  13.5f},
								    { -6.6f,  -1.9f,   3.3f,   6.4f, -13.0f,  -4.4f, -19.6f,   3.1f,  -3.3f,   5.8f, -16.3f,  -9.9f,  -9.6f,   5.2f,   9.7f,   7.7f,  15.8f,  14.8f,   1.4f,  -0.6f,   8.6f,  12.9f,  16.8f,   7.1f,  -2.5f,   6.4f},
								    { -8.8f,  -0.2f,  -7.5f,  -7.3f, -25.3f,  -3.2f, -10.3f,  -0.9f, -16.3f,   1.5f, -17.8f,  -5.8f,  -5.7f,   7.1f,   9.0f,   5.5f,  12.4f,  12.8f,   7.7f,   1.8f,   6.9f,  12.6f,   5.3f,   7.3f,   1.6f,  16.5f},
								    { -5.6f,  -7.3f,  -4.6f,  -4.5f, -12.2f, -15.3f,  -9.2f,  -5.3f,  -7.1f, -11.9f, -10.7f, -10.3f, -15.0f,   7.4f,   5.1f,  10.0f,   4.7f,  14.7f,   4.7f,  19.3f,   0.1f,   9.7f,   5.0f,  14.7f,  12.0f,   6.6f},
								    { -9.2f,   4.7f,  -9.5f, -11.1f, -20.7f,  -5.0f, -14.4f,  -4.5f, -15.9f,  -4.7f, -13.9f,  -9.7f,   4.2f,   1.1f,   4.9f,   0.5f,   2.7f,  -7.1f,   9.3f,  -0.5f,   6.0f,  -2.2f,   5.2f,  -3.8f,   4.4f,  14.3f},
								    { -7.9f,  -3.8f,   6.0f,   2.4f,  -8.2f,  -9.6f, -26.4f,  -3.6f,   1.2f,   2.2f, -17.3f, -17.0f,  -9.8f,   3.3f,   9.4f,   9.2f,  21.4f,  10.0f,   3.0f,  -2.0f,  12.2f,  12.5f,  18.5f,   4.0f,  -5.8f,   0.3f},
								    { -9.1f,  -3.1f,  -5.6f, -12.7f, -13.4f, -11.7f,  -5.5f, -12.2f,  -9.8f,  -3.6f,  -7.8f,  -8.6f,  -2.6f,  -0.6f,   3.6f,   2.1f,   4.1f,  -1.1f,   5.0f,  -2.2f,   2.0f,   1.5f,   4.7f,  -3.2f,   1.4f,   9.2f},
								    { -2.1f,  -4.3f,  -7.2f, -10.5f, -11.4f,   3.9f,  -8.2f,  -3.3f,  -6.2f, -11.5f,  -8.6f,  -0.7f,   2.9f,   1.0f,   7.5f,  -0.4f,   6.0f,  -1.9f,   7.2f,  12.5f,   6.4f,  -1.5f,   7.1f,   5.3f,   8.2f,   9.3f},
								    { -6.6f,  -9.7f,  -4.5f,  -7.3f, -20.4f, -13.9f, -11.4f, -10.6f, -11.1f,  -6.4f, -15.9f,  -9.4f, -13.0f,  -0.9f,   9.3f,   3.3f,  -1.8f,  12.1f,   7.2f,   5.5f,   2.7f,   2.4f,   4.8f,   8.8f,  -2.1f,  13.8f}};
	
	checkTreeDistances(inputVerts_16, expected_16_6, expected_16_14, expected_16_18, expected_16_26);
		
	//Generate 17 leaf nodes with KDOP from vertices
	std::vector<Vector3f> leaf_17_1;
	leaf_17_1.push_back(Vector3f( 6.2f, -3.2f,  3.0f));
	leaf_17_1.push_back(Vector3f(-3.1f, -9.2f,  7.7f));
	leaf_17_1.push_back(Vector3f( 9.5f, -7.1f, -2.3f));
	
	std::vector<Vector3f> leaf_17_2;
	leaf_17_2.push_back(Vector3f(-7.1f,  1.3f, -5.7f));
	leaf_17_2.push_back(Vector3f( 6.2f,  0.7f,  4.3f));
	leaf_17_2.push_back(Vector3f( 4.0f,  0.0f,  4.6f));
	
	std::vector<Vector3f> leaf_17_3;
	leaf_17_3.push_back(Vector3f( 3.3f,  5.2f, -8.2f));
	leaf_17_3.push_back(Vector3f(-2.2f, -4.6f,  0.6f));
	leaf_17_3.push_back(Vector3f( 3.1f,  1.7f, -3.4f));
	
	std::vector<Vector3f> leaf_17_4;
	leaf_17_4.push_back(Vector3f(-3.7f, -8.1f,  6.6f));
	leaf_17_4.push_back(Vector3f(-5.7f, -3.5f, -0.6f));
	leaf_17_4.push_back(Vector3f(-2.2f,  4.2f,  7.8f));
	
	std::vector<Vector3f> leaf_17_5;
	leaf_17_5.push_back(Vector3f(-9.7f,  9.2f, -7.1f));
	leaf_17_5.push_back(Vector3f( 6.4f, -9.9f,  9.9f));
	leaf_17_5.push_back(Vector3f( 3.6f, -1.0f,  3.2f));
	
	std::vector<Vector3f> leaf_17_6;
	leaf_17_6.push_back(Vector3f(-4.9f,  2.1f,  7.4f));
	leaf_17_6.push_back(Vector3f( 2.6f,  8.7f, -1.3f));
	leaf_17_6.push_back(Vector3f(-9.1f, -0.7f, -0.7f));
	
	std::vector<Vector3f> leaf_17_7;
	leaf_17_7.push_back(Vector3f(-2.6f, -1.9f,  4.0f));
	leaf_17_7.push_back(Vector3f(-7.7f,  8.9f,  5.5f));
	leaf_17_7.push_back(Vector3f( 9.1f, -8.3f,  9.2f));
	
	std::vector<Vector3f> leaf_17_8;
	leaf_17_8.push_back(Vector3f(-2.0f,  1.6f, -3.0f));
	leaf_17_8.push_back(Vector3f(-6.6f, -6.8f,  6.5f));
	leaf_17_8.push_back(Vector3f( 2.9f,  5.8f, -8.6f));
	
	std::vector<Vector3f> leaf_17_9;
	leaf_17_9.push_back(Vector3f( 8.2f,  5.1f,  2.3f));
	leaf_17_9.push_back(Vector3f(-8.1f,  4.2f, -6.3f));
	leaf_17_9.push_back(Vector3f(-8.4f, -5.0f, -0.6f));
	
	std::vector<Vector3f> leaf_17_10;
	leaf_17_10.push_back(Vector3f( 8.5f,  0.4f,  4.9f));
	leaf_17_10.push_back(Vector3f(-7.1f, -0.2f,  0.3f));
	leaf_17_10.push_back(Vector3f( 3.5f, -9.9f,  3.9f));
	
	std::vector<Vector3f> leaf_17_11;
	leaf_17_11.push_back(Vector3f(-8.0f, -5.9f,  5.3f));
	leaf_17_11.push_back(Vector3f(-3.6f,  3.4f, -8.2f));
	leaf_17_11.push_back(Vector3f(-4.2f,  0.9f, -0.1f));
	
	std::vector<Vector3f> leaf_17_12;
	leaf_17_12.push_back(Vector3f(-4.0f, -4.4f, -6.1f));
	leaf_17_12.push_back(Vector3f( 9.5f, -6.8f,  1.3f));
	leaf_17_12.push_back(Vector3f(-3.5f, -8.7f,  3.2f));
	
	std::vector<Vector3f> leaf_17_13;
	leaf_17_13.push_back(Vector3f( 9.7f, -9.4f,  4.2f));
	leaf_17_13.push_back(Vector3f( 5.7f,  6.6f, -7.5f));
	leaf_17_13.push_back(Vector3f(-6.1f,  5.4f, -0.5f));
	
	std::vector<Vector3f> leaf_17_14;
	leaf_17_14.push_back(Vector3f(-5.6f,  0.2f, -4.0f));
	leaf_17_14.push_back(Vector3f( 9.8f, -2.8f,  9.5f));
	leaf_17_14.push_back(Vector3f(-5.8f, -6.3f, -6.3f));
	
	std::vector<Vector3f> leaf_17_15;
	leaf_17_15.push_back(Vector3f(-2.4f, -2.2f, -2.1f));
	leaf_17_15.push_back(Vector3f(10.0f,  4.8f,  2.4f));
	leaf_17_15.push_back(Vector3f(-8.4f, -4.3f, -5.7f));
	
	std::vector<Vector3f> leaf_17_16;
	leaf_17_16.push_back(Vector3f( 1.0f,  9.6f,  9.9f));
	leaf_17_16.push_back(Vector3f(-4.9f, -1.3f,  6.8f));
	leaf_17_16.push_back(Vector3f( 6.0f, -9.0f,  5.7f));
	
	std::vector<Vector3f> leaf_17_17;
	leaf_17_17.push_back(Vector3f( 1.1f, -5.6f,  5.6f));
	leaf_17_17.push_back(Vector3f(-5.6f, -6.9f, -9.7f));
	leaf_17_17.push_back(Vector3f(-7.7f,  2.7f, -0.5f));
	
	std::vector< std::vector<Vector3f> > inputVerts_17;
	inputVerts_17.push_back(leaf_17_1);
	inputVerts_17.push_back(leaf_17_2);
	inputVerts_17.push_back(leaf_17_3);
	inputVerts_17.push_back(leaf_17_4);
	inputVerts_17.push_back(leaf_17_5);
	inputVerts_17.push_back(leaf_17_6);
	inputVerts_17.push_back(leaf_17_7);
	inputVerts_17.push_back(leaf_17_8);
	inputVerts_17.push_back(leaf_17_9);
	inputVerts_17.push_back(leaf_17_10);
	inputVerts_17.push_back(leaf_17_11);
	inputVerts_17.push_back(leaf_17_12);
	inputVerts_17.push_back(leaf_17_13);
	inputVerts_17.push_back(leaf_17_14);
	inputVerts_17.push_back(leaf_17_15);
	inputVerts_17.push_back(leaf_17_16);
	inputVerts_17.push_back(leaf_17_17);
	
	float expected_17_6 [25][6 ] = {{ -9.7f,  -9.9f,  -9.7f,  10.0f,   9.6f,   9.9f},
									{ -9.7f,  -9.9f,  -8.6f,  10.0f,   9.6f,   9.9f},
									{ -7.7f,  -6.9f,  -9.7f,   1.1f,   2.7f,   5.6f},
									{ -7.1f,  -9.2f,  -8.2f,   9.5f,   5.2f,   7.8f},
									{ -9.7f,  -9.9f,  -8.6f,   9.1f,   9.2f,   9.9f},
									{ -8.4f,  -9.9f,  -8.2f,   9.5f,   5.1f,   5.3f},
									{ -8.4f,  -9.4f,  -7.5f,  10.0f,   9.6f,   9.9f},
									{ -7.7f,  -6.9f,  -9.7f,   1.1f,   2.7f,   5.6f},
									{ -3.1f,  -9.2f,  -2.3f,   9.5f,  -3.2f,   7.7f},
									{ -7.1f,   0.0f,  -5.7f,   6.2f,   1.3f,   4.6f},
									{ -2.2f,  -4.6f,  -8.2f,   3.3f,   5.2f,   0.6f},
									{ -5.7f,  -8.1f,  -0.6f,  -2.2f,   4.2f,   7.8f},
									{ -9.7f,  -9.9f,  -7.1f,   6.4f,   9.2f,   9.9f},
									{ -9.1f,  -0.7f,  -1.3f,   2.6f,   8.7f,   7.4f},
									{ -7.7f,  -8.3f,   4.0f,   9.1f,   8.9f,   9.2f},
									{ -6.6f,  -6.8f,  -8.6f,   2.9f,   5.8f,   6.5f},
									{ -8.4f,  -5.0f,  -6.3f,   8.2f,   5.1f,   2.3f},
									{ -7.1f,  -9.9f,   0.3f,   8.5f,   0.4f,   4.9f},
									{ -8.0f,  -5.9f,  -8.2f,  -3.6f,   3.4f,   5.3f},
									{ -4.0f,  -8.7f,  -6.1f,   9.5f,  -4.4f,   3.2f},
									{ -6.1f,  -9.4f,  -7.5f,   9.7f,   6.6f,   4.2f},
									{ -5.8f,  -6.3f,  -6.3f,   9.8f,   0.2f,   9.5f},
									{ -8.4f,  -4.3f,  -5.7f,  10.0f,   4.8f,   2.4f},
									{ -4.9f,  -9.0f,   5.7f,   6.0f,   9.6f,   9.9f},
									{ -7.7f,  -6.9f,  -9.7f,   1.1f,   2.7f,   5.6f}};
	float expected_17_14[25][14] = {{ -9.7f,  -9.9f,  -9.7f, -22.2f, -26.0f, -20.0f, -22.1f,  10.0f,   9.6f,   9.9f,  20.5f,  26.6f,  19.8f,  18.9f},
									{ -9.7f,  -9.9f,  -8.6f, -18.4f, -26.0f, -20.0f, -22.1f,  10.0f,   9.6f,   9.9f,  20.5f,  26.6f,  19.8f,  18.9f},
									{ -7.7f,  -6.9f,  -9.7f, -22.2f, -10.9f, -10.1f,  -9.9f,   1.1f,   2.7f,   5.6f,   1.1f,  12.3f,  -2.8f,  11.0f},
									{ -7.1f,  -9.2f,  -8.2f, -11.5f, -14.1f, -20.0f, -14.2f,   9.5f,   5.2f,   7.8f,  11.2f,  14.3f,  16.7f,  18.9f},
									{ -9.7f,  -9.9f,  -8.6f, -10.5f, -26.0f, -19.9f, -22.1f,   9.1f,   9.2f,   9.9f,  10.0f,  26.6f,  17.3f,   8.2f},
									{ -8.4f,  -9.9f,  -8.2f, -14.5f, -18.6f, -19.2f,  -7.4f,   9.5f,   5.1f,   5.3f,  15.6f,  17.6f,  11.0f,  15.0f},
									{ -8.4f,  -9.4f,  -7.5f, -18.4f, -12.0f, -13.0f, -18.5f,  10.0f,   9.6f,   9.9f,  20.5f,  23.3f,  19.8f,  14.9f},
									{ -7.7f,  -6.9f,  -9.7f, -22.2f, -10.9f, -10.1f,  -9.9f,   1.1f,   2.7f,   5.6f,   1.1f,  12.3f,  -2.8f,  11.0f},
									{ -3.1f,  -9.2f,  -2.3f,  -4.6f,  12.4f, -20.0f,  -1.6f,   9.5f,  -3.2f,   7.7f,   6.0f,  14.3f,   4.7f,  18.9f},
									{ -7.1f,   0.0f,  -5.7f, -11.5f, -14.1f,  -0.6f,  -2.7f,   6.2f,   1.3f,   4.6f,  11.2f,   9.8f,   2.6f,   1.2f},
									{ -2.2f,  -4.6f,  -8.2f,  -6.2f, -10.1f,  -7.4f,   1.8f,   3.3f,   5.2f,   0.6f,   1.4f,   3.0f,  16.7f,   6.3f},
									{ -5.7f,  -8.1f,  -0.6f,  -9.8f,  -2.8f, -18.4f, -14.2f,  -2.2f,   4.2f,   7.8f,   9.8f,  11.0f,  -5.8f,  -1.6f},
									{ -9.7f,  -9.9f,  -7.1f,  -7.6f, -26.0f, -13.4f, -11.8f,   6.4f,   9.2f,   9.9f,   6.4f,  26.2f,   6.6f,   6.4f},
									{ -9.1f,  -0.7f,  -1.3f, -10.5f,  -9.1f, -10.2f, -14.4f,   2.6f,   8.7f,   7.4f,  10.0f,   0.4f,  12.6f,  -4.8f},
									{ -7.7f,  -8.3f,   4.0f,  -0.5f, -11.1f,  -8.5f, -22.1f,   9.1f,   8.9f,   9.2f,  10.0f,  26.6f,  -4.3f,   8.2f},
									{ -6.6f,  -6.8f,  -8.6f,  -6.9f, -11.5f, -19.9f,  -6.3f,   2.9f,   5.8f,   6.5f,   0.1f,   6.7f,  17.3f,   5.7f},
									{ -8.4f,  -5.0f,  -6.3f, -14.0f, -18.6f, -12.8f,  -6.0f,   8.2f,   5.1f,   2.3f,  15.6f,   5.4f,  11.0f,   0.8f},
									{ -7.1f,  -9.9f,   0.3f,  -7.0f,  -6.6f, -10.3f,  -7.2f,   8.5f,   0.4f,   4.9f,  13.8f,  17.3f,   4.0f,   9.5f},
									{ -8.0f,  -5.9f,  -8.2f,  -8.6f, -15.2f, -19.2f,  -7.4f,  -3.6f,   3.4f,   5.3f,  -3.4f,   3.2f,   8.0f,   1.2f},
									{ -4.0f,  -8.7f,  -6.1f, -14.5f,  -5.7f, -15.4f,   2.0f,   9.5f,  -4.4f,   3.2f,   4.0f,  17.6f,   1.4f,  15.0f},
									{ -6.1f,  -9.4f,  -7.5f,  -1.2f, -12.0f,  -3.9f, -11.0f,   9.7f,   6.6f,   4.2f,   4.8f,  23.3f,  19.8f,  14.9f},
									{ -5.8f,  -6.3f,  -6.3f, -18.4f,  -9.8f,  -5.8f,  -1.8f,   9.8f,   0.2f,   9.5f,  16.5f,  22.1f,  -1.4f,   6.8f},
									{ -8.4f,  -4.3f,  -5.7f, -18.4f,  -9.8f,  -7.0f,   1.6f,  10.0f,   4.8f,   2.4f,  17.2f,   7.6f,  12.4f,   2.8f},
									{ -4.9f,  -9.0f,   5.7f,   0.6f,   1.3f, -13.0f, -18.5f,   6.0f,   9.6f,   9.9f,  20.5f,  20.7f,   0.7f,   9.3f},
									{ -7.7f,  -6.9f,  -9.7f, -22.2f, -10.9f, -10.1f,  -9.9f,   1.1f,   2.7f,   5.6f,   1.1f,  12.3f,  -2.8f,  11.0f}};
	float expected_17_18[25][18] = {{ -9.7f,  -9.9f,  -9.7f, -13.9f, -16.8f, -16.6f, -18.9f, -13.3f, -19.8f,  10.0f,   9.6f,   9.9f,  14.8f,  19.3f,  19.5f,  19.1f,  13.2f,  16.3f},
									{ -9.7f,  -9.9f,  -8.6f, -13.9f, -16.8f, -12.6f, -18.9f, -13.3f, -19.8f,  10.0f,   9.6f,   9.9f,  14.8f,  19.3f,  19.5f,  19.1f,  13.2f,  16.3f},
									{ -7.7f,  -6.9f,  -9.7f, -12.5f, -15.3f, -16.6f, -10.4f,  -7.2f, -11.2f,   1.1f,   2.7f,   5.6f,  -4.5f,   6.7f,   2.2f,   6.7f,   4.1f,   3.2f},
									{ -7.1f,  -9.2f,  -8.2f, -12.3f, -12.8f,  -9.4f,  -8.4f, -10.8f, -16.9f,   9.5f,   5.2f,   7.8f,   8.5f,  10.5f,  12.0f,  16.6f,  11.8f,  13.4f},
									{ -9.7f,  -9.9f,  -8.6f, -13.4f, -16.8f,  -2.8f, -18.9f, -13.2f, -19.8f,   9.1f,   9.2f,   9.9f,  11.3f,  18.3f,  14.4f,  17.4f,  11.5f,  16.3f},
									{ -8.4f,  -9.9f,  -8.2f, -13.9f, -14.4f, -10.5f, -12.3f, -13.3f, -13.8f,   9.5f,   5.1f,   5.3f,  13.3f,  13.4f,   7.4f,  16.3f,   8.2f,  11.6f},
									{ -8.4f,  -9.4f,  -7.5f, -12.7f, -14.1f, -12.6f, -11.5f, -11.7f, -14.7f,  10.0f,   9.6f,   9.9f,  14.8f,  19.3f,  19.5f,  19.1f,  13.2f,  14.1f},
									{ -7.7f,  -6.9f,  -9.7f, -12.5f, -15.3f, -16.6f, -10.4f,  -7.2f, -11.2f,   1.1f,   2.7f,   5.6f,  -4.5f,   6.7f,   2.2f,   6.7f,   4.1f,   3.2f},
									{ -3.1f,  -9.2f,  -2.3f, -12.3f,   4.6f,  -9.4f,   6.1f, -10.8f, -16.9f,   9.5f,  -3.2f,   7.7f,   3.0f,   9.2f,  -0.2f,  16.6f,  11.8f,  -4.8f},
									{ -7.1f,   0.0f,  -5.7f,  -5.8f, -12.8f,  -4.4f,  -8.4f,  -1.4f,  -4.6f,   6.2f,   1.3f,   4.6f,   6.9f,  10.5f,   5.0f,   5.5f,   1.9f,   7.0f},
									{ -2.2f,  -4.6f,  -8.2f,  -6.8f,  -4.9f,  -4.0f,  -1.9f,  -2.8f,  -5.2f,   3.3f,   5.2f,   0.6f,   8.5f,  -0.3f,  -1.7f,   2.4f,  11.5f,  13.4f},
									{ -5.7f,  -8.1f,  -0.6f, -11.8f,  -6.3f,  -4.1f,  -6.4f, -10.3f, -14.7f,  -2.2f,   4.2f,   7.8f,   2.0f,   5.6f,  12.0f,   4.4f,  -5.1f,  -2.9f},
									{ -9.7f,  -9.9f,  -7.1f,  -3.5f, -16.8f,   0.0f, -18.9f,  -3.5f, -19.8f,   6.4f,   9.2f,   9.9f,   2.6f,  16.3f,   2.2f,  16.3f,   0.4f,  16.3f},
									{ -9.1f,  -0.7f,  -1.3f,  -9.8f,  -9.8f,  -1.4f,  -8.4f, -12.3f,  -5.3f,   2.6f,   8.7f,   7.4f,  11.3f,   2.5f,   9.5f,  -6.1f,   3.9f,  10.0f},
									{ -7.7f,  -8.3f,   4.0f,  -4.5f,  -2.2f,   0.9f, -16.6f, -13.2f, -17.5f,   9.1f,   8.9f,   9.2f,   1.2f,  18.3f,  14.4f,  17.4f,  -0.1f,   3.4f},
									{ -6.6f,  -6.8f,  -8.6f, -13.4f,  -5.7f,  -2.8f,  -3.6f, -13.1f, -13.3f,   2.9f,   5.8f,   6.5f,   8.7f,  -0.1f,  -0.3f,   0.2f,  11.5f,  14.4f},
									{ -8.4f,  -5.0f,  -6.3f, -13.4f, -14.4f,  -5.6f, -12.3f,  -7.8f,  -4.4f,   8.2f,   5.1f,   2.3f,  13.3f,  10.5f,   7.4f,   3.1f,   5.9f,  10.5f},
									{ -7.1f,  -9.9f,   0.3f,  -7.3f,  -6.8f,  -6.0f,  -6.9f,  -7.4f, -13.8f,   8.5f,   0.4f,   4.9f,   8.9f,  13.4f,   5.3f,  13.4f,   3.6f,  -0.5f},
									{ -8.0f,  -5.9f,  -8.2f, -13.9f, -11.8f,  -4.8f,  -7.0f, -13.3f, -11.2f,  -3.6f,   3.4f,   5.3f,  -0.2f,  -2.7f,   0.8f,  -2.1f,   4.6f,  11.6f},
									{ -4.0f,  -8.7f,  -6.1f, -12.2f, -10.1f, -10.5f,   0.4f,  -6.7f, -11.9f,   9.5f,  -4.4f,   3.2f,   2.7f,  10.8f,  -5.5f,  16.3f,   8.2f,   1.7f},
									{ -6.1f,  -9.4f,  -7.5f,  -0.7f,  -6.6f,  -5.2f, -11.5f,  -5.6f, -13.6f,   9.7f,   6.6f,   4.2f,  12.3f,  13.9f,   4.9f,  19.1f,  13.2f,  14.1f},
									{ -5.8f,  -6.3f,  -6.3f, -12.1f, -12.1f, -12.6f,  -5.8f,  -1.6f, -12.3f,   9.8f,   0.2f,   9.5f,   7.0f,  19.3f,   6.7f,  12.6f,   0.5f,   4.2f},
									{ -8.4f,  -4.3f,  -5.7f, -12.7f, -14.1f, -10.0f,  -4.1f,  -2.7f,  -0.1f,  10.0f,   4.8f,   2.4f,  14.8f,  12.4f,   7.2f,   5.2f,   7.6f,   2.4f},
									{ -4.9f,  -9.0f,   5.7f,  -6.2f,   1.9f,  -3.3f,  -8.6f, -11.7f, -14.7f,   6.0f,   9.6f,   9.9f,  10.6f,  11.7f,  19.5f,  15.0f,   0.3f,  -0.3f},
									{ -7.7f,  -6.9f,  -9.7f, -12.5f, -15.3f, -16.6f, -10.4f,  -7.2f, -11.2f,   1.1f,   2.7f,   5.6f,  -4.5f,   6.7f,   2.2f,   6.7f,   4.1f,   3.2f}};
	float expected_17_26[25][26] = {{ -9.7f,  -9.9f,  -9.7f, -22.2f, -26.0f, -20.0f, -22.1f, -13.9f, -16.8f, -16.6f, -18.9f, -13.3f, -19.8f,  10.0f,   9.6f,   9.9f,  20.5f,  26.6f,  19.8f,  18.9f,  14.8f,  19.3f,  19.5f,  19.1f,  13.2f,  16.3f},
									{ -9.7f,  -9.9f,  -8.6f, -18.4f, -26.0f, -20.0f, -22.1f, -13.9f, -16.8f, -12.6f, -18.9f, -13.3f, -19.8f,  10.0f,   9.6f,   9.9f,  20.5f,  26.6f,  19.8f,  18.9f,  14.8f,  19.3f,  19.5f,  19.1f,  13.2f,  16.3f},
									{ -7.7f,  -6.9f,  -9.7f, -22.2f, -10.9f, -10.1f,  -9.9f, -12.5f, -15.3f, -16.6f, -10.4f,  -7.2f, -11.2f,   1.1f,   2.7f,   5.6f,   1.1f,  12.3f,  -2.8f,  11.0f,  -4.5f,   6.7f,   2.2f,   6.7f,   4.1f,   3.2f},
									{ -7.1f,  -9.2f,  -8.2f, -11.5f, -14.1f, -20.0f, -14.2f, -12.3f, -12.8f,  -9.4f,  -8.4f, -10.8f, -16.9f,   9.5f,   5.2f,   7.8f,  11.2f,  14.3f,  16.7f,  18.9f,   8.5f,  10.5f,  12.0f,  16.6f,  11.8f,  13.4f},
									{ -9.7f,  -9.9f,  -8.6f, -10.5f, -26.0f, -19.9f, -22.1f, -13.4f, -16.8f,  -2.8f, -18.9f, -13.2f, -19.8f,   9.1f,   9.2f,   9.9f,  10.0f,  26.6f,  17.3f,   8.2f,  11.3f,  18.3f,  14.4f,  17.4f,  11.5f,  16.3f},
									{ -8.4f,  -9.9f,  -8.2f, -14.5f, -18.6f, -19.2f,  -7.4f, -13.9f, -14.4f, -10.5f, -12.3f, -13.3f, -13.8f,   9.5f,   5.1f,   5.3f,  15.6f,  17.6f,  11.0f,  15.0f,  13.3f,  13.4f,   7.4f,  16.3f,   8.2f,  11.6f},
									{ -8.4f,  -9.4f,  -7.5f, -18.4f, -12.0f, -13.0f, -18.5f, -12.7f, -14.1f, -12.6f, -11.5f, -11.7f, -14.7f,  10.0f,   9.6f,   9.9f,  20.5f,  23.3f,  19.8f,  14.9f,  14.8f,  19.3f,  19.5f,  19.1f,  13.2f,  14.1f},
									{ -7.7f,  -6.9f,  -9.7f, -22.2f, -10.9f, -10.1f,  -9.9f, -12.5f, -15.3f, -16.6f, -10.4f,  -7.2f, -11.2f,   1.1f,   2.7f,   5.6f,   1.1f,  12.3f,  -2.8f,  11.0f,  -4.5f,   6.7f,   2.2f,   6.7f,   4.1f,   3.2f},
									{ -3.1f,  -9.2f,  -2.3f,  -4.6f,  12.4f, -20.0f,  -1.6f, -12.3f,   4.6f,  -9.4f,   6.1f, -10.8f, -16.9f,   9.5f,  -3.2f,   7.7f,   6.0f,  14.3f,   4.7f,  18.9f,   3.0f,   9.2f,  -0.2f,  16.6f,  11.8f,  -4.8f},
									{ -7.1f,   0.0f,  -5.7f, -11.5f, -14.1f,  -0.6f,  -2.7f,  -5.8f, -12.8f,  -4.4f,  -8.4f,  -1.4f,  -4.6f,   6.2f,   1.3f,   4.6f,  11.2f,   9.8f,   2.6f,   1.2f,   6.9f,  10.5f,   5.0f,   5.5f,   1.9f,   7.0f},
									{ -2.2f,  -4.6f,  -8.2f,  -6.2f, -10.1f,  -7.4f,   1.8f,  -6.8f,  -4.9f,  -4.0f,  -1.9f,  -2.8f,  -5.2f,   3.3f,   5.2f,   0.6f,   1.4f,   3.0f,  16.7f,   6.3f,   8.5f,  -0.3f,  -1.7f,   2.4f,  11.5f,  13.4f},
									{ -5.7f,  -8.1f,  -0.6f,  -9.8f,  -2.8f, -18.4f, -14.2f, -11.8f,  -6.3f,  -4.1f,  -6.4f, -10.3f, -14.7f,  -2.2f,   4.2f,   7.8f,   9.8f,  11.0f,  -5.8f,  -1.6f,   2.0f,   5.6f,  12.0f,   4.4f,  -5.1f,  -2.9f},
									{ -9.7f,  -9.9f,  -7.1f,  -7.6f, -26.0f, -13.4f, -11.8f,  -3.5f, -16.8f,   0.0f, -18.9f,  -3.5f, -19.8f,   6.4f,   9.2f,   9.9f,   6.4f,  26.2f,   6.6f,   6.4f,   2.6f,  16.3f,   2.2f,  16.3f,   0.4f,  16.3f},
									{ -9.1f,  -0.7f,  -1.3f, -10.5f,  -9.1f, -10.2f, -14.4f,  -9.8f,  -9.8f,  -1.4f,  -8.4f, -12.3f,  -5.3f,   2.6f,   8.7f,   7.4f,  10.0f,   0.4f,  12.6f,  -4.8f,  11.3f,   2.5f,   9.5f,  -6.1f,   3.9f,  10.0f},
									{ -7.7f,  -8.3f,   4.0f,  -0.5f, -11.1f,  -8.5f, -22.1f,  -4.5f,  -2.2f,   0.9f, -16.6f, -13.2f, -17.5f,   9.1f,   8.9f,   9.2f,  10.0f,  26.6f,  -4.3f,   8.2f,   1.2f,  18.3f,  14.4f,  17.4f,  -0.1f,   3.4f},
									{ -6.6f,  -6.8f,  -8.6f,  -6.9f, -11.5f, -19.9f,  -6.3f, -13.4f,  -5.7f,  -2.8f,  -3.6f, -13.1f, -13.3f,   2.9f,   5.8f,   6.5f,   0.1f,   6.7f,  17.3f,   5.7f,   8.7f,  -0.1f,  -0.3f,   0.2f,  11.5f,  14.4f},
									{ -8.4f,  -5.0f,  -6.3f, -14.0f, -18.6f, -12.8f,  -6.0f, -13.4f, -14.4f,  -5.6f, -12.3f,  -7.8f,  -4.4f,   8.2f,   5.1f,   2.3f,  15.6f,   5.4f,  11.0f,   0.8f,  13.3f,  10.5f,   7.4f,   3.1f,   5.9f,  10.5f},
									{ -7.1f,  -9.9f,   0.3f,  -7.0f,  -6.6f, -10.3f,  -7.2f,  -7.3f,  -6.8f,  -6.0f,  -6.9f,  -7.4f, -13.8f,   8.5f,   0.4f,   4.9f,  13.8f,  17.3f,   4.0f,   9.5f,   8.9f,  13.4f,   5.3f,  13.4f,   3.6f,  -0.5f},
									{ -8.0f,  -5.9f,  -8.2f,  -8.6f, -15.2f, -19.2f,  -7.4f, -13.9f, -11.8f,  -4.8f,  -7.0f, -13.3f, -11.2f,  -3.6f,   3.4f,   5.3f,  -3.4f,   3.2f,   8.0f,   1.2f,  -0.2f,  -2.7f,   0.8f,  -2.1f,   4.6f,  11.6f},
									{ -4.0f,  -8.7f,  -6.1f, -14.5f,  -5.7f, -15.4f,   2.0f, -12.2f, -10.1f, -10.5f,   0.4f,  -6.7f, -11.9f,   9.5f,  -4.4f,   3.2f,   4.0f,  17.6f,   1.4f,  15.0f,   2.7f,  10.8f,  -5.5f,  16.3f,   8.2f,   1.7f},
									{ -6.1f,  -9.4f,  -7.5f,  -1.2f, -12.0f,  -3.9f, -11.0f,  -0.7f,  -6.6f,  -5.2f, -11.5f,  -5.6f, -13.6f,   9.7f,   6.6f,   4.2f,   4.8f,  23.3f,  19.8f,  14.9f,  12.3f,  13.9f,   4.9f,  19.1f,  13.2f,  14.1f},
									{ -5.8f,  -6.3f,  -6.3f, -18.4f,  -9.8f,  -5.8f,  -1.8f, -12.1f, -12.1f, -12.6f,  -5.8f,  -1.6f, -12.3f,   9.8f,   0.2f,   9.5f,  16.5f,  22.1f,  -1.4f,   6.8f,   7.0f,  19.3f,   6.7f,  12.6f,   0.5f,   4.2f},
									{ -8.4f,  -4.3f,  -5.7f, -18.4f,  -9.8f,  -7.0f,   1.6f, -12.7f, -14.1f, -10.0f,  -4.1f,  -2.7f,  -0.1f,  10.0f,   4.8f,   2.4f,  17.2f,   7.6f,  12.4f,   2.8f,  14.8f,  12.4f,   7.2f,   5.2f,   7.6f,   2.4f},
									{ -4.9f,  -9.0f,   5.7f,   0.6f,   1.3f, -13.0f, -18.5f,  -6.2f,   1.9f,  -3.3f,  -8.6f, -11.7f, -14.7f,   6.0f,   9.6f,   9.9f,  20.5f,  20.7f,   0.7f,   9.3f,  10.6f,  11.7f,  19.5f,  15.0f,   0.3f,  -0.3f},
									{ -7.7f,  -6.9f,  -9.7f, -22.2f, -10.9f, -10.1f,  -9.9f, -12.5f, -15.3f, -16.6f, -10.4f,  -7.2f, -11.2f,   1.1f,   2.7f,   5.6f,   1.1f,  12.3f,  -2.8f,  11.0f,  -4.5f,   6.7f,   2.2f,   6.7f,   4.1f,   3.2f}};
	
	checkTreeDistances(inputVerts_17, expected_17_6, expected_17_14, expected_17_18, expected_17_26);
}

void TreeTest::checkTreeDistances(std::vector< std::vector<Vector3f> >& verts,
								  float expected_6 [][6],
								  float expected_14[][14],
								  float expected_18[][18],
								  float expected_26[][26])
{
	//Generate list of leaf nodes with KDOP from vertices
	std::list<Node*> leafNodes_6;
	std::list<Node*> leafNodes_14;
	std::list<Node*> leafNodes_18;
	std::list<Node*> leafNodes_26;
	
	for(int i = 0; i < verts.size(); i++)
	{
		leafNodes_6.push_back (new Node(new KDOP(verts[i], 6)));
		leafNodes_14.push_back(new Node(new KDOP(verts[i], 14)));
		leafNodes_18.push_back(new Node(new KDOP(verts[i], 18)));
		leafNodes_26.push_back(new Node(new KDOP(verts[i], 26)));
	}
	
	//Build tree
	Node* root_6  = Node::buildTree(leafNodes_6);
	Node* root_14 = Node::buildTree(leafNodes_14);
	Node* root_18 = Node::buildTree(leafNodes_18);
	Node* root_26 = Node::buildTree(leafNodes_26);
	
	//Get list of all tree nodes
	std::list<Node*> actual_6;
	std::list<Node*> actual_14;
	std::list<Node*> actual_18;
	std::list<Node*> actual_26;
	
	Node::breadthWalk(root_6,  actual_6);
	Node::breadthWalk(root_14, actual_14);
	Node::breadthWalk(root_18, actual_18);
	Node::breadthWalk(root_26, actual_26);
	
	//Check list of tree nodes' distances against expected output
	float delta =  0.000005f;
	int count_6 = 0;
	int count_14 = 0;
	int count_18 = 0;
	int count_26 = 0;
	
	while(!actual_6.empty())
	{
		const float* distances = actual_6.front()->getKDOP()->getDistances();
		
		for(int i = 0; i < actual_6.front()->getKDOP()->K; i++)
			CPPUNIT_ASSERT_DOUBLES_EQUAL(expected_6[count_6][i], distances[i], delta);
		
		count_6++;
		actual_6.pop_front();
	}
	
	while(!actual_14.empty())
	{
		const float* distances = actual_14.front()->getKDOP()->getDistances();
		
		for(int i = 0; i < actual_14.front()->getKDOP()->K; i++)
			CPPUNIT_ASSERT_DOUBLES_EQUAL(expected_14[count_14][i], distances[i], delta);
		
		count_14++;
		actual_14.pop_front();
	}
	
	while(!actual_18.empty())
	{
		const float* distances = actual_18.front()->getKDOP()->getDistances();
		
		for(int i = 0; i < actual_18.front()->getKDOP()->K; i++)
			CPPUNIT_ASSERT_DOUBLES_EQUAL(expected_18[count_18][i], distances[i], delta);
		
		count_18++;
		actual_18.pop_front();
	}
	
	while(!actual_26.empty())
	{
		const float* distances = actual_26.front()->getKDOP()->getDistances();
		
		for(int i = 0; i < actual_26.front()->getKDOP()->K; i++)
			CPPUNIT_ASSERT_DOUBLES_EQUAL(expected_26[count_26][i], distances[i], delta);
		
		count_26++;
		actual_26.pop_front();
	}
	
	//Release memory
	delete root_6;
	delete root_14;
	delete root_18;
	delete root_26;
}

void TreeTest::testUpdateTree()
{
	//1 leaf node, 1 leaf changed
	std::vector<Vector3f> leaf_1_1_b;
	leaf_1_1_b.push_back(Vector3f( 0.5f, -2.4f, -7.5f));
	leaf_1_1_b.push_back(Vector3f( 6.4f, -4.7f, -1.6f));
	leaf_1_1_b.push_back(Vector3f( 4.0f,  2.3f,  1.6f));
	
	std::vector< std::vector<Vector3f> > inputVerts_1_before;
	inputVerts_1_before.push_back(leaf_1_1_b);
	
	std::vector<Vector3f> leaf_1_1_a;
	leaf_1_1_a.push_back(Vector3f(10.0f,  5.8f,  7.2f));
	leaf_1_1_a.push_back(Vector3f(-4.9f, -5.2f, -3.0f));
	leaf_1_1_a.push_back(Vector3f(-0.9f,  7.2f,  1.4f));
	
	std::vector< std::vector<Vector3f> > inputVerts_1_after;
	inputVerts_1_after.push_back(leaf_1_1_a);
	
	//Flags indicating which leaf nodes in the list will be updated
	std::vector<bool> leaf_1_flags;
	leaf_1_flags.push_back(true);
	
	float expected_1_6 [2][6 ] = {{ -4.9f,  -5.2f,  -3.0f,  10.0f,   7.2f,   7.2f},
								  { -4.9f,  -5.2f,  -3.0f,  10.0f,   7.2f,   7.2f}};
	float expected_1_14[2][14] = {{ -4.9f,  -5.2f,  -3.0f, -13.1f,  -6.7f,  -7.1f,  -9.5f,  10.0f,   7.2f,   7.2f,  23.0f,  11.4f,   8.6f,   3.3f},
								  { -4.9f,  -5.2f,  -3.0f, -13.1f,  -6.7f,  -7.1f,  -9.5f,  10.0f,   7.2f,   7.2f,  23.0f,  11.4f,   8.6f,   3.3f}};
	float expected_1_18[2][18] = {{ -4.9f,  -5.2f,  -3.0f, -10.1f,  -7.9f,  -8.2f,  -8.1f,  -2.3f,  -2.2f,  10.0f,   7.2f,   7.2f,  15.8f,  17.2f,  13.0f,   4.2f,   2.8f,   5.8f},
								  { -4.9f,  -5.2f,  -3.0f, -10.1f,  -7.9f,  -8.2f,  -8.1f,  -2.3f,  -2.2f,  10.0f,   7.2f,   7.2f,  15.8f,  17.2f,  13.0f,   4.2f,   2.8f,   5.8f}};
	float expected_1_26[2][26] = {{ -4.9f,  -5.2f,  -3.0f, -13.1f,  -6.7f,  -7.1f,  -9.5f, -10.1f,  -7.9f,  -8.2f,  -8.1f,  -2.3f,  -2.2f,  10.0f,   7.2f,   7.2f,  23.0f,  11.4f,   8.6f,   3.3f,  15.8f,  17.2f,  13.0f,   4.2f,   2.8f,   5.8f},
								  { -4.9f,  -5.2f,  -3.0f, -13.1f,  -6.7f,  -7.1f,  -9.5f, -10.1f,  -7.9f,  -8.2f,  -8.1f,  -2.3f,  -2.2f,  10.0f,   7.2f,   7.2f,  23.0f,  11.4f,   8.6f,   3.3f,  15.8f,  17.2f,  13.0f,   4.2f,   2.8f,   5.8f}};
	
	checkTreeUpdate(inputVerts_1_before, inputVerts_1_after, leaf_1_flags, expected_1_6, expected_1_14, expected_1_18, expected_1_26);
	
	//4 leaf nodes, 2 leaf nodes changed
	std::vector<Vector3f> leaf_4_1_b;
	leaf_4_1_b.push_back(Vector3f( 8.9f, -9.8f,  6.6f));
	leaf_4_1_b.push_back(Vector3f(-1.1f,  6.6f, -5.7f));
	leaf_4_1_b.push_back(Vector3f( 3.8f,  8.4f,  2.9f));
	
	std::vector<Vector3f> leaf_4_2_b;
	leaf_4_2_b.push_back(Vector3f());
	leaf_4_2_b.push_back(Vector3f());
	leaf_4_2_b.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_4_3_b;
	leaf_4_3_b.push_back(Vector3f(-1.4f, -7.9f,  9.4f));
	leaf_4_3_b.push_back(Vector3f( 4.1f,  0.5f,  7.0f));
	leaf_4_3_b.push_back(Vector3f( 2.9f,  6.8f,  7.6f));
	
	std::vector<Vector3f> leaf_4_4_b;
	leaf_4_4_b.push_back(Vector3f());
	leaf_4_4_b.push_back(Vector3f());
	leaf_4_4_b.push_back(Vector3f());
	
	std::vector< std::vector<Vector3f> > inputVerts_4_before;
	inputVerts_4_before.push_back(leaf_4_1_b);
	inputVerts_4_before.push_back(leaf_4_2_b);
	inputVerts_4_before.push_back(leaf_4_3_b);
	inputVerts_4_before.push_back(leaf_4_4_b);
	
	std::vector<Vector3f> leaf_4_1_a;
	leaf_4_1_a.push_back(Vector3f());
	leaf_4_1_a.push_back(Vector3f());
	leaf_4_1_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_4_2_a;
	leaf_4_2_a.push_back(Vector3f(-3.7f,  8.7f,  1.9f));
	leaf_4_2_a.push_back(Vector3f(-3.7f, -8.2f, -7.0f));
	leaf_4_2_a.push_back(Vector3f( 5.7f,  5.2f,  3.1f));
	
	std::vector<Vector3f> leaf_4_3_a;
	leaf_4_3_a.push_back(Vector3f());
	leaf_4_3_a.push_back(Vector3f());
	leaf_4_3_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_4_4_a;
	leaf_4_4_a.push_back(Vector3f(-2.3f, -1.9f, -2.1f));
	leaf_4_4_a.push_back(Vector3f( 6.3f, -6.2f, -3.3f));
	leaf_4_4_a.push_back(Vector3f( 6.5f, -6.8f,  6.1f));
	
	std::vector< std::vector<Vector3f> > inputVerts_4_after;
	inputVerts_4_after.push_back(leaf_4_1_a);
	inputVerts_4_after.push_back(leaf_4_2_a);
	inputVerts_4_after.push_back(leaf_4_3_a);
	inputVerts_4_after.push_back(leaf_4_4_a);
	
	std::vector<bool> leaf_4_flags;
	leaf_4_flags.push_back(false);
	leaf_4_flags.push_back(true);
	leaf_4_flags.push_back(false);
	leaf_4_flags.push_back(true);
	
	float expected_4_6 [5][6 ] = {{ -3.7f,  -9.8f,  -7.0f,   8.9f,   8.7f,   9.4f},
								  { -1.1f,  -9.8f,  -5.7f,   8.9f,   8.4f,   6.6f},
								  { -3.7f,  -8.2f,  -7.0f,   5.7f,   8.7f,   3.1f},
								  { -1.4f,  -7.9f,   7.0f,   4.1f,   6.8f,   9.4f},
								  { -2.3f,  -6.8f,  -3.3f,   6.5f,  -1.9f,   6.1f}};
	float expected_4_14[5][14] = {{ -3.7f,  -9.8f,  -7.0f, -18.9f, -13.4f, -18.7f, -14.3f,   8.9f,   8.7f,   9.4f,  17.3f,  25.3f,  11.2f,  15.8f},
								  { -1.1f,  -9.8f,  -5.7f,  -0.2f, -13.4f,  -7.5f,  -7.5f,   8.9f,   8.4f,   6.6f,  15.1f,  25.3f,  11.2f,  12.1f},
								  { -3.7f,  -8.2f,  -7.0f, -18.9f, -10.5f,  -4.9f, -14.3f,   5.7f,   8.7f,   3.1f,  14.0f,   3.6f,   7.8f,  11.5f},
								  { -1.4f,  -7.9f,   7.0f,   0.1f,   3.7f, -18.7f, -11.5f,   4.1f,   6.8f,   9.4f,  17.3f,  15.9f,   2.1f,  -2.9f},
								  { -2.3f,  -6.8f,  -3.3f,  -6.3f,  -2.5f,  -6.4f,   1.7f,   6.5f,  -1.9f,   6.1f,   5.8f,  19.4f,   3.4f,  15.8f}};
	float expected_4_18[5][18] = {{ -3.7f,  -9.8f,  -7.0f, -11.9f, -10.7f, -15.2f, -12.4f, -10.8f, -17.3f,   8.9f,   8.7f,   9.4f,  12.2f,  15.5f,  14.4f,  18.7f,   9.6f,  12.3f},
								  { -1.1f,  -9.8f,  -5.7f,  -0.9f,  -6.8f,  -3.2f,  -7.7f,   0.9f, -16.4f,   8.9f,   8.4f,   6.6f,  12.2f,  15.5f,  11.3f,  18.7f,   4.6f,  12.3f},
								  { -3.7f,  -8.2f,  -7.0f, -11.9f, -10.7f, -15.2f, -12.4f,  -5.6f,  -1.2f,   5.7f,   8.7f,   3.1f,  10.9f,   8.8f,  10.6f,   4.5f,   3.3f,   6.8f},
								  { -1.4f,  -7.9f,   7.0f,  -9.3f,   8.0f,   1.5f,  -3.9f, -10.8f, -17.3f,   4.1f,   6.8f,   9.4f,   9.7f,  11.1f,  14.4f,   6.5f,  -2.9f,  -0.8f},
								  { -2.3f,  -6.8f,  -3.3f,  -4.2f,  -4.4f,  -9.5f,  -0.4f,  -0.2f, -12.9f,   6.5f,  -1.9f,   6.1f,   0.1f,  12.6f,  -0.7f,  13.3f,   9.6f,   0.2f}};
	float expected_4_26[5][26] = {{ -3.7f,  -9.8f,  -7.0f, -18.9f, -13.4f, -18.7f, -14.3f, -11.9f, -10.7f, -15.2f, -12.4f, -10.8f, -17.3f,   8.9f,   8.7f,   9.4f,  17.3f,  25.3f,  11.2f,  15.8f,  12.2f,  15.5f,  14.4f,  18.7f,   9.6f,  12.3f},
								  { -1.1f,  -9.8f,  -5.7f,  -0.2f, -13.4f,  -7.5f,  -7.5f,  -0.9f,  -6.8f,  -3.2f,  -7.7f,   0.9f, -16.4f,   8.9f,   8.4f,   6.6f,  15.1f,  25.3f,  11.2f,  12.1f,  12.2f,  15.5f,  11.3f,  18.7f,   4.6f,  12.3f},
								  { -3.7f,  -8.2f,  -7.0f, -18.9f, -10.5f,  -4.9f, -14.3f, -11.9f, -10.7f, -15.2f, -12.4f,  -5.6f,  -1.2f,   5.7f,   8.7f,   3.1f,  14.0f,   3.6f,   7.8f,  11.5f,  10.9f,   8.8f,  10.6f,   4.5f,   3.3f,   6.8f},
								  { -1.4f,  -7.9f,   7.0f,   0.1f,   3.7f, -18.7f, -11.5f,  -9.3f,   8.0f,   1.5f,  -3.9f, -10.8f, -17.3f,   4.1f,   6.8f,   9.4f,  17.3f,  15.9f,   2.1f,  -2.9f,   9.7f,  11.1f,  14.4f,   6.5f,  -2.9f,  -0.8f},
								  { -2.3f,  -6.8f,  -3.3f,  -6.3f,  -2.5f,  -6.4f,   1.7f,  -4.2f,  -4.4f,  -9.5f,  -0.4f,  -0.2f, -12.9f,   6.5f,  -1.9f,   6.1f,   5.8f,  19.4f,   3.4f,  15.8f,   0.1f,  12.6f,  -0.7f,  13.3f,   9.6f,   0.2f}};
	
	checkTreeUpdate(inputVerts_4_before, inputVerts_4_after, leaf_4_flags, expected_4_6, expected_4_14, expected_4_18, expected_4_26);
	
	//5 leaf nodes, 2 leaf nodes changed
	std::vector<Vector3f> leaf_5_1_b;
	leaf_5_1_b.push_back(Vector3f());
	leaf_5_1_b.push_back(Vector3f());
	leaf_5_1_b.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_5_2_b;
	leaf_5_2_b.push_back(Vector3f(-0.3f, -4.4f,  8.1f));
	leaf_5_2_b.push_back(Vector3f(-0.1f, -3.6f, -3.3f));
	leaf_5_2_b.push_back(Vector3f( 0.8f,  5.3f, -0.2f));
	
	std::vector<Vector3f> leaf_5_3_b;
	leaf_5_3_b.push_back(Vector3f( 1.4f,  5.7f,  7.8f));
	leaf_5_3_b.push_back(Vector3f(-6.0f, -6.6f, -1.4f));
	leaf_5_3_b.push_back(Vector3f( 3.4f, -2.8f, -2.0f));
	
	std::vector<Vector3f> leaf_5_4_b;
	leaf_5_4_b.push_back(Vector3f( 8.1f, -4.2f, -1.5f));
	leaf_5_4_b.push_back(Vector3f( 5.0f,  9.9f,  9.9f));
	leaf_5_4_b.push_back(Vector3f( 7.8f,  5.2f,  8.4f));
	
	std::vector<Vector3f> leaf_5_5_b;
	leaf_5_5_b.push_back(Vector3f());
	leaf_5_5_b.push_back(Vector3f());
	leaf_5_5_b.push_back(Vector3f());
	
	std::vector< std::vector<Vector3f> > inputVerts_5_before;
	inputVerts_5_before.push_back(leaf_5_1_b);
	inputVerts_5_before.push_back(leaf_5_2_b);
	inputVerts_5_before.push_back(leaf_5_3_b);
	inputVerts_5_before.push_back(leaf_5_4_b);
	inputVerts_5_before.push_back(leaf_5_5_b);
	
	std::vector<Vector3f> leaf_5_1_a;
	leaf_5_1_a.push_back(Vector3f( 6.6f,  0.5f, -7.6f));
	leaf_5_1_a.push_back(Vector3f( 6.2f, -8.9f,  4.8f));
	leaf_5_1_a.push_back(Vector3f(-2.2f, -6.6f,  7.8f));
	
	std::vector<Vector3f> leaf_5_2_a;
	leaf_5_2_a.push_back(Vector3f());
	leaf_5_2_a.push_back(Vector3f());
	leaf_5_2_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_5_3_a;
	leaf_5_3_a.push_back(Vector3f());
	leaf_5_3_a.push_back(Vector3f());
	leaf_5_3_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_5_4_a;
	leaf_5_4_a.push_back(Vector3f());
	leaf_5_4_a.push_back(Vector3f());
	leaf_5_4_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_5_5_a;
	leaf_5_5_a.push_back(Vector3f( 6.5f,  7.7f, -7.3f));
	leaf_5_5_a.push_back(Vector3f( 9.4f, -7.8f,  6.5f));
	leaf_5_5_a.push_back(Vector3f( 6.0f, -6.4f,  2.4f));
	
	std::vector< std::vector<Vector3f> > inputVerts_5_after;
	inputVerts_5_after.push_back(leaf_5_1_a);
	inputVerts_5_after.push_back(leaf_5_2_a);
	inputVerts_5_after.push_back(leaf_5_3_a);
	inputVerts_5_after.push_back(leaf_5_4_a);
	inputVerts_5_after.push_back(leaf_5_5_a);
	
	std::vector<bool> leaf_5_flags;
	leaf_5_flags.push_back(true);
	leaf_5_flags.push_back(false);
	leaf_5_flags.push_back(false);
	leaf_5_flags.push_back(false);
	leaf_5_flags.push_back(true);
	
	float expected_5_6 [8][6 ] = {{ -6.0f,  -8.9f,  -7.6f,   9.4f,   9.9f,   9.9f},
								  { -6.0f,  -8.9f,  -7.6f,   8.1f,   9.9f,   9.9f},
								  {  6.0f,  -7.8f,  -7.3f,   9.4f,   7.7f,   6.5f},
								  { -2.2f,  -8.9f,  -7.6f,   6.6f,   0.5f,   7.8f},
								  { -0.3f,  -4.4f,  -3.3f,   0.8f,   5.3f,   8.1f},
								  { -6.0f,  -6.6f,  -2.0f,   3.4f,   5.7f,   7.8f},
								  {  5.0f,  -4.2f,  -1.5f,   8.1f,   9.9f,   9.9f},
								  {  6.0f,  -7.8f,  -7.3f,   9.4f,   7.7f,   6.5f}};
	float expected_5_14[8][14] = {{ -6.0f,  -8.9f,  -7.6f, -14.0f,  -8.5f, -16.6f, -14.8f,   9.4f,   9.9f,   9.9f,  24.8f,  23.7f,  21.5f,  13.8f},
								  { -6.0f,  -8.9f,  -7.6f, -14.0f,  -4.7f, -16.6f, -14.8f,   8.1f,   9.9f,   9.9f,  24.8f,  19.9f,  14.7f,  13.8f},
								  {  6.0f,  -7.8f,  -7.3f,   2.0f,  -8.5f,  -4.9f,   6.1f,   9.4f,   7.7f,   6.5f,   8.1f,  23.7f,  21.5f,  10.7f},
								  { -2.2f,  -8.9f,  -7.6f,  -1.0f,  -1.5f, -16.6f,  -3.4f,   6.6f,   0.5f,   7.8f,   2.1f,  19.9f,  14.7f,  13.7f},
								  { -0.3f,  -4.4f,  -3.3f,  -7.0f,  -4.7f, -12.8f,  -4.3f,   0.8f,   5.3f,   8.1f,   5.9f,  12.2f,   6.3f,   6.8f},
								  { -6.0f,  -6.6f,  -2.0f, -14.0f,  -0.8f, -11.2f, -12.1f,   3.4f,   5.7f,   7.8f,  14.9f,   4.2f,   2.6f,   8.2f},
								  {  5.0f,  -4.2f,  -1.5f,   2.4f,   5.0f,   4.6f, -14.8f,   8.1f,   9.9f,   9.9f,  24.8f,  11.0f,   5.4f,  13.8f},
								  {  6.0f,  -7.8f,  -7.3f,   2.0f,  -8.5f,  -4.9f,   6.1f,   9.4f,   7.7f,   6.5f,   8.1f,  23.7f,  21.5f,  10.7f}};
	float expected_5_18[8][18] = {{ -6.0f,  -8.9f,  -7.6f, -12.6f,  -7.4f,  -8.0f,  -4.9f, -10.0f, -14.4f,   9.4f,   9.9f,   9.9f,  14.9f,  16.2f,  19.8f,  17.2f,  14.2f,  15.0f},
								  { -6.0f,  -8.9f,  -7.6f, -12.6f,  -7.4f,  -8.0f,  -4.9f, -10.0f, -14.4f,   8.1f,   9.9f,   9.9f,  14.9f,  16.2f,  19.8f,  15.1f,  14.2f,   8.1f},
								  {  6.0f,  -7.8f,  -7.3f,  -0.4f,  -0.8f,  -4.0f,  -1.2f,   2.9f, -14.3f,   9.4f,   7.7f,   6.5f,  14.2f,  15.9f,   0.4f,  17.2f,  13.8f,  15.0f},
								  { -2.2f,  -8.9f,  -7.6f,  -8.8f,  -1.0f,  -7.1f,   4.4f, -10.0f, -14.4f,   6.6f,   0.5f,   7.8f,   7.1f,  11.0f,   1.2f,  15.1f,  14.2f,   8.1f},
								  { -0.3f,  -4.4f,  -3.3f,  -4.7f,  -3.4f,  -6.9f,  -4.5f,  -8.4f, -12.5f,   0.8f,   5.3f,   8.1f,   6.1f,   7.8f,   5.1f,   4.1f,   3.2f,   5.5f},
								  { -6.0f,  -6.6f,  -2.0f, -12.6f,  -7.4f,  -8.0f,  -4.3f,  -6.4f,  -5.2f,   3.4f,   5.7f,   7.8f,   7.1f,   9.2f,  13.5f,   6.2f,   5.4f,  -0.8f},
								  {  5.0f,  -4.2f,  -1.5f,   3.9f,   6.6f,  -5.7f,  -4.9f,  -4.9f,  -3.2f,   8.1f,   9.9f,   9.9f,  14.9f,  16.2f,  19.8f,  12.3f,   9.6f,   0.0f},
								  {  6.0f,  -7.8f,  -7.3f,  -0.4f,  -0.8f,  -4.0f,  -1.2f,   2.9f, -14.3f,   9.4f,   7.7f,   6.5f,  14.2f,  15.9f,   0.4f,  17.2f,  13.8f,  15.0f}};
	float expected_5_26[8][26] = {{ -6.0f,  -8.9f,  -7.6f, -14.0f,  -8.5f, -16.6f, -14.8f, -12.6f,  -7.4f,  -8.0f,  -4.9f, -10.0f, -14.4f,   9.4f,   9.9f,   9.9f,  24.8f,  23.7f,  21.5f,  13.8f,  14.9f,  16.2f,  19.8f,  17.2f,  14.2f,  15.0f},
								  { -6.0f,  -8.9f,  -7.6f, -14.0f,  -4.7f, -16.6f, -14.8f, -12.6f,  -7.4f,  -8.0f,  -4.9f, -10.0f, -14.4f,   8.1f,   9.9f,   9.9f,  24.8f,  19.9f,  14.7f,  13.8f,  14.9f,  16.2f,  19.8f,  15.1f,  14.2f,   8.1f},
								  {  6.0f,  -7.8f,  -7.3f,   2.0f,  -8.5f,  -4.9f,   6.1f,  -0.4f,  -0.8f,  -4.0f,  -1.2f,   2.9f, -14.3f,   9.4f,   7.7f,   6.5f,   8.1f,  23.7f,  21.5f,  10.7f,  14.2f,  15.9f,   0.4f,  17.2f,  13.8f,  15.0f},
								  { -2.2f,  -8.9f,  -7.6f,  -1.0f,  -1.5f, -16.6f,  -3.4f,  -8.8f,  -1.0f,  -7.1f,   4.4f, -10.0f, -14.4f,   6.6f,   0.5f,   7.8f,   2.1f,  19.9f,  14.7f,  13.7f,   7.1f,  11.0f,   1.2f,  15.1f,  14.2f,   8.1f},
								  { -0.3f,  -4.4f,  -3.3f,  -7.0f,  -4.7f, -12.8f,  -4.3f,  -4.7f,  -3.4f,  -6.9f,  -4.5f,  -8.4f, -12.5f,   0.8f,   5.3f,   8.1f,   5.9f,  12.2f,   6.3f,   6.8f,   6.1f,   7.8f,   5.1f,   4.1f,   3.2f,   5.5f},
								  { -6.0f,  -6.6f,  -2.0f, -14.0f,  -0.8f, -11.2f, -12.1f, -12.6f,  -7.4f,  -8.0f,  -4.3f,  -6.4f,  -5.2f,   3.4f,   5.7f,   7.8f,  14.9f,   4.2f,   2.6f,   8.2f,   7.1f,   9.2f,  13.5f,   6.2f,   5.4f,  -0.8f},
								  {  5.0f,  -4.2f,  -1.5f,   2.4f,   5.0f,   4.6f, -14.8f,   3.9f,   6.6f,  -5.7f,  -4.9f,  -4.9f,  -3.2f,   8.1f,   9.9f,   9.9f,  24.8f,  11.0f,   5.4f,  13.8f,  14.9f,  16.2f,  19.8f,  12.3f,   9.6f,   0.0f},
								  {  6.0f,  -7.8f,  -7.3f,   2.0f,  -8.5f,  -4.9f,   6.1f,  -0.4f,  -0.8f,  -4.0f,  -1.2f,   2.9f, -14.3f,   9.4f,   7.7f,   6.5f,   8.1f,  23.7f,  21.5f,  10.7f,  14.2f,  15.9f,   0.4f,  17.2f,  13.8f,  15.0f}};
	
	checkTreeUpdate(inputVerts_5_before, inputVerts_5_after, leaf_5_flags, expected_5_6, expected_5_14, expected_5_18, expected_5_26);
	
	//16 leaf nodes, 6 leaf nodes changed
	std::vector<Vector3f> leaf_16_1_b;
	leaf_16_1_b.push_back(Vector3f( 7.6f,  1.9f, -3.9f));
	leaf_16_1_b.push_back(Vector3f(-5.8f, -7.3f, -7.4f));
	leaf_16_1_b.push_back(Vector3f( 2.9f, -6.8f,  1.8f));
	
	std::vector<Vector3f> leaf_16_2_b;
	leaf_16_2_b.push_back(Vector3f(-6.8f,  3.1f,  9.0f));
	leaf_16_2_b.push_back(Vector3f( 6.2f, -2.1f,  7.9f));
	leaf_16_2_b.push_back(Vector3f(-3.3f,  6.0f, -7.6f));
	
	std::vector<Vector3f> leaf_16_3_b;
	leaf_16_3_b.push_back(Vector3f());
	leaf_16_3_b.push_back(Vector3f());
	leaf_16_3_b.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_4_b;
	leaf_16_4_b.push_back(Vector3f( 6.9f, -5.0f,  7.4f));
	leaf_16_4_b.push_back(Vector3f(-7.7f, -8.3f, -4.8f));
	leaf_16_4_b.push_back(Vector3f(-3.8f,  6.6f,  2.0f));
	
	std::vector<Vector3f> leaf_16_5_b;
	leaf_16_5_b.push_back(Vector3f( 9.2f, -0.6f, -2.7f));
	leaf_16_5_b.push_back(Vector3f(-2.9f,  7.5f,  2.7f));
	leaf_16_5_b.push_back(Vector3f( 4.7f,  1.7f,  2.9f));
	
	std::vector<Vector3f> leaf_16_6_b;
	leaf_16_6_b.push_back(Vector3f(-4.1f, -5.7f, -7.5f));
	leaf_16_6_b.push_back(Vector3f( 3.7f,  3.0f, -6.2f));
	leaf_16_6_b.push_back(Vector3f( 7.9f,  1.4f, -0.2f));
	
	std::vector<Vector3f> leaf_16_7_b;
	leaf_16_7_b.push_back(Vector3f(-1.3f,  8.8f, -4.6f));
	leaf_16_7_b.push_back(Vector3f(-4.9f,  0.3f,  1.1f));
	leaf_16_7_b.push_back(Vector3f(-9.0f,  8.1f, -1.9f));
	
	std::vector<Vector3f> leaf_16_8_b;
	leaf_16_8_b.push_back(Vector3f(-2.2f, -9.5f,  0.8f));
	leaf_16_8_b.push_back(Vector3f(-9.5f,  1.2f, -3.4f));
	leaf_16_8_b.push_back(Vector3f( 2.2f, -0.7f, -2.2f));
	
	std::vector<Vector3f> leaf_16_9_b;
	leaf_16_9_b.push_back(Vector3f());
	leaf_16_9_b.push_back(Vector3f());
	leaf_16_9_b.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_10_b;
	leaf_16_10_b.push_back(Vector3f());
	leaf_16_10_b.push_back(Vector3f());
	leaf_16_10_b.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_11_b;
	leaf_16_11_b.push_back(Vector3f());
	leaf_16_11_b.push_back(Vector3f());
	leaf_16_11_b.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_12_b;
	leaf_16_12_b.push_back(Vector3f());
	leaf_16_12_b.push_back(Vector3f());
	leaf_16_12_b.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_13_b;
	leaf_16_13_b.push_back(Vector3f());
	leaf_16_13_b.push_back(Vector3f());
	leaf_16_13_b.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_14_b;
	leaf_16_14_b.push_back(Vector3f( 3.6f,  5.2f, -9.0f));
	leaf_16_14_b.push_back(Vector3f( 5.1f,  7.1f, -7.3f));
	leaf_16_14_b.push_back(Vector3f(-7.8f, -3.4f,  7.8f));
	
	std::vector<Vector3f> leaf_16_15_b;
	leaf_16_15_b.push_back(Vector3f(-6.9f,  1.6f,  4.4f));
	leaf_16_15_b.push_back(Vector3f( 3.5f, -0.2f,  1.1f));
	leaf_16_15_b.push_back(Vector3f( 3.9f,  9.0f,  5.0f));
	
	std::vector<Vector3f> leaf_16_16_b;
	leaf_16_16_b.push_back(Vector3f(-1.7f,  0.0f, -0.2f));
	leaf_16_16_b.push_back(Vector3f( 1.7f,  7.7f,  9.4f));
	leaf_16_16_b.push_back(Vector3f(-3.1f,  4.1f,  2.0f));
	
	std::vector< std::vector<Vector3f> > inputVerts_16_before;
	inputVerts_16_before.push_back(leaf_16_1_b);
	inputVerts_16_before.push_back(leaf_16_2_b);
	inputVerts_16_before.push_back(leaf_16_3_b);
	inputVerts_16_before.push_back(leaf_16_4_b);
	inputVerts_16_before.push_back(leaf_16_5_b);
	inputVerts_16_before.push_back(leaf_16_6_b);
	inputVerts_16_before.push_back(leaf_16_7_b);
	inputVerts_16_before.push_back(leaf_16_8_b);
	inputVerts_16_before.push_back(leaf_16_9_b);
	inputVerts_16_before.push_back(leaf_16_10_b);
	inputVerts_16_before.push_back(leaf_16_11_b);
	inputVerts_16_before.push_back(leaf_16_12_b);
	inputVerts_16_before.push_back(leaf_16_13_b);
	inputVerts_16_before.push_back(leaf_16_14_b);
	inputVerts_16_before.push_back(leaf_16_15_b);
	inputVerts_16_before.push_back(leaf_16_16_b);
	
	std::vector<Vector3f> leaf_16_1_a;
	leaf_16_1_a.push_back(Vector3f());
	leaf_16_1_a.push_back(Vector3f());
	leaf_16_1_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_2_a;
	leaf_16_2_a.push_back(Vector3f());
	leaf_16_2_a.push_back(Vector3f());
	leaf_16_2_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_3_a;
	leaf_16_3_a.push_back(Vector3f(-9.2f, -2.7f,  1.7f));
	leaf_16_3_a.push_back(Vector3f( 4.6f,  5.9f, -4.0f));
	leaf_16_3_a.push_back(Vector3f(-6.4f,  8.5f,  7.1f));
	
	std::vector<Vector3f> leaf_16_4_a;
	leaf_16_4_a.push_back(Vector3f());
	leaf_16_4_a.push_back(Vector3f());
	leaf_16_4_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_5_a;
	leaf_16_5_a.push_back(Vector3f());
	leaf_16_5_a.push_back(Vector3f());
	leaf_16_5_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_6_a;
	leaf_16_6_a.push_back(Vector3f());
	leaf_16_6_a.push_back(Vector3f());
	leaf_16_6_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_7_a;
	leaf_16_7_a.push_back(Vector3f());
	leaf_16_7_a.push_back(Vector3f());
	leaf_16_7_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_8_a;
	leaf_16_8_a.push_back(Vector3f());
	leaf_16_8_a.push_back(Vector3f());
	leaf_16_8_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_9_a;
	leaf_16_9_a.push_back(Vector3f( 1.9f,  3.1f,  2.3f));
	leaf_16_9_a.push_back(Vector3f(-6.9f, -5.1f, -9.8f));
	leaf_16_9_a.push_back(Vector3f(-2.9f,  9.3f,  5.5f));
	
	std::vector<Vector3f> leaf_16_10_a;
	leaf_16_10_a.push_back(Vector3f( 2.6f,  4.9f, -5.9f));
	leaf_16_10_a.push_back(Vector3f(-2.1f, -7.8f,  3.9f));
	leaf_16_10_a.push_back(Vector3f(-5.0f,  1.3f, -3.5f));
	
	std::vector<Vector3f> leaf_16_11_a;
	leaf_16_11_a.push_back(Vector3f( 5.4f,  1.0f, -3.0f));
	leaf_16_11_a.push_back(Vector3f(-0.2f,  9.0f, -3.4f));
	leaf_16_11_a.push_back(Vector3f(-3.1f,  4.2f, -3.3f));
	
	std::vector<Vector3f> leaf_16_12_a;
	leaf_16_12_a.push_back(Vector3f(-0.1f, -0.7f, -1.1f));
	leaf_16_12_a.push_back(Vector3f(-7.5f,  3.8f, -9.0f));
	leaf_16_12_a.push_back(Vector3f(-7.3f,  7.5f, -0.6f));
	
	std::vector<Vector3f> leaf_16_13_a;
	leaf_16_13_a.push_back(Vector3f(-0.6f,  2.7f,  1.0f));
	leaf_16_13_a.push_back(Vector3f( 0.6f, -3.9f,  4.8f));
	leaf_16_13_a.push_back(Vector3f(-9.1f, -4.0f,  7.6f));
	
	std::vector<Vector3f> leaf_16_14_a;
	leaf_16_14_a.push_back(Vector3f());
	leaf_16_14_a.push_back(Vector3f());
	leaf_16_14_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_15_a;
	leaf_16_15_a.push_back(Vector3f());
	leaf_16_15_a.push_back(Vector3f());
	leaf_16_15_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_16_16_a;
	leaf_16_16_a.push_back(Vector3f());
	leaf_16_16_a.push_back(Vector3f());
	leaf_16_16_a.push_back(Vector3f());
	
	std::vector< std::vector<Vector3f> > inputVerts_16_after;
	inputVerts_16_after.push_back(leaf_16_1_a);
	inputVerts_16_after.push_back(leaf_16_2_a);
	inputVerts_16_after.push_back(leaf_16_3_a);
	inputVerts_16_after.push_back(leaf_16_4_a);
	inputVerts_16_after.push_back(leaf_16_5_a);
	inputVerts_16_after.push_back(leaf_16_6_a);
	inputVerts_16_after.push_back(leaf_16_7_a);
	inputVerts_16_after.push_back(leaf_16_8_a);
	inputVerts_16_after.push_back(leaf_16_9_a);
	inputVerts_16_after.push_back(leaf_16_10_a);
	inputVerts_16_after.push_back(leaf_16_11_a);
	inputVerts_16_after.push_back(leaf_16_12_a);
	inputVerts_16_after.push_back(leaf_16_13_a);
	inputVerts_16_after.push_back(leaf_16_14_a);
	inputVerts_16_after.push_back(leaf_16_15_a);
	inputVerts_16_after.push_back(leaf_16_16_a);
	
	std::vector<bool> leaf_16_flags;
	leaf_16_flags.push_back(false);
	leaf_16_flags.push_back(false);
	leaf_16_flags.push_back(true);
	leaf_16_flags.push_back(false);
	leaf_16_flags.push_back(false);
	leaf_16_flags.push_back(false);
	leaf_16_flags.push_back(false);
	leaf_16_flags.push_back(false);
	leaf_16_flags.push_back(true);
	leaf_16_flags.push_back(true);
	leaf_16_flags.push_back(true);
	leaf_16_flags.push_back(true);
	leaf_16_flags.push_back(true);
	leaf_16_flags.push_back(false);
	leaf_16_flags.push_back(false);
	leaf_16_flags.push_back(false);
	
	float expected_16_6 [21][6 ] = {{ -9.5f,  -9.5f,  -9.8f,   9.2f,   9.3f,   9.4f},
									{ -9.2f,  -8.3f,  -7.6f,   7.6f,   8.5f,   9.0f},
									{ -9.5f,  -9.5f,  -7.5f,   9.2f,   8.8f,   2.9f},
									{ -7.5f,  -7.8f,  -9.8f,   5.4f,   9.3f,   5.5f},
									{ -9.1f,  -4.0f,  -9.0f,   5.1f,   9.0f,   9.4f},
									{ -5.8f,  -7.3f,  -7.4f,   7.6f,   1.9f,   1.8f},
									{ -6.8f,  -2.1f,  -7.6f,   6.2f,   6.0f,   9.0f},
									{ -9.2f,  -2.7f,  -4.0f,   4.6f,   8.5f,   7.1f},
									{ -7.7f,  -8.3f,  -4.8f,   6.9f,   6.6f,   7.4f},
									{ -2.9f,  -0.6f,  -2.7f,   9.2f,   7.5f,   2.9f},
									{ -4.1f,  -5.7f,  -7.5f,   7.9f,   3.0f,  -0.2f},
									{ -9.0f,   0.3f,  -4.6f,  -1.3f,   8.8f,   1.1f},
									{ -9.5f,  -9.5f,  -3.4f,   2.2f,   1.2f,   0.8f},
									{ -6.9f,  -5.1f,  -9.8f,   1.9f,   9.3f,   5.5f},
									{ -5.0f,  -7.8f,  -5.9f,   2.6f,   4.9f,   3.9f},
									{ -3.1f,   1.0f,  -3.4f,   5.4f,   9.0f,  -3.0f},
									{ -7.5f,  -0.7f,  -9.0f,  -0.1f,   7.5f,  -0.6f},
									{ -9.1f,  -4.0f,   1.0f,   0.6f,   2.7f,   7.6f},
									{ -7.8f,  -3.4f,  -9.0f,   5.1f,   7.1f,   7.8f},
									{ -6.9f,  -0.2f,   1.1f,   3.9f,   9.0f,   5.0f},
									{ -3.1f,   0.0f,  -0.2f,   1.7f,   7.7f,   9.4f}};
	float expected_16_14[21][14] = {{ -9.5f,  -9.5f,  -9.8f, -21.8f, -20.3f, -20.7f, -22.0f,   9.2f,   9.3f,   9.4f,  18.8f,  19.3f,  19.5f,  12.5f},
									{ -9.2f,  -8.3f,  -7.6f, -20.8f, -16.9f, -13.6f, -22.0f,   7.6f,   8.5f,   9.0f,  12.0f,  19.3f,  14.5f,   9.6f},
									{ -9.5f,  -9.5f,  -7.5f, -17.3f, -19.0f, -12.5f, -15.2f,   9.2f,   8.8f,   2.9f,   9.3f,   8.1f,  12.9f,  12.5f},
									{ -7.5f,  -7.8f,  -9.8f, -21.8f, -20.3f, -13.8f, -17.7f,   5.4f,   9.3f,   5.5f,  11.9f,   9.6f,  13.4f,   8.0f},
									{ -9.1f,  -4.0f,  -9.0f,  -5.5f, -10.6f, -20.7f, -15.4f,   5.1f,   9.0f,   9.4f,  18.8f,   9.3f,  19.5f,   7.4f},
									{ -5.8f,  -7.3f,  -7.4f, -20.5f,  -5.9f,  -5.7f,   7.9f,   7.6f,   1.9f,   1.8f,   5.6f,  11.5f,  13.4f,   9.6f},
									{ -6.8f,  -2.1f,  -7.6f,  -4.9f, -16.9f, -12.7f, -18.9f,   6.2f,   6.0f,   9.0f,  12.0f,  16.2f,  10.3f,   0.4f},
									{ -9.2f,  -2.7f,  -4.0f, -10.2f,  -7.8f, -13.6f, -22.0f,   4.6f,   8.5f,   7.1f,   9.2f,  -4.8f,  14.5f,   2.7f},
									{ -7.7f,  -8.3f,  -4.8f, -20.8f,  -8.4f, -11.2f, -12.4f,   6.9f,   6.6f,   7.4f,   9.3f,  19.3f,   0.8f,   5.4f},
									{ -2.9f,  -0.6f,  -2.7f,   5.9f,  -7.7f,   1.9f, -13.1f,   9.2f,   7.5f,   2.9f,   9.3f,   7.1f,  11.3f,  12.5f},
									{ -4.1f,  -5.7f,  -7.5f, -17.3f,  -5.9f,  -2.3f,   6.7f,   7.9f,   3.0f,  -0.2f,   9.1f,   6.3f,  12.9f,   9.1f},
									{ -9.0f,   0.3f,  -4.6f,  -3.5f, -19.0f,  -5.7f, -15.2f,  -1.3f,   8.8f,   1.1f,   2.9f,  -4.1f,  12.1f,  -5.5f},
									{ -9.5f,  -9.5f,  -3.4f, -11.7f, -14.1f, -12.5f,  -7.3f,   2.2f,   1.2f,   0.8f,  -0.7f,   8.1f,   3.7f,   6.5f},
									{ -6.9f,  -5.1f,  -9.8f, -21.8f, -11.6f,  -2.2f, -17.7f,   1.9f,   9.3f,   5.5f,  11.9f,   1.1f,   2.7f,   8.0f},
									{ -5.0f,  -7.8f,  -5.9f,  -7.2f,  -9.8f, -13.8f,  -2.8f,   2.6f,   4.9f,   3.9f,   1.6f,   9.6f,  13.4f,   3.6f},
									{ -3.1f,   1.0f,  -3.4f,  -2.2f, -12.6f,   4.4f,  -5.8f,   5.4f,   9.0f,  -3.0f,   5.4f,   1.4f,  12.2f,   7.4f},
									{ -7.5f,  -0.7f,  -9.0f, -12.7f, -20.3f,   0.3f, -14.2f,  -0.1f,   7.5f,  -0.6f,  -0.4f,  -0.5f,   5.3f,   1.7f},
									{ -9.1f,  -4.0f,   1.0f,  -5.5f,  -2.3f, -20.7f, -12.7f,   0.6f,   2.7f,   7.6f,   3.1f,   9.3f,   1.1f,  -0.3f},
									{ -7.8f,  -3.4f,  -9.0f,  -3.4f, -10.6f, -19.0f, -12.2f,   5.1f,   7.1f,   7.8f,   4.9f,   3.4f,  19.5f,   7.4f},
									{ -6.9f,  -0.2f,   1.1f,  -0.9f,  -4.1f,  -9.7f, -12.9f,   3.9f,   9.0f,   5.0f,  17.9f,   4.8f,   7.9f,   2.6f},
									{ -3.1f,   0.0f,  -0.2f,  -1.9f,  -5.2f,  -1.5f, -15.4f,   1.7f,   7.7f,   9.4f,  18.8f,   3.4f,   0.0f,  -1.5f}};
	float expected_16_18[21][18] = {{ -9.5f,  -9.5f,  -9.8f, -16.0f, -16.7f, -14.9f, -17.1f, -16.7f, -12.4f,   9.2f,   9.3f,   9.4f,  12.9f,  14.3f,  17.1f,  11.9f,  12.6f,  14.4f},
									{ -9.2f,  -8.3f,  -7.6f, -16.0f, -13.2f, -14.7f, -14.9f, -15.8f, -12.4f,   7.6f,   8.5f,   9.0f,  10.5f,  14.3f,  15.6f,  11.9f,  11.5f,  13.6f},
									{ -9.5f,  -9.5f,  -7.5f, -11.7f, -12.9f, -13.2f, -17.1f,  -7.1f, -10.3f,   9.2f,   8.8f,   2.9f,   9.3f,   7.7f,  10.2f,   9.8f,  11.9f,  13.4f},
									{ -7.5f,  -7.8f,  -9.8f, -12.0f, -16.7f, -14.9f, -14.8f,  -8.4f, -11.7f,   5.4f,   9.3f,   5.5f,   8.8f,   4.2f,  14.8f,   5.7f,   8.5f,  12.8f},
									{ -9.1f,  -4.0f,  -9.0f, -13.1f,  -5.4f,  -3.8f,  -8.5f, -16.7f, -11.6f,   5.1f,   9.0f,   9.4f,  12.9f,  11.1f,  17.1f,   4.5f,  12.6f,  14.4f},
									{ -5.8f,  -7.3f,  -7.4f, -13.1f, -13.2f, -14.7f,   1.5f,   1.1f,  -8.6f,   7.6f,   1.9f,   1.8f,   9.5f,   4.7f,  -2.0f,   9.7f,  11.5f,   5.8f},
									{ -6.8f,  -2.1f,  -7.6f,  -3.7f, -10.9f,  -1.6f,  -9.9f, -15.8f, -10.0f,   6.2f,   6.0f,   9.0f,   4.1f,  14.1f,  12.1f,   8.3f,   4.3f,  13.6f},
									{ -9.2f,  -2.7f,  -4.0f, -11.9f,  -7.5f,  -1.0f, -14.9f, -13.5f,  -4.4f,   4.6f,   8.5f,   7.1f,  10.5f,   0.7f,  15.6f,  -1.3f,   8.6f,   9.9f},
									{ -7.7f,  -8.3f,  -4.8f, -16.0f, -12.5f, -13.1f, -10.4f,  -5.8f, -12.4f,   6.9f,   6.6f,   7.4f,   2.8f,  14.3f,   8.6f,  11.9f,  -0.5f,   4.6f},
									{ -2.9f,  -0.6f,  -2.7f,   4.6f,  -0.2f,  -3.3f, -10.4f,  -5.6f,  -1.2f,   9.2f,   7.5f,   2.9f,   8.6f,   7.6f,  10.2f,   9.8f,  11.9f,   4.8f},
									{ -4.1f,  -5.7f,  -7.5f,  -9.8f, -11.6f, -13.2f,   0.7f,   3.4f,   1.6f,   7.9f,   3.0f,  -0.2f,   9.3f,   7.7f,   1.2f,   6.5f,   9.9f,   9.2f},
									{ -9.0f,   0.3f,  -4.6f,  -4.6f, -10.9f,   1.4f, -17.1f,  -7.1f,  -0.8f,  -1.3f,   8.8f,   1.1f,   7.5f,  -3.8f,   6.2f,  -5.2f,   3.3f,  13.4f},
									{ -9.5f,  -9.5f,  -3.4f, -11.7f, -12.9f,  -8.7f, -10.7f,  -6.1f, -10.3f,   2.2f,   1.2f,   0.8f,   1.5f,   0.0f,  -2.2f,   7.3f,   4.4f,   4.6f},
									{ -6.9f,  -5.1f,  -9.8f, -12.0f, -16.7f, -14.9f, -12.2f,  -8.4f,   0.8f,   1.9f,   9.3f,   5.5f,   6.4f,   4.2f,  14.8f,  -1.2f,   2.9f,   4.7f},
									{ -5.0f,  -7.8f,  -5.9f,  -9.9f,  -8.5f,  -3.9f,  -6.3f,  -6.0f, -11.7f,   2.6f,   4.9f,   3.9f,   7.5f,   1.8f,  -1.0f,   5.7f,   8.5f,  10.8f},
									{ -3.1f,   1.0f,  -3.4f,   1.1f,  -6.4f,  -2.0f,  -9.2f,   0.2f,   4.0f,   5.4f,   9.0f,  -3.0f,   8.8f,   2.4f,   5.6f,   4.4f,   8.4f,  12.4f},
									{ -7.5f,  -0.7f,  -9.0f,  -3.7f, -16.5f,  -5.2f, -14.8f,  -6.7f,   0.4f,  -0.1f,   7.5f,  -0.6f,   0.2f,  -1.2f,   6.9f,   0.6f,   1.5f,  12.8f},
									{ -9.1f,  -4.0f,   1.0f, -13.1f,  -1.5f,   0.9f,  -5.1f, -16.7f, -11.6f,   0.6f,   2.7f,   7.6f,   2.1f,   5.4f,   3.7f,   4.5f,  -1.6f,   1.7f},
									{ -7.8f,  -3.4f,  -9.0f, -11.2f,  -5.4f,  -3.8f,  -4.4f, -15.6f, -11.2f,   5.1f,   7.1f,   7.8f,  12.2f,   0.0f,   4.4f,  -1.6f,  12.6f,  14.4f},
									{ -6.9f,  -0.2f,   1.1f,  -5.3f,  -2.5f,   0.9f,  -8.5f, -11.3f,  -2.8f,   3.9f,   9.0f,   5.0f,  12.9f,   8.9f,  14.0f,   3.7f,   2.4f,   4.0f},
									{ -3.1f,   0.0f,  -0.2f,  -1.7f,  -1.9f,  -0.2f,  -7.2f,  -7.7f,  -1.7f,   1.7f,   7.7f,   9.4f,   9.4f,  11.1f,  17.1f,  -1.7f,  -1.5f,   2.1f}};
	float expected_16_26[21][26] = {{-9.5f,  -9.5f,  -9.8f, -21.8f, -20.3f, -20.7f, -22.0f, -16.0f, -16.7f, -14.9f, -17.1f, -16.7f, -12.4f,   9.2f,   9.3f,   9.4f,  18.8f,  19.3f,  19.5f,  12.5f,  12.9f,  14.3f,  17.1f,  11.9f,  12.6f,  14.4f},
									{ -9.2f,  -8.3f,  -7.6f, -20.8f, -16.9f, -13.6f, -22.0f, -16.0f, -13.2f, -14.7f, -14.9f, -15.8f, -12.4f,   7.6f,   8.5f,   9.0f,  12.0f,  19.3f,  14.5f,   9.6f,  10.5f,  14.3f,  15.6f,  11.9f,  11.5f,  13.6f},
									{ -9.5f,  -9.5f,  -7.5f, -17.3f, -19.0f, -12.5f, -15.2f, -11.7f, -12.9f, -13.2f, -17.1f,  -7.1f, -10.3f,   9.2f,   8.8f,   2.9f,   9.3f,   8.1f,  12.9f,  12.5f,   9.3f,   7.7f,  10.2f,   9.8f,  11.9f,  13.4f},
									{ -7.5f,  -7.8f,  -9.8f, -21.8f, -20.3f, -13.8f, -17.7f, -12.0f, -16.7f, -14.9f, -14.8f,  -8.4f, -11.7f,   5.4f,   9.3f,   5.5f,  11.9f,   9.6f,  13.4f,   8.0f,   8.8f,   4.2f,  14.8f,   5.7f,   8.5f,  12.8f},
									{ -9.1f,  -4.0f,  -9.0f,  -5.5f, -10.6f, -20.7f, -15.4f, -13.1f,  -5.4f,  -3.8f,  -8.5f, -16.7f, -11.6f,   5.1f,   9.0f,   9.4f,  18.8f,   9.3f,  19.5f,   7.4f,  12.9f,  11.1f,  17.1f,   4.5f,  12.6f,  14.4f},
									{ -5.8f,  -7.3f,  -7.4f, -20.5f,  -5.9f,  -5.7f,   7.9f, -13.1f, -13.2f, -14.7f,   1.5f,   1.1f,  -8.6f,   7.6f,   1.9f,   1.8f,   5.6f,  11.5f,  13.4f,   9.6f,   9.5f,   4.7f,  -2.0f,   9.7f,  11.5f,   5.8f},
									{ -6.8f,  -2.1f,  -7.6f,  -4.9f, -16.9f, -12.7f, -18.9f,  -3.7f, -10.9f,  -1.6f,  -9.9f, -15.8f, -10.0f,   6.2f,   6.0f,   9.0f,  12.0f,  16.2f,  10.3f,   0.4f,   4.1f,  14.1f,  12.1f,   8.3f,   4.3f,  13.6f},
									{ -9.2f,  -2.7f,  -4.0f, -10.2f,  -7.8f, -13.6f, -22.0f, -11.9f,  -7.5f,  -1.0f, -14.9f, -13.5f,  -4.4f,   4.6f,   8.5f,   7.1f,   9.2f,  -4.8f,  14.5f,   2.7f,  10.5f,   0.7f,  15.6f,  -1.3f,   8.6f,   9.9f},
									{ -7.7f,  -8.3f,  -4.8f, -20.8f,  -8.4f, -11.2f, -12.4f, -16.0f, -12.5f, -13.1f, -10.4f,  -5.8f, -12.4f,   6.9f,   6.6f,   7.4f,   9.3f,  19.3f,   0.8f,   5.4f,   2.8f,  14.3f,   8.6f,  11.9f,  -0.5f,   4.6f},
									{ -2.9f,  -0.6f,  -2.7f,   5.9f,  -7.7f,   1.9f, -13.1f,   4.6f,  -0.2f,  -3.3f, -10.4f,  -5.6f,  -1.2f,   9.2f,   7.5f,   2.9f,   9.3f,   7.1f,  11.3f,  12.5f,   8.6f,   7.6f,  10.2f,   9.8f,  11.9f,   4.8f},
									{ -4.1f,  -5.7f,  -7.5f, -17.3f,  -5.9f,  -2.3f,   6.7f,  -9.8f, -11.6f, -13.2f,   0.7f,   3.4f,   1.6f,   7.9f,   3.0f,  -0.2f,   9.1f,   6.3f,  12.9f,   9.1f,   9.3f,   7.7f,   1.2f,   6.5f,   9.9f,   9.2f},
									{ -9.0f,   0.3f,  -4.6f,  -3.5f, -19.0f,  -5.7f, -15.2f,  -4.6f, -10.9f,   1.4f, -17.1f,  -7.1f,  -0.8f,  -1.3f,   8.8f,   1.1f,   2.9f,  -4.1f,  12.1f,  -5.5f,   7.5f,  -3.8f,   6.2f,  -5.2f,   3.3f,  13.4f},
									{ -9.5f,  -9.5f,  -3.4f, -11.7f, -14.1f, -12.5f,  -7.3f, -11.7f, -12.9f,  -8.7f, -10.7f,  -6.1f, -10.3f,   2.2f,   1.2f,   0.8f,  -0.7f,   8.1f,   3.7f,   6.5f,   1.5f,   0.0f,  -2.2f,   7.3f,   4.4f,   4.6f},
									{ -6.9f,  -5.1f,  -9.8f, -21.8f, -11.6f,  -2.2f, -17.7f, -12.0f, -16.7f, -14.9f, -12.2f,  -8.4f,   0.8f,   1.9f,   9.3f,   5.5f,  11.9f,   1.1f,   2.7f,   8.0f,   6.4f,   4.2f,  14.8f,  -1.2f,   2.9f,   4.7f},
									{ -5.0f,  -7.8f,  -5.9f,  -7.2f,  -9.8f, -13.8f,  -2.8f,  -9.9f,  -8.5f,  -3.9f,  -6.3f,  -6.0f, -11.7f,   2.6f,   4.9f,   3.9f,   1.6f,   9.6f,  13.4f,   3.6f,   7.5f,   1.8f,  -1.0f,   5.7f,   8.5f,  10.8f},
									{ -3.1f,   1.0f,  -3.4f,  -2.2f, -12.6f,   4.4f,  -5.8f,   1.1f,  -6.4f,  -2.0f,  -9.2f,   0.2f,   4.0f,   5.4f,   9.0f,  -3.0f,   5.4f,   1.4f,  12.2f,   7.4f,   8.8f,   2.4f,   5.6f,   4.4f,   8.4f,  12.4f},
									{ -7.5f,  -0.7f,  -9.0f, -12.7f, -20.3f,   0.3f, -14.2f,  -3.7f, -16.5f,  -5.2f, -14.8f,  -6.7f,   0.4f,  -0.1f,   7.5f,  -0.6f,  -0.4f,  -0.5f,   5.3f,   1.7f,   0.2f,  -1.2f,   6.9f,   0.6f,   1.5f,  12.8f},
									{ -9.1f,  -4.0f,   1.0f,  -5.5f,  -2.3f, -20.7f, -12.7f, -13.1f,  -1.5f,   0.9f,  -5.1f, -16.7f, -11.6f,   0.6f,   2.7f,   7.6f,   3.1f,   9.3f,   1.1f,  -0.3f,   2.1f,   5.4f,   3.7f,   4.5f,  -1.6f,   1.7f},
									{ -7.8f,  -3.4f,  -9.0f,  -3.4f, -10.6f, -19.0f, -12.2f, -11.2f,  -5.4f,  -3.8f,  -4.4f, -15.6f, -11.2f,   5.1f,   7.1f,   7.8f,   4.9f,   3.4f,  19.5f,   7.4f,  12.2f,   0.0f,   4.4f,  -1.6f,  12.6f,  14.4f},
									{ -6.9f,  -0.2f,   1.1f,  -0.9f,  -4.1f,  -9.7f, -12.9f,  -5.3f,  -2.5f,   0.9f,  -8.5f, -11.3f,  -2.8f,   3.9f,   9.0f,   5.0f,  17.9f,   4.8f,   7.9f,   2.6f,  12.9f,   8.9f,  14.0f,   3.7f,   2.4f,   4.0f},
									{ -3.1f,   0.0f,  -0.2f,  -1.9f,  -5.2f,  -1.5f, -15.4f,  -1.7f,  -1.9f,  -0.2f,  -7.2f,  -7.7f,  -1.7f,   1.7f,   7.7f,   9.4f,  18.8f,   3.4f,   0.0f,  -1.5f,   9.4f,  11.1f,  17.1f,  -1.7f,  -1.5f,   2.1f}};
	
	checkTreeUpdate(inputVerts_16_before, inputVerts_16_after, leaf_16_flags, expected_16_6, expected_16_14, expected_16_18, expected_16_26);
	
	//17 leaf nodes, 1 leaf node changed
	std::vector<Vector3f> leaf_17_1_b;
	leaf_17_1_b.push_back(Vector3f(-7.9f,  1.3f,  8.4f));
	leaf_17_1_b.push_back(Vector3f( 3.4f,  4.0f, -7.8f));
	leaf_17_1_b.push_back(Vector3f( 9.3f, -7.5f, -6.9f));

	std::vector<Vector3f> leaf_17_2_b;
	leaf_17_2_b.push_back(Vector3f( 6.4f,  5.4f,  4.8f));
	leaf_17_2_b.push_back(Vector3f( 6.5f, -3.9f, -5.0f));
	leaf_17_2_b.push_back(Vector3f(-5.9f, -5.8f, -5.5f));

	std::vector<Vector3f> leaf_17_3_b;
	leaf_17_3_b.push_back(Vector3f( 5.6f, -1.8f, -4.3f));
	leaf_17_3_b.push_back(Vector3f(-6.7f, -4.7f, -4.1f));
	leaf_17_3_b.push_back(Vector3f( 7.2f,  9.2f,  2.8f));

	std::vector<Vector3f> leaf_17_4_b;
	leaf_17_4_b.push_back(Vector3f( 1.2f, -5.6f, -5.2f));
	leaf_17_4_b.push_back(Vector3f( 7.5f,  9.9f,  0.2f));
	leaf_17_4_b.push_back(Vector3f(-2.7f, -1.4f,  4.6f));

	std::vector<Vector3f> leaf_17_5_b;
	leaf_17_5_b.push_back(Vector3f( 2.5f,  3.5f, -9.4f));
	leaf_17_5_b.push_back(Vector3f(-8.4f,  7.1f,  8.6f));
	leaf_17_5_b.push_back(Vector3f(-5.1f,  6.3f,  3.7f));

	std::vector<Vector3f> leaf_17_6_b;
	leaf_17_6_b.push_back(Vector3f( 3.7f, -5.7f, -3.0f));
	leaf_17_6_b.push_back(Vector3f(-8.8f,  7.9f,  7.4f));
	leaf_17_6_b.push_back(Vector3f( 9.3f,  7.7f, -3.3f));

	std::vector<Vector3f> leaf_17_7_b;
	leaf_17_7_b.push_back(Vector3f(-8.4f,  6.3f, -4.2f));
	leaf_17_7_b.push_back(Vector3f(-2.2f, -0.3f, -5.4f));
	leaf_17_7_b.push_back(Vector3f( 9.1f,  7.0f, -8.8f));

	std::vector<Vector3f> leaf_17_8_b;
	leaf_17_8_b.push_back(Vector3f(-7.7f,  0.5f, -7.4f));
	leaf_17_8_b.push_back(Vector3f( 9.4f, -4.1f, -3.6f));
	leaf_17_8_b.push_back(Vector3f( 8.8f, -4.3f, -2.9f));

	std::vector<Vector3f> leaf_17_9_b;
	leaf_17_9_b.push_back(Vector3f(-7.1f, -7.3f, -5.2f));
	leaf_17_9_b.push_back(Vector3f( 3.2f, -3.5f, -6.2f));
	leaf_17_9_b.push_back(Vector3f( 6.0f, -7.6f,  1.5f));

	std::vector<Vector3f> leaf_17_10_b;
	leaf_17_10_b.push_back(Vector3f( 2.4f, -5.1f,  6.1f));
	leaf_17_10_b.push_back(Vector3f( 3.9f, -4.4f, -5.4f));
	leaf_17_10_b.push_back(Vector3f(-5.2f,  8.7f, -0.3f));

	std::vector<Vector3f> leaf_17_11_b;
	leaf_17_11_b.push_back(Vector3f( 4.5f,  4.7f,  9.9f));
	leaf_17_11_b.push_back(Vector3f(-4.2f, -7.4f,  0.4f));
	leaf_17_11_b.push_back(Vector3f( 6.7f,  3.1f, -1.4f));

	std::vector<Vector3f> leaf_17_12_b;
	leaf_17_12_b.push_back(Vector3f( 7.1f,  5.3f,  5.8f));
	leaf_17_12_b.push_back(Vector3f( 2.1f, -2.5f, -6.6f));
	leaf_17_12_b.push_back(Vector3f(-4.1f, -3.3f, -3.6f));

	std::vector<Vector3f> leaf_17_13_b;
	leaf_17_13_b.push_back(Vector3f(-6.3f, -8.5f,  9.7f));
	leaf_17_13_b.push_back(Vector3f( 7.2f, -5.7f,  1.8f));
	leaf_17_13_b.push_back(Vector3f(-4.7f,  5.6f,  0.1f));

	std::vector<Vector3f> leaf_17_14_b;
	leaf_17_14_b.push_back(Vector3f(-4.9f,  2.4f, -6.9f));
	leaf_17_14_b.push_back(Vector3f( 4.5f,  6.2f, -4.3f));
	leaf_17_14_b.push_back(Vector3f(-1.1f, -0.2f,  4.9f));

	std::vector<Vector3f> leaf_17_15_b;
	leaf_17_15_b.push_back(Vector3f( 7.1f, -4.6f,  0.1f));
	leaf_17_15_b.push_back(Vector3f(-5.1f,  1.8f,  8.5f));
	leaf_17_15_b.push_back(Vector3f(-5.3f, -3.3f, -5.2f));

	std::vector<Vector3f> leaf_17_16_b;
	leaf_17_16_b.push_back(Vector3f(-1.8f, -6.3f,  6.4f));
	leaf_17_16_b.push_back(Vector3f(-2.6f, -7.9f, -0.9f));
	leaf_17_16_b.push_back(Vector3f( 3.0f,  2.8f,  4.5f));

	std::vector<Vector3f> leaf_17_17_b;
	leaf_17_17_b.push_back(Vector3f());
	leaf_17_17_b.push_back(Vector3f());
	leaf_17_17_b.push_back(Vector3f());
	
	std::vector< std::vector<Vector3f> > inputVerts_17_before;
	inputVerts_17_before.push_back(leaf_17_1_b);
	inputVerts_17_before.push_back(leaf_17_2_b);
	inputVerts_17_before.push_back(leaf_17_3_b);
	inputVerts_17_before.push_back(leaf_17_4_b);
	inputVerts_17_before.push_back(leaf_17_5_b);
	inputVerts_17_before.push_back(leaf_17_6_b);
	inputVerts_17_before.push_back(leaf_17_7_b);
	inputVerts_17_before.push_back(leaf_17_8_b);
	inputVerts_17_before.push_back(leaf_17_9_b);
	inputVerts_17_before.push_back(leaf_17_10_b);
	inputVerts_17_before.push_back(leaf_17_11_b);
	inputVerts_17_before.push_back(leaf_17_12_b);
	inputVerts_17_before.push_back(leaf_17_13_b);
	inputVerts_17_before.push_back(leaf_17_14_b);
	inputVerts_17_before.push_back(leaf_17_15_b);
	inputVerts_17_before.push_back(leaf_17_16_b);
	inputVerts_17_before.push_back(leaf_17_17_b);
	
	std::vector<Vector3f> leaf_17_1_a;
	leaf_17_1_a.push_back(Vector3f());
	leaf_17_1_a.push_back(Vector3f());
	leaf_17_1_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_2_a;
	leaf_17_2_a.push_back(Vector3f());
	leaf_17_2_a.push_back(Vector3f());
	leaf_17_2_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_3_a;
	leaf_17_3_a.push_back(Vector3f());
	leaf_17_3_a.push_back(Vector3f());
	leaf_17_3_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_4_a;
	leaf_17_4_a.push_back(Vector3f());
	leaf_17_4_a.push_back(Vector3f());
	leaf_17_4_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_5_a;
	leaf_17_5_a.push_back(Vector3f());
	leaf_17_5_a.push_back(Vector3f());
	leaf_17_5_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_6_a;
	leaf_17_6_a.push_back(Vector3f());
	leaf_17_6_a.push_back(Vector3f());
	leaf_17_6_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_7_a;
	leaf_17_7_a.push_back(Vector3f());
	leaf_17_7_a.push_back(Vector3f());
	leaf_17_7_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_8_a;
	leaf_17_8_a.push_back(Vector3f());
	leaf_17_8_a.push_back(Vector3f());
	leaf_17_8_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_9_a;
	leaf_17_9_a.push_back(Vector3f());
	leaf_17_9_a.push_back(Vector3f());
	leaf_17_9_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_10_a;
	leaf_17_10_a.push_back(Vector3f());
	leaf_17_10_a.push_back(Vector3f());
	leaf_17_10_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_11_a;
	leaf_17_11_a.push_back(Vector3f());
	leaf_17_11_a.push_back(Vector3f());
	leaf_17_11_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_12_a;
	leaf_17_12_a.push_back(Vector3f());
	leaf_17_12_a.push_back(Vector3f());
	leaf_17_12_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_13_a;
	leaf_17_13_a.push_back(Vector3f());
	leaf_17_13_a.push_back(Vector3f());
	leaf_17_13_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_14_a;
	leaf_17_14_a.push_back(Vector3f());
	leaf_17_14_a.push_back(Vector3f());
	leaf_17_14_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_15_a;
	leaf_17_15_a.push_back(Vector3f());
	leaf_17_15_a.push_back(Vector3f());
	leaf_17_15_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_16_a;
	leaf_17_16_a.push_back(Vector3f());
	leaf_17_16_a.push_back(Vector3f());
	leaf_17_16_a.push_back(Vector3f());
	
	std::vector<Vector3f> leaf_17_17_a;
	leaf_17_17_a.push_back(Vector3f( 7.1f, -0.2f, -9.2f));
	leaf_17_17_a.push_back(Vector3f( 7.2f,  4.3f,  4.6f));
	leaf_17_17_a.push_back(Vector3f(-4.5f,  8.4f,  8.5f));
	
	std::vector< std::vector<Vector3f> > inputVerts_17_after;
	inputVerts_17_after.push_back(leaf_17_1_a);
	inputVerts_17_after.push_back(leaf_17_2_a);
	inputVerts_17_after.push_back(leaf_17_3_a);
	inputVerts_17_after.push_back(leaf_17_4_a);
	inputVerts_17_after.push_back(leaf_17_5_a);
	inputVerts_17_after.push_back(leaf_17_6_a);
	inputVerts_17_after.push_back(leaf_17_7_a);
	inputVerts_17_after.push_back(leaf_17_8_a);
	inputVerts_17_after.push_back(leaf_17_9_a);
	inputVerts_17_after.push_back(leaf_17_10_a);
	inputVerts_17_after.push_back(leaf_17_11_a);
	inputVerts_17_after.push_back(leaf_17_12_a);
	inputVerts_17_after.push_back(leaf_17_13_a);
	inputVerts_17_after.push_back(leaf_17_14_a);
	inputVerts_17_after.push_back(leaf_17_15_a);
	inputVerts_17_after.push_back(leaf_17_16_a);
	inputVerts_17_after.push_back(leaf_17_17_a);
	
	std::vector<bool> leaf_17_flags;
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(false);
	leaf_17_flags.push_back(true);
	
	float expected_17_6 [25][6 ] = {{ -8.8f,  -8.5f,  -9.4f,   9.4f,   9.9f,   9.9f},
									{ -8.8f,  -8.5f,  -9.4f,   9.4f,   9.9f,   9.9f},
									{ -4.5f,  -0.2f,  -9.2f,   7.2f,   8.4f,   8.5f},
									{ -7.9f,  -7.5f,  -7.8f,   9.3f,   9.9f,   8.4f},
									{ -8.8f,  -5.7f,  -9.4f,   9.4f,   7.9f,   8.6f},
									{ -7.1f,  -7.6f,  -6.6f,   7.1f,   8.7f,   9.9f},
									{ -6.3f,  -8.5f,  -6.9f,   7.2f,   6.2f,   9.7f},
									{ -4.5f,  -0.2f,  -9.2f,   7.2f,   8.4f,   8.5f},
									{ -7.9f,  -7.5f,  -7.8f,   9.3f,   4.0f,   8.4f},
									{ -5.9f,  -5.8f,  -5.5f,   6.5f,   5.4f,   4.8f},
									{ -6.7f,  -4.7f,  -4.3f,   7.2f,   9.2f,   2.8f},
									{ -2.7f,  -5.6f,  -5.2f,   7.5f,   9.9f,   4.6f},
									{ -8.4f,   3.5f,  -9.4f,   2.5f,   7.1f,   8.6f},
									{ -8.8f,  -5.7f,  -3.3f,   9.3f,   7.9f,   7.4f},
									{ -8.4f,  -0.3f,  -8.8f,   9.1f,   7.0f,  -4.2f},
									{ -7.7f,  -4.3f,  -7.4f,   9.4f,   0.5f,  -2.9f},
									{ -7.1f,  -7.6f,  -6.2f,   6.0f,  -3.5f,   1.5f},
									{ -5.2f,  -5.1f,  -5.4f,   3.9f,   8.7f,   6.1f},
									{ -4.2f,  -7.4f,  -1.4f,   6.7f,   4.7f,   9.9f},
									{ -4.1f,  -3.3f,  -6.6f,   7.1f,   5.3f,   5.8f},
									{ -6.3f,  -8.5f,   0.1f,   7.2f,   5.6f,   9.7f},
									{ -4.9f,  -0.2f,  -6.9f,   4.5f,   6.2f,   4.9f},
									{ -5.3f,  -4.6f,  -5.2f,   7.1f,   1.8f,   8.5f},
									{ -2.6f,  -7.9f,  -0.9f,   3.0f,   2.8f,   6.4f},
									{ -4.5f,  -0.2f,  -9.2f,   7.2f,   8.4f,   8.5f}};
	float expected_17_14[25][14] = {{ -8.8f,  -8.5f,  -9.4f, -19.6f, -18.9f, -24.5f, -24.1f,   9.4f,   9.9f,   9.9f,  19.2f,  15.1f,  24.9f,  23.7f},
									{ -8.8f,  -8.5f,  -9.4f, -19.6f, -18.9f, -24.5f, -24.1f,   9.4f,   9.9f,   9.9f,  19.2f,  15.1f,  24.9f,  23.7f},
									{ -4.5f,  -0.2f,  -9.2f,  -2.3f,  -4.4f,  -4.6f, -21.4f,   7.2f,   8.4f,   8.5f,  16.1f,   7.5f,  16.1f,  16.5f},
									{ -7.9f,  -7.5f,  -7.8f, -17.2f,  -8.4f, -15.0f, -17.6f,   9.3f,   9.9f,   8.4f,  19.2f,   9.9f,  17.2f,  23.7f},
									{ -8.8f,  -5.7f,  -9.4f, -14.6f, -18.9f,  -9.9f, -24.1f,   9.4f,   7.9f,   8.6f,  13.7f,  10.2f,  24.9f,  17.1f},
									{ -7.1f,  -7.6f,  -6.6f, -19.6f, -14.2f, -12.0f, -13.6f,   7.1f,   8.7f,   9.9f,  19.1f,  15.1f,  11.2f,  13.7f},
									{ -6.3f,  -8.5f,  -6.9f, -13.8f, -14.2f, -24.5f, -15.4f,   7.2f,   6.2f,   9.7f,  10.3f,  14.7f,  15.0f,  11.6f},
									{ -4.5f,  -0.2f,  -9.2f,  -2.3f,  -4.4f,  -4.6f, -21.4f,   7.2f,   8.4f,   8.5f,  16.1f,   7.5f,  16.1f,  16.5f},
									{ -7.9f,  -7.5f,  -7.8f,  -5.1f,  -8.4f, -15.0f, -17.6f,   9.3f,   4.0f,   8.4f,   1.8f,   9.9f,  15.2f,  23.7f},
									{ -5.9f,  -5.8f,  -5.5f, -17.2f,  -5.6f,  -6.2f,  -3.8f,   6.5f,   5.4f,   4.8f,  16.6f,   5.8f,   7.6f,  15.4f},
									{ -6.7f,  -4.7f,  -4.3f, -15.5f,  -6.1f,  -7.3f,  -4.8f,   7.2f,   9.2f,   2.8f,  19.2f,   3.1f,  13.6f,  11.7f},
									{ -2.7f,  -5.6f,  -5.2f,  -9.6f,  -2.2f,  -8.7f,  -5.9f,   7.5f,   9.9f,   4.6f,  17.6f,   3.3f,  17.2f,  12.0f},
									{ -8.4f,   3.5f,  -9.4f,  -3.4f, -10.4f,  -9.9f, -24.1f,   2.5f,   7.1f,   8.6f,   7.3f,  -6.9f,  15.4f,   8.4f},
									{ -8.8f,  -5.7f,  -3.3f,  -5.0f,  -9.3f,  -8.3f, -24.1f,   9.3f,   7.9f,   7.4f,  13.7f,   6.4f,  20.3f,  12.4f},
									{ -8.4f,  -0.3f,  -8.8f,  -7.9f, -18.9f,   2.1f, -10.5f,   9.1f,   7.0f,  -4.2f,   7.3f,  -6.7f,  24.9f,  10.9f},
									{ -7.7f,  -4.3f,  -7.4f, -14.6f, -15.6f,   0.2f,  -0.8f,   9.4f,   0.5f,  -2.9f,   1.7f,  10.2f,   8.9f,  17.1f},
									{ -7.1f,  -7.6f,  -6.2f, -19.6f,  -5.0f,  -9.2f,   5.4f,   6.0f,  -3.5f,   1.5f,  -0.1f,  15.1f,   5.9f,  12.9f},
									{ -5.2f,  -5.1f,  -5.4f,  -5.9f, -14.2f,  -8.8f, -13.6f,   3.9f,   8.7f,   6.1f,   3.4f,  13.6f,   4.9f,  13.7f},
									{ -4.2f,  -7.4f,  -1.4f, -11.2f,   2.2f, -12.0f, -10.1f,   6.7f,   4.7f,   9.9f,  19.1f,   9.7f,  11.2f,   5.0f},
									{ -4.1f,  -3.3f,  -6.6f, -11.0f,  -4.4f,  -3.8f,  -4.0f,   7.1f,   5.3f,   5.8f,  18.2f,   7.6f,   6.6f,  11.2f},
									{ -6.3f,  -8.5f,   0.1f,  -5.1f, -10.2f, -24.5f, -10.4f,   7.2f,   5.6f,   9.7f,   3.3f,  14.7f,   0.8f,  11.1f},
									{ -4.9f,  -0.2f,  -6.9f,  -9.4f, -14.2f,  -6.2f,  -5.8f,   4.5f,   6.2f,   4.9f,   6.4f,   4.0f,  15.0f,   2.6f},
									{ -5.3f,  -4.6f,  -5.2f, -13.8f,  -7.2f, -11.8f, -15.4f,   7.1f,   1.8f,   8.5f,   5.2f,  11.8f,   2.4f,  11.6f},
									{ -2.6f,  -7.9f,  -0.9f, -11.4f,   4.4f, -14.5f,  -4.3f,   3.0f,   2.8f,   6.4f,  10.3f,  10.9f,   1.3f,   6.2f},
									{ -4.5f,  -0.2f,  -9.2f,  -2.3f,  -4.4f,  -4.6f, -21.4f,   7.2f,   8.4f,   8.5f,  16.1f,   7.5f,  16.1f,  16.5f}};
	float expected_17_18[25][18] = {{ -8.8f,  -8.5f,  -9.4f, -14.8f, -15.1f, -14.4f, -16.7f, -17.0f, -18.2f,   9.4f,   9.9f,   9.9f,  17.4f,  14.4f,  16.9f,  16.8f,  17.9f,  15.8f},
									{ -8.8f,  -8.5f,  -9.4f, -14.8f, -15.1f, -14.4f, -16.7f, -17.0f, -18.2f,   9.4f,   9.9f,   9.9f,  17.4f,  14.4f,  15.7f,  16.8f,  17.9f,  15.8f},
									{ -4.5f,  -0.2f,  -9.2f,   3.9f,  -2.1f,  -9.4f, -12.9f, -13.0f,  -0.3f,   7.2f,   8.4f,   8.5f,  11.5f,  11.8f,  16.9f,   7.3f,  16.3f,   9.0f},
									{ -7.9f,  -7.5f,  -7.8f, -11.7f, -11.4f, -14.4f,  -9.2f, -16.3f,  -7.1f,   9.3f,   9.9f,   8.4f,  17.4f,  11.2f,  12.0f,  16.8f,  16.2f,  11.8f},
									{ -8.8f,  -5.7f,  -9.4f,  -7.2f, -15.1f,  -8.7f, -16.7f, -17.0f,  -2.7f,   9.4f,   7.9f,   8.6f,  17.0f,   6.0f,  15.7f,  13.5f,  17.9f,  15.8f},
									{ -7.1f,  -7.6f,  -6.6f, -14.4f, -12.3f, -12.5f, -13.9f,  -5.4f, -11.2f,   7.1f,   8.7f,   9.9f,  12.4f,  14.4f,  14.6f,  13.6f,   9.4f,   9.0f},
									{ -6.3f,  -8.5f,  -6.9f, -14.8f, -11.8f,  -8.8f, -10.3f, -16.0f, -18.2f,   7.2f,   6.2f,   9.7f,  10.7f,   9.0f,  10.3f,  12.9f,   8.8f,  10.5f},
									{ -4.5f,  -0.2f,  -9.2f,   3.9f,  -2.1f,  -9.4f, -12.9f, -13.0f,  -0.3f,   7.2f,   8.4f,   8.5f,  11.5f,  11.8f,  16.9f,   7.3f,  16.3f,   9.0f},
									{ -7.9f,  -7.5f,  -7.8f,  -6.6f,  -4.4f, -14.4f,  -9.2f, -16.3f,  -7.1f,   9.3f,   4.0f,   8.4f,   7.4f,   2.4f,   9.7f,  16.8f,  16.2f,  11.8f},
									{ -5.9f,  -5.8f,  -5.5f, -11.7f, -11.4f, -11.3f,  -0.1f,  -0.4f,  -0.3f,   6.5f,   5.4f,   4.8f,  11.8f,  11.2f,  10.2f,  10.4f,  11.5f,   1.1f},
									{ -6.7f,  -4.7f,  -4.3f, -11.4f, -10.8f,  -8.8f,  -2.0f,  -2.6f,  -0.6f,   7.2f,   9.2f,   2.8f,  16.4f,  10.0f,  12.0f,   7.4f,   9.9f,   6.4f},
									{ -2.7f,  -5.6f,  -5.2f,  -4.4f,  -4.0f, -10.8f,  -2.4f,  -7.3f,  -6.0f,   7.5f,   9.9f,   4.6f,  17.4f,   7.7f,  10.1f,   6.8f,   7.3f,   9.7f},
									{ -8.4f,   3.5f,  -9.4f,  -1.3f,  -6.9f,  -5.9f, -15.5f, -17.0f,  -1.5f,   2.5f,   7.1f,   8.6f,   6.0f,   0.2f,  15.7f,  -1.0f,  11.9f,  12.9f},
									{ -8.8f,  -5.7f,  -3.3f,  -2.0f,  -1.4f,  -8.7f, -16.7f, -16.2f,  -2.7f,   9.3f,   7.9f,   7.4f,  17.0f,   6.0f,  15.3f,   9.4f,  12.6f,  11.0f},
									{ -8.4f,  -0.3f,  -8.8f,  -2.5f, -12.6f,  -5.7f, -14.7f,  -4.2f,   5.1f,   9.1f,   7.0f,  -4.2f,  16.1f,   0.3f,   2.1f,   2.1f,  17.9f,  15.8f},
									{ -7.7f,  -4.3f,  -7.4f,  -7.2f, -15.1f,  -7.7f,  -8.2f,  -0.3f,  -1.4f,   9.4f,   0.5f,  -2.9f,   5.3f,   5.9f,  -6.9f,  13.5f,  13.0f,   7.9f},
									{ -7.1f,  -7.6f,  -6.2f, -14.4f, -12.3f, -12.5f,   0.2f,  -1.9f,  -9.1f,   6.0f,  -3.5f,   1.5f,  -0.3f,   7.5f,  -6.1f,  13.6f,   9.4f,   2.7f},
									{ -5.2f,  -5.1f,  -5.4f,  -2.7f,  -5.5f,  -9.8f, -13.9f,  -4.9f, -11.2f,   3.9f,   8.7f,   6.1f,   3.5f,   8.5f,   8.4f,   8.3f,   9.3f,   9.0f},
									{ -4.2f,  -7.4f,  -1.4f, -11.6f,  -3.8f,  -7.0f,  -0.2f,  -5.4f,  -7.8f,   6.7f,   4.7f,   9.9f,   9.8f,  14.4f,  14.6f,   3.6f,   8.1f,   4.5f},
									{ -4.1f,  -3.3f,  -6.6f,  -7.4f,  -7.7f,  -9.1f,  -0.8f,  -0.5f,  -0.5f,   7.1f,   5.3f,   5.8f,  12.4f,  12.9f,  11.1f,   4.6f,   8.7f,   4.1f},
									{ -6.3f,  -8.5f,   0.1f, -14.8f,  -4.6f,  -3.9f, -10.3f, -16.0f, -18.2f,   7.2f,   5.6f,   9.7f,   1.5f,   9.0f,   5.7f,  12.9f,   5.4f,   5.5f},
									{ -4.9f,  -0.2f,  -6.9f,  -2.5f, -11.8f,  -4.5f,  -7.3f,  -6.0f,  -5.1f,   4.5f,   6.2f,   4.9f,  10.7f,   3.8f,   4.7f,  -0.9f,   8.8f,  10.5f},
									{ -5.3f,  -4.6f,  -5.2f,  -8.6f, -10.5f,  -8.5f,  -6.9f, -13.6f,  -6.7f,   7.1f,   1.8f,   8.5f,   2.5f,   7.2f,  10.3f,  11.7f,   7.0f,   1.9f},
									{ -2.6f,  -7.9f,  -0.9f, -10.5f,  -3.5f,  -8.8f,   0.2f,  -8.2f, -12.7f,   3.0f,   2.8f,   6.4f,   5.8f,   7.5f,   7.3f,   5.3f,  -1.5f,  -1.7f},
									{ -4.5f,  -0.2f,  -9.2f,   3.9f,  -2.1f,  -9.4f, -12.9f, -13.0f,  -0.3f,   7.2f,   8.4f,   8.5f,  11.5f,  11.8f,  16.9f,   7.3f,  16.3f,   9.0f}};
	float expected_17_26[25][26] = {{ -8.8f,  -8.5f,  -9.4f, -19.6f, -18.9f, -24.5f, -24.1f, -14.8f, -15.1f, -14.4f, -16.7f, -17.0f, -18.2f,   9.4f,   9.9f,   9.9f,  19.2f,  15.1f,  24.9f,  23.7f,  17.4f,  14.4f,  16.9f,  16.8f,  17.9f,  15.8f},
									{ -8.8f,  -8.5f,  -9.4f, -19.6f, -18.9f, -24.5f, -24.1f, -14.8f, -15.1f, -14.4f, -16.7f, -17.0f, -18.2f,   9.4f,   9.9f,   9.9f,  19.2f,  15.1f,  24.9f,  23.7f,  17.4f,  14.4f,  15.7f,  16.8f,  17.9f,  15.8f},
									{ -4.5f,  -0.2f,  -9.2f,  -2.3f,  -4.4f,  -4.6f, -21.4f,   3.9f,  -2.1f,  -9.4f, -12.9f, -13.0f,  -0.3f,   7.2f,   8.4f,   8.5f,  16.1f,   7.5f,  16.1f,  16.5f,  11.5f,  11.8f,  16.9f,   7.3f,  16.3f,   9.0f},
									{ -7.9f,  -7.5f,  -7.8f, -17.2f,  -8.4f, -15.0f, -17.6f, -11.7f, -11.4f, -14.4f,  -9.2f, -16.3f,  -7.1f,   9.3f,   9.9f,   8.4f,  19.2f,   9.9f,  17.2f,  23.7f,  17.4f,  11.2f,  12.0f,  16.8f,  16.2f,  11.8f},
									{ -8.8f,  -5.7f,  -9.4f, -14.6f, -18.9f,  -9.9f, -24.1f,  -7.2f, -15.1f,  -8.7f, -16.7f, -17.0f,  -2.7f,   9.4f,   7.9f,   8.6f,  13.7f,  10.2f,  24.9f,  17.1f,  17.0f,   6.0f,  15.7f,  13.5f,  17.9f,  15.8f},
									{ -7.1f,  -7.6f,  -6.6f, -19.6f, -14.2f, -12.0f, -13.6f, -14.4f, -12.3f, -12.5f, -13.9f,  -5.4f, -11.2f,   7.1f,   8.7f,   9.9f,  19.1f,  15.1f,  11.2f,  13.7f,  12.4f,  14.4f,  14.6f,  13.6f,   9.4f,   9.0f},
									{ -6.3f,  -8.5f,  -6.9f, -13.8f, -14.2f, -24.5f, -15.4f, -14.8f, -11.8f,  -8.8f, -10.3f, -16.0f, -18.2f,   7.2f,   6.2f,   9.7f,  10.3f,  14.7f,  15.0f,  11.6f,  10.7f,   9.0f,  10.3f,  12.9f,   8.8f,  10.5f},
									{ -4.5f,  -0.2f,  -9.2f,  -2.3f,  -4.4f,  -4.6f, -21.4f,   3.9f,  -2.1f,  -9.4f, -12.9f, -13.0f,  -0.3f,   7.2f,   8.4f,   8.5f,  16.1f,   7.5f,  16.1f,  16.5f,  11.5f,  11.8f,  16.9f,   7.3f,  16.3f,   9.0f},
									{ -7.9f,  -7.5f,  -7.8f,  -5.1f,  -8.4f, -15.0f, -17.6f,  -6.6f,  -4.4f, -14.4f,  -9.2f, -16.3f,  -7.1f,   9.3f,   4.0f,   8.4f,   1.8f,   9.9f,  15.2f,  23.7f,   7.4f,   2.4f,   9.7f,  16.8f,  16.2f,  11.8f},
									{ -5.9f,  -5.8f,  -5.5f, -17.2f,  -5.6f,  -6.2f,  -3.8f, -11.7f, -11.4f, -11.3f,  -0.1f,  -0.4f,  -0.3f,   6.5f,   5.4f,   4.8f,  16.6f,   5.8f,   7.6f,  15.4f,  11.8f,  11.2f,  10.2f,  10.4f,  11.5f,   1.1f},
									{ -6.7f,  -4.7f,  -4.3f, -15.5f,  -6.1f,  -7.3f,  -4.8f, -11.4f, -10.8f,  -8.8f,  -2.0f,  -2.6f,  -0.6f,   7.2f,   9.2f,   2.8f,  19.2f,   3.1f,  13.6f,  11.7f,  16.4f,  10.0f,  12.0f,   7.4f,   9.9f,   6.4f},
									{ -2.7f,  -5.6f,  -5.2f,  -9.6f,  -2.2f,  -8.7f,  -5.9f,  -4.4f,  -4.0f, -10.8f,  -2.4f,  -7.3f,  -6.0f,   7.5f,   9.9f,   4.6f,  17.6f,   3.3f,  17.2f,  12.0f,  17.4f,   7.7f,  10.1f,   6.8f,   7.3f,   9.7f},
									{ -8.4f,   3.5f,  -9.4f,  -3.4f, -10.4f,  -9.9f, -24.1f,  -1.3f,  -6.9f,  -5.9f, -15.5f, -17.0f,  -1.5f,   2.5f,   7.1f,   8.6f,   7.3f,  -6.9f,  15.4f,   8.4f,   6.0f,   0.2f,  15.7f,  -1.0f,  11.9f,  12.9f},
									{ -8.8f,  -5.7f,  -3.3f,  -5.0f,  -9.3f,  -8.3f, -24.1f,  -2.0f,  -1.4f,  -8.7f, -16.7f, -16.2f,  -2.7f,   9.3f,   7.9f,   7.4f,  13.7f,   6.4f,  20.3f,  12.4f,  17.0f,   6.0f,  15.3f,   9.4f,  12.6f,  11.0f},
									{ -8.4f,  -0.3f,  -8.8f,  -7.9f, -18.9f,   2.1f, -10.5f,  -2.5f, -12.6f,  -5.7f, -14.7f,  -4.2f,   5.1f,   9.1f,   7.0f,  -4.2f,   7.3f,  -6.7f,  24.9f,  10.9f,  16.1f,   0.3f,   2.1f,   2.1f,  17.9f,  15.8f},
									{ -7.7f,  -4.3f,  -7.4f, -14.6f, -15.6f,   0.2f,  -0.8f,  -7.2f, -15.1f,  -7.7f,  -8.2f,  -0.3f,  -1.4f,   9.4f,   0.5f,  -2.9f,   1.7f,  10.2f,   8.9f,  17.1f,   5.3f,   5.9f,  -6.9f,  13.5f,  13.0f,   7.9f},
									{ -7.1f,  -7.6f,  -6.2f, -19.6f,  -5.0f,  -9.2f,   5.4f, -14.4f, -12.3f, -12.5f,   0.2f,  -1.9f,  -9.1f,   6.0f,  -3.5f,   1.5f,  -0.1f,  15.1f,   5.9f,  12.9f,  -0.3f,   7.5f,  -6.1f,  13.6f,   9.4f,   2.7f},
									{ -5.2f,  -5.1f,  -5.4f,  -5.9f, -14.2f,  -8.8f, -13.6f,  -2.7f,  -5.5f,  -9.8f, -13.9f,  -4.9f, -11.2f,   3.9f,   8.7f,   6.1f,   3.4f,  13.6f,   4.9f,  13.7f,   3.5f,   8.5f,   8.4f,   8.3f,   9.3f,   9.0f},
									{ -4.2f,  -7.4f,  -1.4f, -11.2f,   2.2f, -12.0f, -10.1f, -11.6f,  -3.8f,  -7.0f,  -0.2f,  -5.4f,  -7.8f,   6.7f,   4.7f,   9.9f,  19.1f,   9.7f,  11.2f,   5.0f,   9.8f,  14.4f,  14.6f,   3.6f,   8.1f,   4.5f},
									{ -4.1f,  -3.3f,  -6.6f, -11.0f,  -4.4f,  -3.8f,  -4.0f,  -7.4f,  -7.7f,  -9.1f,  -0.8f,  -0.5f,  -0.5f,   7.1f,   5.3f,   5.8f,  18.2f,   7.6f,   6.6f,  11.2f,  12.4f,  12.9f,  11.1f,   4.6f,   8.7f,   4.1f},
									{ -6.3f,  -8.5f,   0.1f,  -5.1f, -10.2f, -24.5f, -10.4f, -14.8f,  -4.6f,  -3.9f, -10.3f, -16.0f, -18.2f,   7.2f,   5.6f,   9.7f,   3.3f,  14.7f,   0.8f,  11.1f,   1.5f,   9.0f,   5.7f,  12.9f,   5.4f,   5.5f},
									{ -4.9f,  -0.2f,  -6.9f,  -9.4f, -14.2f,  -6.2f,  -5.8f,  -2.5f, -11.8f,  -4.5f,  -7.3f,  -6.0f,  -5.1f,   4.5f,   6.2f,   4.9f,   6.4f,   4.0f,  15.0f,   2.6f,  10.7f,   3.8f,   4.7f,  -0.9f,   8.8f,  10.5f},
									{ -5.3f,  -4.6f,  -5.2f, -13.8f,  -7.2f, -11.8f, -15.4f,  -8.6f, -10.5f,  -8.5f,  -6.9f, -13.6f,  -6.7f,   7.1f,   1.8f,   8.5f,   5.2f,  11.8f,   2.4f,  11.6f,   2.5f,   7.2f,  10.3f,  11.7f,   7.0f,   1.9f},
									{ -2.6f,  -7.9f,  -0.9f, -11.4f,   4.4f, -14.5f,  -4.3f, -10.5f,  -3.5f,  -8.8f,   0.2f,  -8.2f, -12.7f,   3.0f,   2.8f,   6.4f,  10.3f,  10.9f,   1.3f,   6.2f,   5.8f,   7.5f,   7.3f,   5.3f,  -1.5f,  -1.7f},
									{ -4.5f,  -0.2f,  -9.2f,  -2.3f,  -4.4f,  -4.6f, -21.4f,   3.9f,  -2.1f,  -9.4f, -12.9f, -13.0f,  -0.3f,   7.2f,   8.4f,   8.5f,  16.1f,   7.5f,  16.1f,  16.5f,  11.5f,  11.8f,  16.9f,   7.3f,  16.3f,   9.0f}};
	
	checkTreeUpdate(inputVerts_17_before, inputVerts_17_after, leaf_17_flags, expected_17_6, expected_17_14, expected_17_18, expected_17_26);
}

void TreeTest::checkTreeUpdate(std::vector< std::vector<Vector3f> >& verts,
							   std::vector< std::vector<Vector3f> >& updatedVerts,
							   std::vector<bool>& flags,
							   float expected_6 [][6],
							   float expected_14[][14],
							   float expected_18[][18],
							   float expected_26[][26])
{
	//Generate list of leaf nodes with KDOP from vertices
	std::list<Node*> leafNodes_6;
	std::list<Node*> leafNodes_14;
	std::list<Node*> leafNodes_18;
	std::list<Node*> leafNodes_26;
	
	//List of all KDOPs for updating later
	std::vector<KDOP*> kdop_6;
	std::vector<KDOP*> kdop_14;
	std::vector<KDOP*> kdop_18;
	std::vector<KDOP*> kdop_26;
	
	for(int i = 0; i < verts.size(); i++)
	{
		KDOP* k6  = new KDOP(verts[i], 6 );
		KDOP* k14 = new KDOP(verts[i], 14);
		KDOP* k18 = new KDOP(verts[i], 18);
		KDOP* k26 = new KDOP(verts[i], 26);
		
		kdop_6.push_back (k6);
		kdop_14.push_back(k14);
		kdop_18.push_back(k18);
		kdop_26.push_back(k26);
		
		leafNodes_6.push_back (new Node(k6));
		leafNodes_14.push_back(new Node(k14));
		leafNodes_18.push_back(new Node(k18));
		leafNodes_26.push_back(new Node(k26));
	}
	
	//Build tree
	Node* root_6  = Node::buildTree(leafNodes_6);
	Node* root_14 = Node::buildTree(leafNodes_14);
	Node* root_18 = Node::buildTree(leafNodes_18);
	Node* root_26 = Node::buildTree(leafNodes_26);
	
	
	//Update KDOPs with new vertices
	for(int i = 0; i < updatedVerts.size(); i++)
	{
		//Boolean array of leaf KDOP changes
		if(flags[i])
		{
			kdop_6 [i]->update(updatedVerts[i]);
			kdop_14[i]->update(updatedVerts[i]);
			kdop_18[i]->update(updatedVerts[i]);
			kdop_26[i]->update(updatedVerts[i]);
		}
	}
	
	//Propagate KDOP changes up the tree
	Node::updateTree(root_6 );
	Node::updateTree(root_14);
	Node::updateTree(root_18);
	Node::updateTree(root_26);
	
	//Get list of all tree nodes
	std::list<Node*> actual_6;
	std::list<Node*> actual_14;
	std::list<Node*> actual_18;
	std::list<Node*> actual_26;
	
	Node::breadthWalk(root_6,  actual_6);
	Node::breadthWalk(root_14, actual_14);
	Node::breadthWalk(root_18, actual_18);
	Node::breadthWalk(root_26, actual_26);
	
	//Check list of tree nodes' distances against expected output
	float delta =  0.000005f;
	int count_6  = 0;
	int count_14 = 0;
	int count_18 = 0;
	int count_26 = 0;
	
	while(!actual_6.empty())
	{
		const float* distances = actual_6.front()->getKDOP()->getDistances();
		
		for(int i = 0; i < actual_6.front()->getKDOP()->K; i++)
			CPPUNIT_ASSERT_DOUBLES_EQUAL(expected_6[count_6][i], distances[i], delta);
		
		count_6++;
		actual_6.pop_front();
	}
	
	while(!actual_14.empty())
	{
		const float* distances = actual_14.front()->getKDOP()->getDistances();
		
		for(int i = 0; i < actual_14.front()->getKDOP()->K; i++)
			CPPUNIT_ASSERT_DOUBLES_EQUAL(expected_14[count_14][i], distances[i], delta);
		
		count_14++;
		actual_14.pop_front();
	}
	
	while(!actual_18.empty())
	{
		const float* distances = actual_18.front()->getKDOP()->getDistances();
		
		for(int i = 0; i < actual_18.front()->getKDOP()->K; i++)
			CPPUNIT_ASSERT_DOUBLES_EQUAL(expected_18[count_18][i], distances[i], delta);
		
		count_18++;
		actual_18.pop_front();
	}
	
	while(!actual_26.empty())
	{
		const float* distances = actual_26.front()->getKDOP()->getDistances();
		
		for(int i = 0; i < actual_26.front()->getKDOP()->K; i++)
			CPPUNIT_ASSERT_DOUBLES_EQUAL(expected_26[count_26][i], distances[i], delta);
		
		count_26++;
		actual_26.pop_front();
	}
	
	//Release memory
	delete root_6;
	delete root_14;
	delete root_18;
	delete root_26;
}

void TreeTest::testCollisions()
{
	//1 leaf node, 1 collision
	std::vector<Vector3f> leaf_1_1_a;
	leaf_1_1_a.push_back(Vector3f( 1.400000f, -1.100000f,  0.300000f));
	leaf_1_1_a.push_back(Vector3f( 0.800000f, -0.900000f, -1.200000f));
	leaf_1_1_a.push_back(Vector3f( 0.200000f, -0.400000f, -1.600000f));
	
	std::vector< std::vector<Vector3f> > inputVerts_1_a;
	inputVerts_1_a.push_back(leaf_1_1_a);
	
	std::vector<Vector3f> leaf_1_1_b;
	leaf_1_1_b.push_back(Vector3f( 1.700000f, -1.600000f, -0.700000f));
	leaf_1_1_b.push_back(Vector3f( 0.500000f, -0.397149f, -0.600000f));
	leaf_1_1_b.push_back(Vector3f( 0.000000f, -0.097149f, -0.600000f));
	
	std::vector< std::vector<Vector3f> > inputVerts_1_b;
	inputVerts_1_b.push_back(leaf_1_1_b);
	
	std::vector<int> pairs_1;
	pairs_1.push_back(0);
	pairs_1.push_back(0);
	
	checkTreeCollisions(inputVerts_1_a, inputVerts_1_b, pairs_1);
	
	//4 leaf nodes, 3 collisions
	std::vector<Vector3f> leaf_4_1_a;
	leaf_4_1_a.push_back(Vector3f( 1.574532f, -0.763787f,  1.294852f));
	leaf_4_1_a.push_back(Vector3f( 0.231932f, -1.310113f,  2.928904f));
	leaf_4_1_a.push_back(Vector3f( 1.390009f, -0.376967f, -0.014745f));
	
	std::vector<Vector3f> leaf_4_2_a;
	leaf_4_2_a.push_back(Vector3f(-3.831625f,  0.115817f,  0.704995f));
	leaf_4_2_a.push_back(Vector3f(-3.966924f,  2.131703f, -0.125107f));
	leaf_4_2_a.push_back(Vector3f(-2.849538f, -0.850340f,  0.732703f));
	
	std::vector<Vector3f> leaf_4_3_a;
	leaf_4_3_a.push_back(Vector3f(-2.559088f,  1.061652f, -2.002430f));
	leaf_4_3_a.push_back(Vector3f(-3.336901f, -0.694021f, -3.043482f));
	leaf_4_3_a.push_back(Vector3f(-1.556441f,  1.268728f, -1.080182f));
	
	std::vector<Vector3f> leaf_4_4_a;
	leaf_4_4_a.push_back(Vector3f( 0.265566f,  0.123395f, -3.665494f));
	leaf_4_4_a.push_back(Vector3f(-1.126831f, -0.397365f, -2.065115f));
	leaf_4_4_a.push_back(Vector3f( 0.122258f,  0.476261f, -4.989757f));
	
	std::vector< std::vector<Vector3f> > inputVerts_4_a;
	inputVerts_4_a.push_back(leaf_4_1_a);
	inputVerts_4_a.push_back(leaf_4_2_a);
	inputVerts_4_a.push_back(leaf_4_3_a);
	inputVerts_4_a.push_back(leaf_4_4_a);
	
	std::vector<Vector3f> leaf_4_1_b;
	leaf_4_1_b.push_back(Vector3f( 1.031942f, -1.519253f,  2.649092f));
	leaf_4_1_b.push_back(Vector3f(-0.880038f, -0.613848f,  2.105268f));
	leaf_4_1_b.push_back(Vector3f( 2.350224f, -1.144466f,  2.506298f));
	
	std::vector<Vector3f> leaf_4_2_b;
	leaf_4_2_b.push_back(Vector3f(-2.077391f,  0.650538f, -2.240488f));
	leaf_4_2_b.push_back(Vector3f(-3.785367f, -0.120043f, -1.117904f));
	leaf_4_2_b.push_back(Vector3f(-1.923243f,  1.234150f, -3.479180f));
	
	std::vector<Vector3f> leaf_4_3_b;
	leaf_4_3_b.push_back(Vector3f( 1.566814f, -0.257903f,  0.899254f));
	leaf_4_3_b.push_back(Vector3f(-0.094342f,  0.055686f,  0.932030f));
	leaf_4_3_b.push_back(Vector3f( 2.159272f, -1.115021f,  0.671123f));
	
	std::vector<Vector3f> leaf_4_4_b;
	leaf_4_4_b.push_back(Vector3f( 2.351233f, -0.571589f, -2.761934f));
	leaf_4_4_b.push_back(Vector3f( 0.958836f, -1.092349f, -1.161555f));
	leaf_4_4_b.push_back(Vector3f( 2.207925f, -0.218723f, -4.086196f));
	
	std::vector< std::vector<Vector3f> > inputVerts_4_b;
	inputVerts_4_b.push_back(leaf_4_1_b);
	inputVerts_4_b.push_back(leaf_4_2_b);
	inputVerts_4_b.push_back(leaf_4_3_b);
	inputVerts_4_b.push_back(leaf_4_4_b);
	
	std::vector<int> pairs_4;
	pairs_4.push_back(0);
	pairs_4.push_back(0);
	pairs_4.push_back(0);
	pairs_4.push_back(2);
	pairs_4.push_back(2);
	pairs_4.push_back(1);
	
	checkTreeCollisions(inputVerts_4_a, inputVerts_4_b, pairs_4);
	
	//5 leaf nodes, 2 collisions
	std::vector<Vector3f> leaf_5_1_a;
	leaf_5_1_a.push_back(Vector3f(-1.772131f, -2.963633f, -4.047726f));
	leaf_5_1_a.push_back(Vector3f(-2.288299f, -0.407529f, -3.977117f));
	leaf_5_1_a.push_back(Vector3f(-3.502098f, -1.227650f, -4.522547f));
	
	std::vector<Vector3f> leaf_5_2_a;
	leaf_5_2_a.push_back(Vector3f( 0.256196f,  0.070743f,  0.463481f));
	leaf_5_2_a.push_back(Vector3f( 1.483595f,  1.453295f,  2.374323f));
	leaf_5_2_a.push_back(Vector3f(-0.095249f,  1.428745f,  2.586246f));
	
	std::vector<Vector3f> leaf_5_3_a;
	leaf_5_3_a.push_back(Vector3f( 0.891515f, -3.062978f,  0.401789f));
	leaf_5_3_a.push_back(Vector3f( 0.781265f, -1.753007f,  0.762489f));
	leaf_5_3_a.push_back(Vector3f(-0.081961f, -2.465397f, -0.280587f));
	
	std::vector<Vector3f> leaf_5_4_a;
	leaf_5_4_a.push_back(Vector3f(-0.250608f, -1.574968f, -1.528848f));
	leaf_5_4_a.push_back(Vector3f(-0.346470f, -0.635551f, -1.368668f));
	leaf_5_4_a.push_back(Vector3f(-1.125735f, -2.096814f, -2.738292f));
	
	std::vector<Vector3f> leaf_5_5_a;
	leaf_5_5_a.push_back(Vector3f(-1.432238f,  0.813846f,  0.020190f));
	leaf_5_5_a.push_back(Vector3f(-1.876911f,  1.458416f, -0.495737f));
	leaf_5_5_a.push_back(Vector3f(-2.801894f,  0.608256f, -1.709585f));
	
	std::vector< std::vector<Vector3f> > inputVerts_5_a;
	inputVerts_5_a.push_back(leaf_5_1_a);
	inputVerts_5_a.push_back(leaf_5_2_a);
	inputVerts_5_a.push_back(leaf_5_3_a);
	inputVerts_5_a.push_back(leaf_5_4_a);
	inputVerts_5_a.push_back(leaf_5_5_a);
	
	std::vector<Vector3f> leaf_5_1_b;
	leaf_5_1_b.push_back(Vector3f( 4.711471f, -4.450969f, -1.470511f));
	leaf_5_1_b.push_back(Vector3f( 4.943280f, -2.443605f, -3.120310f));
	leaf_5_1_b.push_back(Vector3f( 4.151547f, -3.629679f, -3.760456f));
	
	std::vector<Vector3f> leaf_5_2_b;
	leaf_5_2_b.push_back(Vector3f(-1.582365f, -0.714796f, -1.396917f));
	leaf_5_2_b.push_back(Vector3f(-1.464936f,  1.697528f, -0.285079f));
	leaf_5_2_b.push_back(Vector3f(-2.785033f,  1.514637f, -1.158095f));
	
	std::vector<Vector3f> leaf_5_3_b;
	leaf_5_3_b.push_back(Vector3f( 3.979349f, -1.543050f,  2.761634f));
	leaf_5_3_b.push_back(Vector3f( 4.016621f, -0.303539f,  2.195498f));
	leaf_5_3_b.push_back(Vector3f( 3.818469f, -1.620864f,  1.443129f));
	
	std::vector<Vector3f> leaf_5_4_b;
	leaf_5_4_b.push_back(Vector3f( 4.545391f, -1.684424f, -0.017296f));
	leaf_5_4_b.push_back(Vector3f( 4.619395f, -0.854126f, -0.488972f));
	leaf_5_4_b.push_back(Vector3f( 4.488820f, -2.944209f, -0.971583f));
	
	std::vector<Vector3f> leaf_5_5_b;
	leaf_5_5_b.push_back(Vector3f(-0.265524f, -1.729322f, -1.105380f));
	leaf_5_5_b.push_back(Vector3f(-0.259936f, -1.938282f, -2.019541f));
	leaf_5_5_b.push_back(Vector3f(-1.246257f, -0.747945f, -2.833213f));
	
	std::vector< std::vector<Vector3f> > inputVerts_5_b;
	inputVerts_5_b.push_back(leaf_5_1_b);
	inputVerts_5_b.push_back(leaf_5_2_b);
	inputVerts_5_b.push_back(leaf_5_3_b);
	inputVerts_5_b.push_back(leaf_5_4_b);
	inputVerts_5_b.push_back(leaf_5_5_b);
	
	std::vector<int> pairs_5;
	pairs_5.push_back(3);
	pairs_5.push_back(4);
	pairs_5.push_back(4);
	pairs_5.push_back(1);
	
	checkTreeCollisions(inputVerts_5_a, inputVerts_5_b, pairs_5);
	
	//16 leaf nodes, 8 collisions
	std::vector<Vector3f> leaf_16_1_a;
	leaf_16_1_a.push_back(Vector3f(0.046561f, -0.501890f, -0.451597f));
	leaf_16_1_a.push_back(Vector3f(-0.629853f, -2.037045f, 1.096185f));
	leaf_16_1_a.push_back(Vector3f(-0.247535f, -2.177179f, -0.909129f));
	
	std::vector<Vector3f> leaf_16_2_a;
	leaf_16_2_a.push_back(Vector3f(-0.384107f, 0.220356f, -1.615699f));
	leaf_16_2_a.push_back(Vector3f(-1.936718f, 0.459121f, -3.355261f));
	leaf_16_2_a.push_back(Vector3f(-1.904248f, 1.171373f, -1.378690f));
	
	std::vector<Vector3f> leaf_16_3_a;
	leaf_16_3_a.push_back(Vector3f(0.376662f, 2.337635f, -1.930313f));
	leaf_16_3_a.push_back(Vector3f(2.348446f, 2.880273f, -0.900072f));
	leaf_16_3_a.push_back(Vector3f(0.675292f, 3.980533f, -1.352103f));
	
	std::vector<Vector3f> leaf_16_4_a;
	leaf_16_4_a.push_back(Vector3f(-0.772902f, 4.459019f, -2.779322f));
	leaf_16_4_a.push_back(Vector3f(0.303099f, 3.435085f, -4.533741f));
	leaf_16_4_a.push_back(Vector3f(-1.416944f, 2.970639f, -3.498089f));
	
	std::vector<Vector3f> leaf_16_5_a;
	leaf_16_5_a.push_back(Vector3f(-1.759878f, -3.896771f, 2.482803f));
	leaf_16_5_a.push_back(Vector3f(-3.183883f, -2.109862f, 2.360242f));
	leaf_16_5_a.push_back(Vector3f(-2.594913f, -3.505458f, 0.976984f));
	
	std::vector<Vector3f> leaf_16_6_a;
	leaf_16_6_a.push_back(Vector3f(-2.675344f, 1.280964f, 0.040762f));
	leaf_16_6_a.push_back(Vector3f(-1.511736f, -0.418504f, -0.938020f));
	leaf_16_6_a.push_back(Vector3f(-0.944024f, 1.104657f, 0.301737f));
	
	std::vector<Vector3f> leaf_16_7_a;
	leaf_16_7_a.push_back(Vector3f(-1.657359f, 4.915109f, 1.869840f));
	leaf_16_7_a.push_back(Vector3f(-0.925516f, 4.437210f, -0.247738f));
	leaf_16_7_a.push_back(Vector3f(-2.802923f, 4.708728f, 0.539317f));
	
	std::vector<Vector3f> leaf_16_8_a;
	leaf_16_8_a.push_back(Vector3f(-3.326378f, -3.271925f, -0.578551f));
	leaf_16_8_a.push_back(Vector3f(-4.635870f, -1.696221f, 0.510735f));
	leaf_16_8_a.push_back(Vector3f(-4.014005f, -1.877044f, -1.466068f));
	
	std::vector<Vector3f> leaf_16_9_a;
	leaf_16_9_a.push_back(Vector3f(-3.556411f, 0.761185f, 0.642864f));
	leaf_16_9_a.push_back(Vector3f(-3.281598f, 1.184519f, -1.635044f));
	leaf_16_9_a.push_back(Vector3f(-4.809733f, 0.258315f, -0.547897f));
	
	std::vector<Vector3f> leaf_16_10_a;
	leaf_16_10_a.push_back(Vector3f(1.846043f, 0.271329f, 2.268304f));
	leaf_16_10_a.push_back(Vector3f(0.100227f, 1.098520f, 1.063400f));
	leaf_16_10_a.push_back(Vector3f(0.846342f, 1.568724f, 2.904175f));
	
	std::vector<Vector3f> leaf_16_11_a;
	leaf_16_11_a.push_back(Vector3f(1.849724f, 3.954232f, -0.286736f));
	leaf_16_11_a.push_back(Vector3f(-0.188749f, 3.201731f, 0.402736f));
	leaf_16_11_a.push_back(Vector3f(0.820218f, 4.947222f, 0.737398f));
	
	std::vector<Vector3f> leaf_16_12_a;
	leaf_16_12_a.push_back(Vector3f(-4.072422f, 5.278626f, -1.167872f));
	leaf_16_12_a.push_back(Vector3f(-3.121515f, 3.211508f, -1.116810f));
	leaf_16_12_a.push_back(Vector3f(-2.453041f, 5.047299f, -0.528640f));
	
	std::vector<Vector3f> leaf_16_13_a;
	leaf_16_13_a.push_back(Vector3f(-1.259073f, 2.083495f, -1.908982f));
	leaf_16_13_a.push_back(Vector3f(-3.024761f, 2.666392f, -3.296870f));
	leaf_16_13_a.push_back(Vector3f(-1.241740f, 1.654008f, -3.647129f));
	
	std::vector<Vector3f> leaf_16_14_a;
	leaf_16_14_a.push_back(Vector3f(-3.329320f, 0.659912f, 2.964651f));
	leaf_16_14_a.push_back(Vector3f(-1.785352f, -0.723338f, 3.909220f));
	leaf_16_14_a.push_back(Vector3f(-3.825694f, -0.780003f, 3.842438f));
	
	std::vector<Vector3f> leaf_16_15_a;
	leaf_16_15_a.push_back(Vector3f(-5.681414f, 2.847592f, -2.423082f));
	leaf_16_15_a.push_back(Vector3f(-5.409488f, 1.288340f, -0.721562f));
	leaf_16_15_a.push_back(Vector3f(-6.757284f, 1.417765f, -2.304881f));
	
	std::vector<Vector3f> leaf_16_16_a;
	leaf_16_16_a.push_back(Vector3f(-3.441827f, -0.796088f, -2.686616f));
	leaf_16_16_a.push_back(Vector3f(-2.230807f, -2.313786f, -3.876888f));
	leaf_16_16_a.push_back(Vector3f(-4.155067f, -1.652501f, -4.045471f));
	
	std::vector< std::vector<Vector3f> > inputVerts_16_a;
	inputVerts_16_a.push_back(leaf_16_1_a);
	inputVerts_16_a.push_back(leaf_16_2_a);
	inputVerts_16_a.push_back(leaf_16_3_a);
	inputVerts_16_a.push_back(leaf_16_4_a);
	inputVerts_16_a.push_back(leaf_16_5_a);
	inputVerts_16_a.push_back(leaf_16_6_a);
	inputVerts_16_a.push_back(leaf_16_7_a);
	inputVerts_16_a.push_back(leaf_16_8_a);
	inputVerts_16_a.push_back(leaf_16_9_a);
	inputVerts_16_a.push_back(leaf_16_10_a);
	inputVerts_16_a.push_back(leaf_16_11_a);
	inputVerts_16_a.push_back(leaf_16_12_a);
	inputVerts_16_a.push_back(leaf_16_13_a);
	inputVerts_16_a.push_back(leaf_16_14_a);
	inputVerts_16_a.push_back(leaf_16_15_a);
	inputVerts_16_a.push_back(leaf_16_16_a);
	
	std::vector<Vector3f> leaf_16_1_b;
	leaf_16_1_b.push_back(Vector3f(-0.154947f, -0.956837f, 0.856644f));
	leaf_16_1_b.push_back(Vector3f(-0.875605f, -2.491992f, -0.671039f));
	leaf_16_1_b.push_back(Vector3f(0.383110f, -2.632127f, 0.936159f));
	
	std::vector<Vector3f> leaf_16_2_b;
	leaf_16_2_b.push_back(Vector3f(-2.279210f, -0.206869f, -1.745838f));
	leaf_16_2_b.push_back(Vector3f(0.052325f, 0.031896f, -1.770702f));
	leaf_16_2_b.push_back(Vector3f(-1.457626f, 0.744147f, -3.046608f));
	
	std::vector<Vector3f> leaf_16_3_b;
	leaf_16_3_b.push_back(Vector3f(2.183417f, 2.631353f, -1.414751f));
	leaf_16_3_b.push_back(Vector3f(0.110316f, 3.173991f, -0.607547f));
	leaf_16_3_b.push_back(Vector3f(1.551493f, 4.274251f, -1.570244f));
	
	std::vector<Vector3f> leaf_16_4_b;
	leaf_16_4_b.push_back(Vector3f(-1.140225f, 4.459019f, -4.103487f));
	leaf_16_4_b.push_back(Vector3f(-0.526827f, 3.435085f, -2.138923f));
	leaf_16_4_b.push_back(Vector3f(-0.175203f, 2.970639f, -4.115660f));
	
	std::vector<Vector3f> leaf_16_5_b;
	leaf_16_5_b.push_back(Vector3f(5.525642f, -3.896770f, -2.434318f));
	leaf_16_5_b.push_back(Vector3f(6.554777f, -2.109861f, -3.426130f));
	leaf_16_5_b.push_back(Vector3f(7.209058f, -3.505457f, -2.072540f));
	
	std::vector<Vector3f> leaf_16_6_b;
	leaf_16_6_b.push_back(Vector3f(-3.722518f, -0.748359f, -2.451743f));
	leaf_16_6_b.push_back(Vector3f(-3.750902f, -2.447827f, -0.931483f));
	leaf_16_6_b.push_back(Vector3f(-5.058074f, -0.924665f, -1.319548f));
	
	std::vector<Vector3f> leaf_16_7_b;
	leaf_16_7_b.push_back(Vector3f(-4.467101f, -2.694852f, -2.033954f));
	leaf_16_7_b.push_back(Vector3f(-3.353785f, -3.172751f, -0.089666f));
	leaf_16_7_b.push_back(Vector3f(-2.711410f, -2.901233f, -2.021366f));
	
	std::vector<Vector3f> leaf_16_8_b;
	leaf_16_8_b.push_back(Vector3f(8.861779f, -3.271924f, -1.600055f));
	leaf_16_8_b.push_back(Vector3f(8.902915f, -1.696221f, -3.302882f));
	leaf_16_8_b.push_back(Vector3f(9.982564f, -1.877043f, -1.534033f));
	
	std::vector<Vector3f> leaf_16_9_b;
	leaf_16_9_b.push_back(Vector3f(-5.337668f, -1.615259f, -0.066895f));
	leaf_16_9_b.push_back(Vector3f(-3.802931f, -1.191924f, 1.638674f));
	leaf_16_9_b.push_back(Vector3f(-3.616341f, -2.118129f, -0.227409f));
	
	std::vector<Vector3f> leaf_16_10_b;
	leaf_16_10_b.push_back(Vector3f(-4.295061f, -3.760616f, 0.823004f));
	leaf_16_10_b.push_back(Vector3f(-2.239079f, -2.933424f, 0.300890f));
	leaf_16_10_b.push_back(Vector3f(-4.116258f, -2.463221f, -0.348219f));
	
	std::vector<Vector3f> leaf_16_11_b;
	leaf_16_11_b.push_back(Vector3f(5.236723f, 3.954232f, 2.601045f));
	leaf_16_11_b.push_back(Vector3f(6.058553f, 3.201731f, 0.612242f));
	leaf_16_11_b.push_back(Vector3f(5.142727f, 4.947222f, 1.151942f));
	
	std::vector<Vector3f> leaf_16_12_b;
	leaf_16_12_b.push_back(Vector3f(9.796420f, 5.278626f, -1.774206f));
	leaf_16_12_b.push_back(Vector3f(9.132376f, 3.211508f, -1.091655f));
	leaf_16_12_b.push_back(Vector3f(8.249635f, 5.047299f, -0.975164f));
	
	std::vector<Vector3f> leaf_16_13_b;
	leaf_16_13_b.push_back(Vector3f(8.503699f, 2.083495f, 0.832141f));
	leaf_16_13_b.push_back(Vector3f(10.710564f, 2.666392f, 0.415443f));
	leaf_16_13_b.push_back(Vector3f(9.801325f, 1.654008f, 1.988697f));
	
	std::vector<Vector3f> leaf_16_14_b;
	leaf_16_14_b.push_back(Vector3f(6.195268f, 0.659912f, -3.933293f));
	leaf_16_14_b.push_back(Vector3f(4.468142f, -0.723338f, -3.391923f));
	leaf_16_14_b.push_back(Vector3f(5.860749f, -0.780003f, -4.884605f));
	
	std::vector<Vector3f> leaf_16_15_b;
	leaf_16_15_b.push_back(Vector3f(11.800273f, 2.847592f, -2.160182f));
	leaf_16_15_b.push_back(Vector3f(10.339933f, 1.288340f, -3.074796f));
	leaf_16_15_b.push_back(Vector3f(12.419054f, 1.417765f, -3.048201f));
	
	std::vector<Vector3f> leaf_16_16_b;
	leaf_16_16_b.push_back(Vector3f(10.342760f, -0.796088f, -0.459638f));
	leaf_16_16_b.push_back(Vector3f(10.442462f, -2.313786f, 1.235466f));
	leaf_16_16_b.push_back(Vector3f(11.835369f, -1.652501f, -0.102819f));
	
	std::vector< std::vector<Vector3f> > inputVerts_16_b;
	inputVerts_16_b.push_back(leaf_16_1_b);
	inputVerts_16_b.push_back(leaf_16_2_b);
	inputVerts_16_b.push_back(leaf_16_3_b);
	inputVerts_16_b.push_back(leaf_16_4_b);
	inputVerts_16_b.push_back(leaf_16_5_b);
	inputVerts_16_b.push_back(leaf_16_6_b);
	inputVerts_16_b.push_back(leaf_16_7_b);
	inputVerts_16_b.push_back(leaf_16_8_b);
	inputVerts_16_b.push_back(leaf_16_9_b);
	inputVerts_16_b.push_back(leaf_16_10_b);
	inputVerts_16_b.push_back(leaf_16_11_b);
	inputVerts_16_b.push_back(leaf_16_12_b);
	inputVerts_16_b.push_back(leaf_16_13_b);
	inputVerts_16_b.push_back(leaf_16_14_b);
	inputVerts_16_b.push_back(leaf_16_15_b);
	inputVerts_16_b.push_back(leaf_16_16_b);
	
	std::vector<int> pairs_16;
	pairs_16.push_back(0);
	pairs_16.push_back(0);
	pairs_16.push_back(1);
	pairs_16.push_back(1);
	pairs_16.push_back(2);
	pairs_16.push_back(2);
	pairs_16.push_back(3);
	pairs_16.push_back(3);
	pairs_16.push_back(7);
	pairs_16.push_back(5);
	pairs_16.push_back(7);
	pairs_16.push_back(6);
	pairs_16.push_back(7);
	pairs_16.push_back(8);
	pairs_16.push_back(7);
	pairs_16.push_back(9);
	
	checkTreeCollisions(inputVerts_16_a, inputVerts_16_b, pairs_16);
	
	//17 leaf nodes, 2 collisions
	std::vector<Vector3f> leaf_17_1_a;
	leaf_17_1_a.push_back(Vector3f(4.473518f, -3.834034f, -4.859109f));
	leaf_17_1_a.push_back(Vector3f(1.563300f, -1.631267f, -1.841853f));
	leaf_17_1_a.push_back(Vector3f(3.039510f, -3.823056f, -2.822797f));
	
	std::vector<Vector3f> leaf_17_2_a;
	leaf_17_2_a.push_back(Vector3f(2.289555f, -2.541396f, 1.329847f));
	leaf_17_2_a.push_back(Vector3f(1.223940f, -3.617396f, 4.515975f));
	leaf_17_2_a.push_back(Vector3f(1.413130f, -2.244961f, 2.938055f));
	
	std::vector<Vector3f> leaf_17_3_a;
	leaf_17_3_a.push_back(Vector3f(-1.481662f, -1.047214f, -1.528986f));
	leaf_17_3_a.push_back(Vector3f(-1.241089f, 0.830550f, -3.813329f));
	leaf_17_3_a.push_back(Vector3f(-1.910553f, 0.247660f, -2.286677f));
	
	std::vector<Vector3f> leaf_17_4_a;
	leaf_17_4_a.push_back(Vector3f(-1.316523f, -1.005728f, 4.257849f));
	leaf_17_4_a.push_back(Vector3f(0.326903f, -1.995537f, 3.130802f));
	leaf_17_4_a.push_back(Vector3f(-0.806807f, -1.321852f, 3.253014f));
	
	std::vector<Vector3f> leaf_17_5_a;
	leaf_17_5_a.push_back(Vector3f(-2.354557f, 0.285792f, -1.011755f));
	leaf_17_5_a.push_back(Vector3f(0.912654f, -2.561575f, 0.896967f));
	leaf_17_5_a.push_back(Vector3f(-1.143009f, -0.633982f, 0.960359f));
	
	std::vector<Vector3f> leaf_17_6_a;
	leaf_17_6_a.push_back(Vector3f(-4.238669f, 3.021340f, 3.344484f));
	leaf_17_6_a.push_back(Vector3f(-3.882025f, 1.505825f, 1.754884f));
	leaf_17_6_a.push_back(Vector3f(-3.616779f, 2.142743f, 2.885400f));
	
	std::vector<Vector3f> leaf_17_7_a;
	leaf_17_7_a.push_back(Vector3f(-2.795316f, 0.408414f, 3.945242f));
	leaf_17_7_a.push_back(Vector3f(-4.063000f, 0.187267f, 6.618474f));
	leaf_17_7_a.push_back(Vector3f(-3.427996f, 1.054115f, 5.217063f));
	
	std::vector<Vector3f> leaf_17_8_a;
	leaf_17_8_a.push_back(Vector3f(-2.356369f, -0.249571f, 2.402338f));
	leaf_17_8_a.push_back(Vector3f(-4.787738f, -1.660417f, 4.533692f));
	leaf_17_8_a.push_back(Vector3f(-4.120627f, -0.637987f, 2.825243f));
	
	std::vector<Vector3f> leaf_17_9_a;
	leaf_17_9_a.push_back(Vector3f(-1.673104f, 2.838676f, -5.818596f));
	leaf_17_9_a.push_back(Vector3f(-2.567120f, 1.002105f, -8.694785f));
	leaf_17_9_a.push_back(Vector3f(-1.563865f, 1.319170f, -6.877603f));
	
	std::vector<Vector3f> leaf_17_10_a;
	leaf_17_10_a.push_back(Vector3f(0.721270f, -0.343845f, -4.882560f));
	leaf_17_10_a.push_back(Vector3f(-1.569342f, -0.630755f, -6.746125f));
	leaf_17_10_a.push_back(Vector3f(-0.636821f, -0.994475f, -5.291230f));
	
	std::vector<Vector3f> leaf_17_11_a;
	leaf_17_11_a.push_back(Vector3f(0.516547f, 0.582832f, 2.060946f));
	leaf_17_11_a.push_back(Vector3f(-1.307790f, -0.666241f, 1.811102f));
	leaf_17_11_a.push_back(Vector3f(-0.264835f, -0.185760f, 2.471004f));
	
	std::vector<Vector3f> leaf_17_12_a;
	leaf_17_12_a.push_back(Vector3f(5.345932f, -4.047254f, 0.411145f));
	leaf_17_12_a.push_back(Vector3f(3.291012f, -2.325683f, 4.314848f));
	leaf_17_12_a.push_back(Vector3f(3.835518f, -2.427907f, 1.551074f));
	
	std::vector<Vector3f> leaf_17_13_a;
	leaf_17_13_a.push_back(Vector3f(1.283195f, -2.502704f, 5.614092f));
	leaf_17_13_a.push_back(Vector3f(-2.114996f, -2.323045f, 8.907330f));
	leaf_17_13_a.push_back(Vector3f(-0.105477f, -1.241785f, 7.252612f));
	
	std::vector<Vector3f> leaf_17_14_a;
	leaf_17_14_a.push_back(Vector3f(0.584571f, -2.081781f, 9.851192f));
	leaf_17_14_a.push_back(Vector3f(-1.454935f, -1.192375f, 9.837819f));
	leaf_17_14_a.push_back(Vector3f(-0.287978f, -1.523852f, 9.306387f));
	
	std::vector<Vector3f> leaf_17_15_a;
	leaf_17_15_a.push_back(Vector3f(-4.911549f, 2.790556f, -1.314090f));
	leaf_17_15_a.push_back(Vector3f(-5.752580f, 1.834651f, 1.365650f));
	leaf_17_15_a.push_back(Vector3f(-5.748091f, 2.946678f, -0.006205f));
	
	std::vector<Vector3f> leaf_17_16_a;
	leaf_17_16_a.push_back(Vector3f(1.292112f, 2.327368f, -1.932742f));
	leaf_17_16_a.push_back(Vector3f(-0.944891f, 1.049298f, 0.477046f));
	leaf_17_16_a.push_back(Vector3f(0.710677f, 2.162133f, -0.178616f));
	
	std::vector<Vector3f> leaf_17_17_a;
	leaf_17_17_a.push_back(Vector3f(-1.072293f, 2.570147f, -2.756880f));
	leaf_17_17_a.push_back(Vector3f(-2.298294f, 4.122057f, -3.776344f));
	leaf_17_17_a.push_back(Vector3f(-1.456313f, 3.100944f, -3.726532f));
	
	std::vector< std::vector<Vector3f> > inputVerts_17_a;
	inputVerts_17_a.push_back(leaf_17_1_a);
	inputVerts_17_a.push_back(leaf_17_2_a);
	inputVerts_17_a.push_back(leaf_17_3_a);
	inputVerts_17_a.push_back(leaf_17_4_a);
	inputVerts_17_a.push_back(leaf_17_5_a);
	inputVerts_17_a.push_back(leaf_17_6_a);
	inputVerts_17_a.push_back(leaf_17_7_a);
	inputVerts_17_a.push_back(leaf_17_8_a);
	inputVerts_17_a.push_back(leaf_17_9_a);
	inputVerts_17_a.push_back(leaf_17_10_a);
	inputVerts_17_a.push_back(leaf_17_11_a);
	inputVerts_17_a.push_back(leaf_17_12_a);
	inputVerts_17_a.push_back(leaf_17_13_a);
	inputVerts_17_a.push_back(leaf_17_14_a);
	inputVerts_17_a.push_back(leaf_17_15_a);
	inputVerts_17_a.push_back(leaf_17_16_a);
	inputVerts_17_a.push_back(leaf_17_17_a);
	
	std::vector<Vector3f> leaf_17_1_b;
	leaf_17_1_b.push_back(Vector3f(-2.593107f, -5.996761f, -5.302939f));
	leaf_17_1_b.push_back(Vector3f(-5.231448f, -3.793994f, -2.045277f));
	leaf_17_1_b.push_back(Vector3f(-3.845615f, -5.985783f, -3.150226f));
	
	std::vector<Vector3f> leaf_17_2_b;
	leaf_17_2_b.push_back(Vector3f(-4.233586f, -4.704123f, 1.051722f));
	leaf_17_2_b.push_back(Vector3f(-5.019629f, -5.780124f, 4.318078f));
	leaf_17_2_b.push_back(Vector3f(-4.967628f, -4.407689f, 2.729707f));
	
	std::vector<Vector3f> leaf_17_3_b;
	leaf_17_3_b.push_back(Vector3f(-8.237937f, -3.209941f, -1.470214f));
	leaf_17_3_b.push_back(Vector3f(-8.195846f, -1.332178f, -3.766804f));
	leaf_17_3_b.push_back(Vector3f(-8.730756f, -1.915068f, -2.187969f));
	
	std::vector<Vector3f> leaf_17_4_b;
	leaf_17_4_b.push_back(Vector3f(-7.572899f, -3.168456f, 4.280651f));
	leaf_17_4_b.push_back(Vector3f(-6.033112f, -4.158264f, 3.015683f));
	leaf_17_4_b.push_back(Vector3f(-7.152004f, -3.484580f, 3.235495f));
	
	std::vector<Vector3f> leaf_17_5_b;
	leaf_17_5_b.push_back(Vector3f(-9.062824f, -1.876935f, -0.879423f));
	leaf_17_5_b.push_back(Vector3f(-5.642767f, -4.724303f, 0.739556f));
	leaf_17_5_b.push_back(Vector3f(-7.685243f, -2.796710f, 0.980511f));
	
	std::vector<Vector3f> leaf_17_6_b;
	leaf_17_6_b.push_back(Vector3f(-10.563093f, 0.858612f, 3.623454f));
	leaf_17_6_b.push_back(Vector3f(-10.345275f, -0.656902f, 2.008964f));
	leaf_17_6_b.push_back(Vector3f(-9.983241f, -0.019985f, 3.112301f));
	
	std::vector<Vector3f> leaf_17_7_b;
	leaf_17_7_b.push_back(Vector3f(-9.073188f, -1.754314f, 4.097122f));
	leaf_17_7_b.push_back(Vector3f(-10.104905f, -1.975461f, 6.869981f));
	leaf_17_7_b.push_back(Vector3f(-9.593493f, -1.108613f, 5.418899f));
	
	std::vector<Vector3f> leaf_17_8_b;
	leaf_17_8_b.push_back(Vector3f(-8.769335f, -2.412298f, 2.522033f));
	leaf_17_8_b.push_back(Vector3f(-11.007246f, -3.823145f, 4.855697f));
	leaf_17_8_b.push_back(Vector3f(-10.490404f, -2.800715f, 3.095949f));
	
	std::vector<Vector3f> leaf_17_9_b;
	leaf_17_9_b.push_back(Vector3f(-8.799684f, 0.675948f, -5.727191f));
	leaf_17_9_b.push_back(Vector3f(-9.939118f, -1.160622f, -8.515276f));
	leaf_17_9_b.push_back(Vector3f(-8.782451f, -0.843557f, -6.791677f));
	
	std::vector<Vector3f> leaf_17_10_b;
	leaf_17_10_b.push_back(Vector3f(-6.333322f, -2.506573f, -5.001759f));
	leaf_17_10_b.push_back(Vector3f(-8.776535f, -2.793482f, -6.660219f));
	leaf_17_10_b.push_back(Vector3f(-7.721671f, -3.157202f, -5.291431f));
	
	std::vector<Vector3f> leaf_17_11_b;
	leaf_17_11_b.push_back(Vector3f(-5.936714f, -1.579896f, 1.933434f));
	leaf_17_11_b.push_back(Vector3f(-7.775824f, -2.828969f, 1.842318f));
	leaf_17_11_b.push_back(Vector3f(-6.679701f, -2.348488f, 2.409539f));
	
	std::vector<Vector3f> leaf_17_12_b;
	leaf_17_12_b.push_back(Vector3f(-1.268124f, -6.209981f, -0.127892f));
	leaf_17_12_b.push_back(Vector3f(-2.977700f, -4.488411f, 3.938918f));
	leaf_17_12_b.push_back(Vector3f(-2.674281f, -4.590634f, 1.138405f));
	
	std::vector<Vector3f> leaf_17_13_b;
	leaf_17_13_b.push_back(Vector3f(-0.112956f, 2.757263f, -5.433385f));
	leaf_17_13_b.push_back(Vector3f(-3.213570f, 2.936923f, -1.858570f));
	leaf_17_13_b.push_back(Vector3f(-1.354703f, 4.018183f, -3.680896f));
	
	std::vector<Vector3f> leaf_17_14_b;
	leaf_17_14_b.push_back(Vector3f(-5.195145f, -4.244509f, 9.688601f));
	leaf_17_14_b.push_back(Vector3f(-7.228164f, -3.355103f, 9.851682f));
	leaf_17_14_b.push_back(Vector3f(-6.111546f, -3.686580f, 9.221308f));
	
	std::vector<Vector3f> leaf_17_15_b;
	leaf_17_15_b.push_back(Vector3f(-11.636384f, 0.627829f, -0.959463f));
	leaf_17_15_b.push_back(Vector3f(-12.242485f, -0.328077f, 1.782978f));
	leaf_17_15_b.push_back(Vector3f(-12.356668f, 0.783950f, 0.415876f));
	
	std::vector<Vector3f> leaf_17_16_b;
	leaf_17_16_b.push_back(Vector3f(-5.509481f, 0.164640f, -2.112369f));
	leaf_17_16_b.push_back(Vector3f(-7.529671f, -1.113429f, 0.481873f));
	leaf_17_16_b.push_back(Vector3f(-5.937017f, -0.000594f, -0.314527f));
	
	std::vector<Vector3f> leaf_17_17_b;
	leaf_17_17_b.push_back(Vector3f(3.226803f, -3.391035f, -2.728914f));
	leaf_17_17_b.push_back(Vector3f(1.917221f, -1.839125f, -3.638517f));
	leaf_17_17_b.push_back(Vector3f(2.760355f, -2.860238f, -3.661717f));
	
	std::vector< std::vector<Vector3f> > inputVerts_17_b;
	inputVerts_17_b.push_back(leaf_17_1_b);
	inputVerts_17_b.push_back(leaf_17_2_b);
	inputVerts_17_b.push_back(leaf_17_3_b);
	inputVerts_17_b.push_back(leaf_17_4_b);
	inputVerts_17_b.push_back(leaf_17_5_b);
	inputVerts_17_b.push_back(leaf_17_6_b);
	inputVerts_17_b.push_back(leaf_17_7_b);
	inputVerts_17_b.push_back(leaf_17_8_b);
	inputVerts_17_b.push_back(leaf_17_9_b);
	inputVerts_17_b.push_back(leaf_17_10_b);
	inputVerts_17_b.push_back(leaf_17_11_b);
	inputVerts_17_b.push_back(leaf_17_12_b);
	inputVerts_17_b.push_back(leaf_17_13_b);
	inputVerts_17_b.push_back(leaf_17_14_b);
	inputVerts_17_b.push_back(leaf_17_15_b);
	inputVerts_17_b.push_back(leaf_17_16_b);
	inputVerts_17_b.push_back(leaf_17_17_b);
	
	std::vector<int> pairs_17;
	pairs_17.push_back(0);
	pairs_17.push_back(16);
	pairs_17.push_back(16);
	pairs_17.push_back(12);
	
	checkTreeCollisions(inputVerts_17_a, inputVerts_17_b, pairs_17);
}


void TreeTest::checkTreeCollisions(std::vector< std::vector<Vector3f> >& vertsA,
								   std::vector< std::vector<Vector3f> >& vertsB,
								   std::vector<int>& expectedPairs)
{
	//Generate list of leaf nodes with KDOP from vertices
	std::list<Node*> leafNodes_6_a;
	std::list<Node*> leafNodes_14_a;
	std::list<Node*> leafNodes_18_a;
	std::list<Node*> leafNodes_26_a;
	
	std::list<Node*> leafNodes_6_b;
	std::list<Node*> leafNodes_14_b;
	std::list<Node*> leafNodes_18_b;
	std::list<Node*> leafNodes_26_b;
	
	for(int i = 0; i < vertsA.size(); i++)
	{
		leafNodes_6_a.push_back (new Node(new KDOP(vertsA[i], 6)));
		leafNodes_14_a.push_back(new Node(new KDOP(vertsA[i], 14)));
		leafNodes_18_a.push_back(new Node(new KDOP(vertsA[i], 18)));
		leafNodes_26_a.push_back(new Node(new KDOP(vertsA[i], 26)));
	}
	
	for(int i = 0; i < vertsB.size(); i++)
	{
		leafNodes_6_b.push_back (new Node(new KDOP(vertsB[i], 6)));
		leafNodes_14_b.push_back(new Node(new KDOP(vertsB[i], 14)));
		leafNodes_18_b.push_back(new Node(new KDOP(vertsB[i], 18)));
		leafNodes_26_b.push_back(new Node(new KDOP(vertsB[i], 26)));
	}
	
	std::vector<NodePair> expectedPairs_6;
	std::vector<NodePair> expectedPairs_14;
	std::vector<NodePair> expectedPairs_18;
	std::vector<NodePair> expectedPairs_26;
	
	for(int i = 0; i < expectedPairs.size()/2; i++)
	{
		int index1 = i*2;
		int index2 = i*2 + 1;
		
		int count1_6  = 0;
		int count2_6  = 0;
		int count1_14 = 0;
		int count2_14 = 0;
		int count1_18 = 0;
		int count2_18 = 0;
		int count1_26 = 0;
		int count2_26 = 0;
		
		Node* one_6  = 0;
		Node* two_6  = 0;
		Node* one_14 = 0;
		Node* two_14 = 0;
		Node* one_18 = 0;
		Node* two_18 = 0;
		Node* one_26 = 0;
		Node* two_26 = 0;
		
		for(std::list<Node*>::const_iterator child = leafNodes_6_a.begin(); child != leafNodes_6_a.end(); child++)
		{
			if(count1_6 == expectedPairs[index1])
			{
				one_6 = (*child); break;
			}
			count1_6++;
		}
		
		for(std::list<Node*>::const_iterator child = leafNodes_6_b.begin(); child != leafNodes_6_b.end(); child++)
		{
			if(count2_6 == expectedPairs[index2])
			{
				two_6 = (*child); break;
			}
			count2_6++;
		}
		
		for(std::list<Node*>::const_iterator child = leafNodes_14_a.begin(); child != leafNodes_14_a.end(); child++)
		{
			if(count1_14 == expectedPairs[index1])
			{
				one_14 = (*child); break;
			}
			count1_14++;
		}
		
		for(std::list<Node*>::const_iterator child = leafNodes_14_b.begin(); child != leafNodes_14_b.end(); child++)
		{
			if(count2_14 == expectedPairs[index2])
			{
				two_14 = (*child); break;
			}
			count2_14++;
		}
		
		for(std::list<Node*>::const_iterator child = leafNodes_18_a.begin(); child != leafNodes_18_a.end(); child++)
		{
			if(count1_18 == expectedPairs[index1])
			{
				one_18 = (*child); break;
			}
			count1_18++;
		}
		
		for(std::list<Node*>::const_iterator child = leafNodes_18_b.begin(); child != leafNodes_18_b.end(); child++)
		{
			if(count2_18 == expectedPairs[index2])
			{
				two_18 = (*child); break;
			}
			count2_18++;
		}
		
		for(std::list<Node*>::const_iterator child = leafNodes_26_a.begin(); child != leafNodes_26_a.end(); child++)
		{
			if(count1_26 == expectedPairs[index1])
			{
				one_26 = (*child); break;
			}
			count1_26++;
		}
		
		for(std::list<Node*>::const_iterator child = leafNodes_26_b.begin(); child != leafNodes_26_b.end(); child++)
		{
			if(count2_26 == expectedPairs[index2])
			{
				two_26 = (*child); break;
			}
			count2_26++;
		}
		
		expectedPairs_6.push_back (NodePair(one_6,  two_6));
		expectedPairs_14.push_back(NodePair(one_14, two_14));
		expectedPairs_18.push_back(NodePair(one_18, two_18));
		expectedPairs_26.push_back(NodePair(one_26, two_26));
	}
	
	//Build the two trees
	Node* root_6_a  = Node::buildTree(leafNodes_6_a);
	Node* root_14_a = Node::buildTree(leafNodes_14_a);
	Node* root_18_a = Node::buildTree(leafNodes_18_a);
	Node* root_26_a = Node::buildTree(leafNodes_26_a);
	
	Node* root_6_b  = Node::buildTree(leafNodes_6_b);
	Node* root_14_b = Node::buildTree(leafNodes_14_b);
	Node* root_18_b = Node::buildTree(leafNodes_18_b);
	Node* root_26_b = Node::buildTree(leafNodes_26_b);
	
	//Save collision data in these data structures
	std::vector<NodePair> actualPairs_6;
	std::vector<NodePair> actualPairs_14;
	std::vector<NodePair> actualPairs_18;
	std::vector<NodePair> actualPairs_26;
	
	//Check for collisions between the two trees
	Node::collides(root_6_a,  root_6_b,  actualPairs_6);
	Node::collides(root_14_a, root_14_b, actualPairs_14);
	Node::collides(root_18_a, root_18_b, actualPairs_18);
	Node::collides(root_26_a, root_26_b, actualPairs_26);
	
	for(int i = 0; i < actualPairs_6.size(); i++)
	{
		CPPUNIT_ASSERT_EQUAL(expectedPairs_6[i].one, actualPairs_6[i].one);
		CPPUNIT_ASSERT_EQUAL(expectedPairs_6[i].two, actualPairs_6[i].two);
	}
	
	for(int i = 0; i < actualPairs_14.size(); i++)
	{
		CPPUNIT_ASSERT_EQUAL(expectedPairs_14[i].one, actualPairs_14[i].one);
		CPPUNIT_ASSERT_EQUAL(expectedPairs_14[i].two, actualPairs_14[i].two);
	}
	
	for(int i = 0; i < actualPairs_18.size(); i++)
	{
		CPPUNIT_ASSERT_EQUAL(expectedPairs_18[i].one, actualPairs_18[i].one);
		CPPUNIT_ASSERT_EQUAL(expectedPairs_18[i].two, actualPairs_18[i].two);
	}
	
	for(int i = 0; i < actualPairs_26.size(); i++)
	{
		CPPUNIT_ASSERT_EQUAL(expectedPairs_26[i].one, actualPairs_26[i].one);
		CPPUNIT_ASSERT_EQUAL(expectedPairs_26[i].two, actualPairs_26[i].two);
	}
	
	//Release memory
	delete root_6_a;
	delete root_14_a;
	delete root_18_a;
	delete root_26_a;
	
	delete root_6_b;
	delete root_14_b;
	delete root_18_b;
	delete root_26_b;
}
