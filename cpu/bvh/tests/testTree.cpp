
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
