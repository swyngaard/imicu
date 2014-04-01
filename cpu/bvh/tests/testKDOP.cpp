
#include "testKDOP.h"

void KDOPTest::testNormals()
{
	std::vector<Vector3f> vertices;
	vertices.push_back(Vector3f(3.0f, -4.4f, 1.4f));
	vertices.push_back(Vector3f(2.0f, -5.7f, 9.4f));
	vertices.push_back(Vector3f(-1.4f, 8.4f, -6.4f));
	
	KDOP dop6 (vertices,  6);
	std::vector<Vector3i> normals6 = dop6.getNormals();
	
	std::vector<Vector3i> expected6;
	expected6.push_back(Vector3i(-1,  0,  0));
	expected6.push_back(Vector3i( 0, -1,  0));
	expected6.push_back(Vector3i( 0,  0, -1));
	expected6.push_back(Vector3i( 1,  0,  0));
	expected6.push_back(Vector3i( 0,  1,  0));
	expected6.push_back(Vector3i( 0,  0,  1));
	
	CPPUNIT_ASSERT(expected6[0] == normals6[0]);
	CPPUNIT_ASSERT(expected6[1] == normals6[1]);
	CPPUNIT_ASSERT(expected6[2] == normals6[2]);
	CPPUNIT_ASSERT(expected6[3] == normals6[3]);
	CPPUNIT_ASSERT(expected6[4] == normals6[4]);
	CPPUNIT_ASSERT(expected6[5] == normals6[5]);
	
	KDOP dop14(vertices, 14);
	std::vector<Vector3i> normals14 = dop14.getNormals();
	
	std::vector<Vector3i> expected14;
	expected14.push_back(Vector3i(-1, 0, 0));
	expected14.push_back(Vector3i( 0,-1, 0));
	expected14.push_back(Vector3i( 0, 0,-1));
	expected14.push_back(Vector3i(-1,-1,-1));
	expected14.push_back(Vector3i(-1, 1,-1));
	expected14.push_back(Vector3i(-1,-1, 1));
	expected14.push_back(Vector3i(-1, 1, 1));
	expected14.push_back(Vector3i( 1, 0, 0));
	expected14.push_back(Vector3i( 0, 1, 0));
	expected14.push_back(Vector3i( 0, 0, 1));
	expected14.push_back(Vector3i( 1, 1, 1));
	expected14.push_back(Vector3i( 1,-1, 1));
	expected14.push_back(Vector3i( 1, 1,-1));
	expected14.push_back(Vector3i( 1,-1,-1));
    
    CPPUNIT_ASSERT(expected14[ 0] == normals14[ 0]);
    CPPUNIT_ASSERT(expected14[ 1] == normals14[ 1]);
    CPPUNIT_ASSERT(expected14[ 2] == normals14[ 2]);
    CPPUNIT_ASSERT(expected14[ 3] == normals14[ 3]);
    CPPUNIT_ASSERT(expected14[ 4] == normals14[ 4]);
    CPPUNIT_ASSERT(expected14[ 5] == normals14[ 5]);
    CPPUNIT_ASSERT(expected14[ 6] == normals14[ 6]);
    CPPUNIT_ASSERT(expected14[ 7] == normals14[ 7]);
    CPPUNIT_ASSERT(expected14[ 8] == normals14[ 8]);
    CPPUNIT_ASSERT(expected14[ 9] == normals14[ 9]);
    CPPUNIT_ASSERT(expected14[10] == normals14[10]);
    CPPUNIT_ASSERT(expected14[11] == normals14[11]);
    CPPUNIT_ASSERT(expected14[12] == normals14[12]);
    CPPUNIT_ASSERT(expected14[13] == normals14[13]);
    
	KDOP dop18(vertices, 18);
	std::vector<Vector3i> normals18 = dop18.getNormals();
	
	std::vector<Vector3i> expected18;
	expected18.push_back(Vector3i(-1, 0, 0));
	expected18.push_back(Vector3i( 0,-1, 0));
	expected18.push_back(Vector3i( 0, 0,-1));
	expected18.push_back(Vector3i(-1,-1, 0));
	expected18.push_back(Vector3i(-1, 0,-1));
	expected18.push_back(Vector3i( 0,-1,-1));
	expected18.push_back(Vector3i(-1, 1, 0));
	expected18.push_back(Vector3i(-1, 0, 1));
	expected18.push_back(Vector3i( 0,-1, 1));
	expected18.push_back(Vector3i( 1, 0, 0));
	expected18.push_back(Vector3i( 0, 1, 0));
	expected18.push_back(Vector3i( 0, 0, 1));
	expected18.push_back(Vector3i( 1, 1, 0));
	expected18.push_back(Vector3i( 1, 0, 1));
	expected18.push_back(Vector3i( 0, 1, 1));
	expected18.push_back(Vector3i( 1,-1, 0));
	expected18.push_back(Vector3i( 1, 0,-1));
	expected18.push_back(Vector3i( 0, 1,-1));
	
	CPPUNIT_ASSERT(expected18[ 0] == normals18[ 0]);
	CPPUNIT_ASSERT(expected18[ 1] == normals18[ 1]);
	CPPUNIT_ASSERT(expected18[ 2] == normals18[ 2]);
	CPPUNIT_ASSERT(expected18[ 3] == normals18[ 3]);
	CPPUNIT_ASSERT(expected18[ 4] == normals18[ 4]);
	CPPUNIT_ASSERT(expected18[ 5] == normals18[ 5]);
	CPPUNIT_ASSERT(expected18[ 6] == normals18[ 6]);
	CPPUNIT_ASSERT(expected18[ 7] == normals18[ 7]);
	CPPUNIT_ASSERT(expected18[ 8] == normals18[ 8]);
	CPPUNIT_ASSERT(expected18[ 9] == normals18[ 9]);
	CPPUNIT_ASSERT(expected18[10] == normals18[10]);
	CPPUNIT_ASSERT(expected18[11] == normals18[11]);
	CPPUNIT_ASSERT(expected18[12] == normals18[12]);
	CPPUNIT_ASSERT(expected18[13] == normals18[13]);
	CPPUNIT_ASSERT(expected18[14] == normals18[14]);
	CPPUNIT_ASSERT(expected18[15] == normals18[15]);
	CPPUNIT_ASSERT(expected18[16] == normals18[16]);
	CPPUNIT_ASSERT(expected18[17] == normals18[17]);
	
	KDOP dop26(vertices, 26);
	std::vector<Vector3i> normals26 = dop26.getNormals();
	
	std::vector<Vector3i> expected26;
	expected26.push_back(Vector3i(-1, 0, 0));
    expected26.push_back(Vector3i( 0,-1, 0));
    expected26.push_back(Vector3i( 0, 0,-1));
    expected26.push_back(Vector3i(-1,-1,-1));
    expected26.push_back(Vector3i(-1, 1,-1));
    expected26.push_back(Vector3i(-1,-1, 1));
    expected26.push_back(Vector3i(-1, 1, 1));
    expected26.push_back(Vector3i(-1,-1, 0));
    expected26.push_back(Vector3i(-1, 0,-1));
    expected26.push_back(Vector3i( 0,-1,-1));
    expected26.push_back(Vector3i(-1, 1, 0));
    expected26.push_back(Vector3i(-1, 0, 1));
    expected26.push_back(Vector3i( 0,-1, 1));
    expected26.push_back(Vector3i( 1, 0, 0));
    expected26.push_back(Vector3i( 0, 1, 0));
    expected26.push_back(Vector3i( 0, 0, 1));
    expected26.push_back(Vector3i( 1, 1, 1));
    expected26.push_back(Vector3i( 1,-1, 1));
    expected26.push_back(Vector3i( 1, 1,-1));
    expected26.push_back(Vector3i( 1,-1,-1));
    expected26.push_back(Vector3i( 1, 1, 0));
    expected26.push_back(Vector3i( 1, 0, 1));
    expected26.push_back(Vector3i( 0, 1, 1));
    expected26.push_back(Vector3i( 1,-1, 0));
    expected26.push_back(Vector3i( 1, 0,-1));
    expected26.push_back(Vector3i( 0, 1,-1));
    
    CPPUNIT_ASSERT(expected26[ 0] == normals26[ 0]);
	CPPUNIT_ASSERT(expected26[ 1] == normals26[ 1]);
	CPPUNIT_ASSERT(expected26[ 2] == normals26[ 2]);
	CPPUNIT_ASSERT(expected26[ 3] == normals26[ 3]);
	CPPUNIT_ASSERT(expected26[ 4] == normals26[ 4]);
	CPPUNIT_ASSERT(expected26[ 5] == normals26[ 5]);
	CPPUNIT_ASSERT(expected26[ 6] == normals26[ 6]);
	CPPUNIT_ASSERT(expected26[ 7] == normals26[ 7]);
	CPPUNIT_ASSERT(expected26[ 8] == normals26[ 8]);
	CPPUNIT_ASSERT(expected26[ 9] == normals26[ 9]);
	CPPUNIT_ASSERT(expected26[10] == normals26[10]);
	CPPUNIT_ASSERT(expected26[11] == normals26[11]);
	CPPUNIT_ASSERT(expected26[12] == normals26[12]);
	CPPUNIT_ASSERT(expected26[13] == normals26[13]);
	CPPUNIT_ASSERT(expected26[14] == normals26[14]);
	CPPUNIT_ASSERT(expected26[15] == normals26[15]);
	CPPUNIT_ASSERT(expected26[16] == normals26[16]);
	CPPUNIT_ASSERT(expected26[17] == normals26[17]);
	CPPUNIT_ASSERT(expected26[18] == normals26[18]);
	CPPUNIT_ASSERT(expected26[19] == normals26[19]);
	CPPUNIT_ASSERT(expected26[20] == normals26[20]);
	CPPUNIT_ASSERT(expected26[21] == normals26[21]);
	CPPUNIT_ASSERT(expected26[22] == normals26[22]);
	CPPUNIT_ASSERT(expected26[23] == normals26[23]);
	CPPUNIT_ASSERT(expected26[24] == normals26[24]);
	CPPUNIT_ASSERT(expected26[25] == normals26[25]);
}

void KDOPTest::testBuildDegenerateMatrix()
{
	std::vector<Vector3f> vertices;
	vertices.push_back(Vector3f(3.0f, -4.4f, 1.4f));
	vertices.push_back(Vector3f(2.0f, -5.7f, 9.4f));
	vertices.push_back(Vector3f(-1.4f, 8.4f, -6.4f));
	
	KDOP dop6 (vertices, 6);
	
	bool expected6[6][6] = {{0,1,1,0,1,1},
						    {0,0,1,1,0,1},
						    {0,0,0,1,1,0},
						    {0,0,0,0,1,1},
						    {0,0,0,0,0,1},
						    {0,0,0,0,0,0}};
	
	dop6.buildDegenerateMatrix();
	bool** actual6 = dop6.getDegenerateMatrix();
	
	for(int i = 0; i < 6; i++)
		for(int j = 0; j < 6; j++)
			CPPUNIT_ASSERT_EQUAL(expected6[i][j], actual6[i][j]);
	
	KDOP dop14(vertices, 14);
	
	bool expected14[14][14] = {{0,1,1,1,1,1,1,0,1,1,1,1,1,1},
							   {0,0,1,1,1,1,1,1,0,1,1,1,1,1},
							   {0,0,0,1,1,1,1,1,1,0,1,1,1,1},
							   {0,0,0,0,1,1,1,1,1,1,0,1,1,1},
							   {0,0,0,0,0,1,1,1,1,1,1,0,1,1},
							   {0,0,0,0,0,0,1,1,1,1,1,1,0,1},
							   {0,0,0,0,0,0,0,1,1,1,1,1,1,0},
							   {0,0,0,0,0,0,0,0,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
	
	dop14.buildDegenerateMatrix();
	bool** actual14 = dop14.getDegenerateMatrix();
	
	for(int i = 0; i < 14; i++)
		for(int j = 0; j < 14; j++)
			CPPUNIT_ASSERT_EQUAL(expected14[i][j], actual14[i][j]);
	
	
	KDOP dop18(vertices, 18);
	
	bool expected18[18][18] = {{0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1},
							   {0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1},
							   {0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1},
							   {0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1},
							   {0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1},
							   {0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1},
							   {0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1},
							   {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1},
							   {0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0},
							   {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
	
	dop18.buildDegenerateMatrix();
	bool** actual18 = dop18.getDegenerateMatrix();
	
	for(int i = 0; i < 18; i++)
		for(int j = 0; j < 18; j++)
			CPPUNIT_ASSERT_EQUAL(expected18[i][j], actual18[i][j]);
	
	KDOP dop26(vertices, 26);
	
	bool expected26[26][26] = {{0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1},
							   {0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1},
							   {0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1},
							   {0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
	
	dop26.buildDegenerateMatrix();
	bool** actual26 = dop26.getDegenerateMatrix();
	
	for(int i = 0; i < 26; i++)
		for(int j = 0; j < 26; j++)
			CPPUNIT_ASSERT_EQUAL(expected26[i][j], actual26[i][j]);
}

void KDOPTest::testSetDegenerateMatrix()
{
	std::vector<Vector3f> vertices;
	vertices.push_back(Vector3f(3.0f, -4.4f, 1.4f));
	vertices.push_back(Vector3f(2.0f, -5.7f, 9.4f));
	vertices.push_back(Vector3f(-1.4f, 8.4f, -6.4f));
	
	KDOP dop6 (vertices, 6);
	
	bool expected6[6][6] = {{0,1,1,0,1,1},
						    {0,0,1,1,0,1},
						    {0,0,0,1,1,0},
						    {0,0,0,0,1,1},
						    {0,0,0,0,0,1},
						    {0,0,0,0,0,0}};
	
	bool** actual6 = dop6.getDegenerateMatrix();
	
	for(int i = 0; i < 6; i++)
		for(int j = 0; j < 6; j++)
			CPPUNIT_ASSERT_EQUAL(expected6[i][j], actual6[i][j]);
	
	KDOP dop14(vertices, 14);
	
	bool expected14[14][14] = {{0,1,1,1,1,1,1,0,1,1,1,1,1,1},
							   {0,0,1,1,1,1,1,1,0,1,1,1,1,1},
							   {0,0,0,1,1,1,1,1,1,0,1,1,1,1},
							   {0,0,0,0,1,1,1,1,1,1,0,1,1,1},
							   {0,0,0,0,0,1,1,1,1,1,1,0,1,1},
							   {0,0,0,0,0,0,1,1,1,1,1,1,0,1},
							   {0,0,0,0,0,0,0,1,1,1,1,1,1,0},
							   {0,0,0,0,0,0,0,0,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
	
	bool** actual14 = dop14.getDegenerateMatrix();
	
	for(int i = 0; i < 14; i++)
		for(int j = 0; j < 14; j++)
			CPPUNIT_ASSERT_EQUAL(expected14[i][j], actual14[i][j]);
	
	
	KDOP dop18(vertices, 18);
	
	bool expected18[18][18] = {{0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1},
							   {0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1},
							   {0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1},
							   {0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1},
							   {0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1},
							   {0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,1},
							   {0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1},
							   {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1},
							   {0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0},
							   {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
	
	bool** actual18 = dop18.getDegenerateMatrix();
	
	for(int i = 0; i < 18; i++)
		for(int j = 0; j < 18; j++)
			CPPUNIT_ASSERT_EQUAL(expected18[i][j], actual18[i][j]);
	
	KDOP dop26(vertices, 26);
	
	bool expected26[26][26] = {{0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1},
							   {0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1},
							   {0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1},
							   {0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
							   {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};
	
	bool** actual26 = dop26.getDegenerateMatrix();
	
	for(int i = 0; i < 26; i++)
		for(int j = 0; j < 26; j++)
			CPPUNIT_ASSERT_EQUAL(expected26[i][j], actual26[i][j]);
}

void KDOPTest::testPlaneDistances()
{
	//Default constructor zero vertices
	KDOP input00_6 (6);
	KDOP input00_14(14);
	KDOP input00_18(18);
	KDOP input00_26(26);
	
	float expected00 = 0.0f;
	
	const float* actual00_6  = input00_6.getDistances();
	const float* actual00_14 = input00_14.getDistances();
	const float* actual00_18 = input00_18.getDistances();
	const float* actual00_26 = input00_26.getDistances();
	
	for(int i = 0; i < input00_6.K; i++)
		CPPUNIT_ASSERT_EQUAL(expected00, actual00_6[i]);
	
	for(int i = 0; i < input00_14.K; i++)
		CPPUNIT_ASSERT_EQUAL(expected00, actual00_14[i]);
	
	for(int i = 0; i < input00_18.K; i++)
		CPPUNIT_ASSERT_EQUAL(expected00, actual00_18[i]);
	
	for(int i = 0; i < input00_26.K; i++)
		CPPUNIT_ASSERT_EQUAL(expected00, actual00_26[i]);
	
	//Zero vertices
	std::vector<Vector3f> inputVerts0;
	
	KDOP input0_6 (inputVerts0, 6);
	KDOP input0_14(inputVerts0, 14);
	KDOP input0_18(inputVerts0, 18);
	KDOP input0_26(inputVerts0, 26);
	
	float expected0 = 0.0f;
	
	const float* actual0_6  = input0_6.getDistances();
	const float* actual0_14 = input0_14.getDistances();
	const float* actual0_18 = input0_18.getDistances();
	const float* actual0_26 = input0_26.getDistances();
	
	for(int i = 0; i < input0_6.K; i++)
		CPPUNIT_ASSERT_EQUAL(expected0, actual0_6[i]);
	
	for(int i = 0; i < input0_14.K; i++)
		CPPUNIT_ASSERT_EQUAL(expected0, actual0_14[i]);
	
	for(int i = 0; i < input0_18.K; i++)
		CPPUNIT_ASSERT_EQUAL(expected0, actual0_18[i]);
	
	for(int i = 0; i < input0_26.K; i++)
		CPPUNIT_ASSERT_EQUAL(expected0, actual0_26[i]);
	
	//One vertex
	const Vector3f inputPointA(0.17f, -1.2f, 5.6f);
	
	std::vector<Vector3f> inputVerts1;
	inputVerts1.push_back(inputPointA);
	
	KDOP input1_6 (inputVerts1, 6);
	KDOP input1_14(inputVerts1, 14);
	KDOP input1_18(inputVerts1, 18);
	KDOP input1_26(inputVerts1, 26);
	
	float expected1_6 [6]  = {0.17f, -1.2f, 5.6f, 0.17f, -1.2f, 5.6f};
	float expected1_14[14] = {0.17f, -1.2f, 5.6f, 4.57f, 6.97f, -6.63f, -4.23, 0.17f, -1.2f, 5.6f, 4.57f, 6.97f, -6.63f, -4.23f};
	float expected1_18[18] = {0.17f, -1.2f, 5.6f, -1.03f, 5.77f, 4.4f, 1.37f, -5.43f, -6.8f, 0.17f, -1.2f, 5.6f, -1.03f, 5.77f, 4.4f, 1.37f, -5.43f, -6.8f};
	float expected1_26[26] = {0.17f, -1.2f, 5.6f, 4.57f, 6.97f, -6.63f, -4.23, -1.03f, 5.77f, 4.4f, 1.37f, -5.43f, -6.8f, 0.17f, -1.2f, 5.6f, 4.57f, 6.97f, -6.63f, -4.23, -1.03f, 5.77f, 4.4f, 1.37f, -5.43f, -6.8f};
	float delta = 0.000005f;
	
	const float* actual1_6  = input1_6.getDistances();
	const float* actual1_14 = input1_14.getDistances();
	const float* actual1_18 = input1_18.getDistances();
	const float* actual1_26 = input1_26.getDistances();
	
	for(int i = 0; i < input1_6.K; i++)
		CPPUNIT_ASSERT_DOUBLES_EQUAL(expected1_6[i], actual1_6[i], delta);
	
	for(int i = 0; i < input1_14.K; i++)
		CPPUNIT_ASSERT_DOUBLES_EQUAL(expected1_14[i], actual1_14[i], delta);
	
	for(int i = 0; i < input1_18.K; i++)
		CPPUNIT_ASSERT_DOUBLES_EQUAL(expected1_18[i], actual1_18[i], delta);
	
	for(int i = 0; i < input1_26.K; i++)
		CPPUNIT_ASSERT_DOUBLES_EQUAL(expected1_26[i], actual1_26[i], delta);
	
	//Two vertices
	const Vector3f inputPointB(-1.4f, 8.4f, -6.4f);
	
	std::vector<Vector3f> inputVerts2;
	inputVerts2.push_back(inputPointA);
	inputVerts2.push_back(inputPointB);
	
	KDOP input2_6 (inputVerts2, 6);
	KDOP input2_14(inputVerts2, 14);
	KDOP input2_18(inputVerts2, 18);
	KDOP input2_26(inputVerts2, 26);
	
	float expected2_6 [6]  = {-1.4f, -1.2f, -6.4f,  0.17f,  8.4f ,  5.6f };
	float expected2_14[14] = {-1.4f, -1.2f, -6.4f,  0.6f , -16.2f, -6.63f, -4.23f,  0.17f,  8.4f, 5.6f ,  4.57f,  6.970, 13.4f, -3.40f};
	float expected2_18[18] = {-1.4f, -1.2f, -6.4f, -1.03f, -7.8f ,  2.0f , -9.8f , -5.43f, -6.8f, 0.17f,  8.4f ,  5.600,  7.0f,  5.77f, 4.4f, 1.37f, 5.0f , 14.8f};
	float expected2_26[26] = {-1.4f, -1.2f, -6.4f,  0.6f , -16.2f, -6.63f, -4.23f, -1.03f, -7.8f, 2.0f , -9.8f , -5.430, -6.8f,  0.17f, 8.4f, 5.6f , 4.57f,  6.97f, 13.4f, -3.4f, 7.0f, 5.77f, 4.4f, 1.37f, 5.0f, 14.8f};
	
	const float* actual2_6  = input2_6.getDistances();
	const float* actual2_14 = input2_14.getDistances();
	const float* actual2_18 = input2_18.getDistances();
	const float* actual2_26 = input2_26.getDistances();
	
	for(int i = 0; i < input2_6.K; i++)
		CPPUNIT_ASSERT_DOUBLES_EQUAL(expected2_6[i], actual2_6[i], delta);
	
	for(int i = 0; i < input2_14.K; i++)
		CPPUNIT_ASSERT_DOUBLES_EQUAL(expected2_14[i], actual2_14[i], delta);
	
	for(int i = 0; i < input2_18.K; i++)
		CPPUNIT_ASSERT_DOUBLES_EQUAL(expected2_18[i], actual2_18[i], delta);
	
	for(int i = 0; i < input2_26.K; i++)
		CPPUNIT_ASSERT_DOUBLES_EQUAL(expected2_26[i], actual2_26[i], delta);
	
	//Three vertices
	const Vector3f inputPointC(2.0f, -5.7f, 9.4f);
	
	std::vector<Vector3f> inputVerts3;
	inputVerts3.push_back(inputPointA);
	inputVerts3.push_back(inputPointB);
	inputVerts3.push_back(inputPointC);
	
	KDOP input3_6 (inputVerts3, 6);
	KDOP input3_14(inputVerts3, 14);
	KDOP input3_18(inputVerts3, 18);
	KDOP input3_26(inputVerts3, 26);
	
	float expected3_6 [6]  = {-1.400f, -5.700f, -6.400f, 2.000f, 8.400f, 9.400f};
	float expected3_14[14] = {-1.400f, -5.700f, -6.400f, 0.600f, -16.200f, -13.100f, -4.230f, 2.000f, 8.400f, 9.400f, 5.700f, 17.100f, 13.400f, -1.700f};
	float expected3_18[18] = {-1.400f, -5.700f, -6.400f, -3.700f, -7.800f, 2.000f, -9.800f, -7.400f, -15.100f, 2.000f, 8.400f, 9.400f, 7.000f, 11.400f, 4.400f, 7.700f, 5.000f, 14.800f};
	float expected3_26[26] = {-1.400f, -5.700f, -6.400f, 0.600f, -16.200f, -13.100f, -4.230f, -3.700f, -7.800f, 2.000f, -9.800f, -7.400f, -15.100f, 2.000f, 8.400f, 9.400f, 5.700f, 17.100f, 13.400f, -1.700f, 7.000f, 11.400f, 4.400f, 7.700f, 5.000f, 14.800f};
	
	const float* actual3_6  = input3_6.getDistances();
	const float* actual3_14 = input3_14.getDistances();
	const float* actual3_18 = input3_18.getDistances();
	const float* actual3_26 = input3_26.getDistances();
	
	for(int i = 0; i < input3_6.K; i++)
		CPPUNIT_ASSERT_DOUBLES_EQUAL(expected3_6[i], actual3_6[i], delta);
	
	for(int i = 0; i < input3_14.K; i++)
		CPPUNIT_ASSERT_DOUBLES_EQUAL(expected3_14[i], actual3_14[i], delta);
	
	for(int i = 0; i < input3_18.K; i++)
		CPPUNIT_ASSERT_DOUBLES_EQUAL(expected3_18[i], actual3_18[i], delta);
	
	for(int i = 0; i < input3_26.K; i++)
		CPPUNIT_ASSERT_DOUBLES_EQUAL(expected3_26[i], actual3_26[i], delta);
}

void KDOPTest::testCollisions()
{
	//Point-Point
	Vector3f inputPointA(-1.1f, 3.2f, -4.5f);
	Vector3f inputPointB(7.3f, -9.2f, -6.0f);
	
	std::vector<Vector3f> pointAVerts;
	pointAVerts.push_back(inputPointA);
	std::vector<Vector3f> pointBVerts;
	pointBVerts.push_back(inputPointB);
	
	KDOP inputKDOP_pointA_6 (pointAVerts, 6);
	KDOP inputKDOP_pointA_14(pointAVerts, 14);
	KDOP inputKDOP_pointA_18(pointAVerts, 18);
	KDOP inputKDOP_pointA_26(pointAVerts, 26);
	
	KDOP inputKDOP_pointB_6 (pointBVerts, 6);
	KDOP inputKDOP_pointB_14(pointBVerts, 14);
	KDOP inputKDOP_pointB_18(pointBVerts, 18);
	KDOP inputKDOP_pointB_26(pointBVerts, 26);
	
	bool expectedPointPointAB_6  = false;
	bool expectedPointPointAB_14 = false;
	bool expectedPointPointAB_18 = false;
	bool expectedPointPointAB_26 = false;
	
	bool expectedPointPointAA_6  = true;
	bool expectedPointPointAA_14 = true;
	bool expectedPointPointAA_18 = true;
	bool expectedPointPointAA_26 = true;
	
	bool actualPointPointAB_6  = inputKDOP_pointA_6.collides (&inputKDOP_pointB_6);
	bool actualPointPointAB_14 = inputKDOP_pointA_14.collides(&inputKDOP_pointB_14);
	bool actualPointPointAB_18 = inputKDOP_pointA_18.collides(&inputKDOP_pointB_18);
	bool actualPointPointAB_26 = inputKDOP_pointA_26.collides(&inputKDOP_pointB_26);
	
	bool actualPointPointAA_6  = inputKDOP_pointA_6.collides (&inputKDOP_pointA_6);
	bool actualPointPointAA_14 = inputKDOP_pointA_14.collides(&inputKDOP_pointA_14);
	bool actualPointPointAA_18 = inputKDOP_pointA_18.collides(&inputKDOP_pointA_18);
	bool actualPointPointAA_26 = inputKDOP_pointA_26.collides(&inputKDOP_pointA_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedPointPointAB_6 , actualPointPointAB_6 );
	CPPUNIT_ASSERT_EQUAL(expectedPointPointAB_14, actualPointPointAB_14);
	CPPUNIT_ASSERT_EQUAL(expectedPointPointAB_18, actualPointPointAB_18);
	CPPUNIT_ASSERT_EQUAL(expectedPointPointAB_26, actualPointPointAB_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedPointPointAA_6 , actualPointPointAA_6 );
	CPPUNIT_ASSERT_EQUAL(expectedPointPointAA_14, actualPointPointAA_14);
	CPPUNIT_ASSERT_EQUAL(expectedPointPointAA_18, actualPointPointAA_18);
	CPPUNIT_ASSERT_EQUAL(expectedPointPointAA_26, actualPointPointAA_26);
	
	//Point-Line
	Vector3f inputPointC(3.0f, -4.6f, 2.2f);
	Vector3f inputPointD(-1.1f, 3.2f, -4.5f);
	Vector3f inputPointE(7.3f, -9.2f, -6.0f);
	
	std::vector<Vector3f> pointCVerts;
	pointCVerts.push_back(inputPointC);
	
	std::vector<Vector3f> pointDVerts;
	pointDVerts.push_back(inputPointD);
	
	std::vector<Vector3f> lineAVerts;
	lineAVerts.push_back(inputPointD);
	lineAVerts.push_back(inputPointE);
	
	KDOP inputKDOP_pointC_6 (pointCVerts, 6);
	KDOP inputKDOP_pointC_14(pointCVerts, 14);
	KDOP inputKDOP_pointC_18(pointCVerts, 18);
	KDOP inputKDOP_pointC_26(pointCVerts, 26);
	
	KDOP inputKDOP_pointD_6 (pointDVerts, 6);
	KDOP inputKDOP_pointD_14(pointDVerts, 14);
	KDOP inputKDOP_pointD_18(pointDVerts, 18);
	KDOP inputKDOP_pointD_26(pointDVerts, 26);
	
	KDOP inputKDOP_lineA_6 (lineAVerts, 6);
	KDOP inputKDOP_lineA_14(lineAVerts, 14);
	KDOP inputKDOP_lineA_18(lineAVerts, 18);
	KDOP inputKDOP_lineA_26(lineAVerts, 26);
	
	bool expectedPointLineCA_6  = false;
	bool expectedPointLineCA_14 = false;
	bool expectedPointLineCA_18 = false;
	bool expectedPointLineCA_26 = false;
	
	bool expectedPointLineDA_6  = true;
	bool expectedPointLineDA_14 = true;
	bool expectedPointLineDA_18 = true;
	bool expectedPointLineDA_26 = true;
	
	bool actualPointLineCA_6  = inputKDOP_pointC_6.collides (&inputKDOP_lineA_6);
	bool actualPointLineCA_14 = inputKDOP_pointC_14.collides(&inputKDOP_lineA_14);
	bool actualPointLineCA_18 = inputKDOP_pointC_18.collides(&inputKDOP_lineA_18);
	bool actualPointLineCA_26 = inputKDOP_pointC_26.collides(&inputKDOP_lineA_26);
	
	bool actualPointLineDA_6  = inputKDOP_pointD_6.collides (&inputKDOP_lineA_6);
	bool actualPointLineDA_14 = inputKDOP_pointD_14.collides(&inputKDOP_lineA_14);
	bool actualPointLineDA_18 = inputKDOP_pointD_18.collides(&inputKDOP_lineA_18);
	bool actualPointLineDA_26 = inputKDOP_pointD_26.collides(&inputKDOP_lineA_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedPointLineCA_6,  actualPointLineCA_6);
	CPPUNIT_ASSERT_EQUAL(expectedPointLineCA_14, actualPointLineCA_14);
	CPPUNIT_ASSERT_EQUAL(expectedPointLineCA_18, actualPointLineCA_18);
	CPPUNIT_ASSERT_EQUAL(expectedPointLineCA_26, actualPointLineCA_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedPointLineDA_6,  actualPointLineDA_6);
	CPPUNIT_ASSERT_EQUAL(expectedPointLineDA_14, actualPointLineDA_14);
	CPPUNIT_ASSERT_EQUAL(expectedPointLineDA_18, actualPointLineDA_18);
	CPPUNIT_ASSERT_EQUAL(expectedPointLineDA_26, actualPointLineDA_26);
	
	//Point-Triangle
	Vector3f inputPointF(6.2f, 2.8f, 10.5f);
	Vector3f inputPointG(-1.1f, 3.2f, -4.5f);
	Vector3f inputPointH(7.3f, -9.2f, -6.0f);
	Vector3f inputPointI(3.0f, -4.6f, 2.2f);
	
	std::vector<Vector3f> pointFVerts;
	pointFVerts.push_back(inputPointF);
	
	std::vector<Vector3f> pointGVerts;
	pointGVerts.push_back(inputPointG);
	
	std::vector<Vector3f> triangleAVerts;
	triangleAVerts.push_back(inputPointG);
	triangleAVerts.push_back(inputPointH);
	triangleAVerts.push_back(inputPointI);
	
	KDOP inputKDOP_pointF_6 (pointFVerts, 6);
	KDOP inputKDOP_pointF_14(pointFVerts, 14);
	KDOP inputKDOP_pointF_18(pointFVerts, 18);
	KDOP inputKDOP_pointF_26(pointFVerts, 26);
	
	KDOP inputKDOP_pointG_6 (pointGVerts, 6);
	KDOP inputKDOP_pointG_14(pointGVerts, 14);
	KDOP inputKDOP_pointG_18(pointGVerts, 18);
	KDOP inputKDOP_pointG_26(pointGVerts, 26);
	
	KDOP inputKDOP_triangleA_6 (triangleAVerts, 6);
	KDOP inputKDOP_triangleA_14(triangleAVerts, 14);
	KDOP inputKDOP_triangleA_18(triangleAVerts, 18);
	KDOP inputKDOP_triangleA_26(triangleAVerts, 26);
	
	bool expectedPointTriangleFA_6  = false;
	bool expectedPointTriangleFA_14 = false;
	bool expectedPointTriangleFA_18 = false;
	bool expectedPointTriangleFA_26 = false;
	
	bool expectedPointTriangleGA_6  = true;
	bool expectedPointTriangleGA_14 = true;
	bool expectedPointTriangleGA_18 = true;
	bool expectedPointTriangleGA_26 = true;
	
	bool actualPointTriangleFA_6  = inputKDOP_pointF_6.collides (&inputKDOP_triangleA_6);
	bool actualPointTriangleFA_14 = inputKDOP_pointF_14.collides(&inputKDOP_triangleA_14);
	bool actualPointTriangleFA_18 = inputKDOP_pointF_18.collides(&inputKDOP_triangleA_18);
	bool actualPointTriangleFA_26 = inputKDOP_pointF_26.collides(&inputKDOP_triangleA_26);
	
	bool actualPointTriangleGA_6  = inputKDOP_pointG_6.collides (&inputKDOP_triangleA_6);
	bool actualPointTriangleGA_14 = inputKDOP_pointG_14.collides(&inputKDOP_triangleA_14);
	bool actualPointTriangleGA_18 = inputKDOP_pointG_18.collides(&inputKDOP_triangleA_18);
	bool actualPointTriangleGA_26 = inputKDOP_pointG_26.collides(&inputKDOP_triangleA_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedPointTriangleFA_6,  actualPointTriangleFA_6);
	CPPUNIT_ASSERT_EQUAL(expectedPointTriangleFA_14, actualPointTriangleFA_14);
	CPPUNIT_ASSERT_EQUAL(expectedPointTriangleFA_18, actualPointTriangleFA_18);
	CPPUNIT_ASSERT_EQUAL(expectedPointTriangleFA_26, actualPointTriangleFA_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedPointTriangleGA_6,  actualPointTriangleGA_6);
	CPPUNIT_ASSERT_EQUAL(expectedPointTriangleGA_14, actualPointTriangleGA_14);
	CPPUNIT_ASSERT_EQUAL(expectedPointTriangleGA_18, actualPointTriangleGA_18);
	CPPUNIT_ASSERT_EQUAL(expectedPointTriangleGA_26, actualPointTriangleGA_26);
	
	//Point-Cube
	Vector3f inputPointJ(3.0f, -4.6f, 2.2f);
	Vector3f inputPointK(-1.0f, 9.8f, -5.2f);
	Vector3f inputPointL(-1.0f, 9.8f, -4.2f);
	Vector3f inputPointM(-1.0f, 7.8f, -4.2f);
	Vector3f inputPointN(2.0f, 7.8f, -4.2f);
	
	std::vector<Vector3f> pointJVerts;
	pointJVerts.push_back(inputPointJ);
	
	std::vector<Vector3f> pointKVerts;
	pointKVerts.push_back(inputPointK);
	
	std::vector<Vector3f> cubeAVerts;
	cubeAVerts.push_back(inputPointK);
	cubeAVerts.push_back(inputPointL);
	cubeAVerts.push_back(inputPointM);
	cubeAVerts.push_back(inputPointN);
	
	KDOP inputKDOP_pointJ_6 (pointJVerts, 6);
	KDOP inputKDOP_pointJ_14(pointJVerts, 14);
	KDOP inputKDOP_pointJ_18(pointJVerts, 18);
	KDOP inputKDOP_pointJ_26(pointJVerts, 26);
	
	KDOP inputKDOP_pointK_6 (pointKVerts, 6);
	KDOP inputKDOP_pointK_14(pointKVerts, 14);
	KDOP inputKDOP_pointK_18(pointKVerts, 18);
	KDOP inputKDOP_pointK_26(pointKVerts, 26);
	
	KDOP inputKDOP_cubeA_6 (cubeAVerts, 6);
	KDOP inputKDOP_cubeA_14(cubeAVerts, 14);
	KDOP inputKDOP_cubeA_18(cubeAVerts, 18);
	KDOP inputKDOP_cubeA_26(cubeAVerts, 26);
	
	bool expectedPointCubeJA_6  = false;
	bool expectedPointCubeJA_14 = false;
	bool expectedPointCubeJA_18 = false;
	bool expectedPointCubeJA_26 = false;
	
	bool expectedPointCubeKA_6  = true;
	bool expectedPointCubeKA_14 = true;
	bool expectedPointCubeKA_18 = true;
	bool expectedPointCubeKA_26 = true;
	
	bool actualPointCubeJA_6  = inputKDOP_pointJ_6.collides (&inputKDOP_cubeA_6);
	bool actualPointCubeJA_14 = inputKDOP_pointJ_14.collides(&inputKDOP_cubeA_14);
	bool actualPointCubeJA_18 = inputKDOP_pointJ_18.collides(&inputKDOP_cubeA_18);
	bool actualPointCubeJA_26 = inputKDOP_pointJ_26.collides(&inputKDOP_cubeA_26);
	
	bool actualPointCubeKA_6  = inputKDOP_pointK_6.collides (&inputKDOP_cubeA_6);
	bool actualPointCubeKA_14 = inputKDOP_pointK_14.collides(&inputKDOP_cubeA_14);
	bool actualPointCubeKA_18 = inputKDOP_pointK_18.collides(&inputKDOP_cubeA_18);
	bool actualPointCubeKA_26 = inputKDOP_pointK_26.collides(&inputKDOP_cubeA_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedPointCubeJA_6,  actualPointCubeJA_6);
	CPPUNIT_ASSERT_EQUAL(expectedPointCubeJA_14, actualPointCubeJA_14);
	CPPUNIT_ASSERT_EQUAL(expectedPointCubeJA_18, actualPointCubeJA_18);
	CPPUNIT_ASSERT_EQUAL(expectedPointCubeJA_26, actualPointCubeJA_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedPointCubeKA_6,  actualPointCubeKA_6);
	CPPUNIT_ASSERT_EQUAL(expectedPointCubeKA_14, actualPointCubeKA_14);
	CPPUNIT_ASSERT_EQUAL(expectedPointCubeKA_18, actualPointCubeKA_18);
	CPPUNIT_ASSERT_EQUAL(expectedPointCubeKA_26, actualPointCubeKA_26);
	
	//Line-Line
	std::vector<Vector3f> lineBVerts;
	lineBVerts.push_back(Vector3f(2.41616f, -1.97972f, 2.626f));
	lineBVerts.push_back(Vector3f(-2.41616f, 1.97972f, -2.626f));
	
	std::vector<Vector3f> lineCVerts;
	lineCVerts.push_back(Vector3f(0.25411f, -7.8508f, -0.72124f));
	lineCVerts.push_back(Vector3f(2.02617f, -4.92754f, -1.72528f));
	
	std::vector<Vector3f> lineDVerts;
	lineDVerts.push_back(Vector3f(1.21756f, 1.48599f, 1.61504f));
	lineDVerts.push_back(Vector3f(-1.21756f, -1.48599f, -1.61504f));
	
	KDOP inputKDOP_lineB_6 (lineBVerts, 6);
	KDOP inputKDOP_lineB_14(lineBVerts, 14);
	KDOP inputKDOP_lineB_18(lineBVerts, 18);
	KDOP inputKDOP_lineB_26(lineBVerts, 26);
	
	KDOP inputKDOP_lineC_6 (lineCVerts, 6);
	KDOP inputKDOP_lineC_14(lineCVerts, 14);
	KDOP inputKDOP_lineC_18(lineCVerts, 18);
	KDOP inputKDOP_lineC_26(lineCVerts, 26);
	
	KDOP inputKDOP_lineD_6 (lineDVerts, 6);
	KDOP inputKDOP_lineD_14(lineDVerts, 14);
	KDOP inputKDOP_lineD_18(lineDVerts, 18);
	KDOP inputKDOP_lineD_26(lineDVerts, 26);
	
	bool expectedLineLineBC_6  = false;
	bool expectedLineLineBC_14 = false;
	bool expectedLineLineBC_18 = false;
	bool expectedLineLineBC_26 = false;
	
	bool expectedLineLineBD_6  = true;
	bool expectedLineLineBD_14 = true;
	bool expectedLineLineBD_18 = true;
	bool expectedLineLineBD_26 = true;
	
	bool actualLineLineBC_6  = inputKDOP_lineB_6.collides (&inputKDOP_lineC_6);
	bool actualLineLineBC_14 = inputKDOP_lineB_14.collides(&inputKDOP_lineC_14);
	bool actualLineLineBC_18 = inputKDOP_lineB_18.collides(&inputKDOP_lineC_18);
	bool actualLineLineBC_26 = inputKDOP_lineB_26.collides(&inputKDOP_lineC_26);
	
	bool actualLineLineBD_6  = inputKDOP_lineB_6.collides (&inputKDOP_lineD_6);
	bool actualLineLineBD_14 = inputKDOP_lineB_14.collides(&inputKDOP_lineD_14);
	bool actualLineLineBD_18 = inputKDOP_lineB_18.collides(&inputKDOP_lineD_18);
	bool actualLineLineBD_26 = inputKDOP_lineB_26.collides(&inputKDOP_lineD_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedLineLineBC_6,  actualLineLineBC_6);
	CPPUNIT_ASSERT_EQUAL(expectedLineLineBC_14, actualLineLineBC_14);
	CPPUNIT_ASSERT_EQUAL(expectedLineLineBC_18, actualLineLineBC_18);
	CPPUNIT_ASSERT_EQUAL(expectedLineLineBC_26, actualLineLineBC_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedLineLineBD_6,  actualLineLineBD_6);
	CPPUNIT_ASSERT_EQUAL(expectedLineLineBD_14, actualLineLineBD_14);
	CPPUNIT_ASSERT_EQUAL(expectedLineLineBD_18, actualLineLineBD_18);
	CPPUNIT_ASSERT_EQUAL(expectedLineLineBD_26, actualLineLineBD_26);
	
	//Line-Triangle
	std::vector<Vector3f> lineEVerts;
	lineEVerts.push_back(Vector3f(3.47173f, -1.76828f, 3.08134f));
	lineEVerts.push_back(Vector3f(-1.36446f, 0.17415f, 0.0707f));
	
	std::vector<Vector3f> triangleBVerts;
	triangleBVerts.push_back(Vector3f(2.39661f, -4.65902f, -1.59556f));
	triangleBVerts.push_back(Vector3f(1.61819f, -2.56889f, -1.354f));
	triangleBVerts.push_back(Vector3f(1.29779f, -3.94503f, -0.28312f));
	
	std::vector<Vector3f> triangleCVerts;
	triangleCVerts.push_back(Vector3f(1.25973f, -3.47013f, 2.04335f));
	triangleCVerts.push_back(Vector3f(1.15458f, 0.90818f, 0.66879f));
	triangleCVerts.push_back(Vector3f(1.09922f, -0.47046f, 2.26525f));
	
	KDOP inputKDOP_lineE_6 (lineEVerts, 6);
	KDOP inputKDOP_lineE_14(lineEVerts, 14);
	KDOP inputKDOP_lineE_18(lineEVerts, 18);
	KDOP inputKDOP_lineE_26(lineEVerts, 26);
	
	KDOP inputKDOP_triangleB_6 (triangleBVerts, 6);
	KDOP inputKDOP_triangleB_14(triangleBVerts, 14);
	KDOP inputKDOP_triangleB_18(triangleBVerts, 18);
	KDOP inputKDOP_triangleB_26(triangleBVerts, 26);
	
	KDOP inputKDOP_triangleC_6 (triangleCVerts, 6);
	KDOP inputKDOP_triangleC_14(triangleCVerts, 14);
	KDOP inputKDOP_triangleC_18(triangleCVerts, 18);
	KDOP inputKDOP_triangleC_26(triangleCVerts, 26);
	
	bool expectedLineTriangleEB_6  = false;
	bool expectedLineTriangleEB_14 = false;
	bool expectedLineTriangleEB_18 = false;
	bool expectedLineTriangleEB_26 = false;
	
	bool expectedLineTriangleEC_6  = true;
	bool expectedLineTriangleEC_14 = true;
	bool expectedLineTriangleEC_18 = true;
	bool expectedLineTriangleEC_26 = true;
	
	bool actualLineTriangleEB_6  = inputKDOP_lineE_6.collides (&inputKDOP_triangleB_6);
	bool actualLineTriangleEB_14 = inputKDOP_lineE_14.collides(&inputKDOP_triangleB_14);
	bool actualLineTriangleEB_18 = inputKDOP_lineE_18.collides(&inputKDOP_triangleB_18);
	bool actualLineTriangleEB_26 = inputKDOP_lineE_26.collides(&inputKDOP_triangleB_26);
	
	bool actualLineTriangleEC_6  = inputKDOP_lineE_6.collides (&inputKDOP_triangleC_6);
	bool actualLineTriangleEC_14 = inputKDOP_lineE_14.collides(&inputKDOP_triangleC_14);
	bool actualLineTriangleEC_18 = inputKDOP_lineE_18.collides(&inputKDOP_triangleC_18);
	bool actualLineTriangleEC_26 = inputKDOP_lineE_26.collides(&inputKDOP_triangleC_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedLineTriangleEB_6,  actualLineTriangleEB_6);
	CPPUNIT_ASSERT_EQUAL(expectedLineTriangleEB_14, actualLineTriangleEB_14);
	CPPUNIT_ASSERT_EQUAL(expectedLineTriangleEB_18, actualLineTriangleEB_18);
	CPPUNIT_ASSERT_EQUAL(expectedLineTriangleEB_26, actualLineTriangleEB_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedLineTriangleEC_6,  actualLineTriangleEC_6);
	CPPUNIT_ASSERT_EQUAL(expectedLineTriangleEC_14, actualLineTriangleEC_14);
	CPPUNIT_ASSERT_EQUAL(expectedLineTriangleEC_18, actualLineTriangleEC_18);
	CPPUNIT_ASSERT_EQUAL(expectedLineTriangleEC_26, actualLineTriangleEC_26);
	
	//Line-Cube
	std::vector<Vector3f> lineFVerts;
	lineFVerts.push_back(Vector3f(2.16923f, 1.73874f, 2.26035f));
	lineFVerts.push_back(Vector3f(-0.87078f, -2.40713f, -1.77364f));
	
	std::vector<Vector3f> cubeBVerts;
	cubeBVerts.push_back(Vector3f(-4.31167f, 3.60703f, 0.94583f));
	cubeBVerts.push_back(Vector3f(-4.19776f, 3.21684f, -0.27951f));
	cubeBVerts.push_back(Vector3f(-3.82977f, 2.8319f, -0.12272f));
	cubeBVerts.push_back(Vector3f(-3.94369f, 3.22208f, 1.10262f));
	cubeBVerts.push_back(Vector3f(-3.50668f, 4.31519f, 0.79517f));
	cubeBVerts.push_back(Vector3f(-3.39277f, 3.925f, -0.43018f));
	cubeBVerts.push_back(Vector3f(-3.02478f, 3.54006f, -0.27339f));
	cubeBVerts.push_back(Vector3f(-3.1387f, 3.93025f, 0.95195f));
	
	std::vector<Vector3f> cubeCVerts;
	cubeCVerts.push_back(Vector3f(1.55895f, -1.13508f, 0.74341f));
	cubeCVerts.push_back(Vector3f(1.35127f, -1.44557f, -0.59646f));
	cubeCVerts.push_back(Vector3f(1.81921f, -0.2914f, -0.93646f));
	cubeCVerts.push_back(Vector3f(2.02689f, 0.0191f, 0.40342f));
	cubeCVerts.push_back(Vector3f(-2.02689f, -0.0191f, -0.40342f));
	cubeCVerts.push_back(Vector3f(-1.55895f, 1.13508f, -0.74341f));
	cubeCVerts.push_back(Vector3f(-1.35127f, 1.44557f, 0.59646f));
	cubeCVerts.push_back(Vector3f(-1.81921f, 0.2914f, 0.93646f));
	
	KDOP inputKDOP_lineF_6 (lineFVerts, 6);
	KDOP inputKDOP_lineF_14(lineFVerts, 14);
	KDOP inputKDOP_lineF_18(lineFVerts, 18);
	KDOP inputKDOP_lineF_26(lineFVerts, 26);
	
	KDOP inputKDOP_cubeB_6 (cubeBVerts, 6);
	KDOP inputKDOP_cubeB_14(cubeBVerts, 14);
	KDOP inputKDOP_cubeB_18(cubeBVerts, 18);
	KDOP inputKDOP_cubeB_26(cubeBVerts, 26);
	
	KDOP inputKDOP_cubeC_6 (cubeCVerts, 6);
	KDOP inputKDOP_cubeC_14(cubeCVerts, 14);
	KDOP inputKDOP_cubeC_18(cubeCVerts, 18);
	KDOP inputKDOP_cubeC_26(cubeCVerts, 26);
	
	bool expectedLineCubeFB_6  = false;
	bool expectedLineCubeFB_14 = false;
	bool expectedLineCubeFB_18 = false;
	bool expectedLineCubeFB_26 = false;
	
	bool expectedLineCubeFC_6  = true;
	bool expectedLineCubeFC_14 = true;
	bool expectedLineCubeFC_18 = true;
	bool expectedLineCubeFC_26 = true;
	
	bool actualLineCubeFB_6  = inputKDOP_lineF_6.collides (&inputKDOP_cubeB_6);
	bool actualLineCubeFB_14 = inputKDOP_lineF_14.collides(&inputKDOP_cubeB_14);
	bool actualLineCubeFB_18 = inputKDOP_lineF_18.collides(&inputKDOP_cubeB_18);
	bool actualLineCubeFB_26 = inputKDOP_lineF_26.collides(&inputKDOP_cubeB_26);
	
	bool actualLineCubeFC_6  = inputKDOP_lineF_6.collides (&inputKDOP_cubeC_6);
	bool actualLineCubeFC_14 = inputKDOP_lineF_14.collides(&inputKDOP_cubeC_14);
	bool actualLineCubeFC_18 = inputKDOP_lineF_18.collides(&inputKDOP_cubeC_18);
	bool actualLineCubeFC_26 = inputKDOP_lineF_26.collides(&inputKDOP_cubeC_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedLineCubeFB_6,  actualLineCubeFB_6);
	CPPUNIT_ASSERT_EQUAL(expectedLineCubeFB_14, actualLineCubeFB_14);
	CPPUNIT_ASSERT_EQUAL(expectedLineCubeFB_18, actualLineCubeFB_18);
	CPPUNIT_ASSERT_EQUAL(expectedLineCubeFB_26, actualLineCubeFB_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedLineCubeFC_6,  actualLineCubeFC_6);
	CPPUNIT_ASSERT_EQUAL(expectedLineCubeFC_14, actualLineCubeFC_14);
	CPPUNIT_ASSERT_EQUAL(expectedLineCubeFC_18, actualLineCubeFC_18);
	CPPUNIT_ASSERT_EQUAL(expectedLineCubeFC_26, actualLineCubeFC_26);
	
	//Triangle-Triangle
	std::vector<Vector3f> triangleDVerts;
	triangleDVerts.push_back(Vector3f(-1.10626f, 0.54072f, 0.46685f));
	triangleDVerts.push_back(Vector3f(-3.2041f, 1.11084f, -0.24841f));
	triangleDVerts.push_back(Vector3f(0.22791f, -0.04845f, -2.06917f));
	
	std::vector<Vector3f> triangleEVerts;
	triangleEVerts.push_back(Vector3f(3.28322f, -2.82146f, -0.00034f));
	triangleEVerts.push_back(Vector3f(5.14088f, -2.61298f, -0.80339f));
	triangleEVerts.push_back(Vector3f(2.31253f, -3.3921f, 1.4642f));
	
	std::vector<Vector3f> triangleFVerts;
	triangleFVerts.push_back(Vector3f(-1.36052f, -1.11336f, -0.72951f));
	triangleFVerts.push_back(Vector3f(-1.69752f, 2.69428f, -1.59045f));
	triangleFVerts.push_back(Vector3f(-0.76924f, -1.59401f, -0.08939f));
	
	KDOP inputKDOP_triangleD_6 (triangleDVerts, 6);
	KDOP inputKDOP_triangleD_14(triangleDVerts, 14);
	KDOP inputKDOP_triangleD_18(triangleDVerts, 18);
	KDOP inputKDOP_triangleD_26(triangleDVerts, 26);
	
	KDOP inputKDOP_triangleE_6 (triangleEVerts, 6);
	KDOP inputKDOP_triangleE_14(triangleEVerts, 14);
	KDOP inputKDOP_triangleE_18(triangleEVerts, 18);
	KDOP inputKDOP_triangleE_26(triangleEVerts, 26);
	
	KDOP inputKDOP_triangleF_6 (triangleFVerts, 6);
	KDOP inputKDOP_triangleF_14(triangleFVerts, 14);
	KDOP inputKDOP_triangleF_18(triangleFVerts, 18);
	KDOP inputKDOP_triangleF_26(triangleFVerts, 26);
	
	bool expectedTriangleTriangleDE_6  = false;
	bool expectedTriangleTriangleDE_14 = false;
	bool expectedTriangleTriangleDE_18 = false;
	bool expectedTriangleTriangleDE_26 = false;
	
	bool expectedTriangleTriangleDF_6  = true;
	bool expectedTriangleTriangleDF_14 = true;
	bool expectedTriangleTriangleDF_18 = true;
	bool expectedTriangleTriangleDF_26 = true;
	
	bool actualTriangleTriangleDE_6  = inputKDOP_triangleD_6.collides (&inputKDOP_triangleE_6);
	bool actualTriangleTriangleDE_14 = inputKDOP_triangleD_14.collides(&inputKDOP_triangleE_14);
	bool actualTriangleTriangleDE_18 = inputKDOP_triangleD_18.collides(&inputKDOP_triangleE_18);
	bool actualTriangleTriangleDE_26 = inputKDOP_triangleD_26.collides(&inputKDOP_triangleE_26);
	
	bool actualTriangleTriangleDF_6  = inputKDOP_triangleD_6.collides (&inputKDOP_triangleF_6);
	bool actualTriangleTriangleDF_14 = inputKDOP_triangleD_14.collides(&inputKDOP_triangleF_14);
	bool actualTriangleTriangleDF_18 = inputKDOP_triangleD_18.collides(&inputKDOP_triangleF_18);
	bool actualTriangleTriangleDF_26 = inputKDOP_triangleD_26.collides(&inputKDOP_triangleF_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedTriangleTriangleDE_6,  actualTriangleTriangleDE_6);
	CPPUNIT_ASSERT_EQUAL(expectedTriangleTriangleDE_14, actualTriangleTriangleDE_14);
	CPPUNIT_ASSERT_EQUAL(expectedTriangleTriangleDE_18, actualTriangleTriangleDE_18);
	CPPUNIT_ASSERT_EQUAL(expectedTriangleTriangleDE_26, actualTriangleTriangleDE_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedTriangleTriangleDF_6,  actualTriangleTriangleDF_6);
	CPPUNIT_ASSERT_EQUAL(expectedTriangleTriangleDF_14, actualTriangleTriangleDF_14);
	CPPUNIT_ASSERT_EQUAL(expectedTriangleTriangleDF_18, actualTriangleTriangleDF_18);
	CPPUNIT_ASSERT_EQUAL(expectedTriangleTriangleDF_26, actualTriangleTriangleDF_26);
	
	//Triangle-Cube
	std::vector<Vector3f> triangleGVerts;
	triangleGVerts.push_back(Vector3f(-0.76924f, -1.59401f, -0.08939f));
	triangleGVerts.push_back(Vector3f(-1.36052f, -1.11336f, -0.72951f));
	triangleGVerts.push_back(Vector3f(-1.69752f, 2.69428f, -1.59045f));
	
	std::vector<Vector3f> cubeDVerts;
	cubeDVerts.push_back(Vector3f(3.77965f, -5.5799f, 2.64883f));
	cubeDVerts.push_back(Vector3f(3.03598f, -5.76758f, 1.96043f));
	cubeDVerts.push_back(Vector3f(3.1395f, -5.40167f, 1.74883f));
	cubeDVerts.push_back(Vector3f(3.88318f, -5.214f, 2.43723f));
	cubeDVerts.push_back(Vector3f(1.30876f, -3.64259f, 4.78997f));
	cubeDVerts.push_back(Vector3f(0.56508f, -3.83027f, 4.10157f));
	cubeDVerts.push_back(Vector3f(0.66861f, -3.46436f, 3.88997f));
	cubeDVerts.push_back(Vector3f(1.41229f, -3.27668f, 4.57837f));
	
	std::vector<Vector3f> cubeEVerts;
	cubeEVerts.push_back(Vector3f(-0.38811f, -0.72419f, 0.37762f));
	cubeEVerts.push_back(Vector3f(-0.32274f, -1.10924f, -0.57613f));
	cubeEVerts.push_back(Vector3f(0.80002f, 0.45081f, -1.12899f));
	cubeEVerts.push_back(Vector3f(0.73466f, 0.83585f, -0.17525f));
	cubeEVerts.push_back(Vector3f(-3.5239f, 1.18352f, -0.60748f));
	cubeEVerts.push_back(Vector3f(-3.45853f, 0.79848f, -1.56122f));
	cubeEVerts.push_back(Vector3f(-2.33577f, 2.35852f, -2.11409f));
	cubeEVerts.push_back(Vector3f(-2.40114f, 2.74357f, -1.16035f));
	
	KDOP inputKDOP_triangleG_6 (triangleGVerts, 6);
	KDOP inputKDOP_triangleG_14(triangleGVerts, 14);
	KDOP inputKDOP_triangleG_18(triangleGVerts, 18);
	KDOP inputKDOP_triangleG_26(triangleGVerts, 26);
	
	KDOP inputKDOP_cubeD_6 (cubeDVerts, 6);
	KDOP inputKDOP_cubeD_14(cubeDVerts, 14);
	KDOP inputKDOP_cubeD_18(cubeDVerts, 18);
	KDOP inputKDOP_cubeD_26(cubeDVerts, 26);
	
	KDOP inputKDOP_cubeE_6 (cubeEVerts, 6);
	KDOP inputKDOP_cubeE_14(cubeEVerts, 14);
	KDOP inputKDOP_cubeE_18(cubeEVerts, 18);
	KDOP inputKDOP_cubeE_26(cubeEVerts, 26);
	
	bool expectedTriangleCubeGD_6  = false;
	bool expectedTriangleCubeGD_14 = false;
	bool expectedTriangleCubeGD_18 = false;
	bool expectedTriangleCubeGD_26 = false;
	
	bool expectedTriangleCubeGE_6  = true;
	bool expectedTriangleCubeGE_14 = true;
	bool expectedTriangleCubeGE_18 = true;
	bool expectedTriangleCubeGE_26 = true;
	
	bool actualTriangleCubeGD_6  = inputKDOP_triangleG_6.collides (&inputKDOP_cubeD_6);
	bool actualTriangleCubeGD_14 = inputKDOP_triangleG_14.collides(&inputKDOP_cubeD_14);
	bool actualTriangleCubeGD_18 = inputKDOP_triangleG_18.collides(&inputKDOP_cubeD_18);
	bool actualTriangleCubeGD_26 = inputKDOP_triangleG_26.collides(&inputKDOP_cubeD_26);
	
	bool actualTriangleCubeGE_6  = inputKDOP_triangleG_6.collides (&inputKDOP_cubeE_6);
	bool actualTriangleCubeGE_14 = inputKDOP_triangleG_14.collides(&inputKDOP_cubeE_14);
	bool actualTriangleCubeGE_18 = inputKDOP_triangleG_18.collides(&inputKDOP_cubeE_18);
	bool actualTriangleCubeGE_26 = inputKDOP_triangleG_26.collides(&inputKDOP_cubeE_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedTriangleCubeGD_6,  actualTriangleCubeGD_6);
	CPPUNIT_ASSERT_EQUAL(expectedTriangleCubeGD_14, actualTriangleCubeGD_14);
	CPPUNIT_ASSERT_EQUAL(expectedTriangleCubeGD_18, actualTriangleCubeGD_18);
	CPPUNIT_ASSERT_EQUAL(expectedTriangleCubeGD_26, actualTriangleCubeGD_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedTriangleCubeGE_6,  actualTriangleCubeGE_6);
	CPPUNIT_ASSERT_EQUAL(expectedTriangleCubeGE_14, actualTriangleCubeGE_14);
	CPPUNIT_ASSERT_EQUAL(expectedTriangleCubeGE_18, actualTriangleCubeGE_18);
	CPPUNIT_ASSERT_EQUAL(expectedTriangleCubeGE_26, actualTriangleCubeGE_26);
	
	//Cube-Cube
	std::vector<Vector3f> cubeFVerts;
	cubeFVerts.push_back(Vector3f(-0.60062f, -0.44106f, 1.295f));
	cubeFVerts.push_back(Vector3f(-0.06143f, -1.45268f, -0.34387f));
	cubeFVerts.push_back(Vector3f(0.85847f, -1.20759f, -0.19251f));
	cubeFVerts.push_back(Vector3f(0.31928f, -0.19597f, 1.44636f));
	cubeFVerts.push_back(Vector3f(-0.85847f, 1.20759f, 0.19251f));
	cubeFVerts.push_back(Vector3f(-0.31928f, 0.19597f, -1.44636f));
	cubeFVerts.push_back(Vector3f(0.60062f, 0.44106f, -1.295f));
	cubeFVerts.push_back(Vector3f(0.06143f, 1.45268f, 0.34387f));
	
	std::vector<Vector3f> cubeGVerts;
	cubeGVerts.push_back(Vector3f(3.77965f, -5.5799f, 2.64883f));
	cubeGVerts.push_back(Vector3f(3.03598f, -5.76758f, 1.96043f));
	cubeGVerts.push_back(Vector3f(3.1395f, -5.40167f, 1.74883f));
	cubeGVerts.push_back(Vector3f(3.88318f, -5.214f, 2.43723f));
	cubeGVerts.push_back(Vector3f(1.30876f, -3.64259f, 4.78997f));
	cubeGVerts.push_back(Vector3f(0.56508f, -3.83027f, 4.10157f));
	cubeGVerts.push_back(Vector3f(0.66861f, -3.46436f, 3.88997f));
	cubeGVerts.push_back(Vector3f(1.41229f, -3.27668f, 4.57837f));
	
	std::vector<Vector3f> cubeHVerts;
	cubeHVerts.push_back(Vector3f(-0.38811f, -0.72419f, 0.37762f));
	cubeHVerts.push_back(Vector3f(-0.32274f, -1.10924f, -0.57613f));
	cubeHVerts.push_back(Vector3f(0.80002f, 0.45081f, -1.12899f));
	cubeHVerts.push_back(Vector3f(0.73466f, 0.83585f, -0.17525f));
	cubeHVerts.push_back(Vector3f(-3.5239f, 1.18352f, -0.60748f));
	cubeHVerts.push_back(Vector3f(-3.45853f, 0.79848f, -1.56122f));
	cubeHVerts.push_back(Vector3f(-2.33577f, 2.35852f, -2.11409f));
	cubeHVerts.push_back(Vector3f(-2.40114f, 2.74357f, -1.16035f));
	
	KDOP inputKDOP_cubeF_6 (cubeFVerts, 6);
	KDOP inputKDOP_cubeF_14(cubeFVerts, 14);
	KDOP inputKDOP_cubeF_18(cubeFVerts, 18);
	KDOP inputKDOP_cubeF_26(cubeFVerts, 26);
	
	KDOP inputKDOP_cubeG_6 (cubeGVerts, 6);
	KDOP inputKDOP_cubeG_14(cubeGVerts, 14);
	KDOP inputKDOP_cubeG_18(cubeGVerts, 18);
	KDOP inputKDOP_cubeG_26(cubeGVerts, 26);
	
	KDOP inputKDOP_cubeH_6 (cubeHVerts, 6);
	KDOP inputKDOP_cubeH_14(cubeHVerts, 14);
	KDOP inputKDOP_cubeH_18(cubeHVerts, 18);
	KDOP inputKDOP_cubeH_26(cubeHVerts, 26);
	
	bool expectedCubeCubeFG_6  = false;
	bool expectedCubeCubeFG_14 = false;
	bool expectedCubeCubeFG_18 = false;
	bool expectedCubeCubeFG_26 = false;
	
	bool expectedCubeCubeFH_6  = true;
	bool expectedCubeCubeFH_14 = true;
	bool expectedCubeCubeFH_18 = true;
	bool expectedCubeCubeFH_26 = true;
	
	bool actualCubeCubeFG_6  = inputKDOP_cubeF_6.collides (&inputKDOP_cubeG_6);
	bool actualCubeCubeFG_14 = inputKDOP_cubeF_14.collides(&inputKDOP_cubeG_14);
	bool actualCubeCubeFG_18 = inputKDOP_cubeF_18.collides(&inputKDOP_cubeG_18);
	bool actualCubeCubeFG_26 = inputKDOP_cubeF_26.collides(&inputKDOP_cubeG_26);
	
	bool actualCubeCubeFH_6  = inputKDOP_cubeF_6.collides (&inputKDOP_cubeH_6);
	bool actualCubeCubeFH_14 = inputKDOP_cubeF_14.collides(&inputKDOP_cubeH_14);
	bool actualCubeCubeFH_18 = inputKDOP_cubeF_18.collides(&inputKDOP_cubeH_18);
	bool actualCubeCubeFH_26 = inputKDOP_cubeF_26.collides(&inputKDOP_cubeH_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedCubeCubeFG_6,  actualCubeCubeFG_6);
	CPPUNIT_ASSERT_EQUAL(expectedCubeCubeFG_14, actualCubeCubeFG_14);
	CPPUNIT_ASSERT_EQUAL(expectedCubeCubeFG_18, actualCubeCubeFG_18);
	CPPUNIT_ASSERT_EQUAL(expectedCubeCubeFG_26, actualCubeCubeFG_26);
	
	CPPUNIT_ASSERT_EQUAL(expectedCubeCubeFH_6,  actualCubeCubeFH_6);
	CPPUNIT_ASSERT_EQUAL(expectedCubeCubeFH_14, actualCubeCubeFH_14);
	CPPUNIT_ASSERT_EQUAL(expectedCubeCubeFH_18, actualCubeCubeFH_18);
	CPPUNIT_ASSERT_EQUAL(expectedCubeCubeFH_26, actualCubeCubeFH_26);
}
