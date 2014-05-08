
#ifndef __TEST_TREE_H__
#define __TEST_TREE_H__

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include "tree.h"


class TreeTest : public CppUnit::TestFixture
{	
public:
	void setUp(){}
	void tearDown(){}
	
	void testBuildTree();
	void testKDOPTree();
	void testUpdateTree();
	
	CPPUNIT_TEST_SUITE( TreeTest );
	
	CPPUNIT_TEST( testBuildTree );
	CPPUNIT_TEST( testKDOPTree );
	CPPUNIT_TEST( testUpdateTree );
	
	CPPUNIT_TEST_SUITE_END();
	
protected:
	void checkTreeDepths(int numLeaves, std::vector<int>& expected);
	void checkTreeDistances(std::vector< std::vector<Vector3f> >& verts, 
							float expected_6 [][6],
							float expected_14[][14],
							float expected_18[][18],
							float expected_26[][26]);
	void checkTreeUpdate(std::vector< std::vector<Vector3f> >& verts,
						 std::vector< std::vector<Vector3f> >& updatedVerts,
						 std::vector<bool>& flags,
						 float expected_6 [][6],
						 float expected_14[][14],
						 float expected_18[][18],
						 float expected_26[][26]);
};

#endif
