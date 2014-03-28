
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
	
	CPPUNIT_TEST_SUITE( TreeTest );
	
	CPPUNIT_TEST( testBuildTree );
	
	CPPUNIT_TEST_SUITE_END();
	
protected:
	void buildTree(int numLeaves, std::vector<int>& expected);
};

#endif
