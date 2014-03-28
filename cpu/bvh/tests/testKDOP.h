
#ifndef __TEST_KDOP_H__
#define __TEST_KDOP_H__

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include "kdop.h"

class KDOPTest : public CppUnit::TestFixture
{	
public:
	void setUp(){}
	void tearDown(){}
	
	void testNormals();
	void testBuildDegenerateMatrix();
	void testSetDegenerateMatrix();
	void testPlaneDistances();
	void testCollisions();
	
	
	CPPUNIT_TEST_SUITE( KDOPTest );
	
	CPPUNIT_TEST( testNormals );
	CPPUNIT_TEST( testBuildDegenerateMatrix );
	CPPUNIT_TEST( testSetDegenerateMatrix );
	CPPUNIT_TEST( testPlaneDistances );
	CPPUNIT_TEST( testCollisions );
	
	CPPUNIT_TEST_SUITE_END();
};

#endif
