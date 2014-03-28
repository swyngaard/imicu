
#ifndef __TEST_TOOLS_H__
#define __TEST_TOOLS_H__

#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>
#include "tools.h"

class Vector3iTest : public CppUnit::TestFixture
{	
public:
	void setUp(){}
	void tearDown(){}
	
	void testConstructor();
	void testEquals();
	void testAssignment();
	void testAddition();
	void testDotProduct();
	void testCrossProduct();
	void testDeterminant();
	
	CPPUNIT_TEST_SUITE( Vector3iTest );
	
	CPPUNIT_TEST( testConstructor );
	CPPUNIT_TEST( testEquals );
	CPPUNIT_TEST( testAssignment );
	CPPUNIT_TEST( testAddition );
	CPPUNIT_TEST( testDotProduct );
	CPPUNIT_TEST( testCrossProduct );
	CPPUNIT_TEST( testDeterminant );
	
	CPPUNIT_TEST_SUITE_END();
};

class Vector3fTest : public CppUnit::TestFixture
{
public:
	void setUp(){}
	void tearDown(){}
	
	void testConstructor();
	void testAssignment();
	void testAddition();
	void testSubtraction();
	void testScaling();
	void testNegation();
	void testLengthSquared();
	void testLength();
	void testLengthInverse();
	void testUnitize();
	void testDotProduct();
	void testCrossProduct();
	void testDeterminant();
	
	CPPUNIT_TEST_SUITE( Vector3fTest );
	
	CPPUNIT_TEST( testConstructor );
	CPPUNIT_TEST( testAssignment );
	CPPUNIT_TEST( testAddition );
	CPPUNIT_TEST( testSubtraction );
	CPPUNIT_TEST( testScaling );
	CPPUNIT_TEST( testNegation );
	CPPUNIT_TEST( testLengthSquared );
	CPPUNIT_TEST( testLength );
	CPPUNIT_TEST( testLengthInverse );
	CPPUNIT_TEST( testDotProduct );
	CPPUNIT_TEST( testCrossProduct );
	CPPUNIT_TEST( testDeterminant );
	
	CPPUNIT_TEST_SUITE_END();
};

#endif
