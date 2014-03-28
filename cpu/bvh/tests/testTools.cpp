
//~ #include <cppunit/ui/text/TestRunner.h>

#include "testTools.h"

void Vector3fTest::testConstructor()
{
	//Test default constructor
	Vector3f inputA;
	
	CPPUNIT_ASSERT_EQUAL(inputA.x, 0.0f);
	CPPUNIT_ASSERT_EQUAL(inputA.y, 0.0f);
	CPPUNIT_ASSERT_EQUAL(inputA.z, 0.0f);
	
	//Test assignment constructor
	float expectedX = -5.346456f;
	float expectedY = 7.5475673f;
	float expectedZ = 18.098352f;
	
	Vector3f inputB(expectedX, expectedY, expectedZ);
	
	CPPUNIT_ASSERT_EQUAL(inputB.x, expectedX);
	CPPUNIT_ASSERT_EQUAL(inputB.y, expectedY);
	CPPUNIT_ASSERT_EQUAL(inputB.z, expectedZ);
	
	//Test integer constructor
	Vector3i inputC(-5, 7, 18);
	
	float expectedXFloat = -5.0f;
	float expectedYFloat = 7.0f;
	float expectedZFloat = 18.0f;
	
	Vector3f actual(inputC);
	
	CPPUNIT_ASSERT_EQUAL(expectedXFloat, actual.x);
	CPPUNIT_ASSERT_EQUAL(expectedYFloat, actual.y);
	CPPUNIT_ASSERT_EQUAL(expectedZFloat, actual.z);
}

void Vector3fTest::testAssignment()
{
	Vector3f inputA;
	
	float expectedX = -5.346456f;
	float expectedY = 7.5475673f;
	float expectedZ = 18.098352f;
	
	Vector3f expectedFloat(expectedX, expectedY, expectedZ);
	
	inputA = expectedFloat;
	
	CPPUNIT_ASSERT_EQUAL(expectedFloat.x, inputA.x);
	CPPUNIT_ASSERT_EQUAL(expectedFloat.y, inputA.y);
	CPPUNIT_ASSERT_EQUAL(expectedFloat.z, inputA.z);
	
	float expectedXInt = -5.0f;
	float expectedYInt = 7.0f;
	float expectedZInt = 18.0f;
	
	Vector3i inputB((int)expectedXInt, (int)expectedYInt, (int)expectedZInt);
	
	inputA = inputB;
	
	CPPUNIT_ASSERT_EQUAL(expectedXInt, inputA.x);
	CPPUNIT_ASSERT_EQUAL(expectedYInt, inputA.y);
	CPPUNIT_ASSERT_EQUAL(expectedZInt, inputA.z);
}

void Vector3fTest::testAddition()
{
	Vector3f inputA(-3.3f, 2.2f, 8.8f);
	Vector3f inputB(6.6f, 43.43f, -12.12f);
	
	Vector3f expected(3.3f, 45.63f, -3.32f);
	
	Vector3f actual = inputA + inputB;
	float delta = 0.000005f;
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.x, actual.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.y, actual.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.z, actual.z, delta);
	
	inputA += inputB;
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.x, inputA.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.y, inputA.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.z, inputA.z, delta);
}

void Vector3fTest::testSubtraction()
{
	Vector3f inputA(-3.3f, 2.2f, 8.8f);
	Vector3f inputB(6.6f, 43.43f, -12.12f);
	
	Vector3f expected(-9.9f, -41.23f, 20.92f);
	
	Vector3f actual = inputA - inputB;
	float delta = 0.000005f;
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.x, actual.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.y, actual.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.z, actual.z, delta);
	
	inputA -= inputB;
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.x, inputA.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.y, inputA.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.z, inputA.z, delta);
}

void Vector3fTest::testScaling()
{
	Vector3f inputA(-3.3f, 2.2f, 8.8f);
	float inputFactor = 5.64f;
	
	Vector3f expectedA(-18.612f, 12.408f, 49.632f);
	float delta = 0.000005f;
	
	Vector3f actualA = inputA * inputFactor;
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA.x, actualA.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA.y, actualA.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA.z, actualA.z, delta);
	
	inputA *= inputFactor;
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA.x, inputA.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA.y, inputA.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA.z, inputA.z, delta);
	
	Vector3f inputB(-3.3f, 2.2f, 8.8f);
	
	Vector3f expectedB(-0.585106382978723f, 0.390070921985816f, 1.56028368794326f);
	
	Vector3f actualB = inputB / inputFactor;
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedB.x, actualB.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedB.y, actualB.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedB.z, actualB.z, delta);
	
	inputB /= inputFactor;
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedB.x, inputB.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedB.y, inputB.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedB.z, inputB.z, delta);
}

void Vector3fTest::testNegation()
{
	Vector3f input(-3.3f, 2.2f, 8.8f);
	
	Vector3f expected(3.3f, -2.2f, -8.8f);
	
	Vector3f actual = -input;
	
	CPPUNIT_ASSERT_EQUAL(expected.x, actual.x);
	CPPUNIT_ASSERT_EQUAL(expected.y, actual.y);
	CPPUNIT_ASSERT_EQUAL(expected.z, actual.z);
}

void Vector3fTest::testLengthSquared()
{
	Vector3f inputA(-3.3f, 2.2f, 8.8f);
	
	float expectedA = 93.17f;
	float delta = 0.000005f;
	
	float actualA = inputA.length_sqr();
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA, actualA, delta);
	
	Vector3f inputB;
	
	float expectedB = 0.0f;
	
	float actualB = inputB.length_sqr();
	
	CPPUNIT_ASSERT_EQUAL(expectedB, actualB);
}

void Vector3fTest::testLength()
{
	Vector3f inputA(-3.3f, 2.2f, 8.8f);
	
	float expectedA = 9.652460826f;
	float delta = 0.000005f;
	
	float actualA = inputA.length();
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA, actualA, delta);
	
	Vector3f inputB;
	
	float expectedB = 0.0f;
	
	float actualB = inputB.length();
	
	CPPUNIT_ASSERT_EQUAL(expectedB, actualB);
}

void Vector3fTest::testLengthInverse()
{
	Vector3f inputA(-3.3f, 2.2f, 8.8f);
	
	float expectedA = 0.103600524f;
	float largeDelta = 0.0002f;
	float smallDelta = 0.000005f;
	
	float actualA = inputA.length_inverse();
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA, actualA, largeDelta);
	CPPUNIT_ASSERT_ASSERTION_FAIL( CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA, actualA, smallDelta) );
	
	Vector3f inputB;
	
	float expectedB = 0.0f;
	
	float actualB = inputB.length_inverse();
	
	CPPUNIT_ASSERT_EQUAL(expectedB, actualB);
}

void Vector3fTest::testUnitize()
{
	Vector3f inputA(-3.3f, 2.2f, 8.8f);
	
	Vector3f expectedA(-0.341881729f, 0.227921152, 0.911684611f);
	float delta = 0.000005f;
	
	Vector3f actualA = inputA.unit();
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA.x, actualA.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA.y, actualA.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA.z, actualA.z, delta);
	
	inputA.unitize();
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA.x, inputA.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA.y, inputA.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedA.z, inputA.z, delta);
	
	Vector3f inputB;
	
	Vector3f expectedB;
	
	Vector3f actualB = inputB.unit();
	
	CPPUNIT_ASSERT_EQUAL(expectedB.x, actualB.x);
	CPPUNIT_ASSERT_EQUAL(expectedB.y, actualB.y);
	CPPUNIT_ASSERT_EQUAL(expectedB.z, actualB.z);
	
	inputB.unitize();
	
	CPPUNIT_ASSERT_EQUAL(expectedB.x, inputB.x);
	CPPUNIT_ASSERT_EQUAL(expectedB.y, inputB.y);
	CPPUNIT_ASSERT_EQUAL(expectedB.z, inputB.z);
}

void Vector3fTest::testDotProduct()
{
	Vector3f inputA(-3.3f, 2.2f, 8.8f);
	Vector3f inputB(-5.0f, 7.0f, 18.0f);
	
	float expected = 190.3f;
	float largeDelta = 0.0002f;
	float smallDelta = 0.000005f;
	
	float actualA = inputA.dot(inputB);
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actualA, largeDelta);
	CPPUNIT_ASSERT_ASSERTION_FAIL( CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actualA, smallDelta) );
	
	Vector3i inputC(-5, 7, 18);
	
	float actualB = inputA.dot(inputC);
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actualB, largeDelta);
	CPPUNIT_ASSERT_ASSERTION_FAIL( CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actualB, smallDelta) );
}

void Vector3fTest::testCrossProduct()
{
	Vector3f inputA(-3.3f, 2.2f, 8.8f);
	Vector3f inputB(-5.0f, 7.0f, 18.0f);
	
	Vector3f expected(-22.0f, 15.4f, -12.1f);
	float delta = 0.000005f;
	
	Vector3f actualA = inputA.cross(inputB);
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.x, actualA.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.y, actualA.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.z, actualA.z, delta);
	
	Vector3i inputC(-5, 7, 18);
	
	Vector3f actualB = inputA.cross(inputC);
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.x, actualB.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.y, actualB.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected.z, actualB.z, delta);
}

void Vector3fTest::testDeterminant()
{
	Vector3f inputA(-3.3f, 2.2f, 8.8f);
	Vector3f inputB(-5.0f, 7.0f, 18.0f);
	Vector3f inputC(6.6f, 43.43f, -12.12f);
	
	float expected = 670.274f;
	float largeDelta = 0.0002f;
	float smallDelta = 0.000005f;
	
	float actual = Vector3f::determinant(inputA, inputB, inputC);
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, largeDelta);
	CPPUNIT_ASSERT_ASSERTION_FAIL( CPPUNIT_ASSERT_DOUBLES_EQUAL(expected, actual, smallDelta) );
}

void Vector3iTest::testConstructor()
{
	//Test default constructor
	Vector3i inputA;
	
	CPPUNIT_ASSERT_EQUAL(inputA.x, 0);
	CPPUNIT_ASSERT_EQUAL(inputA.y, 0);
	CPPUNIT_ASSERT_EQUAL(inputA.z, 0);
	
	//Test assignment constructor
	int expectedX = -5;
	int expectedY = 7;
	int expectedZ = 18;
	
	Vector3i inputB(expectedX, expectedY, expectedZ);
	
	CPPUNIT_ASSERT_EQUAL(inputB.x, expectedX);
	CPPUNIT_ASSERT_EQUAL(inputB.y, expectedY);
	CPPUNIT_ASSERT_EQUAL(inputB.z, expectedZ);
}

void Vector3iTest::testEquals()
{
	Vector3i inputA(-1, 22, -90);
	Vector3i inputB(-1, 22, -90);
	Vector3i inputC(42, -32, 0);
	
	CPPUNIT_ASSERT(inputA == inputB);
	CPPUNIT_ASSERT(inputA != inputC);
	
	CPPUNIT_ASSERT_ASSERTION_FAIL( CPPUNIT_ASSERT(inputA != inputB) );
	CPPUNIT_ASSERT_ASSERTION_FAIL( CPPUNIT_ASSERT(inputA == inputC) );
}

void Vector3iTest::testAssignment()
{
	Vector3i inputA;
	Vector3i inputB(-9, 4, -5);
	
	inputA = inputB;
	
	CPPUNIT_ASSERT(inputA == inputB);
}

void Vector3iTest::testAddition()
{
	Vector3i inputA(-3, 2, 8);
	Vector3i inputB(6, 43, -12);
	
	Vector3i expected(3, 45, -4);
	
	Vector3i actual = inputA + inputB;
	
	CPPUNIT_ASSERT(expected == actual);
}

void Vector3iTest::testDotProduct()
{
	Vector3i inputA(-3, 2, 8);
	Vector3i inputB(6, 43, -12);
	
	int expectedInt = -28;
	
	int actualInt = inputA.dot(inputB);
	
	CPPUNIT_ASSERT_EQUAL(expectedInt, actualInt);
	
	Vector3f inputC(6.6f, 43.43f, -12.12);
	
	float expectedFloat = -29.9f;
	float delta = 0.000005f;
	
	float actualFloat = inputA.dot(inputC);
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedFloat, actualFloat, delta);
}

void Vector3iTest::testCrossProduct()
{
	Vector3i inputA(-3, 2, 8);
	Vector3i inputB(6, 43, -12);
	
	Vector3i expectedInt(-368, 12, -141);
	
	Vector3i actualInt = inputA.cross(inputB);
	
	CPPUNIT_ASSERT(expectedInt == actualInt);
	
	Vector3f inputC(6.6f, 43.43f, -12.12f);
	
	Vector3f expectedFloat(-371.68f, 16.44f, -143.49f);
	float delta = 0.000005f;
	
	Vector3f actualFloat = inputA.cross(inputC);
	
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedFloat.x, actualFloat.x, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedFloat.y, actualFloat.y, delta);
	CPPUNIT_ASSERT_DOUBLES_EQUAL(expectedFloat.z, actualFloat.z, delta);
}

void Vector3iTest::testDeterminant()
{
	Vector3i inputA(-3, 2, 8);
	Vector3i inputB(6, 43, -12);
	Vector3i inputC(-1, 22, -90);
	
	int expected = 13322;
	
	int actual = Vector3i::determinant(inputA, inputB, inputC);
	
	CPPUNIT_ASSERT_EQUAL(expected, actual);
}

//~ CPPUNIT_TEST_SUITE_REGISTRATION( Vector3fTest );
//~ CPPUNIT_TEST_SUITE_REGISTRATION( Vector3iTest );
//~ 
//~ int main(int argc, char **argv)
//~ {
	//~ CppUnit::TextTestRunner runner;
	//~ 
	//~ CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
	//~ 
	//~ runner.addTest(registry.makeTest());
	//~ 
	//~ bool wasSuccessful = runner.run("", false);
	//~ 
	//~ //Return error code if a test fails
	//~ return wasSuccessful ? 0 : 1;
//~ }
