
#include <cppunit/ui/text/TestRunner.h>

#include "testTools.h"
#include "testKDOP.h"
#include "testTree.h"

CPPUNIT_TEST_SUITE_REGISTRATION( Vector3fTest );
CPPUNIT_TEST_SUITE_REGISTRATION( Vector3iTest );
CPPUNIT_TEST_SUITE_REGISTRATION( KDOPTest );
CPPUNIT_TEST_SUITE_REGISTRATION( TreeTest );

int main(int argc, char **argv)
{
	CppUnit::TextTestRunner runner;
	
	CppUnit::TestFactoryRegistry &registry = CppUnit::TestFactoryRegistry::getRegistry();
	
	runner.addTest(registry.makeTest());
	
	bool wasSuccessful = runner.run("", false);
	
	//Return error code if a test fails
	return wasSuccessful ? 0 : 1;
}
