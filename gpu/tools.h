
#ifndef __TOOLS_H__
#define __TOOLS_H__

#include <math.h>

namespace pilar
{
	class Vector3f
	{
	public:
		float x;
		float y;
		float z;
		
		Vector3f();
		Vector3f(float x, float y, float z);
		
		Vector3f& operator= (Vector3f v);
		Vector3f operator+ (Vector3f v);
		Vector3f operator- (Vector3f v);
		Vector3f operator* (float value);
		Vector3f operator/ (float value);
		Vector3f& operator+= (Vector3f v);
		Vector3f& operator-= (Vector3f v);
		Vector3f& operator*= (float value);
		Vector3f& operator/= (float value);
		Vector3f operator- ();
		//TODO add dot product operator
		
		float length();
		float length_inverse();
		float length_sqr();
		void unitize();
		Vector3f unit();
	};
}

#endif

