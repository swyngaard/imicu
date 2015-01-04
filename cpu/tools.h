
#ifndef __TOOLS_H__
#define __TOOLS_H__

namespace pilar
{
	class Vector3i;
	
	class Vector3f
	{
	public:
		float x;
		float y;
		float z;
		
		Vector3f();
		Vector3f(float x, float y, float z);
		Vector3f(const Vector3i& v);
		
		Vector3f& operator= (Vector3f v);
		Vector3f& operator= (Vector3i v);
		
		Vector3f operator+ (Vector3f v);
		Vector3f operator- (Vector3f v);
		Vector3f operator* (float value);
		Vector3f operator/ (float value);
		Vector3f& operator+= (Vector3f v);
		Vector3f& operator-= (Vector3f v);
		Vector3f& operator*= (float value);
		Vector3f& operator/= (float value);
		Vector3f operator- ();
		
		float dot(const Vector3f& v);
		float dot(const Vector3i& v);
		Vector3f cross(const Vector3f& v);
		Vector3f cross(const Vector3i& v);
		
		static float determinant(const Vector3f& a, const Vector3f& b, const Vector3f& c);
		static Vector3f random(float low, float high);
		
		float length();
		float length_inverse();
		float length_sqr();
		void unitize();
		Vector3f unit();
	};

	class Vector3i
	{
	public:
		int x;
		int y;
		int z;
		
		Vector3i();
		Vector3i(int x, int y, int z);
		
		bool operator== (const Vector3i& v) const;
		bool operator!= (const Vector3i& v) const;
		
		Vector3i& operator= (Vector3i v);
		Vector3i operator+ (Vector3i v);
		
		int dot(const Vector3i& v);
		float dot(const Vector3f& v);
		
		Vector3i cross(const Vector3i& v);
		Vector3f cross(const Vector3f& v);
		
		static int determinant(const Vector3i& a, const Vector3i& b, const Vector3i& c);
	};
}

#endif

