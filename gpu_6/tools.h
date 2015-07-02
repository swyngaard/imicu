
#ifndef __TOOLS_H__
#define __TOOLS_H__

#include <cuda_runtime.h>

namespace pilar
{
	class Vector3i;

	class Vector3f
	{
	public:
		float x;
		float y;
		float z;
		
		__host__ __device__ Vector3f();
		__host__ __device__ Vector3f(float x, float y, float z);
		__host__ __device__ Vector3f(const Vector3i &v);
		
		__host__ __device__ Vector3f& operator= (Vector3f v);
		__host__ __device__ Vector3f& operator= (Vector3i v);

		__host__ __device__ Vector3f operator+ (Vector3f v);
		__host__ __device__ Vector3f operator- (Vector3f v);
		__host__ __device__ Vector3f operator* (float value);
		__host__ __device__ Vector3f operator/ (float value);
		__host__ __device__ Vector3f& operator+= (Vector3f v);
		__host__ __device__ Vector3f& operator-= (Vector3f v);
		__host__ __device__ Vector3f& operator*= (float value);
		__host__ __device__ Vector3f& operator/= (float value);
		__host__ __device__ Vector3f operator- ();
		
		__host__ __device__ float dot(const Vector3f &v);
		__host__ __device__ float dot(const Vector3i &v);
		__host__ __device__ Vector3f cross(const Vector3f &v);
		__host__ __device__ Vector3f cross(const Vector3i &v);

		__host__ __device__ static float determinant(const Vector3f &a, const Vector3f &b, const Vector3f &c);
//		__host__ __device__ static Vector3f random(float low, float high);

		__host__ __device__ float length();
		__host__ __device__ float length_inverse();
		__host__ __device__ float length_sqr();
		__host__ __device__ void unitize();
		__host__ __device__ Vector3f unit();
	};

	class Vector3i
	{
	public:
		int x;
		int y;
		int z;

		__host__ __device__ Vector3i();
		__host__ __device__ Vector3i(int x, int y, int z);

		__host__ __device__ bool operator== (const Vector3i &v) const;
		__host__ __device__ bool operator!= (const Vector3i &v) const;

		__host__ __device__ Vector3i& operator= (Vector3i v);
		__host__ __device__ Vector3i operator+ (Vector3i v);

		__host__ __device__ int dot(const Vector3i &v);
		__host__ __device__ float dot(const Vector3f &v);

		__host__ __device__ Vector3i cross(const Vector3i &v);
		__host__ __device__ Vector3f cross(const Vector3f &v);

		__host__ __device__ static int determinant(const Vector3i &a, const Vector3i &b, const Vector3i &c);
	};
}

#endif

