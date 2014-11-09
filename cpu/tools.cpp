
#include <math.h>

#include "tools.h"

namespace pilar
{
	Vector3f::Vector3f()
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
	}

	Vector3f::Vector3f(float x, float y, float z)
	{
		this->x = x;
		this->y = y;
		this->z = z;	
	}

	Vector3f::Vector3f(const Vector3i& v)
	{
		this->x = float(v.x);
		this->y = float(v.y);
		this->z = float(v.z);
	}

	Vector3f& Vector3f::operator= (Vector3f v)			// operator= sets values of v to this Vector3f. example: v1 = v2 means that values of v2 are set onto v1
	{
		x = v.x;
		y = v.y;
		z = v.z;
		
		return *this;
	}

	Vector3f& Vector3f::operator= (Vector3i v)
	{
		x = float(v.x);
		y = float(v.y);
		z = float(v.z);
		
		return *this;
	}

	Vector3f Vector3f::operator+ (Vector3f v)				// operator+ is used to add two Vector3f's. operator+ returns a new Vector3f
	{
		return Vector3f(x + v.x, y + v.y, z + v.z);
	}


	Vector3f Vector3f::operator- (Vector3f v)				// operator- is used to take difference of two Vector3f's. operator- returns a new Vector3f
	{
		return Vector3f(x - v.x, y - v.y, z - v.z);
	}

	Vector3f Vector3f::operator* (float value)			// operator* is used to scale a Vector3f by a value. This value multiplies the Vector3f's x, y and z.
	{
		return Vector3f(x * value, y * value, z * value);
	}

	Vector3f Vector3f::operator/ (float value)			// operator/ is used to scale a Vector3f by a value. This value divides the Vector3f's x, y and z.
	{
		return Vector3f(x / value, y / value, z / value);
	}

	Vector3f& Vector3f::operator+= (Vector3f v)			// operator+= is used to add another Vector3f to this Vector3f.
	{
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	Vector3f& Vector3f::operator-= (Vector3f v)			// operator-= is used to subtract another Vector3f from this Vector3f.
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	Vector3f& Vector3f::operator*= (float value)			// operator*= is used to scale this Vector3f by a value.
	{
		x *= value;
		y *= value;
		z *= value;
		return *this;
	}

	Vector3f& Vector3f::operator/= (float value)			// operator/= is used to scale this Vector3f by a value.
	{
		x /= value;
		y /= value;
		z /= value;
		return *this;
	}


	Vector3f Vector3f::operator- ()						// operator- is used to set this Vector3f's x, y, and z to the negative of them.
	{
		return Vector3f(-x, -y, -z);
	}

	float Vector3f::length()								// length() returns the length of this Vector3f
	{
		return sqrtf(length_sqr());
	}

	float Vector3f::length_sqr()							//length_sqr()return the squared length of this Vector3f
	{
		return x*x + y*y + z*z;
	}

	float Vector3f::length_inverse()
	{
		float number = length_sqr();
		
		if(number != 0)
		{
			float xhalf = 0.5f*number;
			int i = *(int*)&number; // get bits for floating value
			i = 0x5f375a86- (i>>1); // gives initial guess y0
			number = *(float*)&i; // convert bits back to float
			number = number*(1.5f-xhalf*number*number); // Newton step, repeating increases accuracy
		}
		
		return number;
	}

	void Vector3f::unitize()								// unitize() normalizes this Vector3f that its direction remains the same but its length is 1.
	{
		float length = this->length();

		if (length == 0)
			return;

		x /= length;
		y /= length;
		z /= length;
	}

	Vector3f Vector3f::unit()								// unit() returns a new Vector3f. The returned value is a unitized version of this Vector3f.
	{
		float length = this->length();

		if (length == 0)
			return *this;

		return Vector3f(x / length, y / length, z / length);
	}

	float Vector3f::dot(const Vector3f& v)
	{
		return x*v.x + y*v.y + z*v.z; 
	}

	float Vector3f::dot(const Vector3i& v)
	{
		return x*v.x + y*v.y + z*v.z; 
	}

	Vector3f Vector3f::cross(const Vector3f& v)
	{
		return Vector3f(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
	}

	Vector3f Vector3f::cross(const Vector3i& v)
	{
		return Vector3f(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
	}

	float Vector3f::determinant(const Vector3f& a, const Vector3f& b, const Vector3f& c)
	{
		float ff = a.x * b.y * c.z - a.x * b.z * c.y;
		float ss = a.y * b.z * c.x - a.y * b.x * c.z;
		float tt = a.z * b.x * c.y - a.z * b.y * c.x;
		
		return ff + ss + tt;
	}

	Vector3i::Vector3i()
	{
		x = 0;
		y = 0;
		z = 0;
	}

	Vector3i::Vector3i(int x, int y, int z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	bool Vector3i::operator== (const Vector3i& v) const
	{
		return this->x == v.x && this->y == v.y && this->z == v.z;
	}

	bool Vector3i::operator!= (const Vector3i& v) const
	{
		return !(*this == v);
	}

	Vector3i& Vector3i::operator= (Vector3i v)			// operator= sets values of v to this Vector3i. example: v1 = v2 means that values of v2 are set onto v1
	{
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}

	Vector3i Vector3i::operator+ (Vector3i v)				// operator+ is used to add two Vector3i's. operator+ returns a new Vector3i
	{
		return Vector3i(x + v.x, y + v.y, z + v.z);
	}

	int Vector3i::dot(const Vector3i& v)
	{
		return x*v.x + y*v.y + z*v.z;
	}

	float Vector3i::dot(const Vector3f& v)
	{
		return x*v.x + y*v.y + z*v.z;
	}

	Vector3i Vector3i::cross(const Vector3i& v)
	{
		return Vector3i(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
	}

	Vector3f Vector3i::cross(const Vector3f& v)
	{
		return Vector3f(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
	}

	int Vector3i::determinant(const Vector3i& a, const Vector3i& b, const Vector3i& c)
	{
		int ff = a.x * b.y * c.z - a.x * b.z * c.y;
		int ss = a.y * b.z * c.x - a.y * b.x * c.z;
		int tt = a.z * b.x * c.y - a.z * b.y * c.x;
		
		return ff + ss + tt;
	}
}

