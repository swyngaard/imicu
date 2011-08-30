
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

	Vector3f& Vector3f::operator= (Vector3f v)			// operator= sets values of v to this Vector3f. example: v1 = v2 means that values of v2 are set onto v1
	{
		x = v.x;
		y = v.y;
		z = v.z;
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
		
		float xhalf = 0.5f*number;
		int i = *(int*)&number; // get bits for floating value
		i = 0x5f375a86- (i>>1); // gives initial guess y0
		number = *(float*)&i; // convert bits back to float
		number = number*(1.5f-xhalf*number*number); // Newton step, repeating increases accuracy
		
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
}

