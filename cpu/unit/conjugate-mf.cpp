//Matrix-free Conjugate Gradient Method

#include <iostream>
#include <math.h>
using namespace std;

struct Particle
{
	float x;
	float vx;
};

Particle particle[4];

float AA(int i, int j, int N)
{
	float h = 1.25f;
	
	if(i == j)
	{
		float d_above = (i == 0) ? -particle[i].x/fabs(-particle[i].x) : (particle[i-1].x-particle[i].x)/fabs(particle[i-1].x-particle[i].x);
		float d_below = (i == (N-1)) ? 0.0f : (particle[i+1].x-particle[i].x)/fabs(particle[i+1].x-particle[i].x);
		
		return 1.0f + h*d_above*d_above + h*d_below*d_below;
	}
	else if(i != 0 && (i - j) == 1)
	{
		float d_above = (particle[i-1].x-particle[i].x)/fabs(particle[i-1].x-particle[i].x);
		
		return -h*d_above*d_above;
	}
	else if(i != (N-1) && (i - j) == -1)
	{
		float d_below = (particle[i+1].x-particle[i].x)/fabs(particle[i+1].x-particle[i].x);
		
		return -h*d_below*d_below;
	}
	
	return 0.0f;
}

float bb(int i, int N)
{
	float g = 0.75f;
	float length = 0.005f;
	
	if(i == 0)
	{
		float x0 = particle[0].x;
		float x1 = particle[1].x;
		float v0 = particle[0].vx;
		float d00 = -x0/fabs(x0);
		float d01 = (x1-x0)/fabs(x1-x0);
		
		return v0 + g*(-x0*d00-length)*d00 + g*((x1-x0)*d01-length)*d01;
	}
	else if(i == (N-1))
	{
		float xn0 = particle[i-1].x;
		float xn1 = particle[i].x;
		float vn1 = particle[i].vx;
		float dn10 = (xn0-xn1)/fabs(xn0-xn1);
		
		return vn1 + g*((xn0-xn1)*dn10-length)*dn10;
	}
	else
	{
		float xn0 = particle[i-1].x;
		float xn1 = particle[i].x;
		float xn2 = particle[i+1].x;
		float vn1 = particle[i].vx;
		
		float dn10 = (xn0-xn1)/fabs(xn0-xn1);
		float dn12 = (xn2-xn1)/fabs(xn2-xn1);
		
		return vn1 + g*((xn0-xn1)*dn10-length)*dn10 + g*((xn2-xn1)*dn12-length)*dn12;
	}
	
	//Shouldn't reach this point in the code
	return 0.0f;
}

void conjugate(int N, const float* A, const float* b, float* x)
{
	float r[N];
	float p[N];

	for(int i = 0; i < N; i++)
	{
		//r = b - Ax
		r[i] = b[i];
		for(int j = 0; j < N; j++)
		{
			r[i] -= A[i*N+j]*x[j];
		}
	
		//p = r
		p[i] = r[i];
	}

	float rsold = 0.0f;

	for(int i = 0; i < N; i ++)
	{
		rsold += r[i] * r[i];
	}

	for(int i = 0; i < N; i++)
	{
		float Ap[N];
	
		for(int j = 0; j < N; j++)
		{
			Ap[j] = 0.0f;
		
			for(int k = 0; k < N; k++)
			{
				Ap[j] += A[j*N+k] * p[k];
			}
		}
	
		float abot = 0.0f;
	
		for(int j = 0; j < N; j++)
		{
			abot += p[j] * Ap[j];
		}
	
		float alpha = rsold / abot;
	
		for(int j = 0; j < N; j++)
		{
			x[j] = x[j] + alpha * p[j];
		
			r[j] = r[j] - alpha * Ap[j];
		}
	
		float rsnew = 0.0f;
	
		for(int j = 0; j < N; j++)
		{
			rsnew += r[j] * r[j];
		}
	
		if(rsnew < 1e-10f)
		{
//			cout << "break " << i << endl;
			break;
		}
		
		for(int j = 0; j < N; j++)
		{
			p[j] = r[j] + rsnew / rsold * p[j];
		}
	
		rsold = rsnew;
	}
}

void conjugate_mf(int N, float* x)
{
	float r[N];
	float p[N];

	for(int i = 0; i < N; i++)
	{
		//r = b - Ax
		r[i] = bb(i,N);
		for(int j = 0; j < N; j++)
		{
			r[i] -= AA(i,j,N)*x[j];
		}
	
		//p = r
		p[i] = r[i];
	}

	float rsold = 0.0f;

	for(int i = 0; i < N; i ++)
	{
		rsold += r[i] * r[i];
	}

	for(int i = 0; i < N; i++)
	{
		float Ap[N];
	
		for(int j = 0; j < N; j++)
		{
			Ap[j] = 0.0f;
		
			for(int k = 0; k < N; k++)
			{
				Ap[j] += AA(j,k,N) * p[k];
			}
		}
	
		float abot = 0.0f;
	
		for(int j = 0; j < N; j++)
		{
			abot += p[j] * Ap[j];
		}
	
		float alpha = rsold / abot;
	
		for(int j = 0; j < N; j++)
		{
			x[j] = x[j] + alpha * p[j];
		
			r[j] = r[j] - alpha * Ap[j];
		}
	
		float rsnew = 0.0f;
	
		for(int j = 0; j < N; j++)
		{
			rsnew += r[j] * r[j];
		}
	
		if(rsnew < 1e-10f)
		{
//			cout << "break " << i << endl;
			break;
		}
		
		for(int j = 0; j < N; j++)
		{
			p[j] = r[j] + rsnew / rsold * p[j];
		}
	
		rsold = rsnew;
	}
}

void test3x3()
{
	float* a;
	float* b;
	float* x;
	float* xm;
	float* ans;
	
	int n = 3;
	
	a = new float[n*n];
	b = new float[n];
	x = new float[n];
	xm = new float[n];
	ans = new float[n];
	
	particle[0].x = 0.25f;
	particle[1].x = 0.5f;
	particle[2].x = 0.75f;
	
	particle[0].vx = 0.5f;
	particle[1].vx = 0.5f;
	particle[2].vx = 0.5f;
	
	float x0 = particle[0].x;
	float x1 = particle[1].x;
	float x2 = particle[2].x;
	
	float v0 = particle[0].vx;
	float v1 = particle[1].vx;
	float v2 = particle[2].vx;
	
	float d00 = -particle[0].x/fabs(-particle[0].x);
	float d01 = (particle[1].x-particle[0].x)/fabs(particle[1].x-particle[0].x);
	float d10 = (particle[0].x-particle[1].x)/fabs(particle[0].x-particle[1].x);
	float d12 = (particle[2].x-particle[1].x)/fabs(particle[2].x-particle[1].x);
	float d21 = (particle[1].x-particle[2].x)/fabs(particle[1].x-particle[2].x);
	
	float h = 1.25f;
	
	a[0] = 1+h*d00*d00+h*d01*d01; a[1] = -h*d01*d01;            a[2] = 0.0f;
	a[3] = -h*d10*d10;            a[4] = 1+h*d10*d10+h*d12*d12; a[5] = -h*d12*d12;
	a[6] = 0.0f;                  a[7] = -h*d21*d21;            a[8] = 1+h*d21*d21;
	
	//for(int i = 0; i < 3; i++)
	//{
		//for(int j = 0; j < 3; j++)
		//{
			//cout << a[i*3+j] << " " << ends;
		//}
		
		//cout << endl;
	//}
	
	//cout << endl;
	
	//for(int i = 0; i < 3; i++)
	//{
		//for(int j = 0; j < 3; j++)
		//{
			//cout << AA(i, j, 3) << " " << ends;
		//}
		
		//cout << endl;
	//}
	
	float g = 0.75f;
	float length = 0.005f;
	
	b[0] = v0 + g*(-x0*d00-length)*d00 + g*((x1-x0)*d01-length)*d01;
	b[1] = v1 + g*((x0-x1)*d10-length)*d10 + g*((x2-x1)*d12-length)*d12;
	b[2] = v2 + g*((x1-x2)*d21-length)*d21;
	
	//~ cout << "b:  " << ends;
	//~ for(int i = 0; i < n; i++)
	//~ {
		//~ cout << b[i] << " " << ends;
	//~ }
	//~ cout << endl;
	//~ 
	//~ cout << "bm: " << ends;
	//~ for(int i = 0; i < n; i++)
	//~ {
		//~ cout << bb(i, n) << " " << ends;
	//~ }
	//~ cout << endl;
	
	x[0] = 2;
	x[1] = 2;
	x[2] = 2;
	
	xm[0] = 2;
	xm[1] = 2;
	xm[2] = 2;
	
	ans[0] =  0.2722f;
	ans[1] =  0.3621f;
	ans[2] =  0.3417f;
	
	conjugate(n, a, b, x);
	conjugate_mf(n, xm);
	
	cout.precision(4);
	cout << "sol: " << ends;
	for(int i = 0; i < n; i++)
	{
		cout << x[i] << " " << ends;
	}
	cout << endl;
	
	cout << "smf: " << ends;
	for(int i = 0; i < n; i++)
	{
		cout << xm[i] << " " << ends;
	}
	cout << endl;
	
	cout << "ans: " << ends;
	for(int i = 0; i < n; i++)
	{
		cout << ans[i] << " " << ends;
	}
	cout << endl;
	
	delete [] a;
	delete [] b;
	delete [] x;
	delete [] xm;
	delete [] ans;
}

void test4x4()
{
	float* a;
	float* b;
	float* x;
	float* xm;
	float* ans;
	
	int n = 4;
	
	a = new float[n*n];
	b = new float[n];
	x = new float[n];
	xm = new float[n];
	ans = new float[n];
	
	particle[0].x = 0.25f;
	particle[1].x = 0.50f;
	particle[2].x = 0.75f;
	particle[3].x = 1.00f;
	
	particle[0].vx = 0.5f;
	particle[1].vx = 0.5f;
	particle[2].vx = 0.5f;
	particle[3].vx = 0.5f;
	
	float x0 = particle[0].x;
	float x1 = particle[1].x;
	float x2 = particle[2].x;
	float x3 = particle[3].x;
	
	float v0 = particle[0].vx;
	float v1 = particle[1].vx;
	float v2 = particle[2].vx;
	float v3 = particle[3].vx;
	
	float d00 = -particle[0].x/fabs(-particle[0].x);
	float d01 = (particle[1].x-particle[0].x)/fabs(particle[1].x-particle[0].x);
	float d10 = (particle[0].x-particle[1].x)/fabs(particle[0].x-particle[1].x);
	float d12 = (particle[2].x-particle[1].x)/fabs(particle[2].x-particle[1].x);
	float d21 = (particle[1].x-particle[2].x)/fabs(particle[1].x-particle[2].x);
	float d23 = (particle[3].x-particle[2].x)/fabs(particle[3].x-particle[2].x);
	float d32 = (particle[2].x-particle[3].x)/fabs(particle[2].x-particle[3].x);
	
	float h = 1.25f;
	
	a[ 0] = 1+h*d00*d00+h*d01*d01; a[ 1] = -h*d01*d01;            a[ 2] = 0.0f;                  a[ 3] = 0.0f;
	a[ 4] = -h*d10*d10;            a[ 5] = 1+h*d10*d10+h*d12*d12; a[ 6] = -h*d12*d12;            a[ 7] = 0.0f;
	a[ 8] = 0.0f;                  a[ 9] = -h*d21*d21;            a[10] = 1+h*d21*d21+h*d23*d23; a[11] = -h*d23*d23;
	a[12] = 0.0f;                  a[13] = 0.0f;                  a[14] = -h*d32*d32;            a[15] = 1+h*d32*d32;
	
	//Check matching output
	/*
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			cout.precision(4);
			cout.width(5);
			cout << a[i*n+j] << " " << ends;
		}
		cout << endl;
	}
	cout << endl;
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			cout.precision(4);
			cout.width(5);
			cout << AA(i, j, n) << " " << ends;
		}
		cout << endl;
	}
	*/
	
	float g = 0.75f;
	float length = 0.005f;
	
	b[0] = v0 + g*(-x0*d00-length)*d00 + g*((x1-x0)*d01-length)*d01;
	b[1] = v1 + g*((x0-x1)*d10-length)*d10 + g*((x2-x1)*d12-length)*d12;
	b[2] = v2 + g*((x1-x2)*d21-length)*d21 + g*((x3-x2)*d23-length)*d23;
	b[3] = v3 + g*((x2-x3)*d32-length)*d32;
	
	//~ cout << "b:  " << ends;
	//~ for(int i = 0; i < n; i++)
	//~ {
		//~ cout << b[i] << " " << ends;
	//~ }
	//~ cout << endl;
	//~ 
	//~ cout << "bm: " << ends;
	//~ for(int i = 0; i < n; i++)
	//~ {
		//~ cout << bb(i, n) << " " << ends;
	//~ }
	//~ cout << endl;
	
	x[0] = 1.0f;
	x[1] = 1.0f;
	x[2] = 1.0f;
	x[3] = 1.0f;
	
	xm[0] = 1.0f;
	xm[1] = 1.0f;
	xm[2] = 1.0f;
	xm[3] = 1.0f;
	
	ans[0] = 0.2830f;
	ans[1] = 0.3924f;
	ans[2] = 0.4157f;
	ans[3] = 0.3715f;
	
	conjugate(n, a, b, x);
	conjugate_mf(n, xm);
	
	cout.precision(4);
	cout << "sol: " << ends;
	for(int i = 0; i < n; i++)
	{
		cout << x[i] << " " << ends;
	}
	cout << endl;
	
	cout << "smf: " << ends;
	for(int i = 0; i < n; i++)
	{
		cout << xm[i] << " " << ends;
	}
	cout << endl;
	
	cout << "ans: " << ends;
	for(int i = 0; i < n; i++)
	{
		cout << ans[i] << " " << ends;
	}
	cout << endl;
	
	delete [] a;
	delete [] b;
	delete [] x;
	delete [] xm;
	delete [] ans;
}

int main()
{
	test3x3();
	
	test4x4();
	
	return 0;
}
