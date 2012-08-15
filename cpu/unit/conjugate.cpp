#include <iostream>

using namespace std;

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

void test3x3()
{
	float* a;
	float* b;
	float* x;
	float* ans;
	
	int n = 3;
	
	a = new float[n*n];
	b = new float[n];
	x = new float[n];
	ans = new float[n];
	
	a[0] =  2; a[1] = -1; a[2] =  0;
	a[3] = -1; a[4] =  2; a[5] = -1;
	a[6] =  0; a[7] = -1; a[8] =  2;
	
	b[0] =  8;
	b[1] =  4;
	b[2] = -8;
	
	x[0] = 2;
	x[1] = 2;
	x[2] = 2;
	
	ans[0] =  6;
	ans[1] =  4;
	ans[2] = -2;
	
	conjugate(n, a, b, x);
	
	cout << "sol: " << ends;
	for(int i = 0; i < n; i++)
	{
		cout << x[i] << " " << ends;
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
	delete [] ans;
}

void test4x4()
{
	float* a;
	float* b;
	float* x;
	float* ans;
	
	int n = 4;
	
	a = new float[n*n];
	b = new float[n];
	x = new float[n];
	ans = new float[n];
	
	a[ 0] =  5.0f; a[ 1] =  2.0f; a[ 2] =  0.0f; a[ 3] =  0.0f;
	a[ 4] =  2.0f; a[ 5] =  3.0f; a[ 6] =  1.0f; a[ 7] =  0.0f;
	a[ 8] =  0.0f; a[ 9] =  1.0f; a[10] =  4.0f; a[11] = -1.0f;
	a[12] =  0.0f; a[13] =  0.0f; a[14] = -1.0f; a[15] =  7.0f;
	
	b[0] =  4.0f;
	b[1] = -2.0f;
	b[2] =  5.0f;
	b[3] =  4.0f;
	
	x[0] = 1.0f;
	x[1] = 1.0f;
	x[2] = 1.0f;
	x[3] = 1.0f;
	
	ans[0] =  1.840f;
	ans[1] = -2.599f;
	ans[2] =  2.118f;
	ans[3] =  0.874f;
	
	conjugate(n, a, b, x);
	
	cout << "sol: " << ends;
	for(int i = 0; i < n; i++)
	{
		cout << x[i] << " " << ends;
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
	delete [] ans;
}

int main()
{
	test3x3();
	
	test4x4();
	
	return 0;
}
