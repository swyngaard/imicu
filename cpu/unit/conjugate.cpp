//Conjugate Gradient Method Unit Tests

#include <iostream>
#include <math.h>

#define NUMPARTICLES	4
#define NUMCOMPONENTS	2

using namespace std;

struct Particle
{
	float x;
	float y;
	float vx;
	float vy;
};

Particle particle[NUMPARTICLES];
float AA[NUMPARTICLES*NUMCOMPONENTS*NUMPARTICLES*NUMCOMPONENTS];
float bb[NUMPARTICLES*NUMCOMPONENTS];
float xx[NUMPARTICLES*NUMCOMPONENTS];

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

void buildAB()
{
	
	float length = 0.005f;
	float mass = 0.000000001f;
	float k_edge = 0.009f;
	float d_edge = 0.0f;
	float dt = 0.006f;
	float h = dt*dt*k_edge/(4.0f*mass*length) + d_edge*dt/(2.0f*mass);
	float g = dt*k_edge/(2.0f*mass*length);
	
	Particle du0;
	Particle dd0;
	du0.x = -particle[0].x/sqrtf(-particle[0].x*-particle[0].x+-particle[0].y*-particle[0].y);
	du0.y = -particle[0].y/sqrtf(-particle[0].x*-particle[0].x+-particle[0].y*-particle[0].y);
	dd0.x = (particle[1].x-particle[0].x)/sqrtf((particle[1].x-particle[0].x)*(particle[1].x-particle[0].x)+(particle[1].y-particle[0].y)*(particle[1].y-particle[0].y));
	dd0.y = (particle[1].y-particle[0].y)/sqrtf((particle[1].x-particle[0].x)*(particle[1].x-particle[0].x)+(particle[1].y-particle[0].y)*(particle[1].y-particle[0].y));
	
	AA[0] = 1.0f + h*du0.x*du0.x + h*dd0.x*dd0.x;
	AA[1] = h*du0.x*du0.y + h*dd0.x*dd0.y;
	AA[2] = -h*dd0.x*dd0.x;
	AA[3] = -h*dd0.x*dd0.y;
	
	AA[NUMPARTICLES*NUMCOMPONENTS  ] = h*du0.x*du0.y + h*dd0.x*dd0.y;
	AA[NUMPARTICLES*NUMCOMPONENTS+1] = 1.0f + h*du0.y*du0.y + h*dd0.y*dd0.y;
	AA[NUMPARTICLES*NUMCOMPONENTS+2] = -h*dd0.x*dd0.y;
	AA[NUMPARTICLES*NUMCOMPONENTS+3] = -h*dd0.y*dd0.y;
	
	bb[0] = particle[0].vx + g*(-particle[0].x*du0.x-particle[0].y*du0.y-length)*du0.x + g*((particle[1].x-particle[0].x)*dd0.x+(particle[1].y-particle[0].y)*dd0.y-length)*dd0.x;
	bb[1] = particle[0].vy + g*(-particle[0].x*du0.x-particle[0].y*du0.y-length)*du0.y + g*((particle[1].x-particle[0].x)*dd0.x+(particle[1].y-particle[0].y)*dd0.y-length)*dd0.y;
	
	//~ cout << bb[0] << " " << bb[1] << endl;
	
	for(int i = 1; i < (NUMPARTICLES-1); i++)
	{
		Particle du;
		Particle dd;
		du.x = (particle[i-1].x-particle[i].x)/sqrtf((particle[i-1].x-particle[i].x)*(particle[i-1].x-particle[i].x)+(particle[i-1].y-particle[i].y)*(particle[i-1].y-particle[i].y));
		du.y = (particle[i-1].y-particle[i].y)/sqrtf((particle[i-1].x-particle[i].x)*(particle[i-1].x-particle[i].x)+(particle[i-1].y-particle[i].y)*(particle[i-1].y-particle[i].y));
		dd.x = (particle[i+1].x-particle[i].x)/sqrtf((particle[i+1].x-particle[i].x)*(particle[i+1].x-particle[i].x)+(particle[i+1].y-particle[i].y)*(particle[i+1].y-particle[i].y));
		dd.y = (particle[i+1].y-particle[i].y)/sqrtf((particle[i+1].x-particle[i].x)*(particle[i+1].x-particle[i].x)+(particle[i+1].y-particle[i].y)*(particle[i+1].y-particle[i].y));
		
		AA[(i*NUMCOMPONENTS)*NUMCOMPONENTS*NUMPARTICLES + i*NUMCOMPONENTS-2] = -h*du.x*du.x;
		AA[(i*NUMCOMPONENTS)*NUMCOMPONENTS*NUMPARTICLES + i*NUMCOMPONENTS-1] = -h*du.x*du.y;
		AA[(i*NUMCOMPONENTS)*NUMCOMPONENTS*NUMPARTICLES + i*NUMCOMPONENTS  ] = 1.0f + h*du.x*du.x + h*dd.x*dd.x; //Diagonal
		AA[(i*NUMCOMPONENTS)*NUMCOMPONENTS*NUMPARTICLES + i*NUMCOMPONENTS+1] = h*du.x*du.y + h*dd.x*dd.y;
		AA[(i*NUMCOMPONENTS)*NUMCOMPONENTS*NUMPARTICLES + i*NUMCOMPONENTS+2] = -h*dd.x*dd.x;
		AA[(i*NUMCOMPONENTS)*NUMCOMPONENTS*NUMPARTICLES + i*NUMCOMPONENTS+3] = -h*dd.x*dd.y;
		
		AA[(i*NUMCOMPONENTS+1)*NUMCOMPONENTS*NUMPARTICLES + i*NUMCOMPONENTS-2] = -h*du.x*du.y;
		AA[(i*NUMCOMPONENTS+1)*NUMCOMPONENTS*NUMPARTICLES + i*NUMCOMPONENTS-1] = -h*du.y*du.y;
		AA[(i*NUMCOMPONENTS+1)*NUMCOMPONENTS*NUMPARTICLES + i*NUMCOMPONENTS  ] = h*du.x*du.y + h*dd.x*dd.y;
		AA[(i*NUMCOMPONENTS+1)*NUMCOMPONENTS*NUMPARTICLES + i*NUMCOMPONENTS+1] = 1.0f + h*du.y*du.y + h*dd.y*dd.y; //Diagonal
		AA[(i*NUMCOMPONENTS+1)*NUMCOMPONENTS*NUMPARTICLES + i*NUMCOMPONENTS+2] = -h*dd.x*dd.y;
		AA[(i*NUMCOMPONENTS+1)*NUMCOMPONENTS*NUMPARTICLES + i*NUMCOMPONENTS+3] = -h*dd.y*dd.y;
		
		bb[i*NUMCOMPONENTS  ] = particle[i].vx + g*((particle[i-1].x-particle[i].x)*du.x+(particle[i-1].y-particle[i].y)*du.y-length)*du.x + g*((particle[i+1].x-particle[i].x)*dd.x+(particle[i+1].y-particle[i].y)*dd.y-length)*dd.x;
		bb[i*NUMCOMPONENTS+1] = particle[i].vy + g*((particle[i-1].x-particle[i].x)*du.x+(particle[i-1].y-particle[i].y)*du.y-length)*du.y + g*((particle[i+1].x-particle[i].x)*dd.x+(particle[i+1].y-particle[i].y)*dd.y-length)*dd.y;
		
		//~ cout << bb[i*NUMCOMPONENTS  ] << " " << bb[i*NUMCOMPONENTS+1] << endl;
	}
	
	Particle duN;
	duN.x = (particle[NUMPARTICLES-2].x-particle[NUMPARTICLES-1].x)/sqrtf((particle[NUMPARTICLES-2].x-particle[NUMPARTICLES-1].x)*(particle[NUMPARTICLES-2].x-particle[NUMPARTICLES-1].x)+(particle[NUMPARTICLES-2].y-particle[NUMPARTICLES-1].y)*(particle[NUMPARTICLES-2].y-particle[NUMPARTICLES-1].y));
	duN.y = (particle[NUMPARTICLES-2].y-particle[NUMPARTICLES-1].y)/sqrtf((particle[NUMPARTICLES-2].x-particle[NUMPARTICLES-1].x)*(particle[NUMPARTICLES-2].x-particle[NUMPARTICLES-1].x)+(particle[NUMPARTICLES-2].y-particle[NUMPARTICLES-1].y)*(particle[NUMPARTICLES-2].y-particle[NUMPARTICLES-1].y));
	
	AA[(NUMPARTICLES-1)*NUMCOMPONENTS*NUMPARTICLES*NUMCOMPONENTS + NUMPARTICLES*NUMCOMPONENTS-4] = -h*duN.x*duN.x;
	AA[(NUMPARTICLES-1)*NUMCOMPONENTS*NUMPARTICLES*NUMCOMPONENTS + NUMPARTICLES*NUMCOMPONENTS-3] = -h*duN.x*duN.y;
	AA[(NUMPARTICLES-1)*NUMCOMPONENTS*NUMPARTICLES*NUMCOMPONENTS + NUMPARTICLES*NUMCOMPONENTS-2] = 1.0f + h*duN.x*duN.x;
	AA[(NUMPARTICLES-1)*NUMCOMPONENTS*NUMPARTICLES*NUMCOMPONENTS + NUMPARTICLES*NUMCOMPONENTS-1] = h*duN.x*duN.y;
	
	AA[NUMPARTICLES*NUMCOMPONENTS*NUMPARTICLES*NUMCOMPONENTS - 4] = -h*duN.x*duN.y;
	AA[NUMPARTICLES*NUMCOMPONENTS*NUMPARTICLES*NUMCOMPONENTS - 3] = -h*duN.y*duN.y;
	AA[NUMPARTICLES*NUMCOMPONENTS*NUMPARTICLES*NUMCOMPONENTS - 2] = h*duN.x*duN.y;
	AA[NUMPARTICLES*NUMCOMPONENTS*NUMPARTICLES*NUMCOMPONENTS - 1] = 1.0f + h*duN.y*duN.y;
	
	bb[NUMPARTICLES*NUMCOMPONENTS-2] = particle[NUMPARTICLES-1].vx + g*((particle[NUMPARTICLES-2].x-particle[NUMPARTICLES-1].x)*duN.x+(particle[NUMPARTICLES-2].y-particle[NUMPARTICLES-1].y)*duN.y-length)*duN.x;
	bb[NUMPARTICLES*NUMCOMPONENTS-1] = particle[NUMPARTICLES-1].vy + g*((particle[NUMPARTICLES-2].x-particle[NUMPARTICLES-1].x)*duN.x+(particle[NUMPARTICLES-2].y-particle[NUMPARTICLES-1].y)*duN.y-length)*duN.y;
	
	//~ cout << bb[NUMPARTICLES*NUMCOMPONENTS-2] << " " << bb[NUMPARTICLES*NUMCOMPONENTS-1] << endl;
	
	cout << "AA:" << endl;
	for(int i = 0; i < NUMPARTICLES*NUMCOMPONENTS; i++)
	{
		cout << "[" << ends;
		for(int j = 0; j < NUMPARTICLES*NUMCOMPONENTS; j++)
		{
			cout.width(12);
			cout.precision(8);
			cout << AA[i*NUMPARTICLES*NUMCOMPONENTS+j];
			
			if(j < (NUMCOMPONENTS*NUMPARTICLES-1))
				cout << ", " << ends;
		}
		
		cout << "]," << endl;
	}
	
	cout << "bb:" << endl;
	cout << "[" << ends;
	for(int i = 0; i < NUMPARTICLES*NUMCOMPONENTS; i++)
	{
		cout.precision(8);
		cout << bb[i] << ends;
		
		if(i < (NUMPARTICLES*NUMCOMPONENTS-1))
			cout << ", " << ends;
	}
	cout << "]" << endl;
}

void test2D()
{
	for(int i = 0; i < NUMPARTICLES; i++)
	{
		particle[i].x = (i+1.0f)*(-0.005f/2.0f);
		particle[i].y = 0.0f;
		particle[i].vx = 0.0f;
		particle[i].vx = 0.0f;
		
		//~ cout.width(12);
		//~ cout << particle[i].x << " " << particle[i].y << endl;
	}
	
	cout << endl;
	
	buildAB();
	
	conjugate(NUMPARTICLES*NUMCOMPONENTS, AA, bb, xx);
	
	cout << "xx:" << endl;
	cout << "[" << ends;
	for(int i = 0; i < NUMPARTICLES*NUMCOMPONENTS; i++)
	{
		cout.precision(8);
		cout << xx[i] << ends; 
		
		if(i < (NUMPARTICLES*NUMCOMPONENTS-1))
			cout << ", " << ends;
	}
	cout << "]" << endl;
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
	//~ test3x3();
	
	//~ test4x4();
	
	test2D();
	
	return 0;
}
