
#include "kdop.h"
#include <cstdlib>
#include <iostream>
#include <cfloat>

void KDOP::initialise(int k)
{
	//Make sure that only a valid value of K is set
	switch(k)
	{
		case  6: addNormals6(); break;
		case 14: addNormals14(); break;
		case 18: addNormals18(); break;
		case 26: addNormals26(); break;
		
		default:
		{
			//unspecified value for K, set to default 
			k = 14;
			addNormals14();
		}
	}
	
	this->K = k;
	distance = new float[K];
	
	for(int i = 0; i < K; i++)
		distance[i] = 0.0f;
	
	setDegenerateMatrix();
}

KDOP::KDOP(int k)
{
	initialise(k);
}

KDOP::KDOP(std::vector<Vector3f>& vertex, int k)
{
	initialise(k);
	
	update(vertex);
}

//TODO add copy constructor
KDOP::KDOP(const KDOP& kdop)
{
	//Make sure that only a valid value of K is set
	switch(kdop.K)
	{
		case  6: addNormals6(); break;
		case 14: addNormals14(); break;
		case 18: addNormals18(); break;
		case 26: addNormals26(); break;
	}
	
	this->K = kdop.K;
	distance = new float[K];
	
	for(int i = 0; i < K; i++)
		distance[i] = kdop.distance[i];
	
	setDegenerateMatrix();
}

KDOP::~KDOP()
{
	for(int i = 0; i < K; i++)
	{
		delete [] ndg[i];
	}
	
	delete [] ndg;
	
	delete [] distance;
}

std::vector<Vector3i>& KDOP::getNormals()
{
	return normal;
}

void KDOP::buildDegenerateMatrix()
{
	for(int i = 0; i < K; i++)
		for(int j = 0; j < K; j++)
			ndg[i][j] = false;
	
	for(int i = 0; i < K; i++)
	{
		for(int j = i+1; j < K; j++)
		{
			Vector3i pair = normal[i] + normal[j];
			
			Vector3i cross = pair.cross(normal[i]);
			
			if(!(cross.x == 0 && cross.y == 0 && cross.z == 0))
				ndg[i][j] = true;
		}
	}
}

bool** KDOP::getDegenerateMatrix()
{
	return ndg;
}

float* KDOP::getDistances()
{
	return distance;
}

void KDOP::update(std::vector<Vector3f>& vertex)
{
	if(vertex.size() == 0) return;
	
	switch(K)
	{
		case  6: build6(vertex); break;
		case 14: build14(vertex); break;
		case 18: build18(vertex); break;
		case 26: build26(vertex); break;
	}
}

void KDOP::addNormals6()
{
	normal.push_back(Vector3i(-1,  0,  0));
	normal.push_back(Vector3i( 0, -1,  0));
	normal.push_back(Vector3i( 0,  0, -1));
	
	normal.push_back(Vector3i( 1,  0,  0));
	normal.push_back(Vector3i( 0,  1,  0));
	normal.push_back(Vector3i( 0,  0,  1));
}

void KDOP::addNormals14()
{
	normal.push_back(Vector3i(-1,  0,  0));
	normal.push_back(Vector3i( 0, -1,  0));
	normal.push_back(Vector3i( 0,  0, -1));
	normal.push_back(Vector3i(-1, -1, -1));
	normal.push_back(Vector3i(-1,  1, -1));
	normal.push_back(Vector3i(-1, -1,  1));
	normal.push_back(Vector3i(-1,  1,  1));
	
	normal.push_back(Vector3i( 1,  0,  0));
	normal.push_back(Vector3i( 0,  1,  0));
	normal.push_back(Vector3i( 0,  0,  1));
	normal.push_back(Vector3i( 1,  1,  1));
	normal.push_back(Vector3i( 1, -1,  1));
	normal.push_back(Vector3i( 1,  1, -1));
	normal.push_back(Vector3i( 1, -1, -1));
}

void KDOP::addNormals18()
{
	normal.push_back(Vector3i(-1,  0,  0));
	normal.push_back(Vector3i( 0, -1,  0));
	normal.push_back(Vector3i( 0,  0, -1));
	normal.push_back(Vector3i(-1, -1,  0));
	normal.push_back(Vector3i(-1,  0, -1));
	normal.push_back(Vector3i( 0, -1, -1));
	normal.push_back(Vector3i(-1,  1,  0));
	normal.push_back(Vector3i(-1,  0,  1));
	normal.push_back(Vector3i( 0, -1,  1));
	
	normal.push_back(Vector3i( 1,  0,  0));
	normal.push_back(Vector3i( 0,  1,  0));
	normal.push_back(Vector3i( 0,  0,  1));
	normal.push_back(Vector3i( 1,  1,  0));
	normal.push_back(Vector3i( 1,  0,  1));
	normal.push_back(Vector3i( 0,  1,  1));
	normal.push_back(Vector3i( 1, -1,  0));
	normal.push_back(Vector3i( 1,  0, -1));
	normal.push_back(Vector3i( 0,  1, -1));
}

void KDOP::addNormals26()
{
	normal.push_back(Vector3i(-1,  0,  0));
	normal.push_back(Vector3i( 0, -1,  0));
	normal.push_back(Vector3i( 0,  0, -1));
	normal.push_back(Vector3i(-1, -1, -1));
	normal.push_back(Vector3i(-1,  1, -1));
	normal.push_back(Vector3i(-1, -1,  1));
	normal.push_back(Vector3i(-1,  1,  1));
	normal.push_back(Vector3i(-1, -1,  0));
	normal.push_back(Vector3i(-1,  0, -1));
	normal.push_back(Vector3i( 0, -1, -1));
	normal.push_back(Vector3i(-1,  1,  0));
	normal.push_back(Vector3i(-1,  0,  1));
	normal.push_back(Vector3i( 0, -1,  1));
	
	normal.push_back(Vector3i(1,  0,  0));
	normal.push_back(Vector3i(0,  1,  0));
	normal.push_back(Vector3i(0,  0,  1));
	normal.push_back(Vector3i(1,  1,  1));
	normal.push_back(Vector3i(1, -1,  1));
	normal.push_back(Vector3i(1,  1, -1));
	normal.push_back(Vector3i(1, -1, -1));
	normal.push_back(Vector3i(1,  1,  0));
	normal.push_back(Vector3i(1,  0,  1));
	normal.push_back(Vector3i(0,  1,  1));
	normal.push_back(Vector3i(1, -1,  0));
	normal.push_back(Vector3i(1,  0, -1));
	normal.push_back(Vector3i(0,  1, -1));
}

void KDOP::build6(std::vector<Vector3f>& vertex)
{
	//Cuboid planes
	float minX = vertex[0].x;
	float maxX = minX;
	float minY = vertex[0].y;
	float maxY = minY;
	float minZ = vertex[0].z;
	float maxZ = minZ;
	
	for(int i = 1; i < vertex.size(); i++)
	{
		//(1,0,0)
		minX = std::min(minX,vertex[i].x);
		maxX = std::max(maxX,vertex[i].x);
		
		//(0,1,0)
		minY = std::min(minY,vertex[i].y);
		maxY = std::max(maxY,vertex[i].y);
		
		//(0,0,1)
		minZ = std::min(minZ,vertex[i].z);
		maxZ = std::max(maxZ,vertex[i].z);
	}
	
	distance[0] = minX;
	distance[1] = minY;
	distance[2] = minZ;
	
	distance[3] = maxX;
	distance[4] = maxY;
	distance[5] = maxZ;
}

void KDOP::build14(std::vector<Vector3f>& vertex)
{
	//Calculate distances to corner planes
	float d111 = vertex[0].x + vertex[0].y + vertex[0].z;
	float d101 = vertex[0].x - vertex[0].y + vertex[0].z;
	float d110 = vertex[0].x + vertex[0].y - vertex[0].z;
	float d100 = vertex[0].x - vertex[0].y - vertex[0].z;
	
	//Cuboid planes
	float minX = vertex[0].x;
	float maxX = minX;
	float minY = vertex[0].y;
	float maxY = minY;
	float minZ = vertex[0].z;
	float maxZ = minZ;
	
	//Min-max to cuboid corner planes
	float min111 = d111;
	float max111 = min111;
	float min101 = d101;
	float max101 = min101;
	float min110 = d110;
	float max110 = min110;
	float min100 = d100;
	float max100 = min100;
	
	for(int i = 1; i < vertex.size(); i++)
	{
		//Calculate distances to corners
		d111 = vertex[i].x + vertex[i].y + vertex[i].z;
		d101 = vertex[i].x - vertex[i].y + vertex[i].z;
		d110 = vertex[i].x + vertex[i].y - vertex[i].z;
		d100 = vertex[i].x - vertex[i].y - vertex[i].z;
		
		//Calculate min-max distances for kdop
		
		//(1,0,0)
		minX = std::min(minX,vertex[i].x);
		maxX = std::max(maxX,vertex[i].x);
		
		//(0,1,0)
		minY = std::min(minY,vertex[i].y);
		maxY = std::max(maxY,vertex[i].y);
		
		//(0,0,1)
		minZ = std::min(minZ,vertex[i].z);
		maxZ = std::max(maxZ,vertex[i].z);
		
		//(1,1,1)
		min111 = std::min(min111, d111);
		max111 = std::max(max111, d111);
		
		//(1,-1,1)
		min101 = std::min(min101, d101);
		max101 = std::max(max101, d101);
		
		//(1,1,-1)
		min110 = std::min(min110, d110);
		max110 = std::max(max110, d110);
		
		//(1,-1,-1)
		min100 = std::min(min100, d100);
		max100 = std::max(max100, d100);
	}
	
	distance[0] = minX;
	distance[1] = minY;
	distance[2] = minZ;
	distance[3] = min111;
	distance[4] = min101;
	distance[5] = min110;
	distance[6] = min100;
	
	distance[ 7] = maxX;
	distance[ 8] = maxY;
	distance[ 9] = maxZ;
	distance[10] = max111;
	distance[11] = max101;
	distance[12] = max110;
	distance[13] = max100;
}

void KDOP::build18(std::vector<Vector3f>& vertex)
{
	//12 edges of cube
	float d110 = vertex[0].x + vertex[0].y;
	float d101 = vertex[0].x + vertex[0].z;
	float d011 = vertex[0].y + vertex[0].z;
	float d120 = vertex[0].x - vertex[0].y;
	float d102 = vertex[0].x - vertex[0].z;
	float d012 = vertex[0].y - vertex[0].z;
	
	//Cube planes
	float minX = vertex[0].x;
	float maxX = minX;
	float minY = vertex[0].y;
	float maxY = minY;
	float minZ = vertex[0].z;
	float maxZ = minZ;
	
	float min110 = d110;
	float max110 = d110;
	float min101 = d101;
	float max101 = d101;
	float min011 = d011;
	float max011 = d011;
	float min120 = d120;
	float max120 = d120;
	float min102 = d102;
	float max102 = d102;
	float min012 = d012;
	float max012 = d012;
	
	for(int i = 1; i < vertex.size(); i++)
	{
		d110 = vertex[i].x + vertex[i].y;
		d101 = vertex[i].x + vertex[i].z;
		d011 = vertex[i].y + vertex[i].z;
		d120 = vertex[i].x - vertex[i].y;
		d102 = vertex[i].x - vertex[i].z;
		d012 = vertex[i].y - vertex[i].z;
		
		//(1,0,0)
		minX = std::min(minX,vertex[i].x);
		maxX = std::max(maxX,vertex[i].x);
		
		//(0,1,0)
		minY = std::min(minY,vertex[i].y);
		maxY = std::max(maxY,vertex[i].y);
		
		//(0,0,1)
		minZ = std::min(minZ,vertex[i].z);
		maxZ = std::max(maxZ,vertex[i].z);
		
		min110 = std::min(min110, d110);
		max110 = std::max(max110, d110);
		min101 = std::min(min101, d101);
		max101 = std::max(max101, d101);
		min011 = std::min(min011, d011);
		max011 = std::max(max011, d011);
		min120 = std::min(min120, d120);
		max120 = std::max(max120, d120);
		min102 = std::min(min102, d102);
		max102 = std::max(max102, d102);
		min012 = std::min(min012, d012);
		max012 = std::max(max012, d012);
	}
	
	distance[0] = minX;
	distance[1] = minY;
	distance[2] = minZ;
	distance[3] = min110;
	distance[4] = min101;
	distance[5] = min011;
	distance[6] = min120;
	distance[7] = min102;
	distance[8] = min012;
	
	distance[ 9] = maxX;
	distance[10] = maxY;
	distance[11] = maxZ;
	distance[12] = max110;
	distance[13] = max101;
	distance[14] = max011;
	distance[15] = max120;
	distance[16] = max102;
	distance[17] = max012;
}

void KDOP::build26(std::vector<Vector3f>& vertex)
{
	float d111 = vertex[0].x + vertex[0].y + vertex[0].z;
	float d121 = vertex[0].x - vertex[0].y + vertex[0].z;
	float d112 = vertex[0].x + vertex[0].y - vertex[0].z;
	float d122 = vertex[0].x - vertex[0].y - vertex[0].z;
	float d110 = vertex[0].x + vertex[0].y;
	float d101 = vertex[0].x + vertex[0].z;
	float d011 = vertex[0].y + vertex[0].z;
	float d120 = vertex[0].x - vertex[0].y;
	float d102 = vertex[0].x - vertex[0].z;
	float d012 = vertex[0].y - vertex[0].z;
	
	float minX = vertex[0].x;
	float maxX = minX;
	float minY = vertex[0].y;
	float maxY = minY;
	float minZ = vertex[0].z;
	float maxZ = minZ;
	
	float min111 = d111;
	float max111 = d111;
	float min121 = d121;
	float max121 = d121;
	float min112 = d112;
	float max112 = d112;
	float min122 = d122;
	float max122 = d122;
	float min110 = d110;
	float max110 = d110;
	float min101 = d101;
	float max101 = d101;
	float min011 = d011;
	float max011 = d011;
	float min120 = d120;
	float max120 = d120;
	float min102 = d102;
	float max102 = d102;
	float min012 = d012;
	float max012 = d012;
	
	for(int i = 1; i < vertex.size(); i++)
	{
		d111 = vertex[i].x + vertex[i].y + vertex[i].z;
		d121 = vertex[i].x - vertex[i].y + vertex[i].z;
		d112 = vertex[i].x + vertex[i].y - vertex[i].z;
		d122 = vertex[i].x - vertex[i].y - vertex[i].z;
		d110 = vertex[i].x + vertex[i].y;
		d101 = vertex[i].x + vertex[i].z;
		d011 = vertex[i].y + vertex[i].z;
		d120 = vertex[i].x - vertex[i].y;
		d102 = vertex[i].x - vertex[i].z;
		d012 = vertex[i].y - vertex[i].z;
		
		//(1,0,0)
		minX = std::min(minX,vertex[i].x);
		maxX = std::max(maxX,vertex[i].x);
		
		//(0,1,0)
		minY = std::min(minY,vertex[i].y);
		maxY = std::max(maxY,vertex[i].y);
		
		//(0,0,1)
		minZ = std::min(minZ,vertex[i].z);
		maxZ = std::max(maxZ,vertex[i].z);
		
		min111 = std::min(min111, d111);
		max111 = std::max(max111, d111);
		
		min121 = std::min(min121, d121);
		max121 = std::max(max121, d121);
		
		min112 = std::min(min112, d112);
		max112 = std::max(max112, d112);
		
		min122 = std::min(min122, d122);
		max122 = std::max(max122, d122);
		
		min110 = std::min(min110, d110);
		max110 = std::max(max110, d110);
		
		min101 = std::min(min101, d101);
		max101 = std::max(max101, d101);
		
		min011 = std::min(min011, d011);
		max011 = std::max(max011, d011);
		
		min120 = std::min(min120, d120);
		max120 = std::max(max120, d120);
		
		min102 = std::min(min102, d102);
		max102 = std::max(max102, d102);
		
		min012 = std::min(min012, d012);
		max012 = std::max(max012, d012);
	}
	
	distance[ 0] = minX;
	distance[ 1] = minY;
	distance[ 2] = minZ;
	distance[ 3] = min111;
	distance[ 4] = min121;
	distance[ 5] = min112;
	distance[ 6] = min122;
	distance[ 7] = min110;
	distance[ 8] = min101;
	distance[ 9] = min011;
	distance[10] = min120;
	distance[11] = min102;
	distance[12] = min012;
	
	distance[13] = maxX;
	distance[14] = maxY;
	distance[15] = maxZ;
	distance[16] = max111;
	distance[17] = max121;
	distance[18] = max112;
	distance[19] = max122;
	distance[20] = max110;
	distance[21] = max101;
	distance[22] = max011;
	distance[23] = max120;
	distance[24] = max102;
	distance[25] = max012;
}

//Return vertices for visualising kdop. Each pair of vertices is a line.
std::vector<Vector3f>& KDOP::debugVertices()
{
	return vertices;
}

void KDOP::setDegenerateMatrix()
{
	ndg = new bool*[K];
	for(int i = 0; i < K; i++)
	{
		ndg[i] = new bool[K];
	}
	
	switch(K)
	{
		case 6:
		{
			ndg[0][0]=0; ndg[0][1]=1; ndg[0][2]=1; ndg[0][3]=0; ndg[0][4]=1; ndg[0][5]=1;
			ndg[1][0]=0; ndg[1][1]=0; ndg[1][2]=1; ndg[1][3]=1; ndg[1][4]=0; ndg[1][5]=1;
			ndg[2][0]=0; ndg[2][1]=0; ndg[2][2]=0; ndg[2][3]=1; ndg[2][4]=1; ndg[2][5]=0;
			ndg[3][0]=0; ndg[3][1]=0; ndg[3][2]=0; ndg[3][3]=0; ndg[3][4]=1; ndg[3][5]=1;
			ndg[4][0]=0; ndg[4][1]=0; ndg[4][2]=0; ndg[4][3]=0; ndg[4][4]=0; ndg[4][5]=1;
			ndg[5][0]=0; ndg[5][1]=0; ndg[5][2]=0; ndg[5][3]=0; ndg[5][4]=0; ndg[5][5]=0;
		}
		break;
		
		case 14:
		{
			ndg[ 0][0]=0; ndg[ 0][1]=1; ndg[ 0][2]=1; ndg[ 0][3]=1; ndg[ 0][4]=1; ndg[ 0][5]=1; ndg[ 0][6]=1; ndg[ 0][7]=0; ndg[ 0][8]=1; ndg[ 0][9]=1; ndg[ 0][10]=1; ndg[ 0][11]=1; ndg[ 0][12]=1; ndg[ 0][13]=1; 
			ndg[ 1][0]=0; ndg[ 1][1]=0; ndg[ 1][2]=1; ndg[ 1][3]=1; ndg[ 1][4]=1; ndg[ 1][5]=1; ndg[ 1][6]=1; ndg[ 1][7]=1; ndg[ 1][8]=0; ndg[ 1][9]=1; ndg[ 1][10]=1; ndg[ 1][11]=1; ndg[ 1][12]=1; ndg[ 1][13]=1;
			ndg[ 2][0]=0; ndg[ 2][1]=0; ndg[ 2][2]=0; ndg[ 2][3]=1; ndg[ 2][4]=1; ndg[ 2][5]=1; ndg[ 2][6]=1; ndg[ 2][7]=1; ndg[ 2][8]=1; ndg[ 2][9]=0; ndg[ 2][10]=1; ndg[ 2][11]=1; ndg[ 2][12]=1; ndg[ 2][13]=1;
			ndg[ 3][0]=0; ndg[ 3][1]=0; ndg[ 3][2]=0; ndg[ 3][3]=0; ndg[ 3][4]=1; ndg[ 3][5]=1; ndg[ 3][6]=1; ndg[ 3][7]=1; ndg[ 3][8]=1; ndg[ 3][9]=1; ndg[ 3][10]=0; ndg[ 3][11]=1; ndg[ 3][12]=1; ndg[ 3][13]=1;
			ndg[ 4][0]=0; ndg[ 4][1]=0; ndg[ 4][2]=0; ndg[ 4][3]=0; ndg[ 4][4]=0; ndg[ 4][5]=1; ndg[ 4][6]=1; ndg[ 4][7]=1; ndg[ 4][8]=1; ndg[ 4][9]=1; ndg[ 4][10]=1; ndg[ 4][11]=0; ndg[ 4][12]=1; ndg[ 4][13]=1;
			ndg[ 5][0]=0; ndg[ 5][1]=0; ndg[ 5][2]=0; ndg[ 5][3]=0; ndg[ 5][4]=0; ndg[ 5][5]=0; ndg[ 5][6]=1; ndg[ 5][7]=1; ndg[ 5][8]=1; ndg[ 5][9]=1; ndg[ 5][10]=1; ndg[ 5][11]=1; ndg[ 5][12]=0; ndg[ 5][13]=1;
			ndg[ 6][0]=0; ndg[ 6][1]=0; ndg[ 6][2]=0; ndg[ 6][3]=0; ndg[ 6][4]=0; ndg[ 6][5]=0; ndg[ 6][6]=0; ndg[ 6][7]=1; ndg[ 6][8]=1; ndg[ 6][9]=1; ndg[ 6][10]=1; ndg[ 6][11]=1; ndg[ 6][12]=1; ndg[ 6][13]=0;
			ndg[ 7][0]=0; ndg[ 7][1]=0; ndg[ 7][2]=0; ndg[ 7][3]=0; ndg[ 7][4]=0; ndg[ 7][5]=0; ndg[ 7][6]=0; ndg[ 7][7]=0; ndg[ 7][8]=1; ndg[ 7][9]=1; ndg[ 7][10]=1; ndg[ 7][11]=1; ndg[ 7][12]=1; ndg[ 7][13]=1;
			ndg[ 8][0]=0; ndg[ 8][1]=0; ndg[ 8][2]=0; ndg[ 8][3]=0; ndg[ 8][4]=0; ndg[ 8][5]=0; ndg[ 8][6]=0; ndg[ 8][7]=0; ndg[ 8][8]=0; ndg[ 8][9]=1; ndg[ 8][10]=1; ndg[ 8][11]=1; ndg[ 8][12]=1; ndg[ 8][13]=1;
			ndg[ 9][0]=0; ndg[ 9][1]=0; ndg[ 9][2]=0; ndg[ 9][3]=0; ndg[ 9][4]=0; ndg[ 9][5]=0; ndg[ 9][6]=0; ndg[ 9][7]=0; ndg[ 9][8]=0; ndg[ 9][9]=0; ndg[ 9][10]=1; ndg[ 9][11]=1; ndg[ 9][12]=1; ndg[ 9][13]=1;
			ndg[10][0]=0; ndg[10][1]=0; ndg[10][2]=0; ndg[10][3]=0; ndg[10][4]=0; ndg[10][5]=0; ndg[10][6]=0; ndg[10][7]=0; ndg[10][8]=0; ndg[10][9]=0; ndg[10][10]=0; ndg[10][11]=1; ndg[10][12]=1; ndg[10][13]=1;
			ndg[11][0]=0; ndg[11][1]=0; ndg[11][2]=0; ndg[11][3]=0; ndg[11][4]=0; ndg[11][5]=0; ndg[11][6]=0; ndg[11][7]=0; ndg[11][8]=0; ndg[11][9]=0; ndg[11][10]=0; ndg[11][11]=0; ndg[11][12]=1; ndg[11][13]=1;
			ndg[12][0]=0; ndg[12][1]=0; ndg[12][2]=0; ndg[12][3]=0; ndg[12][4]=0; ndg[12][5]=0; ndg[12][6]=0; ndg[12][7]=0; ndg[12][8]=0; ndg[12][9]=0; ndg[12][10]=0; ndg[12][11]=0; ndg[12][12]=0; ndg[12][13]=1;
			ndg[13][0]=0; ndg[13][1]=0; ndg[13][2]=0; ndg[13][3]=0; ndg[13][4]=0; ndg[13][5]=0; ndg[13][6]=0; ndg[13][7]=0; ndg[13][8]=0; ndg[13][9]=0; ndg[13][10]=0; ndg[13][11]=0; ndg[13][12]=0; ndg[13][13]=0;
		}
		break;
		
		case 18:
		{
			ndg[ 0][0]=0; ndg[ 0][1]=1; ndg[ 0][2]=1; ndg[ 0][3]=1; ndg[ 0][4]=1; ndg[ 0][5]=1; ndg[ 0][6]=1; ndg[ 0][7]=1; ndg[ 0][8]=1; ndg[ 0][9]=0; ndg[ 0][10]=1; ndg[ 0][11]=1; ndg[ 0][12]=1; ndg[ 0][13]=1; ndg[ 0][14]=1; ndg[ 0][15]=1; ndg[ 0][16]=1; ndg[ 0][17]=1; 
			ndg[ 1][0]=0; ndg[ 1][1]=0; ndg[ 1][2]=1; ndg[ 1][3]=1; ndg[ 1][4]=1; ndg[ 1][5]=1; ndg[ 1][6]=1; ndg[ 1][7]=1; ndg[ 1][8]=1; ndg[ 1][9]=1; ndg[ 1][10]=0; ndg[ 1][11]=1; ndg[ 1][12]=1; ndg[ 1][13]=1; ndg[ 1][14]=1; ndg[ 1][15]=1; ndg[ 1][16]=1; ndg[ 1][17]=1; 
			ndg[ 2][0]=0; ndg[ 2][1]=0; ndg[ 2][2]=0; ndg[ 2][3]=1; ndg[ 2][4]=1; ndg[ 2][5]=1; ndg[ 2][6]=1; ndg[ 2][7]=1; ndg[ 2][8]=1; ndg[ 2][9]=1; ndg[ 2][10]=1; ndg[ 2][11]=0; ndg[ 2][12]=1; ndg[ 2][13]=1; ndg[ 2][14]=1; ndg[ 2][15]=1; ndg[ 2][16]=1; ndg[ 2][17]=1; 
			ndg[ 3][0]=0; ndg[ 3][1]=0; ndg[ 3][2]=0; ndg[ 3][3]=0; ndg[ 3][4]=1; ndg[ 3][5]=1; ndg[ 3][6]=1; ndg[ 3][7]=1; ndg[ 3][8]=1; ndg[ 3][9]=1; ndg[ 3][10]=1; ndg[ 3][11]=1; ndg[ 3][12]=0; ndg[ 3][13]=1; ndg[ 3][14]=1; ndg[ 3][15]=1; ndg[ 3][16]=1; ndg[ 3][17]=1; 
			ndg[ 4][0]=0; ndg[ 4][1]=0; ndg[ 4][2]=0; ndg[ 4][3]=0; ndg[ 4][4]=0; ndg[ 4][5]=1; ndg[ 4][6]=1; ndg[ 4][7]=1; ndg[ 4][8]=1; ndg[ 4][9]=1; ndg[ 4][10]=1; ndg[ 4][11]=1; ndg[ 4][12]=1; ndg[ 4][13]=0; ndg[ 4][14]=1; ndg[ 4][15]=1; ndg[ 4][16]=1; ndg[ 4][17]=1; 
			ndg[ 5][0]=0; ndg[ 5][1]=0; ndg[ 5][2]=0; ndg[ 5][3]=0; ndg[ 5][4]=0; ndg[ 5][5]=0; ndg[ 5][6]=1; ndg[ 5][7]=1; ndg[ 5][8]=1; ndg[ 5][9]=1; ndg[ 5][10]=1; ndg[ 5][11]=1; ndg[ 5][12]=1; ndg[ 5][13]=1; ndg[ 5][14]=0; ndg[ 5][15]=1; ndg[ 5][16]=1; ndg[ 5][17]=1; 
			ndg[ 6][0]=0; ndg[ 6][1]=0; ndg[ 6][2]=0; ndg[ 6][3]=0; ndg[ 6][4]=0; ndg[ 6][5]=0; ndg[ 6][6]=0; ndg[ 6][7]=1; ndg[ 6][8]=1; ndg[ 6][9]=1; ndg[ 6][10]=1; ndg[ 6][11]=1; ndg[ 6][12]=1; ndg[ 6][13]=1; ndg[ 6][14]=1; ndg[ 6][15]=0; ndg[ 6][16]=1; ndg[ 6][17]=1; 
			ndg[ 7][0]=0; ndg[ 7][1]=0; ndg[ 7][2]=0; ndg[ 7][3]=0; ndg[ 7][4]=0; ndg[ 7][5]=0; ndg[ 7][6]=0; ndg[ 7][7]=0; ndg[ 7][8]=1; ndg[ 7][9]=1; ndg[ 7][10]=1; ndg[ 7][11]=1; ndg[ 7][12]=1; ndg[ 7][13]=1; ndg[ 7][14]=1; ndg[ 7][15]=1; ndg[ 7][16]=0; ndg[ 7][17]=1; 
			ndg[ 8][0]=0; ndg[ 8][1]=0; ndg[ 8][2]=0; ndg[ 8][3]=0; ndg[ 8][4]=0; ndg[ 8][5]=0; ndg[ 8][6]=0; ndg[ 8][7]=0; ndg[ 8][8]=0; ndg[ 8][9]=1; ndg[ 8][10]=1; ndg[ 8][11]=1; ndg[ 8][12]=1; ndg[ 8][13]=1; ndg[ 8][14]=1; ndg[ 8][15]=1; ndg[ 8][16]=1; ndg[ 8][17]=0; 
			ndg[ 9][0]=0; ndg[ 9][1]=0; ndg[ 9][2]=0; ndg[ 9][3]=0; ndg[ 9][4]=0; ndg[ 9][5]=0; ndg[ 9][6]=0; ndg[ 9][7]=0; ndg[ 9][8]=0; ndg[ 9][9]=0; ndg[ 9][10]=1; ndg[ 9][11]=1; ndg[ 9][12]=1; ndg[ 9][13]=1; ndg[ 9][14]=1; ndg[ 9][15]=1; ndg[ 9][16]=1; ndg[ 9][17]=1; 
			ndg[10][0]=0; ndg[10][1]=0; ndg[10][2]=0; ndg[10][3]=0; ndg[10][4]=0; ndg[10][5]=0; ndg[10][6]=0; ndg[10][7]=0; ndg[10][8]=0; ndg[10][9]=0; ndg[10][10]=0; ndg[10][11]=1; ndg[10][12]=1; ndg[10][13]=1; ndg[10][14]=1; ndg[10][15]=1; ndg[10][16]=1; ndg[10][17]=1; 
			ndg[11][0]=0; ndg[11][1]=0; ndg[11][2]=0; ndg[11][3]=0; ndg[11][4]=0; ndg[11][5]=0; ndg[11][6]=0; ndg[11][7]=0; ndg[11][8]=0; ndg[11][9]=0; ndg[11][10]=0; ndg[11][11]=0; ndg[11][12]=1; ndg[11][13]=1; ndg[11][14]=1; ndg[11][15]=1; ndg[11][16]=1; ndg[11][17]=1; 
			ndg[12][0]=0; ndg[12][1]=0; ndg[12][2]=0; ndg[12][3]=0; ndg[12][4]=0; ndg[12][5]=0; ndg[12][6]=0; ndg[12][7]=0; ndg[12][8]=0; ndg[12][9]=0; ndg[12][10]=0; ndg[12][11]=0; ndg[12][12]=0; ndg[12][13]=1; ndg[12][14]=1; ndg[12][15]=1; ndg[12][16]=1; ndg[12][17]=1; 
			ndg[13][0]=0; ndg[13][1]=0; ndg[13][2]=0; ndg[13][3]=0; ndg[13][4]=0; ndg[13][5]=0; ndg[13][6]=0; ndg[13][7]=0; ndg[13][8]=0; ndg[13][9]=0; ndg[13][10]=0; ndg[13][11]=0; ndg[13][12]=0; ndg[13][13]=0; ndg[13][14]=1; ndg[13][15]=1; ndg[13][16]=1; ndg[13][17]=1; 
			ndg[14][0]=0; ndg[14][1]=0; ndg[14][2]=0; ndg[14][3]=0; ndg[14][4]=0; ndg[14][5]=0; ndg[14][6]=0; ndg[14][7]=0; ndg[14][8]=0; ndg[14][9]=0; ndg[14][10]=0; ndg[14][11]=0; ndg[14][12]=0; ndg[14][13]=0; ndg[14][14]=0; ndg[14][15]=1; ndg[14][16]=1; ndg[14][17]=1; 
			ndg[15][0]=0; ndg[15][1]=0; ndg[15][2]=0; ndg[15][3]=0; ndg[15][4]=0; ndg[15][5]=0; ndg[15][6]=0; ndg[15][7]=0; ndg[15][8]=0; ndg[15][9]=0; ndg[15][10]=0; ndg[15][11]=0; ndg[15][12]=0; ndg[15][13]=0; ndg[15][14]=0; ndg[15][15]=0; ndg[15][16]=1; ndg[15][17]=1; 
			ndg[16][0]=0; ndg[16][1]=0; ndg[16][2]=0; ndg[16][3]=0; ndg[16][4]=0; ndg[16][5]=0; ndg[16][6]=0; ndg[16][7]=0; ndg[16][8]=0; ndg[16][9]=0; ndg[16][10]=0; ndg[16][11]=0; ndg[16][12]=0; ndg[16][13]=0; ndg[16][14]=0; ndg[16][15]=0; ndg[16][16]=0; ndg[16][17]=1; 
			ndg[17][0]=0; ndg[17][1]=0; ndg[17][2]=0; ndg[17][3]=0; ndg[17][4]=0; ndg[17][5]=0; ndg[17][6]=0; ndg[17][7]=0; ndg[17][8]=0; ndg[17][9]=0; ndg[17][10]=0; ndg[17][11]=0; ndg[17][12]=0; ndg[17][13]=0; ndg[17][14]=0; ndg[17][15]=0; ndg[17][16]=0; ndg[17][17]=0;
		}
		break;
		
		case 26:
		{
			ndg[ 0][0]=0; ndg[ 0][1]=1; ndg[ 0][2]=1; ndg[ 0][3]=1; ndg[ 0][4]=1; ndg[ 0][5]=1; ndg[ 0][6]=1; ndg[ 0][7]=1; ndg[ 0][8]=1; ndg[ 0][9]=1; ndg[ 0][10]=1; ndg[ 0][11]=1; ndg[ 0][12]=1; ndg[ 0][13]=0; ndg[ 0][14]=1; ndg[ 0][15]=1; ndg[ 0][16]=1; ndg[ 0][17]=1; ndg[ 0][18]=1; ndg[ 0][19]=1; ndg[ 0][20]=1; ndg[ 0][21]=1; ndg[ 0][22]=1; ndg[ 0][23]=1; ndg[ 0][24]=1; ndg[ 0][25]=1;
			ndg[ 1][0]=0; ndg[ 1][1]=0; ndg[ 1][2]=1; ndg[ 1][3]=1; ndg[ 1][4]=1; ndg[ 1][5]=1; ndg[ 1][6]=1; ndg[ 1][7]=1; ndg[ 1][8]=1; ndg[ 1][9]=1; ndg[ 1][10]=1; ndg[ 1][11]=1; ndg[ 1][12]=1; ndg[ 1][13]=1; ndg[ 1][14]=0; ndg[ 1][15]=1; ndg[ 1][16]=1; ndg[ 1][17]=1; ndg[ 1][18]=1; ndg[ 1][19]=1; ndg[ 1][20]=1; ndg[ 1][21]=1; ndg[ 1][22]=1; ndg[ 1][23]=1; ndg[ 1][24]=1; ndg[ 1][25]=1;
			ndg[ 2][0]=0; ndg[ 2][1]=0; ndg[ 2][2]=0; ndg[ 2][3]=1; ndg[ 2][4]=1; ndg[ 2][5]=1; ndg[ 2][6]=1; ndg[ 2][7]=1; ndg[ 2][8]=1; ndg[ 2][9]=1; ndg[ 2][10]=1; ndg[ 2][11]=1; ndg[ 2][12]=1; ndg[ 2][13]=1; ndg[ 2][14]=1; ndg[ 2][15]=0; ndg[ 2][16]=1; ndg[ 2][17]=1; ndg[ 2][18]=1; ndg[ 2][19]=1; ndg[ 2][20]=1; ndg[ 2][21]=1; ndg[ 2][22]=1; ndg[ 2][23]=1; ndg[ 2][24]=1; ndg[ 2][25]=1;
			ndg[ 3][0]=0; ndg[ 3][1]=0; ndg[ 3][2]=0; ndg[ 3][3]=0; ndg[ 3][4]=1; ndg[ 3][5]=1; ndg[ 3][6]=1; ndg[ 3][7]=1; ndg[ 3][8]=1; ndg[ 3][9]=1; ndg[ 3][10]=1; ndg[ 3][11]=1; ndg[ 3][12]=1; ndg[ 3][13]=1; ndg[ 3][14]=1; ndg[ 3][15]=1; ndg[ 3][16]=0; ndg[ 3][17]=1; ndg[ 3][18]=1; ndg[ 3][19]=1; ndg[ 3][20]=1; ndg[ 3][21]=1; ndg[ 3][22]=1; ndg[ 3][23]=1; ndg[ 3][24]=1; ndg[ 3][25]=1;
			ndg[ 4][0]=0; ndg[ 4][1]=0; ndg[ 4][2]=0; ndg[ 4][3]=0; ndg[ 4][4]=0; ndg[ 4][5]=1; ndg[ 4][6]=1; ndg[ 4][7]=1; ndg[ 4][8]=1; ndg[ 4][9]=1; ndg[ 4][10]=1; ndg[ 4][11]=1; ndg[ 4][12]=1; ndg[ 4][13]=1; ndg[ 4][14]=1; ndg[ 4][15]=1; ndg[ 4][16]=1; ndg[ 4][17]=0; ndg[ 4][18]=1; ndg[ 4][19]=1; ndg[ 4][20]=1; ndg[ 4][21]=1; ndg[ 4][22]=1; ndg[ 4][23]=1; ndg[ 4][24]=1; ndg[ 4][25]=1;
			ndg[ 5][0]=0; ndg[ 5][1]=0; ndg[ 5][2]=0; ndg[ 5][3]=0; ndg[ 5][4]=0; ndg[ 5][5]=0; ndg[ 5][6]=1; ndg[ 5][7]=1; ndg[ 5][8]=1; ndg[ 5][9]=1; ndg[ 5][10]=1; ndg[ 5][11]=1; ndg[ 5][12]=1; ndg[ 5][13]=1; ndg[ 5][14]=1; ndg[ 5][15]=1; ndg[ 5][16]=1; ndg[ 5][17]=1; ndg[ 5][18]=0; ndg[ 5][19]=1; ndg[ 5][20]=1; ndg[ 5][21]=1; ndg[ 5][22]=1; ndg[ 5][23]=1; ndg[ 5][24]=1; ndg[ 5][25]=1;
			ndg[ 6][0]=0; ndg[ 6][1]=0; ndg[ 6][2]=0; ndg[ 6][3]=0; ndg[ 6][4]=0; ndg[ 6][5]=0; ndg[ 6][6]=0; ndg[ 6][7]=1; ndg[ 6][8]=1; ndg[ 6][9]=1; ndg[ 6][10]=1; ndg[ 6][11]=1; ndg[ 6][12]=1; ndg[ 6][13]=1; ndg[ 6][14]=1; ndg[ 6][15]=1; ndg[ 6][16]=1; ndg[ 6][17]=1; ndg[ 6][18]=1; ndg[ 6][19]=0; ndg[ 6][20]=1; ndg[ 6][21]=1; ndg[ 6][22]=1; ndg[ 6][23]=1; ndg[ 6][24]=1; ndg[ 6][25]=1;
			ndg[ 7][0]=0; ndg[ 7][1]=0; ndg[ 7][2]=0; ndg[ 7][3]=0; ndg[ 7][4]=0; ndg[ 7][5]=0; ndg[ 7][6]=0; ndg[ 7][7]=0; ndg[ 7][8]=1; ndg[ 7][9]=1; ndg[ 7][10]=1; ndg[ 7][11]=1; ndg[ 7][12]=1; ndg[ 7][13]=1; ndg[ 7][14]=1; ndg[ 7][15]=1; ndg[ 7][16]=1; ndg[ 7][17]=1; ndg[ 7][18]=1; ndg[ 7][19]=1; ndg[ 7][20]=0; ndg[ 7][21]=1; ndg[ 7][22]=1; ndg[ 7][23]=1; ndg[ 7][24]=1; ndg[ 7][25]=1;
			ndg[ 8][0]=0; ndg[ 8][1]=0; ndg[ 8][2]=0; ndg[ 8][3]=0; ndg[ 8][4]=0; ndg[ 8][5]=0; ndg[ 8][6]=0; ndg[ 8][7]=0; ndg[ 8][8]=0; ndg[ 8][9]=1; ndg[ 8][10]=1; ndg[ 8][11]=1; ndg[ 8][12]=1; ndg[ 8][13]=1; ndg[ 8][14]=1; ndg[ 8][15]=1; ndg[ 8][16]=1; ndg[ 8][17]=1; ndg[ 8][18]=1; ndg[ 8][19]=1; ndg[ 8][20]=1; ndg[ 8][21]=0; ndg[ 8][22]=1; ndg[ 8][23]=1; ndg[ 8][24]=1; ndg[ 8][25]=1;
			ndg[ 9][0]=0; ndg[ 9][1]=0; ndg[ 9][2]=0; ndg[ 9][3]=0; ndg[ 9][4]=0; ndg[ 9][5]=0; ndg[ 9][6]=0; ndg[ 9][7]=0; ndg[ 9][8]=0; ndg[ 9][9]=0; ndg[ 9][10]=1; ndg[ 9][11]=1; ndg[ 9][12]=1; ndg[ 9][13]=1; ndg[ 9][14]=1; ndg[ 9][15]=1; ndg[ 9][16]=1; ndg[ 9][17]=1; ndg[ 9][18]=1; ndg[ 9][19]=1; ndg[ 9][20]=1; ndg[ 9][21]=1; ndg[ 9][22]=0; ndg[ 9][23]=1; ndg[ 9][24]=1; ndg[ 9][25]=1;
			ndg[10][0]=0; ndg[10][1]=0; ndg[10][2]=0; ndg[10][3]=0; ndg[10][4]=0; ndg[10][5]=0; ndg[10][6]=0; ndg[10][7]=0; ndg[10][8]=0; ndg[10][9]=0; ndg[10][10]=0; ndg[10][11]=1; ndg[10][12]=1; ndg[10][13]=1; ndg[10][14]=1; ndg[10][15]=1; ndg[10][16]=1; ndg[10][17]=1; ndg[10][18]=1; ndg[10][19]=1; ndg[10][20]=1; ndg[10][21]=1; ndg[10][22]=1; ndg[10][23]=0; ndg[10][24]=1; ndg[10][25]=1;
			ndg[11][0]=0; ndg[11][1]=0; ndg[11][2]=0; ndg[11][3]=0; ndg[11][4]=0; ndg[11][5]=0; ndg[11][6]=0; ndg[11][7]=0; ndg[11][8]=0; ndg[11][9]=0; ndg[11][10]=0; ndg[11][11]=0; ndg[11][12]=1; ndg[11][13]=1; ndg[11][14]=1; ndg[11][15]=1; ndg[11][16]=1; ndg[11][17]=1; ndg[11][18]=1; ndg[11][19]=1; ndg[11][20]=1; ndg[11][21]=1; ndg[11][22]=1; ndg[11][23]=1; ndg[11][24]=0; ndg[11][25]=1;
			ndg[12][0]=0; ndg[12][1]=0; ndg[12][2]=0; ndg[12][3]=0; ndg[12][4]=0; ndg[12][5]=0; ndg[12][6]=0; ndg[12][7]=0; ndg[12][8]=0; ndg[12][9]=0; ndg[12][10]=0; ndg[12][11]=0; ndg[12][12]=0; ndg[12][13]=1; ndg[12][14]=1; ndg[12][15]=1; ndg[12][16]=1; ndg[12][17]=1; ndg[12][18]=1; ndg[12][19]=1; ndg[12][20]=1; ndg[12][21]=1; ndg[12][22]=1; ndg[12][23]=1; ndg[12][24]=1; ndg[12][25]=0;
			ndg[13][0]=0; ndg[13][1]=0; ndg[13][2]=0; ndg[13][3]=0; ndg[13][4]=0; ndg[13][5]=0; ndg[13][6]=0; ndg[13][7]=0; ndg[13][8]=0; ndg[13][9]=0; ndg[13][10]=0; ndg[13][11]=0; ndg[13][12]=0; ndg[13][13]=0; ndg[13][14]=1; ndg[13][15]=1; ndg[13][16]=1; ndg[13][17]=1; ndg[13][18]=1; ndg[13][19]=1; ndg[13][20]=1; ndg[13][21]=1; ndg[13][22]=1; ndg[13][23]=1; ndg[13][24]=1; ndg[13][25]=1;
			ndg[14][0]=0; ndg[14][1]=0; ndg[14][2]=0; ndg[14][3]=0; ndg[14][4]=0; ndg[14][5]=0; ndg[14][6]=0; ndg[14][7]=0; ndg[14][8]=0; ndg[14][9]=0; ndg[14][10]=0; ndg[14][11]=0; ndg[14][12]=0; ndg[14][13]=0; ndg[14][14]=0; ndg[14][15]=1; ndg[14][16]=1; ndg[14][17]=1; ndg[14][18]=1; ndg[14][19]=1; ndg[14][20]=1; ndg[14][21]=1; ndg[14][22]=1; ndg[14][23]=1; ndg[14][24]=1; ndg[14][25]=1;
			ndg[15][0]=0; ndg[15][1]=0; ndg[15][2]=0; ndg[15][3]=0; ndg[15][4]=0; ndg[15][5]=0; ndg[15][6]=0; ndg[15][7]=0; ndg[15][8]=0; ndg[15][9]=0; ndg[15][10]=0; ndg[15][11]=0; ndg[15][12]=0; ndg[15][13]=0; ndg[15][14]=0; ndg[15][15]=0; ndg[15][16]=1; ndg[15][17]=1; ndg[15][18]=1; ndg[15][19]=1; ndg[15][20]=1; ndg[15][21]=1; ndg[15][22]=1; ndg[15][23]=1; ndg[15][24]=1; ndg[15][25]=1;
			ndg[16][0]=0; ndg[16][1]=0; ndg[16][2]=0; ndg[16][3]=0; ndg[16][4]=0; ndg[16][5]=0; ndg[16][6]=0; ndg[16][7]=0; ndg[16][8]=0; ndg[16][9]=0; ndg[16][10]=0; ndg[16][11]=0; ndg[16][12]=0; ndg[16][13]=0; ndg[16][14]=0; ndg[16][15]=0; ndg[16][16]=0; ndg[16][17]=1; ndg[16][18]=1; ndg[16][19]=1; ndg[16][20]=1; ndg[16][21]=1; ndg[16][22]=1; ndg[16][23]=1; ndg[16][24]=1; ndg[16][25]=1; 
			ndg[17][0]=0; ndg[17][1]=0; ndg[17][2]=0; ndg[17][3]=0; ndg[17][4]=0; ndg[17][5]=0; ndg[17][6]=0; ndg[17][7]=0; ndg[17][8]=0; ndg[17][9]=0; ndg[17][10]=0; ndg[17][11]=0; ndg[17][12]=0; ndg[17][13]=0; ndg[17][14]=0; ndg[17][15]=0; ndg[17][16]=0; ndg[17][17]=0; ndg[17][18]=1; ndg[17][19]=1; ndg[17][20]=1; ndg[17][21]=1; ndg[17][22]=1; ndg[17][23]=1; ndg[17][24]=1; ndg[17][25]=1;
			ndg[18][0]=0; ndg[18][1]=0; ndg[18][2]=0; ndg[18][3]=0; ndg[18][4]=0; ndg[18][5]=0; ndg[18][6]=0; ndg[18][7]=0; ndg[18][8]=0; ndg[18][9]=0; ndg[18][10]=0; ndg[18][11]=0; ndg[18][12]=0; ndg[18][13]=0; ndg[18][14]=0; ndg[18][15]=0; ndg[18][16]=0; ndg[18][17]=0; ndg[18][18]=0; ndg[18][19]=1; ndg[18][20]=1; ndg[18][21]=1; ndg[18][22]=1; ndg[18][23]=1; ndg[18][24]=1; ndg[18][25]=1;
			ndg[19][0]=0; ndg[19][1]=0; ndg[19][2]=0; ndg[19][3]=0; ndg[19][4]=0; ndg[19][5]=0; ndg[19][6]=0; ndg[19][7]=0; ndg[19][8]=0; ndg[19][9]=0; ndg[19][10]=0; ndg[19][11]=0; ndg[19][12]=0; ndg[19][13]=0; ndg[19][14]=0; ndg[19][15]=0; ndg[19][16]=0; ndg[19][17]=0; ndg[19][18]=0; ndg[19][19]=0; ndg[19][20]=1; ndg[19][21]=1; ndg[19][22]=1; ndg[19][23]=1; ndg[19][24]=1; ndg[19][25]=1;
			ndg[20][0]=0; ndg[20][1]=0; ndg[20][2]=0; ndg[20][3]=0; ndg[20][4]=0; ndg[20][5]=0; ndg[20][6]=0; ndg[20][7]=0; ndg[20][8]=0; ndg[20][9]=0; ndg[20][10]=0; ndg[20][11]=0; ndg[20][12]=0; ndg[20][13]=0; ndg[20][14]=0; ndg[20][15]=0; ndg[20][16]=0; ndg[20][17]=0; ndg[20][18]=0; ndg[20][19]=0; ndg[20][20]=0; ndg[20][21]=1; ndg[20][22]=1; ndg[20][23]=1; ndg[20][24]=1; ndg[20][25]=1;
			ndg[21][0]=0; ndg[21][1]=0; ndg[21][2]=0; ndg[21][3]=0; ndg[21][4]=0; ndg[21][5]=0; ndg[21][6]=0; ndg[21][7]=0; ndg[21][8]=0; ndg[21][9]=0; ndg[21][10]=0; ndg[21][11]=0; ndg[21][12]=0; ndg[21][13]=0; ndg[21][14]=0; ndg[21][15]=0; ndg[21][16]=0; ndg[21][17]=0; ndg[21][18]=0; ndg[21][19]=0; ndg[21][20]=0; ndg[21][21]=0; ndg[21][22]=1; ndg[21][23]=1; ndg[21][24]=1; ndg[21][25]=1;
			ndg[22][0]=0; ndg[22][1]=0; ndg[22][2]=0; ndg[22][3]=0; ndg[22][4]=0; ndg[22][5]=0; ndg[22][6]=0; ndg[22][7]=0; ndg[22][8]=0; ndg[22][9]=0; ndg[22][10]=0; ndg[22][11]=0; ndg[22][12]=0; ndg[22][13]=0; ndg[22][14]=0; ndg[22][15]=0; ndg[22][16]=0; ndg[22][17]=0; ndg[22][18]=0; ndg[22][19]=0; ndg[22][20]=0; ndg[22][21]=0; ndg[22][22]=0; ndg[22][23]=1; ndg[22][24]=1; ndg[22][25]=1;
			ndg[23][0]=0; ndg[23][1]=0; ndg[23][2]=0; ndg[23][3]=0; ndg[23][4]=0; ndg[23][5]=0; ndg[23][6]=0; ndg[23][7]=0; ndg[23][8]=0; ndg[23][9]=0; ndg[23][10]=0; ndg[23][11]=0; ndg[23][12]=0; ndg[23][13]=0; ndg[23][14]=0; ndg[23][15]=0; ndg[23][16]=0; ndg[23][17]=0; ndg[23][18]=0; ndg[23][19]=0; ndg[23][20]=0; ndg[23][21]=0; ndg[23][22]=0; ndg[23][23]=0; ndg[23][24]=1; ndg[23][25]=1;
			ndg[24][0]=0; ndg[24][1]=0; ndg[24][2]=0; ndg[24][3]=0; ndg[24][4]=0; ndg[24][5]=0; ndg[24][6]=0; ndg[24][7]=0; ndg[24][8]=0; ndg[24][9]=0; ndg[24][10]=0; ndg[24][11]=0; ndg[24][12]=0; ndg[24][13]=0; ndg[24][14]=0; ndg[24][15]=0; ndg[24][16]=0; ndg[24][17]=0; ndg[24][18]=0; ndg[24][19]=0; ndg[24][20]=0; ndg[24][21]=0; ndg[24][22]=0; ndg[24][23]=0; ndg[24][24]=0; ndg[24][25]=1;
			ndg[25][0]=0; ndg[25][1]=0; ndg[25][2]=0; ndg[25][3]=0; ndg[25][4]=0; ndg[25][5]=0; ndg[25][6]=0; ndg[25][7]=0; ndg[25][8]=0; ndg[25][9]=0; ndg[25][10]=0; ndg[25][11]=0; ndg[25][12]=0; ndg[25][13]=0; ndg[25][14]=0; ndg[25][15]=0; ndg[25][16]=0; ndg[25][17]=0; ndg[25][18]=0; ndg[25][19]=0; ndg[25][20]=0; ndg[25][21]=0; ndg[25][22]=0; ndg[25][23]=0; ndg[25][24]=0; ndg[25][25]=0;
		}
		break;
	}
}

std::vector<Vector3f>& KDOP::debug()
{
	vertices.clear();
	
	//~ bool ndg[K][K];
	
	CA connection[K][K];
	
	//~ for(int i = 0; i < K; i++)
		//~ for(int j = 0; j < K; j++)
			//~ ndg[i][j] = false;
	
	for(int i = 0; i < K; i++)
	{
		for(int j = i+1; j < K; j++)
		{
			//~ Vector3i pair = normal[i] + normal[j];
			
			//~ Vector3i cross = pair.cross(normal[i]);
			
			//debug
			//~ std::cout << "p: " << pair.x << " " << pair.y << " " << pair.z << std::endl;
			//~ std::cout << "c: " << cross.x << " " << cross.y << " " << cross.z << std::endl;
			//~ std::cout << std::endl;
			
			//~ if(!(cross.x == 0 && cross.y == 0 && cross.z == 0))
				//~ ndg[i][j] = true;
			
			if(ndg[i][j])
			{
				//build list of good indices
				for(int a = 0; a < K; a++)
				{
					if((a != i) && (a != j))
					{
						int det = Vector3i::determinant(normal[i], normal[j], normal[a]);
						
						if(det != 0)
						{
							bool good = true;
							
							for(int b = 0; b < K; b++)
							{
								if((b != i) && (b != j) && (b != a))
								{
									int ijb = Vector3i::determinant(normal[i], normal[j], normal[b]);
									int iba = Vector3i::determinant(normal[i], normal[b], normal[a]);
									int bja = Vector3i::determinant(normal[b], normal[j], normal[a]);
									
									if(det > 0)
									{
										if(!( (ijb < 0) || (iba < 0) || (bja < 0) ))
										{
											good = false;
											break;
										}
									}
									else
									{
										if(!( (ijb > 0) || (iba > 0) || (bja > 0) ))
										{
											good = false;
											break;
										}
									}
								}
							}
							
							if(good)
							{
								//calc normal between p[i] x p[j]
								Vector3f pn = normal[i].cross(normal[j]);
								
								//calc iprojector
								Vector3f iv = normal[j].cross(pn);
								Vector3f iproj = iv/(iv.dot(normal[i]));
								
								//calc jprojector
								Vector3f jv = normal[i].cross(pn);
								Vector3f jproj = jv/(jv.dot(normal[j]));
								
								//debug
								//~ std::cout << "normal[i]: <"<< normal[i].x << "," << normal[i].y << "," << normal[i].z << ">" << std::endl;
								//~ std::cout << "normal[j]: <"<< normal[j].x << "," << normal[j].y << "," << normal[j].z << ">" << std::endl;
								//~ std::cout << "pn       : <"<< pn.x << "," << pn.y << "," << pn.z << ">" << std::endl;
								//~ std::cout << "iv       : <"<< iv.x << "," << iv.y << "," << iv.z << ">" << std::endl;
								//~ std::cout << "iproj    : <"<< iproj.x << "," << iproj.y << "," << iproj.z << ">" << std::endl;
								//~ std::cout << "jv       : <"<< jv.x << "," << jv.y << "," << jv.z << ">" << std::endl;
								//~ std::cout << "jproj    : <"<< jproj.x << "," << jproj.y << "," << jproj.z << ">" << std::endl;
								//~ std::cout << std::endl;
								
								CA::Connection c;
								c.index = a;
								c.iscal = iproj.dot(normal[a]);
								c.jscal = jproj.dot(normal[a]);
								c.nscal = pn.dot(normal[a]);
								
								//debug
								//~ std::cout << "iscal: " << c.iscal << std::endl;
								//~ std::cout << "jscal: " << c.jscal << std::endl;
								//~ std::cout << "nscal: " << c.nscal << std::endl;
								//~ std::cout << std::endl;
								
								connection[i][j].pn = pn;
								connection[i][j].iproj = iproj;
								connection[i][j].jproj = jproj;
								connection[i][j].array.push_back(c);
							}
						}
					}
				}
			}
		}
	}
	
	//debug
	//~ for(int i = 0; i < K; i++)
	//~ {
		//~ for(int j = 0; j < K; j++)
		//~ {
			//~ std::cout << ndg[i][j] << " " << std::ends;
		//~ }
		//~ 
		//~ std::cout << std::endl;
	//~ }
	//~ std::cout << std::endl;
	
	//debug
	//~ for(int i = 0; i < k; i++)
	//~ {
		//~ for(int j = 0; j < k; j++)
		//~ {
			//~ std::cout << connection[i][j].array.size() << " " << std::ends;
		//~ }
		//~ 
		//~ std::cout << std::endl;
	//~ }
	
	//TODO possibly set first half of distances to negative for rendering
	float M[K];
	
	for(int i = 0; i < K/2; i++)
	{
		M[i] = -distance[i];
		M[i+K/2] = distance[i+K/2];
	}
	
	//debug
	//~ for(int i = 0; i < k; i++)
	//~ {
		//~ std::cout << "M: " << M[i] << "\td: " << distance[i] << std::endl;
	//~ }
	
	for(int i = 0; i < K; i++)
	{
		for(int j = 0; j < K; j++)
		{
			if(ndg[i][j])
			{
				int N = connection[i][j].array.size();
				
				if(N > 0)
				{
					float min = -FLT_MAX;
					float max =  FLT_MAX;
					
					for(int a = 0; a < N; a++)
					{
						CA::Connection c;
						c.index = connection[i][j].array[a].index;
						c.nscal = connection[i][j].array[a].nscal;
						c.iscal = connection[i][j].array[a].iscal;
						c.jscal = connection[i][j].array[a].jscal;
						
						if(c.nscal < 0)
						{
							float value = (M[c.index] - M[i] * c.iscal - M[j] * c.jscal)/c.nscal;
							if(value > min)
								min = value;
						}
						else if(c.nscal > 0)
						{
							float value = (M[c.index] - M[i] * c.iscal - M[j] * c.jscal)/c.nscal;
							
							if(value < max)
								max = value;
						}
						
						if(max < min)
						{
							//debug
							//~ std::cout << "max: " << max << " min: " << min << std::endl;
							//~ std::cout << "break!" << std::endl;
							
							break;
						}
					}
					
					if(min < max)
					{
						Vector3f X = (connection[i][j].iproj * M[i]) + (connection[i][j].jproj * M[j]);
						
						Vector3f vmin(X + (connection[i][j].pn * min));
						Vector3f vmax(X + (connection[i][j].pn * max));
						
						//debug
						//~ std::cout << "i: " << i << " j: " << j << std::endl;
						//~ std::cout << "iproj: <" << connection[i][j].iproj.x << "," << connection[i][j].iproj.y << "," << connection[i][j].iproj.z << ">" << std::endl;
						//~ std::cout << "jproj: <" << connection[i][j].jproj.x << "," << connection[i][j].jproj.y << "," << connection[i][j].jproj.z << ">" << std::endl;
						//~ std::cout << "M[i]: " << M[i] << " M[j]: " << M[j] << std::endl;
						//~ std::cout << "Min: " << min << " Max: " << max << std::endl;
						//~ std::cout << "X: <" << X.x << "," << X.y << "," << X.z << ">" << std::endl;
						//~ std::cout << "pn: <" << connection[i][j].pn.x << "," << connection[i][j].pn.y << "," << connection[i][j].pn.z << ">" << std::endl;
						//~ std::cout << "vmin: <" << vmin.x << "," << vmin.y << "," << vmin.z << ">" << std::endl;
						//~ std::cout << "vmax: <" << vmax.x << "," << vmax.y << "," << vmax.z << ">" << std::endl;
						//~ std::cout << std::endl;
						
						vertices.push_back(vmax);
						vertices.push_back(vmin);
					}
				}
			}
		}
	}
	
	//debug
	//~ for(int i = 0; i < vertices.size(); i++)
	//~ {
		//~ std::cout << i << ": " << vertices[i].x << " " << vertices[i].y << " " << vertices[i].z << std::endl;
	//~ }
	
	return vertices;
}

//Returns true if there is a collision with the given KDOP
bool KDOP::collides(const KDOP* kdop)
{
	//Can't compare objects with a different number of planes
	if(kdop->K != K)
		return false;
	
	int halfK = K/2;
	
	for(int i = 0; i < halfK; i++)
	{
		if(distance[i] > kdop->distance[halfK+i])
			return false;
		
		if(distance[halfK+i] < kdop->distance[i])
			return false;
	}
	
	return true;
}

//Returns true if this KDOP can merge with the given KDOP
bool KDOP::merge(const KDOP* kdop)
{
	//Can't merge objects with a different number of planes
	if(kdop->K != K)
		return false;
	
	int halfK = K/2;
	
	for(int i = 0; i < halfK; i++)
	{
		distance[i] = std::min(kdop->distance[i], distance[i]);
		distance[halfK+i] = std::max(kdop->distance[halfK+i], distance[halfK+i]);
	}
	
	return true;
}
