
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <cmath>
#include <cstdlib>

#include "hair.h"
#include "constants.h"

pilar::Hair* hair = NULL;

Model_OBJ obj;

Model_OBJ::Model_OBJ()
{
	this->TotalConnectedTriangles = 0; 
	this->TotalConnectedPoints = 0;
}
 
float* Model_OBJ::calculateNormal( float *coord1, float *coord2, float *coord3, float *norm )
{
	/* calculate Vector1 and Vector2 */
	float va[3], vb[3], vr[3], val;
	va[0] = coord1[0] - coord2[0];
	va[1] = coord1[1] - coord2[1];
	va[2] = coord1[2] - coord2[2];

	vb[0] = coord1[0] - coord3[0];
	vb[1] = coord1[1] - coord3[1];
	vb[2] = coord1[2] - coord3[2];

	/* cross product */
	vr[0] = va[1] * vb[2] - vb[1] * va[2];
	vr[1] = vb[0] * va[2] - va[0] * vb[2];
	vr[2] = va[0] * vb[1] - vb[0] * va[1];

	/* normalization factor */
	val = sqrtf( vr[0]*vr[0] + vr[1]*vr[1] + vr[2]*vr[2] );

	norm[0] = vr[0]/val;
	norm[1] = vr[1]/val;
	norm[2] = vr[2]/val;
}
 
 
int Model_OBJ::Load(const char* filename)
{
	std::string line;
	std::ifstream objFile (filename);	
	if (objFile.is_open())													// If obj file is open, continue
	{
		objFile.seekg (0, std::ios::end);										// Go to end of the file, 
		long fileSize = objFile.tellg();									// get file size
		objFile.seekg (0, std::ios::beg);										// we'll use this to register memory for our 3d model
 
		vertexBuffer = (float*) malloc (fileSize);							// Allocate memory for the verteces
		Faces_Triangles = (float*) malloc(fileSize*sizeof(float));			// Allocate memory for the triangles
		normals  = (float*) malloc(fileSize*sizeof(float));					// Allocate memory for the normals
 
		int triangle_index = 0;												// Set triangle index to zero
		int normal_index = 0;												// Set normal index to zero
 
		while (! objFile.eof() )											// Start reading file data
		{		
			getline (objFile,line);											// Get line from file
 
			if (line.c_str()[0] == 'v')										// The first character is a v: on this line is a vertex stored.
			{
				line[0] = ' ';												// Set first character to 0. This will allow us to use sscanf
 
				sscanf(line.c_str(),"%f %f %f ",							// Read floats from the line: v X Y Z
					&vertexBuffer[TotalConnectedPoints],
					&vertexBuffer[TotalConnectedPoints+1], 
					&vertexBuffer[TotalConnectedPoints+2]);
 
				TotalConnectedPoints += POINTS_PER_VERTEX;					// Add 3 to the total connected points
			}
			if (line.c_str()[0] == 'f')										// The first character is an 'f': on this line is a point stored
			{
		    	line[0] = ' ';												// Set first character to 0. This will allow us to use sscanf
 
				int vertexNumber[4] = { 0, 0, 0 };
                sscanf(line.c_str(),"%i%i%i",								// Read integers from the line:  f 1 2 3
					&vertexNumber[0],										// First point of our triangle. This is an 
					&vertexNumber[1],										// pointer to our vertexBuffer list
					&vertexNumber[2] );										// each point represents an X,Y,Z.
 
				vertexNumber[0] -= 1;										// OBJ file starts counting from 1
				vertexNumber[1] -= 1;										// OBJ file starts counting from 1
				vertexNumber[2] -= 1;										// OBJ file starts counting from 1
 
 
				/********************************************************************
				 * Create triangles (f 1 2 3) from points: (v X Y Z) (v X Y Z) (v X Y Z). 
				 * The vertexBuffer contains all verteces
				 * The triangles will be created using the verteces we read previously
				 */
 
				int tCounter = 0;
				for (int i = 0; i < POINTS_PER_VERTEX; i++)					
				{
					Faces_Triangles[triangle_index + tCounter   ] = vertexBuffer[3*vertexNumber[i] ];
					Faces_Triangles[triangle_index + tCounter +1 ] = vertexBuffer[3*vertexNumber[i]+1 ];
					Faces_Triangles[triangle_index + tCounter +2 ] = vertexBuffer[3*vertexNumber[i]+2 ];
					tCounter += POINTS_PER_VERTEX;
				}
 
				/*********************************************************************
				 * Calculate all normals, used for lighting
				 */ 
				float coord1[3] = { Faces_Triangles[triangle_index], Faces_Triangles[triangle_index+1],Faces_Triangles[triangle_index+2]};
				float coord2[3] = {Faces_Triangles[triangle_index+3],Faces_Triangles[triangle_index+4],Faces_Triangles[triangle_index+5]};
				float coord3[3] = {Faces_Triangles[triangle_index+6],Faces_Triangles[triangle_index+7],Faces_Triangles[triangle_index+8]};
				
				float norm[3];
				this->calculateNormal(coord1, coord2, coord3, norm);
//				float *norm = this->calculateNormal( coord1, coord2, coord3 );
 
				tCounter = 0;
				for (int i = 0; i < POINTS_PER_VERTEX; i++)
				{
					normals[normal_index + tCounter ] = norm[0];
					normals[normal_index + tCounter +1] = norm[1];
					normals[normal_index + tCounter +2] = norm[2];
					tCounter += POINTS_PER_VERTEX;
				}
 				
				triangle_index += TOTAL_FLOATS_IN_TRIANGLE;
				normal_index += TOTAL_FLOATS_IN_TRIANGLE;
				TotalConnectedTriangles += TOTAL_FLOATS_IN_TRIANGLE;			
			}	
		}
		objFile.close();														// Close OBJ file
	}
	else 
	{
		std::cout << "Unable to open file";								
	}
	return 0;
}
 
void Model_OBJ::Release()
{
	free(this->Faces_Triangles);
	free(this->normals);
	free(this->vertexBuffer);
}
 
void Model_OBJ::Draw()
{
 	//~ glEnableClientState(GL_VERTEX_ARRAY);						// Enable vertex arrays
 	//~ glEnableClientState(GL_NORMAL_ARRAY);						// Enable normal arrays
	//~ glVertexPointer(3,GL_FLOAT,	0,Faces_Triangles);				// Vertex Pointer to triangle array
	//~ glNormalPointer(GL_FLOAT, 0, normals);						// Normal pointer to normal array
	//~ glDrawArrays(GL_TRIANGLES, 0, TotalConnectedTriangles);		// Draw the triangles
	//~ glDisableClientState(GL_VERTEX_ARRAY);						// Disable vertex arrays
	//~ glDisableClientState(GL_NORMAL_ARRAY);						// Disable normal arrays
}

void writeToFile(int count)
{
	//Convert the file number to a string
	std::string number;
	std::stringstream sout;
	sout << std::setfill('0') << std::setw(4) << count;
	
	//Construct the output file name
	std::string folder("output");
	mkdir(folder.c_str(), 0777);
	std::string prefix = std::string("frame") + std::string(sout.str());
	std::string file = folder + std::string("/") + prefix + std::string(".lxs");
	std::ofstream fout(file.c_str());
	
	//Output the name of the filemake
	fout << "LookAt 0 0 -35 0 0 0 0 1 0" << std::endl;
	fout << "Film \"fleximage\"" << std::endl;
	fout << "\t\t\"string filename\" [\"" << prefix << "\"]" << std::endl;
	
	//Open new file and write out scene template
	std::ifstream fin("template.lxs");
	char str[2048];
	
	while(!fin.eof())
	{
		fin.getline(str, 2048);
		fout << str << std::endl;
	}
	
	//Close LuxRender scene template
	fin.close();
	
	//Write out individual follicle positions
	for(int i = 0; i < hair->numStrands; i++)
	{
		for(int j = 0; j < hair->strand[i]->numParticles; j++)
		{
			pilar::Particle* particle = hair->strand[i]->particle[j];
			
			fout << "AttributeBegin" << std::endl;
			fout << "Material \"matte\" \"color Kd\" [0.1 0.1 0.8 ]" << std::endl;
			fout << "\tTranslate " << particle->position.x << " " << particle->position.y << " " << particle->position.z << std::endl;
			fout << "\tShape \"sphere\" \"float radius\" 0.25" << std::endl;
			fout << "AttributeEnd" << std::endl;
			fout << std::endl;
		}
	}
	
	fout << "WorldEnd" << std::endl;
	
	//Close the output file
	fout.close();
}


int main()
{
	std::cout << "Hair Simulation" << std::endl;	
	
	float elapsed = 0.0f;
	float dt = 1.0f/50.0f; //50 Frames per second
	float total = 2.0f; //total time of the simulation in seconds
	int fileCount = 0;
	
	obj.Load("spherehalf.obj");
	
	pilar::Vector3f strand00(0.0f, 0.0f, 0.0f);
	pilar::Vector3f strand01(-0.025f, 0.0f, 0.0f);
	
	std::vector<pilar::Vector3f> roots;
	roots.push_back(strand00);
	roots.push_back(strand01);
	
	pilar::Vector3f normal00(-1.0f, 0.0f, 0.0f);
	pilar::Vector3f normal01(1.0f, 0.0f, 0.0f);
	
	std::vector<pilar::Vector3f> normals;
	normals.push_back(normal00);
	normals.push_back(normal01);
	
	hair = new pilar::Hair(roots.size(), NUMPARTICLES, MASS, K_EDGE, K_BEND, K_TWIST, K_EXTRA, D_EDGE, D_BEND, D_TWIST, D_EXTRA, LENGTH_EDGE, LENGTH_BEND, LENGTH_TWIST, roots, normals, obj);
	
	while(elapsed < total)
	{
		hair->update(dt);
		
//		writeToFile(fileCount);
		
		elapsed += dt;
		fileCount++;
	}
	
	obj.Release();
	
	hair->release();
	
	delete hair;
	
	hair = NULL;
	
	return 0;
}

