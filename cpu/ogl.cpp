//#include <GL/glut.h>
#include <GL/freeglut.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include "hair.h"
#include "constants.h"

#ifndef GLUT_KEY_ESCAPE
#define GLUT_KEY_ESCAPE 27

#define POINTS_PER_VERTEX 3
#define TOTAL_FLOATS_IN_TRIANGLE 9

#endif

class Model_OBJ
{
  public: 
	Model_OBJ();			
    float* calculateNormal(float* coord1,float* coord2,float* coord3 );
    int Load(const char *filename);	// Loads the model
	void Draw();					// Draws the model on the screen
	void Release();				// Release the model
 
	float* normals;							// Stores the normals
    float* Faces_Triangles;					// Stores the triangles
	float* vertexBuffer;					// Stores the points which make the object
	long TotalConnectedPoints;				// Stores the total number of connected verteces
	long TotalConnectedTriangles;			// Stores the total number of connected triangles
 
};

Model_OBJ obj;

//float angle = 0.0f;
pilar::Hair* hair = NULL;
int prevTime;

void animate(int milli);
void reshape(int w, int h);
void render(void);
void keyboard(unsigned char key, int x, int y);

void init();
void update(float dt);
void cleanup();



Model_OBJ::Model_OBJ()
{
	this->TotalConnectedTriangles = 0; 
	this->TotalConnectedPoints = 0;
}
 
float* Model_OBJ::calculateNormal( float *coord1, float *coord2, float *coord3 )
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
   val = sqrt( vr[0]*vr[0] + vr[1]*vr[1] + vr[2]*vr[2] );
 
	float norm[3];
	norm[0] = vr[0]/val;
	norm[1] = vr[1]/val;
	norm[2] = vr[2]/val;
 
 
	return norm;
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
				float *norm = this->calculateNormal( coord1, coord2, coord3 );
 
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
 	glEnableClientState(GL_VERTEX_ARRAY);						// Enable vertex arrays
 	glEnableClientState(GL_NORMAL_ARRAY);						// Enable normal arrays
	glVertexPointer(3,GL_FLOAT,	0,Faces_Triangles);				// Vertex Pointer to triangle array
	glNormalPointer(GL_FLOAT, 0, normals);						// Normal pointer to normal array
	glDrawArrays(GL_TRIANGLES, 0, TotalConnectedTriangles);		// Draw the triangles
	glDisableClientState(GL_VERTEX_ARRAY);						// Disable vertex arrays
	glDisableClientState(GL_NORMAL_ARRAY);						// Disable normal arrays
}

int main(int argc, char **argv) {

	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(800,600);
	glutCreateWindow("Simulation");
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
	
	// register callbacks
	glutDisplayFunc(render);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
//	glutIdleFunc(render);
	glutTimerFunc(20, animate, 20);
	
	glPointSize(3.0f);
	glShadeModel(GL_FLAT);
	
	prevTime = glutGet(GLUT_ELAPSED_TIME);
	
	//Initialise hair simulation
	init();
	
	// enter GLUT event processing cycle
	glutMainLoop();
	
	//Release hair memory
	cleanup();	
	
	std::cout << "Exiting cleanly..." << std::endl;
	
	return 0;
}

void init()
{
	pilar::Vector3f root;
	std::vector<pilar::Vector3f> roots;
	
	roots.push_back(root);
	
	hair = new pilar::Hair(roots.size(), NUMPARTICLES, MASS, K_EDGE, K_BEND, K_TWIST, K_EXTRA, D_EDGE, D_BEND, D_TWIST, D_EXTRA, LENGTH, roots);
	obj.Load("monkey.obj");
}

void cleanup()
{
	obj.Release();
	
	hair->release();
	
	delete hair;
	
	hair = NULL;
}

void reshape(int w, int h)
{
	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if (h == 0)
		h = 1;

	float ratio =  w * 1.0 / h;

	// Use the Projection Matrix
	glMatrixMode(GL_PROJECTION);

	// Reset Matrix
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	// Set the correct perspective.
	gluPerspective(45.0f, ratio, 0.1f, 100.0f);

	// Get Back to the Modelview
	glMatrixMode(GL_MODELVIEW);
}

//float angle = 0.0f;

void render(void) {

	// Clear Color and Depth Buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Reset transformations
	glLoadIdentity();
	// Set the camera
	//Ideal camera closeup
	gluLookAt(	0.0f, -0.13f, -0.35f,
				0.0f, -0.13f,  0.0f,
				0.0f, 1.0f,  0.0f);
	//closeup with damping
//	gluLookAt(	0.0f, -0.25f, -0.65f,
//				0.0f, -0.25f,  0.0f,
//				0.0f, 1.0f,  0.0f);
	//Closeup without damping, lots of stretching
//	gluLookAt(	0.0f, -0.4f, -1.0f,
//				0.0f, -0.4f,  0.0f,
//				0.0f, 1.0f,  0.0f);
	
	glPushMatrix();
//		glRotatef(90,0,1,0);
		glTranslatef(0.0f, -0.13f, 0.0f);
		obj.Draw();
	glPopMatrix();
	
	//Draw hair
	glBegin(GL_LINE_STRIP);
	
//	glVertex3f(0.0f, 0.0f, 0.0f);
	
	for(int i = 0; i < hair->numStrands; i++)
	{
		for(int j = 0; j < hair->strand[i]->numParticles; j++)
		{
			pilar::Particle* particle = hair->strand[i]->particle[j];
//			pilar::Particle* p0 = hair->strand[i]->particle[j-1];
			
			//Set the colour of the spring
			
			switch(j%4)
			{
				case 0: glColor3f(1.0f, 1.0f, 1.0f); break; //WHITE
				case 1: glColor3f(1.0f, 0.0f, 0.0f); break; //RED
				case 2: glColor3f(0.0f, 1.0f, 0.0f); break; //GREEN
				case 3: glColor3f(1.0f, 0.0f, 1.0f); break; //PINK
			}
			
			
//			glVertex3f(p0->position.x, p0->position.y, p0->position.z);
			glVertex3f(particle->position.x, particle->position.y, particle->position.z);
			
//			if(j==(hair->strand[i]->numParticles-1))
//				std::cout << particle->position.x << " " << particle->position.y << " " << particle->position.z << std::endl;
		}
	}
	
	glEnd();
	
	glColor3f(1.0f, 1.0f, 1.0f);
	glBegin(GL_POINTS);
		glVertex3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.0f, -0.25f, 0.0f);
	glEnd();
	
	
	glutSwapBuffers();
}

void animate(int milli)
{
	glutTimerFunc(milli, animate, milli);
	
	int currentTime = glutGet(GLUT_ELAPSED_TIME);
	
	float dt =  (currentTime - prevTime)/1000.0f;
	
	hair->update(dt);
		
	glutPostRedisplay();
	
	prevTime = currentTime;
}

void keyboard(unsigned char key, int x, int y)
{
	if(key == GLUT_KEY_ESCAPE)
	{
		glutLeaveMainLoop();
	}
}

