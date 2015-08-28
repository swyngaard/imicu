
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include <fstream>
#include <string>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>

#include "hair.h"
#include "constants.h"

#ifndef GLUT_KEY_ESCAPE
#define GLUT_KEY_ESCAPE 27
#endif

static pilar::Hair* hair = 0;

static int prevTime;

// vbo variables
static GLuint strand_vbo = 0;
static GLuint colour_vbo = 0;
static struct cudaGraphicsResource *strand_vbo_resource = 0;
static struct cudaGraphicsResource *colour_vbo_resource = 0;

static ModelOBJ model = {};

static
void loadOBJ(const char* filename, ModelOBJ* obj)
{
	std::ifstream file(filename);
	
	if(file.is_open())
	{
		file.seekg(0, std::ios::end);
		long fileSize = file.tellg();
		file.seekg(0, std::ios::beg);
		
		obj->totalConnectedPoints = 0;
		obj->totalConnectedTriangles = 0;
		obj->vertices = new float[fileSize/sizeof(float)];
		obj->normals  = new float[fileSize];
		obj->faces	  = new float[fileSize];
		obj->bytes	  = fileSize;
		
		int triangleIndex = 0;
		int normalIndex = 0;
		std::string line;
		
		int POINTS_PER_VERTEX = 3;
		int TOTAL_FLOATS_IN_TRIANGLE = 9;
		
		while(!file.eof())
		{
			getline(file, line);
			
			if (line.c_str()[0] == 'v')
			{
				line[0] = ' ';

				sscanf(line.c_str(), "%f %f %f ",
					   &obj->vertices[obj->totalConnectedPoints  ],
					   &obj->vertices[obj->totalConnectedPoints+1],
					   &obj->vertices[obj->totalConnectedPoints+2]);

				obj->totalConnectedPoints += POINTS_PER_VERTEX;
			}
			
			if (line.c_str()[0] == 'f')
			{
				line[0] = ' ';
				
				int vertexNumber[3] = { 0, 0, 0 };
				sscanf(line.c_str(),"%i%i%i", &vertexNumber[0], &vertexNumber[1], &vertexNumber[2]);
				
				vertexNumber[0] -= 1;
				vertexNumber[1] -= 1;
				vertexNumber[2] -= 1;
				
				int tCounter = 0;
				for (int i = 0; i < POINTS_PER_VERTEX; i++)					
				{
					obj->faces[triangleIndex + tCounter    ] = obj->vertices[3*vertexNumber[i]   ];
					obj->faces[triangleIndex + tCounter + 1] = obj->vertices[3*vertexNumber[i]+1 ];
					obj->faces[triangleIndex + tCounter + 2] = obj->vertices[3*vertexNumber[i]+2 ];
					tCounter += POINTS_PER_VERTEX;
				}
				
				float coord1[3] = {obj->faces[triangleIndex  ], obj->faces[triangleIndex+1], obj->faces[triangleIndex+2]};
				float coord2[3] = {obj->faces[triangleIndex+3], obj->faces[triangleIndex+4], obj->faces[triangleIndex+5]};
				float coord3[3] = {obj->faces[triangleIndex+6], obj->faces[triangleIndex+7], obj->faces[triangleIndex+8]};
				
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
				
				float norm[3];
				norm[0] = vr[0]/val;
				norm[1] = vr[1]/val;
				norm[2] = vr[2]/val;
				
				tCounter = 0;
				for (int i = 0; i < POINTS_PER_VERTEX; i++)
				{
					obj->normals[normalIndex + tCounter    ] = norm[0];
					obj->normals[normalIndex + tCounter + 1] = norm[1];
					obj->normals[normalIndex + tCounter + 2] = norm[2];
					tCounter += POINTS_PER_VERTEX;
				}
				
				triangleIndex += TOTAL_FLOATS_IN_TRIANGLE;
				normalIndex += TOTAL_FLOATS_IN_TRIANGLE;
				obj->totalConnectedTriangles += TOTAL_FLOATS_IN_TRIANGLE;			
			}
		}
		
		file.close();
	}
	else
	{
		std::cout << "Unable to open file: " << filename << std::endl;
	}
}

static
void releaseOBJ(ModelOBJ* obj)
{
	if(obj)
	{
		delete [] obj->vertices;
		delete [] obj->normals;
		delete [] obj->faces;
	}
}

static
void renderOBJ(ModelOBJ* obj)
{
	if(obj)
	{
		//Unbind any buffers before rendering
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		
		glPushMatrix();
		glEnableClientState(GL_VERTEX_ARRAY);						// Enable vertex arrays
		glEnableClientState(GL_NORMAL_ARRAY);						// Enable normal arrays
		glVertexPointer(3,GL_FLOAT,	0, obj->faces);					// Vertex Pointer to triangle array
		glNormalPointer(GL_FLOAT, 0, obj->normals);						// Normal pointer to normal array
		glDrawArrays(GL_TRIANGLES, 0, obj->totalConnectedTriangles);		// Draw the triangles
		glDisableClientState(GL_VERTEX_ARRAY);						// Disable vertex arrays
		glDisableClientState(GL_NORMAL_ARRAY);						// Disable normal arrays
		glPopMatrix();
	}
}

static
void initialise()
{
	//Root positions
	pilar::Vector3f* roots = new pilar::Vector3f[NUMSTRANDS];
	roots[0] = pilar::Vector3f(0.0f, 0.0f, 0.0f);
	roots[1] = pilar::Vector3f(-0.025f, 0.0f, 0.0f);
	
	//Normal directions
	pilar::Vector3f* normals = new pilar::Vector3f[NUMSTRANDS];
	normals[0] = pilar::Vector3f(1.0f, -1.0f, 0.0f);
	normals[1] = pilar::Vector3f(-1.0f, -1.0f, 0.0f);
	
	//Gravity
	pilar::Vector3f gravity(0.0f, GRAVITY, 0.0f);
	
	//Load geometry from file
	loadOBJ("monkey.obj", &model);
	
	//Intialise temp colour buffer
	float* colour_values = new float[NUMSTRANDS*NUMPARTICLES*NUMCOMPONENTS];
	
	//Set values of colours
	for(int i = 0; i < NUMSTRANDS*NUMPARTICLES; i++)
	{
		int index = i * NUMCOMPONENTS;
		
		switch(i%4)
		{
			case 0: //White
				colour_values[index  ] = 1.0f;
				colour_values[index+1] = 1.0f;
				colour_values[index+2] = 1.0f;
			break;
			case 1: //Red
				colour_values[index  ] = 1.0f;
				colour_values[index+1] = 0.0f;
				colour_values[index+2] = 0.0f;
			break;
			case 2: //Green
				colour_values[index  ] = 0.0f;
				colour_values[index+1] = 1.0f;
				colour_values[index+2] = 0.0f;
			break;
			case 3: //Pink
				colour_values[index  ] = 1.0f;
				colour_values[index+1] = 0.0f;
				colour_values[index+2] = 1.0f;
			break;
		}
	}
	
	//Create the VBO for colours
	glGenBuffers(1,&colour_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, colour_vbo);
	glBufferData(GL_ARRAY_BUFFER, NUMSTRANDS*NUMPARTICLES*NUMCOMPONENTS*sizeof(float), (void*)colour_values, GL_DYNAMIC_DRAW);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&colour_vbo_resource, colour_vbo, cudaGraphicsMapFlagsNone));
	
	hair = new pilar::Hair(NUMSTRANDS, NUMPARTICLES, NUMCOMPONENTS, MASS,
						   K_EDGE, K_BEND, K_TWIST, K_EXTRA,
						   D_EDGE, D_BEND, D_TWIST, D_EXTRA,
						   LENGTH_EDGE, LENGTH_BEND, LENGTH_TWIST,
						   gravity,
						   roots,
						   normals,
						   &model);
	
	//Create VBO for all strand particles
	glGenBuffers(1,&strand_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, strand_vbo);
	glBufferData(GL_ARRAY_BUFFER, NUMSTRANDS*NUMPARTICLES*sizeof(pilar::Vector3f), 0, GL_DYNAMIC_DRAW);
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&strand_vbo_resource, strand_vbo, cudaGraphicsMapFlagsNone));
	
	//Map resource, set map flags, write intial data, unmap resource
	size_t strand_size;
	pilar::Vector3f* position;
	checkCudaErrors(cudaGraphicsMapResources(1, &strand_vbo_resource, NULL));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&position, &strand_size, strand_vbo_resource));

	//Initialise positions along normals on the gpu
	
	hair->initialise(position);
	
	
	checkCudaErrors(cudaGraphicsUnmapResources(1, &strand_vbo_resource, 0));
	
	delete [] colour_values;
	delete [] normals;
	delete [] roots;
}

static
void cleanup()
{
	delete hair;
	
	hair = NULL;
	
	releaseOBJ(&model);
	
	//Release VBO for all strand particles
	cudaGraphicsUnregisterResource(strand_vbo_resource);
	glBindBuffer(1, strand_vbo);
	glDeleteBuffers(1, &strand_vbo);
	strand_vbo = 0;

	//Release VBO for strand colours
	cudaGraphicsUnregisterResource(colour_vbo_resource);
	glBindBuffer(1, colour_vbo);
	glDeleteBuffers(1, &colour_vbo);
	colour_vbo = 0;
}

static
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

static
void render()
{
	// Clear Color and Depth Buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Reset transformations
	glLoadIdentity();
	
	// Set the camera
	//Ideal camera closeup
	gluLookAt(	0.0f, -0.13f, -0.35f,
				0.0f, -0.13f,  0.0f,
				0.0f, 1.0f,  0.0f);
	
	renderOBJ(&model);
	
	glColor3f(1.0f, 1.0f, 1.0f);
	glBegin(GL_POINTS);
		glVertex3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.0f, -0.25f, 0.0f);
	glEnd();
	
	//TODO Render line from strand root to first particle
	
	//Render from the VBO for the positions of the all strands
	glBindBuffer(GL_ARRAY_BUFFER, strand_vbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, colour_vbo);
	glColorPointer(3, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	//FIXME Investigate glDrawElements
	for(int i = 0; i < NUMSTRANDS; i++)
	{
		glDrawArrays(GL_LINE_STRIP, i*NUMPARTICLES, NUMPARTICLES);
	}

	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glutSwapBuffers();
}

static
void animate(int milli)
{
	glutTimerFunc(milli, animate, milli);
	
	int currentTime = glutGet(GLUT_ELAPSED_TIME);
	
	float dt =  (currentTime - prevTime)/1000.0f;

	size_t strand_size;
	pilar::Vector3f* position;
	checkCudaErrors(cudaGraphicsMapResources(1, &strand_vbo_resource, NULL));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&position, &strand_size, strand_vbo_resource));

	hair->update(dt, position);

	checkCudaErrors(cudaGraphicsUnmapResources(1, &strand_vbo_resource, 0));

	//Calculate frames per second and display in window title
	static int frames = 0;
	static int baseTime = 0;
	static char titleString[32];

	frames++;
	int diffTime = currentTime - baseTime;

	if(diffTime > 1000)
	{
		sprintf(titleString, "Simulation FPS: %.2f", frames*1000.0f/diffTime);
		baseTime = currentTime;
		frames = 0;
	}

	glutSetWindowTitle(titleString);

	glutPostRedisplay();
	
	prevTime = currentTime;
}

static
void keyboard(unsigned char key, int x, int y)
{
	if(key == GLUT_KEY_ESCAPE)
	{
		glutLeaveMainLoop();
	}
}

int main(int argc, char **argv)
{
	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(1024,768);
	glutCreateWindow("Simulation");
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
	
	// register callbacks
	glutDisplayFunc(render);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
//	glutIdleFunc(render);
	glutTimerFunc(8, animate, 8);
	
	glewInit();
	
	cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );
	
	glPointSize(3.0f);
	glShadeModel(GL_FLAT);
	
	prevTime = glutGet(GLUT_ELAPSED_TIME);
	
	//Initialise hair simulation
	initialise();
	
	// enter GLUT event processing cycle
	glutMainLoop();
	
	//Release hair memory
	cleanup();	
	
	std::cout << "Exiting cleanly..." << std::endl;
	
	return 0;
}
