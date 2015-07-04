
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>

#include "hair.h"
#include "constants.h"

#ifndef GLUT_KEY_ESCAPE
#define GLUT_KEY_ESCAPE 27
#endif

static pilar::Hair* hair = NULL;
static int prevTime;

// vbo variables
static GLuint strand_vbo = 0;
static GLuint colour_vbo = 0;
static struct cudaGraphicsResource *strand_vbo_resource = NULL;
static struct cudaGraphicsResource *colour_vbo_resource = NULL;

static
void init()
{
	//Root positions
	pilar::Vector3f strand00(0.0f, 0.0f, 0.0f);
	pilar::Vector3f strand01(-0.025f, 0.0f, 0.0f);
	
	std::vector<pilar::Vector3f> roots_;
	roots_.push_back(strand00);
	roots_.push_back(strand01);
	
	pilar::Vector3f normal00(1.0f, -1.0f, 0.0f);
	pilar::Vector3f normal01(-1.0f, -1.0f, 0.0f);
	
	std::vector<pilar::Vector3f> normals_;
	normals_.push_back(normal00);
	normals_.push_back(normal01);
	
	//Intialise temp colour buffer
	float* colour_values = (float*)malloc(NUMSTRANDS*NUMPARTICLES*NUMCOMPONENTS*sizeof(float));
	
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
	
	free(colour_values);

	hair = new pilar::Hair(roots_.size(), NUMPARTICLES, NUMCOMPONENTS, MASS, K_EDGE, K_BEND, K_TWIST, K_EXTRA, D_EDGE, D_BEND, D_TWIST, D_EXTRA, LENGTH_EDGE, LENGTH_BEND, LENGTH_TWIST, roots_, normals_);
	
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
}

static
void cleanup()
{
	delete hair;
	
	hair = NULL;
	
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
	//closeup with damping
//	gluLookAt(	0.0f, -0.25f, -0.65f,
//				0.0f, -0.25f,  0.0f,
//				0.0f, 1.0f,  0.0f);
	//Closeup without damping, lots of stretching
//	gluLookAt(	0.0f, -0.4f, -1.0f,
//				0.0f, -0.4f,  0.0f,
//				0.0f, 1.0f,  0.0f);
	
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
	init();
	
	// enter GLUT event processing cycle
	glutMainLoop();
	
	//Release hair memory
	cleanup();	
	
	std::cout << "Exiting cleanly..." << std::endl;
	
	return 0;
}
