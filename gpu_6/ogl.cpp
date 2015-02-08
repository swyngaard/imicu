//#include <GL/glut.h>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>

//#include <cutil_inline.h>
//#include <cutil_gl_inline.h>
//#include <cutil_gl_error.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>

#include "hair.h"
#include "constants.h"

#ifndef GLUT_KEY_ESCAPE
#define GLUT_KEY_ESCAPE 27
#endif

//float angle = 0.0f;
pilar::Hair* hair = NULL;
int prevTime;

// vbo variables
GLuint vbo = 0;
GLuint vbo2 = 0;
struct cudaGraphicsResource *cuda_vbo_resource = NULL;
struct cudaGraphicsResource *cuda_vbo_resource2 = NULL;
static float* colour = NULL;

void animate(int milli);
void reshape(int w, int h);
void render(void);
void keyboard(unsigned char key, int x, int y);

void init();
void update(float dt);
void cleanup();

void createVBO(GLuint* v, int size);
void deleteVBO(GLuint* v);

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
	glutTimerFunc(40, animate, 40);
	
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

void createVBO(std::vector<pilar::Vector3f> &root)
{
	unsigned int size = NUMSTRANDS * NUMPARTICLES * sizeof(float3);
	
	float3* position_h = (float3*) malloc(size);
	
	for(int i = 0; i < NUMSTRANDS; i++)
	{
		for(int j = 0; j < NUMPARTICLES; j++)
		{
			int index = i*NUMPARTICLES + j;
			position_h[index].x = root[i].x + j * LENGTH / 2.0f;
			position_h[index].y = root[i].y;
			position_h[index].z = root[i].z;
			
//			printf("%f %f %f\n", position_h[index].x, position_h[index].y, position_h[index].z);
		}
	}

	//create vertex buffers and register with CUDA
	glGenBuffers(1,&vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, NUMSTRANDS*NUMPARTICLES*sizeof(float3), (void*)position_h, GL_DYNAMIC_DRAW);
	
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone));
	
	glGenBuffers(1,&vbo2);
	glBindBuffer(GL_ARRAY_BUFFER, vbo2);
	glBufferData(GL_ARRAY_BUFFER, NUMPARTICLES*3*sizeof(float), (void*)colour, GL_DYNAMIC_DRAW);
	
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource2, vbo2, cudaGraphicsMapFlagsNone));
	
	free(position_h);
}

void releaseVBO()
{
	//Delete VBO
	cudaGraphicsUnregisterResource(cuda_vbo_resource);
	glBindBuffer(1, vbo);
	glDeleteBuffers(1, &vbo);
	vbo = 0;
	
	cudaGraphicsUnregisterResource(cuda_vbo_resource2);
	glBindBuffer(1,vbo2);
	glDeleteBuffers(1, &vbo2);
	vbo2 = 0;
}

void init()
{
	
	
	pilar::Vector3f root;
	std::vector<pilar::Vector3f> roots;
	
	//TODO randomly generate roots on a plane
	roots.push_back(root);
	
	colour = new float[NUMPARTICLES*3];
	
	for(int i = 0; i < NUMPARTICLES; i++)
	{
		switch(i%4)
		{
			case 0: //WHITE
				colour[i*3  ] = 1.0f;
				colour[i*3+1] = 1.0f;
				colour[i*3+2] = 1.0f;
			break;
			case 1: //RED
				colour[i*3  ] = 1.0f;
				colour[i*3+1] = 0.0f;
				colour[i*3+2] = 0.0f;
			break;
			case 2: //GREEN
				colour[i*3  ] = 0.0f;
				colour[i*3+1] = 1.0f;
				colour[i*3+2] = 0.0f;
			break;
			case 3: //PINK
				colour[i*3  ] = 1.0f;
				colour[i*3+1] = 0.0f;
				colour[i*3+2] = 1.0f;
			break;
		}
	}
	
	createVBO(roots);
	
	size_t size;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, NULL));
	
	hair = new pilar::Hair(roots.size(), NUMPARTICLES, MASS, K_EDGE, K_BEND, K_TWIST, K_EXTRA, D_EDGE, D_BEND, D_TWIST, D_EXTRA, LENGTH, roots);
	
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&hair->position, &size, cuda_vbo_resource));
	
	hair->init();
	
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}

void cleanup()
{
	hair->release();
	
	delete hair;
	
	hair = NULL;
	
	delete [] colour;
	
	colour = NULL;
	
	releaseVBO();
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
	
	// render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo2);
    glColorPointer(3, GL_FLOAT, 0, 0);
    
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    
		glDrawArrays(GL_LINE_STRIP, 0, NUMPARTICLES);
	
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	glutSwapBuffers();
}

void animate(int milli)
{
	glutTimerFunc(milli, animate, milli);
	
	int currentTime = glutGet(GLUT_ELAPSED_TIME);
	
	float dt =  (currentTime - prevTime)/1000.0f;
	
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&hair->position, &num_bytes, cuda_vbo_resource));
	
	hair->update(dt);
	
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
	
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

