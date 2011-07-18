//#include <GL/glut.h>
#include <GL/freeglut.h>
#include <iostream>

#include "hair.h"
#include "constants.h"

#ifndef GLUT_KEY_ESCAPE
#define GLUT_KEY_ESCAPE 27
#endif

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
	
	glPointSize(10.0f);
	
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
	
	hair = new pilar::Hair(roots.size(), NUMPARTICLES, MASS, K_EDGE, K_BEND, K_TWIST, K_EXTRA, LENGTH, roots);
}

void cleanup()
{
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
	gluLookAt(	0.0f, 0.0f, -55.0f,
				0.0f, 0.0f,  0.0f,
				0.0f, 1.0f,  0.0f);

//	glRotatef(angle, 0.0f, 1.0f, 0.0f);

//	glBegin(GL_TRIANGLES);
//		glVertex3f(-2.0f,-2.0f, 0.0f);
//		glVertex3f( 2.0f, 0.0f, 0.0);
//		glVertex3f( 0.0f, 2.0f, 0.0);
//	glEnd();
	
	//TODO Draw hair
	
	glBegin(GL_LINES);
	
	for(int i = 0; i < hair->numStrands; i++)
	{
		for(int j = 0; j < hair->strand[i]->numParticles; j++)
		{
			pilar::Particle* particle = hair->strand[i]->particle[j];
			
			glVertex3f(particle->position.x, particle->position.y, particle->position.z);
		}
	}
	
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

