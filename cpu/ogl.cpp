#include <GL/glut.h>
#include "hair.h"

//float angle = 0.0f;
pilar::Hair* hair = NULL;

void animate(int milli);
void reshape(int w, int h);
void render(void);

int main(int argc, char **argv) {

	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(640,480);
	glutCreateWindow("Simulation");

	// register callbacks
	glutDisplayFunc(render);
	glutReshapeFunc(reshape);
//	glutIdleFunc(render);
	glutTimerFunc(50, animate, 50);
	
	//Initialise hair simulation
	pilar::Vector3f root;
	std::vector<pilar::Vector3f> roots;
	
	roots.push_back(root);
	
	hair = new pilar::Hair(roots.size, 0.0001f, 10000.0f, 0.05f, roots);
	
	//TODO release hair memory
	
	// enter GLUT event processing cycle
	glutMainLoop();
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
	gluLookAt(	0.0f, 0.0f, -35.0f,
				0.0f, 0.0f,  0.0f,
				0.0f, 1.0f,  0.0f);

//	glRotatef(angle, 0.0f, 1.0f, 0.0f);

//	glBegin(GL_TRIANGLES);
//		glVertex3f(-2.0f,-2.0f, 0.0f);
//		glVertex3f( 2.0f, 0.0f, 0.0);
//		glVertex3f( 0.0f, 2.0f, 0.0);
//	glEnd();
	//TODO Draw hair
	
	
	
	glutSwapBuffers();
}

void animate(int milli)
{
	glutTimerFunc(milli, animate, milli);
	
	//TODO update hair

//	angle+=2.0f;
		
	glutPostRedisplay();
}

