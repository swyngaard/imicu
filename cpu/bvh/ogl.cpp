//#include <GL/glut.h>
#include <GL/freeglut.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <math.h>
#include <cfloat>
#include <vector>

#include "ogl.h"
#include "tools.h"
#include "kdop.h"

#ifndef GLUT_KEY_ESCAPE
#define GLUT_KEY_ESCAPE 27
#endif


Model_OBJ obj;
Model_OBJ triTouch;
Model_OBJ triMiss;

KDOP* kdop = NULL;
KDOP* ttKdop = NULL;
KDOP* tmKdop = NULL;

std::vector<Vector3f> ttPoints;
float moveZ = 0.0f;
float moveFactor = 0.001f;

//float angle = 0.0f;
//pilar::Hair* hair = NULL;
int prevTime;

// camera attributes
float viewerPosition[3]		= { 0.0f, -0.13f, -0.35f };
float viewerDirection[3]	= { 0.0, 1.0, 0.0 };

// rotation values for the navigation
float navigationRotation[3]	= { 0.0, 0.0, 0.0 };

// parameters for the navigation

// position of the mouse when pressed
int mousePressedX = 0, mousePressedY = 0;
float lastXOffset = 0.0, lastYOffset = 0.0, lastZOffset = 0.0;
// mouse button states
int leftMouseButtonActive = 0, middleMouseButtonActive = 0, rightMouseButtonActive = 0;
// modifier state
int shiftActive = 0, altActive = 0, ctrlActive = 0;


void *font = GLUT_BITMAP_8_BY_13;
int bitmapHeight = 13;
int frame,timet,timebase=0;
char s[30];

void setOrthographicProjection() {

	// switch to projection mode
	glMatrixMode(GL_PROJECTION);
	// save previous matrix which contains the 
	//settings for the perspective projection
	glPushMatrix();
	// reset matrix
	glLoadIdentity();
	// set a 2D orthographic projection
	gluOrtho2D(0, 800, 0, 600);
	// invert the y axis, down is positive
	glScalef(1, -1, 1);
	// mover the origin from the bottom left corner
	// to the upper left corner
	glTranslatef(0, -600, 0);
	glMatrixMode(GL_MODELVIEW);
}

void resetPerspectiveProjection() {
	// set the current matrix to GL_PROJECTION
	glMatrixMode(GL_PROJECTION);
	// restore previous settings
	glPopMatrix();
	// get back to GL_MODELVIEW matrix
	glMatrixMode(GL_MODELVIEW);
}

void renderBitmapString(float x, float y, void *font,char *string)
{
	char *c;
	
	// set position to start drawing fonts
	glRasterPos2f(x, y);
	
	// loop all the characters in the string
	for (c=string; *c != '\0'; c++)
	{
		glutBitmapCharacter(font, *c);
	}
}

int main(int argc, char **argv)
{
	// init GLUT and create window
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(1024,768);
	glutCreateWindow("Pilar");
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);
	
	// register callbacks
	glutDisplayFunc(render);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutTimerFunc(8, animate, 8);
	glutMouseFunc(mouseFunc);
	glutMotionFunc(mouseMotionFunc);
	
	glPointSize(3.0f);
	glShadeModel(GL_FLAT);
	
	prevTime = glutGet(GLUT_ELAPSED_TIME);
	
	//Initialise hair simulation
	init();
	
	// enter GLUT event processing cycle
	glutMainLoop();
	
	//Release hair memory
	cleanup();	
	
	std::cout << "Exiting..." << std::endl;
	
	return 0;
}

void init()
{
	obj.Load("triangle.obj");
	triTouch.Load("touching-triangle.obj");
	triMiss.Load("non-touching-triangle.obj");
	
	//debug
	//std::cout << "Points: " << obj.TotalConnectedPoints << std::endl;
	
	std::vector<Vector3f> points;
	//~ std::vector<Vector3f> ttPoints;
	std::vector<Vector3f> tmPoints;
	
	for(int i = 0; i < obj.TotalConnectedPoints/POINTS_PER_VERTEX; i++)
	{
		int index = i*POINTS_PER_VERTEX;
		points.push_back(Vector3f(obj.vertexBuffer[index], obj.vertexBuffer[index+1], obj.vertexBuffer[index+2]));
		
		//debug
		//~ std::cout << "Vertex" << i << ": [" << points[i].x << "," << points[i].y << "," << points[i].z << "]" << std::endl;
	}
	
	for(int i = 0; i < triTouch.TotalConnectedPoints/POINTS_PER_VERTEX; i++)
	{
		int index = i*POINTS_PER_VERTEX;
		ttPoints.push_back(Vector3f(triTouch.vertexBuffer[index], triTouch.vertexBuffer[index+1], triTouch.vertexBuffer[index+2]));
	}
	
	for(int i = 0; i < triMiss.TotalConnectedPoints/POINTS_PER_VERTEX; i++)
	{
		int index = i*POINTS_PER_VERTEX;
		tmPoints.push_back(Vector3f(triMiss.vertexBuffer[index], triMiss.vertexBuffer[index+1], triMiss.vertexBuffer[index+2]));
	}
	
	kdop = new KDOP(points,26);
	ttKdop = new KDOP(ttPoints,26);
	tmKdop = new KDOP(tmPoints,26);
	
	//debug
	//~ for(int i = 0; i < kdop->k; i++)
	//~ {
		//~ std::cout << "distance " << i << ": " << kdop->distance[i] << std::endl; 
	//~ }
	
	kdop->debug();
	ttKdop->debug();
	tmKdop->debug();
	
	//debug
	//~ for(int i = 0; i < cage.size()/2; i++)
	//~ {
		//~ int c1 = i * 2;
		//~ int c2 = i * 2 + 1;
		//~ 
		//~ std::cout << "Line " << (i+1) << ": [" << cage[c1].x << "," << cage[c1].y << "," << cage[c1].z << "] -> [" << cage[c2].x << "," << cage[c2].y << "," << cage[c2].z << "]" << std::endl;
	//~ }
	
	//debug
	//~ if(kdop->collides(ttKdop)) std::cout << "Collides with touch triangle!" << std::endl;
	//~ if(kdop->collides(tmKdop)) std::cout << "Collides with miss triangle!" << std::endl;
	
	Vector3f root;
	std::vector<Vector3f> roots;
	
	roots.push_back(root);
}

void cleanup()
{
	delete kdop;
	delete ttKdop;
	delete tmKdop;
	
	obj.Release();
	triTouch.Release();
	triMiss.Release();
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
	gluPerspective(45.0f, ratio, 0.01f, 100.0f);

	// Get Back to the Modelview
	glMatrixMode(GL_MODELVIEW);
	
//	glShadeModel( GL_FLAT );
    glClearColor( 0.2f, 0.2f, 0.2f, 0.5f );
    glClearDepth( 1.0f );

//    glEnable( GL_DEPTH_TEST );
//    glDepthFunc( GL_LEQUAL );
    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );
 
    GLfloat amb_light[] = { 0.1, 0.1, 0.1, 1.0 };
    GLfloat diffuse[] = { 0.6, 0.6, 0.6, 1 };
    GLfloat specular[] = { 0.7, 0.7, 0.3, 1 };
    glLightModelfv( GL_LIGHT_MODEL_AMBIENT, amb_light );
    glLightfv( GL_LIGHT0, GL_DIFFUSE, diffuse );
    glLightfv( GL_LIGHT0, GL_SPECULAR, specular );
    glEnable( GL_LIGHT0 );
    glEnable( GL_COLOR_MATERIAL );
	
	glShadeModel( GL_SMOOTH );
//	glShadeModel( GL_FLAT );

    glLightModeli( GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE );
    glDepthFunc( GL_LEQUAL );
    glEnable( GL_DEPTH_TEST );
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0); 
}

//float angle = 0.0f;

void render(void) {

	// Clear Color and Depth Buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Reset transformations
	glLoadIdentity();
	// Set the camera
	//Ideal camera closeup
	gluLookAt(	0.0f, -0.10f, -0.15f, //eye - position of the eye point
				0.0f, -0.13f,  0.0f,  //centre - position of the reference point
				0.0f, 1.0f,  0.0f);   //up - direction of up vector
	
	
	//closeup
	//~ gluLookAt(	0.0f, 0.0f, -0.15f,
				//~ 0.0f, 0.0f, 0.0f,
				//~ 0.0f, 1.0f,  0.0f);
	//Closeup without damping, lots of stretching
//	gluLookAt(	0.0f, -0.4f, -1.0f,
//				0.0f, -0.4f,  0.0f,
//				0.0f, 1.0f,  0.0f);
	
	// add navigation rotation
//	glRotatef( navigationRotation[0], 1.0f, 0.0f, 0.0f );
	glRotatef( navigationRotation[1], 0.0f, -1.0f, 0.0f );
	
	//Draw triangle object
	glColor3f(1.0f, 0.7f, 1.0f);
	glPushMatrix();
		obj.Draw();
	glPopMatrix();
	
	//Draw intersecting triangle
	glPushMatrix();
		glTranslatef(0.0f, 0.0f, moveZ);
		triTouch.Draw();
	glPopMatrix();
	
	//Draw non-intersecting triangle
	glPushMatrix();
		triMiss.Draw();
	glPopMatrix();
	
	glDisable(GL_LIGHT0);
	glDisable(GL_LIGHTING);
	
	//Get bounding volume vertices
	std::vector<Vector3f> vertices = kdop->debugVertices();
	std::vector<Vector3f> ttVerts = ttKdop->debugVertices();
	std::vector<Vector3f> tmVerts = tmKdop->debugVertices();
	//~ std::vector<Vector3f> vertices;
	
	int totalLines = vertices.size()/2;
	int ttLines = ttVerts.size()/2;
	int tmLines = tmVerts.size()/2;
	
	//debug
	//~ std::cout << totalLines << std::endl;
	
	//Draw bounding volume here
	glColor3f(1.0f, 1.0f, 1.0f);
	
	glBegin(GL_LINES);
	
	for(int i = 0; i < totalLines; i++)
	{
		int v1 = i * 2;
		int v2 = i * 2 + 1;
		
		//~ std::cout << vertices[v1].x << " " << vertices[v1].y << " " << vertices[v1].z << std::endl;
		//~ std::cout << vertices[v2].x << " " << vertices[v2].y << " " << vertices[v2].z << std::endl;
		
		glVertex3f(vertices[v1].x, vertices[v1].y, vertices[v1].z);
		glVertex3f(vertices[v2].x, vertices[v2].y, vertices[v2].z);
		
		//~ glVertex3f(-vertices[v1].x, -vertices[v1].y, -vertices[v1].z);
		//~ glVertex3f(-vertices[v2].x, -vertices[v2].y, -vertices[v2].z);
	}
	
	glEnd();
	
	glBegin(GL_LINES);
	
	for(int i = 0; i < tmLines; i++)
	{
		int v1 = i * 2;
		int v2 = i * 2 + 1;
		
		glVertex3f(tmVerts[v1].x, tmVerts[v1].y, tmVerts[v1].z);
		glVertex3f(tmVerts[v2].x, tmVerts[v2].y, tmVerts[v2].z);
	}
	
	glEnd();
	
	if(ttKdop->collides(kdop))
		glColor3f(0.0f, 1.0f, 0.2f);
	else
		glColor3f(1.0f, 1.0f, 1.0f);
	
	glBegin(GL_LINES);
	
	for(int i = 0; i < ttLines; i++)
	{
		int v1 = i * 2;
		int v2 = i * 2 + 1;
		
		glVertex3f(ttVerts[v1].x, ttVerts[v1].y, ttVerts[v1].z);
		glVertex3f(ttVerts[v2].x, ttVerts[v2].y, ttVerts[v2].z);
	}
	
	glEnd();
	
	glColor3f(1.0f, 1.0f, 1.0f);
	
	glBegin(GL_LINES);
	
	glVertex3f(-0.125f, -0.2f, 0.0f);
	glVertex3f( 0.125f, -0.2f, 0.0f);
	glEnd();
	
	
	
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	
	
/*
	//Draw hair
	glBegin(GL_LINE_STRIP);
	
	glVertex3f(0.0f, 0.0f, 0.0f);
	
	for(int i = 0; i < hair->numStrands; i++)
	{
		for(int j = 0; j < hair->strand[i]->numParticles; j++)
		{
			pilar::Particle* particle = hair->strand[i]->particle[j];
			
			//Set the colour of the spring
			switch(j%4)
			{
				case 0: glColor3f(1.0f, 1.0f, 1.0f); break; //WHITE
				case 1: glColor3f(1.0f, 0.0f, 0.0f); break; //RED
				case 2: glColor3f(0.0f, 1.0f, 0.0f); break; //GREEN
				case 3: glColor3f(1.0f, 0.0f, 1.0f); break; //PINK
			}
			
			glVertex3f(particle->position.x, particle->position.y, particle->position.z);
		}
		//glVertex3f(hair->strand[i]->rootParticle->position.x, hair->strand[i]->rootParticle->position.y, hair->strand[i]->rootParticle->position.z);
		//~ glColor3f(1.0f, 1.0f, 1.0f);
		//~ glVertex3f(hair->strand[i]->particle[0]->position.x, hair->strand[i]->particle[0]->position.y, hair->strand[i]->particle[0]->position.z);
		//~ glColor3f(1.0f, 0.0f, 0.0f);
		//~ glVertex3f(hair->strand[i]->particle[1]->position.x, hair->strand[i]->particle[1]->position.y, hair->strand[i]->particle[1]->position.z);
		//~ glColor3f(1.0f, 0.0f, 1.0f);
		//~ glVertex3f(hair->strand[i]->particle[2]->position.x, hair->strand[i]->particle[2]->position.y, hair->strand[i]->particle[2]->position.z);
	}
	
	glEnd();
	
	
	glColor3f(1.0f, 1.0f, 1.0f);
	glBegin(GL_POINTS);
		glVertex3f(0.0f, 0.0f, 0.0f);
		glVertex3f(0.0f, -0.25f, 0.0f);
		glVertex3f(0.0f, -DOMAIN_HALF, 0.0f);
		glVertex3f(0.0f, -0.005f, 0.0f);
	glEnd();
	
	glPushMatrix();
		glTranslatef(0.0f, -0.125f, 0.0f);
		glutWireCube(CELL_WIDTH * DOMAIN_DIM);
	glPopMatrix();
*/	
//	glPushMatrix();
//	
//		glTranslatef(-DOMAIN_HALF+CELL_HALF, -DOMAIN_HALF-0.125f+CELL_HALF, -DOMAIN_HALF+CELL_HALF);
//		
//		for(int xx = 0; xx < DOMAIN_DIM; xx++)
//		{
//			for(int yy = 0; yy < DOMAIN_DIM; yy++)
//			{
//				for(int zz = 0; zz < DOMAIN_DIM; zz++)
//				{
//					glPushMatrix();
//						glTranslatef(xx*CELL_WIDTH, yy*CELL_WIDTH, zz*CELL_WIDTH);
////						if(hair->grid[xx][yy][zz] < FLT_MAX && hair->grid[xx][yy][zz] < 0.0f)
//						if((xx == 99 && yy == 0 && zz == 99) || (xx == 99 && yy == 0 && zz == 99))
//						{
//							glColor3f(0.0f, 0.0f, 0.0f);
//							glBegin(GL_POINTS);
//								glVertex3f(0.0f, 0.0f, 0.0f);
//							glEnd();
////							glutWireCube(CELL_WIDTH);
//						}
//						
//					glPopMatrix();
//				}
//			}
//		}
//		
//	glPopMatrix();
	
	frame++;
	timet=glutGet(GLUT_ELAPSED_TIME);
	if (timet - timebase > 1000) {
		sprintf(s,"FPS:%4.2f",frame*1000.0/(timet-timebase));
		timebase = timet;		
		frame = 0;
	}

	glColor3f(0.0f,1.0f,0.0f);
	setOrthographicProjection();
	glPushMatrix();
	glLoadIdentity();
//	renderBitmapString(30,15,font,"GLUT Tutorial @ 3D Tech");
	renderBitmapString(30,35,font,s);
//	renderBitmapString(30,55,font,"Esc - Quit");
	glPopMatrix();
	resetPerspectiveProjection();
	
	glutSwapBuffers();
}

void animate(int milli)
{
	glutTimerFunc(milli, animate, milli);
	
	int currentTime = glutGet(GLUT_ELAPSED_TIME);
	
	float dt =  (currentTime - prevTime)/1000.0f;
	
//	hair->update(dt);
	
	if(moveZ > 0.05f)
		moveFactor = -0.001f;
	else if(moveZ < -0.05f)
		moveFactor = 0.001f;
	
	moveZ += dt * moveFactor;
	
	//debug
	//~ std::cout << "moveZ: " << moveZ << std::endl;
	
	ttPoints.clear();
	
	//animate touch triangle
	for(int i = 0; i < triTouch.TotalConnectedPoints/POINTS_PER_VERTEX; i++)
	{
		int index = i*POINTS_PER_VERTEX;
		ttPoints.push_back(Vector3f(triTouch.vertexBuffer[index], triTouch.vertexBuffer[index+1], triTouch.vertexBuffer[index+2]+moveZ));
	}
	
	//update bounding volume
	ttKdop->update(ttPoints);
	
	//update vertices for bounding volume
	ttKdop->debug();
	
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

// mouse callback
void mouseFunc(int button, int state, int x, int y)
{
	
	// get the modifiers
	switch (glutGetModifiers()) {

		case GLUT_ACTIVE_SHIFT:
			shiftActive = 1;
			break;
		case GLUT_ACTIVE_ALT:
			altActive	= 1;
			break;
		case GLUT_ACTIVE_CTRL:
			ctrlActive	= 1;
			break;
		default:
			shiftActive = 0;
			altActive	= 0;
			ctrlActive	= 0;
			break;
	}

	// get the mouse buttons
	if (button == GLUT_LEFT_BUTTON)
		if (state == GLUT_DOWN) {
			leftMouseButtonActive += 1;
		} else
			leftMouseButtonActive -= 1;
	else if (button == GLUT_MIDDLE_BUTTON)
		if (state == GLUT_DOWN) {
			middleMouseButtonActive += 1;
			lastXOffset = 0.0;
			lastYOffset = 0.0;
		} else
			middleMouseButtonActive -= 1;
	else if (button == GLUT_RIGHT_BUTTON)
		if (state == GLUT_DOWN) {
			rightMouseButtonActive += 1;
			lastZOffset = 0.0;
		} else
			rightMouseButtonActive -= 1;

//	if (altActive) {
		mousePressedX = x;
		mousePressedY = y;
//	}
}

//-----------------------------------------------------------------------------

void mouseMotionFunc(int x, int y)
{
	
	float xOffset = 0.0, yOffset = 0.0, zOffset = 0.0;

	// navigation
//	if (altActive) {
	
		// rotatation
		if (leftMouseButtonActive) {

			navigationRotation[0] += ((mousePressedY - y) * 180.0f) / 200.0f;
			navigationRotation[1] += ((mousePressedX - x) * 180.0f) / 200.0f;

			mousePressedY = y;
			mousePressedX = x;

		}
		// panning
		else if (middleMouseButtonActive) {

			xOffset = (mousePressedX + x);
			if (!lastXOffset == 0.0) {
				viewerPosition[0]	-= (xOffset - lastXOffset) / 8.0;
				viewerDirection[0]	-= (xOffset - lastXOffset) / 8.0;
			}
			lastXOffset = xOffset;

			yOffset = (mousePressedY + y);
			if (!lastYOffset == 0.0) {
				viewerPosition[1]	+= (yOffset - lastYOffset) / 8.0;
				viewerDirection[1]	+= (yOffset - lastYOffset) / 8.0;	
			}	
			lastYOffset = yOffset;

		}
		// depth movement
		else if (rightMouseButtonActive) {
			zOffset = (mousePressedX + x);
			if (!lastZOffset == 0.0) {
				viewerPosition[2] -= (zOffset - lastZOffset) / 5.0;
				viewerDirection[2] -= (zOffset - lastZOffset) / 5.0;
			}
			lastZOffset = zOffset;
		}
//	}

}

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
 	glEnableClientState(GL_VERTEX_ARRAY);						// Enable vertex arrays
 	glEnableClientState(GL_NORMAL_ARRAY);						// Enable normal arrays
	glVertexPointer(3,GL_FLOAT,	0,Faces_Triangles);				// Vertex Pointer to triangle array
	glNormalPointer(GL_FLOAT, 0, normals);						// Normal pointer to normal array
	glDrawArrays(GL_TRIANGLES, 0, TotalConnectedTriangles);		// Draw the triangles
	glDisableClientState(GL_VERTEX_ARRAY);						// Disable vertex arrays
	glDisableClientState(GL_NORMAL_ARRAY);						// Disable normal arrays
}

