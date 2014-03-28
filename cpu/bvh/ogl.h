
#ifndef __OGL_H__
#define __OGL_H__

#define POINTS_PER_VERTEX 3
#define TOTAL_FLOATS_IN_TRIANGLE 9

class Model_OBJ
{
  public: 
	Model_OBJ();			
    float* calculateNormal(float* coord1,float* coord2,float* coord3, float* norm);
    int Load(const char *filename);	// Loads the model
	void Draw();					// Draws the model on the screen
	void Release();				// Release the model
 
	float* normals;							// Stores the normals
    float* Faces_Triangles;					// Stores the triangles
	float* vertexBuffer;					// Stores the points which make the object
	long TotalConnectedPoints;				// Stores the total number of connected vertices
	long TotalConnectedTriangles;			// Stores the total number of connected triangles
 
};

void animate(int milli);
void reshape(int w, int h);
void render(void);
void keyboard(unsigned char key, int x, int y);
void mouseFunc(int button, int state, int x, int y);
void mouseMotionFunc(int x, int y);

void init();
void update(float dt);
void cleanup();

#endif

