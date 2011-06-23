CC=g++
CFLAGS=-I.
DEPS = main.cpp tools.h tools.cpp hair.h hair.cpp
OGL = ogl.cpp hair.cpp tools.cpp
LIBS = -lglut -lGL -lGLU
OBJ = main.o tools.o hair.o
EXE = pilar

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

ogl:
	$(CC) $(LIBS) -o gl $(OGL)

.PHONY: clean
	
clean:
	rm -rf *.o core $(EXE) gl output

