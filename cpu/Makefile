CC=g++
CFLAGS=-I.:$(HOME)/local/include/boost -I$(HOME)/local/include
DEPS = main.cpp tools.h tools.cpp hair.h hair.cpp constants.h
OGL = ogl.cpp hair.cpp tools.cpp
LIBS = -lglut -lGL -lGLU
OBJ = main.o tools.o hair.o
EXE = pilar

%.o: %.cpp $(DEPS) $(OGL)
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)
	$(CC) $(LIBS) $(CFLAGS) -o gl $(OGL)

.PHONY: clean
	
clean:
	rm -rf *.o core $(EXE) gl output

