CC = g++ -g

CPP = main.cpp hair.cpp tools.cpp kdop.cpp tree.cpp
HEADERS = hair.h tools.h constants.h kdop.h tree.h
OBJF = hair.o tools.o kdop.o tree.o
OBJ = main.o $(OBJF)
FLAGS = -I$(HOME)/local/include
EXE = pilar

OCPP = ogl.cpp
OOBJ = ogl.o $(OBJF)
OEXE = gl
LIBS = -lglut -lGL -lGLU

all: pilar ogl

obj_pilar: $(HEADERS)
	$(CC) $(FLAGS) -c $(CPP)

pilar: obj_pilar
	$(CC) $(OBJ) -o $(EXE) 

obj_ogl: $(HEADERS)
	$(CC) -c $(OCPP)

ogl: obj_ogl
	$(CC) $(OOBJ) $(LIBS) -o $(OEXE)

memcheck: $(OEXE)
	valgrind --leak-check=full ./$(EXE)

.PHONY: clean
	
clean:
	rm -rf *.o core $(EXE) gl output

