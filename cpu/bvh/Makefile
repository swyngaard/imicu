CC = g++
CPP = tools.cpp kdop.cpp ogl.cpp tree.cpp
HEADERS = tools.h kdop.h ogl.h tree.h
OBJ = tools.o kdop.o ogl.o tree.o
FLAGS = 
BIN = kdop
LIBS = -lglut -lGL -lGLU

TEST_DIR = tests/
TEST_CPP = $(addprefix $(TEST_DIR), main.cpp testTools.cpp testKDOP.cpp testTree.cpp)
TEST_HEADERS = $(addprefix $(TEST_DIR), testTools.h testKDOP.h testTree.h)
TEST_OBJ = tools.o kdop.o tree.o
TEST_FLAGS = -I.
TEST_LIBS = -lcppunit
TEST_BIN = $(TEST_DIR)test

all: kdop test

obj_kdop: $(HEADERS)
	$(CC) $(FLAGS) -c $(CPP)

kdop: obj_kdop
	$(CC) $(OBJ) $(LIBS) -o $(BIN)

test: $(TEST_CPP) $(TEST_HEADERS) $(TEST_OBJ)
	$(CC) $(TEST_FLAGS) $(TEST_LIBS) $(TEST_CPP) $(TEST_OBJ) -o $(TEST_BIN)

check: $(TEST_BIN)
	$(TEST_BIN)

memcheck: $(TEST_BIN)
	valgrind --leak-check=full $(TEST_BIN)

.PHONY: clean
	
clean:
	rm -rf *.o core $(BIN) $(TEST_BIN)
