CC=g++
CFLAGS=-I.
DEPS = main.cpp tools.h tools.cpp hair.h hair.cpp
OBJ = main.o tools.o hair.o
EXE = pilar

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXE): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean
	
clean:
	rm -f *.o core $(EXE)

