CC=nvcc
CFLAGS=-O3 -DDOUBLE_PRECISION 
LFLAGS=-O3 -lm

EXECUTABLE=CG

OBJS=CG.o

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(CC) $(LFLAGS) -o $(EXECUTABLE) $(OBJS)

%.o: %.cu
	$(CC) -c $(CFLAGS) $< -o $@
  
clean:
	-rm -rf $(EXECUTABLE)
	-rm -rf *.o
	-rm -rf *~
