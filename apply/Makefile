CC=nvcc
#CFLAGS=-O3 -Wall -pg -DSINGLE_PRECISION -DTEST -DPREC
#LFLAGS=-O3 -Wall -pg -lm

EXECUTABLE=apply

OBJS=apply.o

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(CC) $(LFLAGS) -o $(EXECUTABLE) $(OBJS)

%.o: %.cu
	$(CC) -c  $< -o $@
  
clean:
	-rm -rf $(EXECUTABLE)
	-rm -rf *.o
	-rm -rf *~
	-rm -rf *.out
