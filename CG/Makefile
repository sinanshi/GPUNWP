CC=gcc
CFLAGS=-O3 -Wall -pg -DDOUBLE_PRECISION -DTEST -DPREC
LFLAGS=-O3 -Wall -pg -lm

EXECUTABLE=CG

OBJS=CG.o

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJS)
	$(CC) $(LFLAGS) -o $(EXECUTABLE) $(OBJS)

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@
  
clean:
	-rm -rf $(EXECUTABLE)
	-rm -rf *.o
	-rm -rf *~
