PG_INC=$(PGHOME)/include/server/
GP_INC=$(GPHOME)/include/postgresql/server/
GP_INC_INTERNAL=$(GPHOME)/include/postgresql/internal/
CFLAGS=-O3 -I../../.. -fpic 
LDFLAGS=-shared
CC=gcc

all: pg gp
pg: array clean
gp: array-gp clean

array:
	$(CC) $(CFLAGS) -I$(PG_INC) -c array.c -o array.o
	$(CC) $(LDFLAGS) -o array.so array.o
	cp array.so $(PGHOME)/lib/bismarck-array.so

array-gp:
	$(CC) $(CFLAGS) -I$(GP_INC) -I$(GP_INC_INTERNAL) -c array.c -o array.o
	$(CC) $(LDFLAGS) -o array.so array.o
	cp array.so $(GPHOME)/lib/postgresql/bismarck-array.so

clean:
	rm *.o *.so

