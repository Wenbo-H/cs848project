PG_INC=$(PGHOME)/include/server/
GP_INC=$(GPHOME)/include/postgresql/server/
GP_INC_INTERNAL=$(GPHOME)/include/postgresql/internal/
CFLAGS=-O3 -I../../.. -fpic 
LDFLAGS=-shared
CC=gcc

all: pg gp
pg: factor factor-agg clean
gp: factor-gp factor-gp-agg clean

factor:
	$(CC) $(CFLAGS) -I$(PG_INC) -c factor.c -o factor.o
	$(CC) $(LDFLAGS) -o factor.so factor.o
	cp factor.so $(PGHOME)/lib/factor-shmem.so

factor-agg:
	$(CC) -DVAGG $(CFLAGS) -I$(PG_INC) -c factor.c -o factor.o
	$(CC) $(LDFLAGS) -o factor.so factor.o
	cp factor.so $(PGHOME)/lib/factor-agg.so

factor-gp:
	$(CC) $(CFLAGS) -I$(GP_INC) -I$(GP_INC_INTERNAL) -c factor.c -o factor.o
	$(CC) $(LDFLAGS) -o factor.so factor.o
	cp factor.so $(GPHOME)/lib/postgresql/factor-shmem.so

factor-gp-agg:
	$(CC) -DVAGG $(CFLAGS) -I$(GP_INC) -I$(GP_INC_INTERNAL) -c factor.c -o factor.o
	$(CC) $(LDFLAGS) -o factor.so factor.o
	cp factor.so $(GPHOME)/lib/postgresql/factor-agg.so

clean:
	rm *.o *.so

