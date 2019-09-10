PG_INC=$(PGHOME)/include/server/
GP_INC=$(GPHOME)/include/postgresql/server/
GP_INC_INTERNAL=$(GPHOME)/include/postgresql/internal/
CFLAGS=-O3 -I../../.. -fpic 
LDFLAGS=-shared
CC=gcc

all: pg gp
pg: crf crf-agg clean
gp: crf-gp crf-gp-agg clean

crf:
	$(CC) $(CFLAGS) -I$(PG_INC) -c crf.c -o crf.o
	$(CC) $(LDFLAGS) -o crf.so crf.o
	cp crf.so $(PGHOME)/lib/crf-shmem.so

crf-agg:
	$(CC) -DVAGG $(CFLAGS) -I$(PG_INC) -c crf.c -o crf.o
	$(CC) $(LDFLAGS) -o crf.so crf.o
	cp crf.so $(PGHOME)/lib/crf-agg.so

crf-gp:
	$(CC) $(CFLAGS) -I$(GP_INC) -I$(GP_INC_INTERNAL) -c crf.c -o crf.o
	$(CC) $(LDFLAGS) -o crf.so crf.o
	cp crf.so $(GPHOME)/lib/postgresql/crf-shmem.so

crf-gp-agg:
	$(CC) -DVAGG $(CFLAGS) -I$(GP_INC) -I$(GP_INC_INTERNAL) -c crf.c -o crf.o
	$(CC) $(LDFLAGS) -o crf.so crf.o
	cp crf.so $(GPHOME)/lib/postgresql/crf-agg.so

clean:
	rm *.o *.so

