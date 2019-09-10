PG_INC=$(PGHOME)/include/server/
GP_INC=$(GPHOME)/include/postgresql/server/
GP_INC_INTERNAL=$(GPHOME)/include/postgresql/internal/
CFLAGS=-O3 -I../../.. -fpic 
LDFLAGS=-shared
CC=gcc

all: pg gp
pg: dense sparse dense-agg sparse-agg clean
gp: dense-gp sparse-gp dense-gp-agg sparse-gp-agg clean

sparse:
	$(CC) -DSPARSE $(CFLAGS) -I$(PG_INC) -c logit.c -o logit.o
	$(CC) $(LDFLAGS) -o logit.so logit.o
	cp logit.so $(PGHOME)/lib/sparse-logit-shmem.so

dense:
	$(CC) $(CFLAGS) -I$(PG_INC) -c logit.c -o logit.o
	$(CC) $(LDFLAGS) -o logit.so logit.o
	cp logit.so $(PGHOME)/lib/dense-logit-shmem.so

sparse-agg:
	$(CC) -DVAGG -DSPARSE $(CFLAGS) -I$(PG_INC) -c logit.c -o logit.o
	$(CC) $(LDFLAGS) -o logit.so logit.o
	cp logit.so $(PGHOME)/lib/sparse-logit-agg.so

dense-agg:
	$(CC) -DVAGG $(CFLAGS) -I$(PG_INC) -c logit.c -o logit.o
	$(CC) $(LDFLAGS) -o logit.so logit.o
	cp logit.so $(PGHOME)/lib/dense-logit-agg.so

sparse-gp:
	$(CC) -DSPARSE $(CFLAGS) -I$(GP_INC) -I$(GP_INC_INTERNAL) -c logit.c -o logit.o
	$(CC) $(LDFLAGS) -o logit.so logit.o
	cp logit.so $(GPHOME)/lib/postgresql/sparse-logit-shmem.so

dense-gp:
	$(CC) $(CFLAGS) -I$(GP_INC) -I$(GP_INC_INTERNAL) -c logit.c -o logit.o
	$(CC) $(LDFLAGS) -o logit.so logit.o
	cp logit.so $(GPHOME)/lib/postgresql/dense-logit-shmem.so

sparse-gp-agg:
	$(CC) -DVAGG -DSPARSE $(CFLAGS) -I$(GP_INC) -I$(GP_INC_INTERNAL) -c logit.c -o logit.o
	$(CC) $(LDFLAGS) -o logit.so logit.o
	cp logit.so $(GPHOME)/lib/postgresql/sparse-logit-agg.so

dense-gp-agg:
	$(CC) -DVAGG $(CFLAGS) -I$(GP_INC) -I$(GP_INC_INTERNAL) -c logit.c -o logit.o
	$(CC) $(LDFLAGS) -o logit.so logit.o
	cp logit.so $(GPHOME)/lib/postgresql/dense-logit-agg.so

clean:
	rm *.o *.so
