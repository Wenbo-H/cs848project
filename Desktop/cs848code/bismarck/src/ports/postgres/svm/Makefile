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
	$(CC) -DSPARSE $(CFLAGS) -I$(PG_INC) -c svm.c -o svm.o
	$(CC) $(LDFLAGS) -o svm.so svm.o
	cp svm.so $(PGHOME)/lib/sparse-svm-shmem.so

dense:
	$(CC) $(CFLAGS) -I$(PG_INC) -c svm.c -o svm.o
	$(CC) $(LDFLAGS) -o svm.so svm.o
	cp svm.so $(PGHOME)/lib/dense-svm-shmem.so

sparse-agg:
	$(CC) -DVAGG -DSPARSE $(CFLAGS) -I$(PG_INC) -c svm.c -o svm.o
	$(CC) $(LDFLAGS) -o svm.so svm.o
	cp svm.so $(PGHOME)/lib/sparse-svm-agg.so

dense-agg:
	$(CC) -DVAGG $(CFLAGS) -I$(PG_INC) -c svm.c -o svm.o
	$(CC) $(LDFLAGS) -o svm.so svm.o
	cp svm.so $(PGHOME)/lib/dense-svm-agg.so

sparse-gp:
	$(CC) -DSPARSE $(CFLAGS) -I$(GP_INC) -I$(GP_INC_INTERNAL) -c svm.c -o svm.o
	$(CC) $(LDFLAGS) -o svm.so svm.o
	cp svm.so $(GPHOME)/lib/postgresql/sparse-svm-shmem.so

dense-gp:
	$(CC) $(CFLAGS) -I$(GP_INC) -I$(GP_INC_INTERNAL) -c svm.c -o svm.o
	$(CC) $(LDFLAGS) -o svm.so svm.o
	cp svm.so $(GPHOME)/lib/postgresql/dense-svm-shmem.so

sparse-gp-agg:
	$(CC) -DVAGG -DSPARSE $(CFLAGS) -I$(GP_INC) -I$(GP_INC_INTERNAL) -c svm.c -o svm.o
	$(CC) $(LDFLAGS) -o svm.so svm.o
	cp svm.so $(GPHOME)/lib/postgresql/sparse-svm-agg.so

dense-gp-agg:
	$(CC) -DVAGG $(CFLAGS) -I$(GP_INC) -I$(GP_INC_INTERNAL) -c svm.c -o svm.o
	$(CC) $(LDFLAGS) -o svm.so svm.o
	cp svm.so $(GPHOME)/lib/postgresql/dense-svm-agg.so

clean:
	rm *.o *.so
