PG_INC=$(PGHOME)/include/server/
GP_INC=$(GPHOME)/include/postgresql/server/
GP_INC_INTERNAL=$(GPHOME)/include/postgresql/internal/
CFLAGS=-O3 -I../../.. -fpic 
LDFLAGS=-shared
CC=gcc

all: pg
pg: linear-reg linear-reg-agg clean

linear-reg:
	$(CC) $(CFLAGS) -I$(PG_INC) -c linear_reg.c -o linear_reg.o
	$(CC) $(LDFLAGS) -o linear_reg.so linear_reg.o
	cp linear_reg.so $(PGHOME)/lib/linear-reg-shmem.so

linear-reg-agg:
	$(CC) -DVAGG $(CFLAGS) -I$(PG_INC) -c linear_reg.c -o linear_reg.o
	$(CC) $(LDFLAGS) -o linear_reg.so linear_reg.o
	cp linear_reg.so $(PGHOME)/lib/linear-reg-agg.so

clean:
	rm *.o *.so

