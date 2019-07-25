/*
Copyright 2012 Xixuan (Aaron) Feng and Arun Kumar and Christopher Re

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"

#include "catalog/pg_type.h"
#include "utils/array.h"
#include "executor/spi.h"
#include "access/tuptoaster.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

/* macros that are subject to change due to system environment */
#define SHARED_MEM_SIZE (1 << 30)
#define ARRAY_HEAD_SIZE (20)

/**
 * string functions def
 */
inline char* 
pack_shmem_name(int mid, char* buf, int size) {
	memset(buf, '\0', size);
	sprintf(buf, "/mid%d", mid);
    return buf;
}

/**
 * create or get a shared memory region of mid
 *
 * args:
 *   mid int, model id
 * return:
 *   pointer char*, start pointer
 */
inline char *
get_model_by_mid(int mid) {
	char* ptrModel;
	char buf[20];
	// open the shared memory
	int shmid = shmget(ftok("/", mid), 0, SHM_R | SHM_W);
	if (shmid == -1) {	elog(ERROR, "In get model by id, shmget failed!\n"); }
	// elog(WARNING, "init: after shmget\n");
	// attach the memory region
	ptrModel = (char *)shmat(shmid, NULL, 0);
	// elog(WARNING, "init: after shmat, model: %x\n", ptrModel);
	return ptrModel;
}



 /* ----------------
  *      Variable-length datatypes all share the 'struct varlena' header.
  *
  * NOTE: for TOASTable types, this is an oversimplification, since the value
  * may be compressed or moved out-of-line.  However datatype-specific routines
  * are mostly content to deal with de-TOASTed values only, and of course
  * client-side routines should never see a TOASTed value.  But even in a
  * de-TOASTed value, beware of touching vl_len_ directly, as its
  * representation is no longer convenient.  It's recommended that code always
  * use macros VARDATA_ANY, VARSIZE_ANY, VARSIZE_ANY_EXHDR, VARDATA, VARSIZE,
  * and SET_VARSIZE instead of relying on direct mentions of the struct fields.
  * See postgres.h for details of the TOASTed form.
  * ----------------

   struct varlena
 {
     char        vl_len_[4];        Do not touch this field directly! 
     char        vl_dat[FLEXIBLE_ARRAY_MEMBER];      Data content is here 
 };
 */


/**
 * parse the array by NO PALLOC?
 *
 * args:
 *   input struct varlena*, variable length struct pointer
 *   typesize int, size of element type
 *   output (void*)*, start pointer of the array elements
 * return:
 *   int, length of the array, # of elements
 */

inline int 
my_parse_array_no_copy(struct varlena* input, int typesize, char** output) {
	//elog(WARNING, "Inside loss(), for v, ISEXTERNAL %d, ISCOMPR %d, ISHORT %d, varsize_short %d", VARATT_IS_EXTERNAL(v2) ? 1 : 0, VARATT_IS_COMPRESSED(v2)  ? 1 : 0, VARATT_IS_SHORT(v2)  ? 1 : 0, VARSIZE_SHORT(v2));
	if (VARATT_IS_EXTERNAL(input) || VARATT_IS_COMPRESSED(input)) {
		// if compressed, palloc is necessary
		input = heap_tuple_untoast_attr(input);
        *output = VARDATA(input) + ARRAY_HEAD_SIZE;
        return (VARSIZE(input) - VARHDRSZ - ARRAY_HEAD_SIZE) / typesize;
	} else if (VARATT_IS_SHORT(input)) {
        *output = VARDATA_SHORT(input) + ARRAY_HEAD_SIZE;
        return (VARSIZE_SHORT(input) - VARHDRSZ_SHORT - ARRAY_HEAD_SIZE) / typesize;
    } else {
        *output = VARDATA(input) + ARRAY_HEAD_SIZE;
        return (VARSIZE(input) - VARHDRSZ - ARRAY_HEAD_SIZE) / typesize;
    }
}

/**
 * construct Postgres array, not null elements assumed
 *
 * args:
 *   nelems int, number of elements
 *   typesize int, size of element type
 *   elemtype Oid, OID of elements
 * return:
 *   ArrayType *, resulting Postgres array with all zeros
 */
inline ArrayType *
my_construct_array(int nelems, int typesize, Oid elemtype) {
	int nbytes = ARR_OVERHEAD_NONULLS(1) + nelems * typesize;
	ArrayType *result = (ArrayType *) palloc0(nbytes);
	SET_VARSIZE(result, nbytes);
	result->ndim = 1;
	result->dataoffset = 0;
	result->elemtype = elemtype;
	*(int *) ARR_DIMS(result) = nelems;
	*(int *) ARR_LBOUND(result) = 1;
	return result;
}
