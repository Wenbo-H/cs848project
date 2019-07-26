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

#include "../c_udf_helper.h"
#include "utils/numeric.h"
#include "modules/linear/linear_model.h"
#include "modules/logit/logit.h"

/* the proof of postgresql version 1 C UDF */
PG_FUNCTION_INFO_V1(init);
PG_FUNCTION_INFO_V1(grad);
PG_FUNCTION_INFO_V1(pre);
PG_FUNCTION_INFO_V1(final);
PG_FUNCTION_INFO_V1(loss);
PG_FUNCTION_INFO_V1(pred);

/**
 * init for a new model instance
 */
Datum
init(PG_FUNCTION_ARGS) {
    // -------------------------------------------------------------------
    // 0. parse lmf_model row type into local variables
    // -------------------------------------------------------------------
    HeapTupleHeader modelTuple = PG_GETARG_HEAPTUPLEHEADER(0);
    bool isnull;
    // meta data
    int mid = DatumGetInt32(GetAttributeByNum(modelTuple, 1, &isnull));
    int ndims = DatumGetInt32(GetAttributeByNum(modelTuple, 2, &isnull));
    int ntuples = DatumGetInt32(GetAttributeByNum(modelTuple, 3, &isnull));
    double mu = DatumGetFloat8(GetAttributeByNum(modelTuple, 4, &isnull));
    double stepsize = DatumGetFloat8(GetAttributeByNum(modelTuple, 5, &isnull));
    double decay = DatumGetFloat8(GetAttributeByNum(modelTuple, 6, &isnull));
    // weight vector
    ArrayType *warray = (ArrayType *) GetAttributeByNum(modelTuple, 7, &isnull);
    double *w;
    int wLen = my_parse_array_no_copy((struct varlena*) warray, 
            sizeof(float8), (char **) &w);

    // temp_v vector ?????
    ArrayType *varray = (ArrayType *)GetAttributeByNum(modelTuple, 8, &isnull);
    //int haibo = DatumGetInt32(GetAttributeByNum(modelTuple, 8, &isnull));
    
    double *temp_v;
    int vLen = my_parse_array_no_copy((struct varlena*) varray, 
            sizeof(float8), (char **) &temp_v); 

    // dimension sanity check
    assert(wLen == ndims);
    assert(vLen == ndims);

#ifdef VAGG
    // -------------------------------------------------------------------
    // 1. allocate w+
    // -------------------------------------------------------------------
    double *wp;
    ArrayType *wparray = my_construct_array(wLen + vLen + META_LEN, sizeof(float8),
            FLOAT8OID);
    int wpLen = my_parse_array_no_copy((struct varlena *) wparray, 
            sizeof(float8), (char **) &wp);

    // -------------------------------------------------------------------
    // 2. serialize meta data into w+
    // -------------------------------------------------------------------
    wp[0] = mid;
    wp[1] = ndims;
    wp[2] = ntuples;
    wp[3] = mu;
    wp[4] = stepsize;
    wp[5] = decay;
    wp[6] = 0;  // count of tuple seen

    // -------------------------------------------------------------------
    // 3. copy weight vector into w+
    // -------------------------------------------------------------------
    memcpy(wp + META_LEN, w, sizeof(double) * wLen);

    memcpy(wp + META_LEN + wLen, temp_v, sizeof(double) * vLen);

    // return
    PG_RETURN_ARRAYTYPE_P(wparray);
#else
    //--------------------------------------------------------------------
    // 1. create a shared memory region for the LinearModel structure
    //    using mid as key
    //--------------------------------------------------------------------
    struct LinearModel* ptrModel;
    int size = sizeof(struct LinearModel) + sizeof(double) * (ndims) * 2;
    // open the shared memory
    int shmid = shmget(ftok("/", mid), size, SHM_R | SHM_W | IPC_CREAT);
    if (shmid == -1) { elog(ERROR, "In init, shmget failed!\n"); }
    // elog(WARNING, "init: after shmget\n");
    // attach the memory region
    ptrModel = (struct LinearModel*) shmat(shmid, NULL, 0);
    // elog(WARNING, "init: after shmat, model: %x\n", ptrModel);

    //--------------------------------------------------------------------
    // 2. init meta data
    //--------------------------------------------------------------------
    // constructor
    LinearModel_init(ptrModel, mid, ndims, ntuples, 
			mu, stepsize, decay);

    // -------------------------------------------------------------------
    // 3. copy weight vector into shared memory
    // -------------------------------------------------------------------
    memcpy(ptrModel->w, w, sizeof(double) * wLen);

    memcpy(ptrModel->temp_v, temp_v, sizeof(double) * vLen);

    PG_RETURN_NULL();
#endif
}

/**
 * gradient function
 */
Datum
grad(PG_FUNCTION_ARGS) {
#if defined(VAGG) && defined(SPARSE)
#define OLD_MODEL (4)
#elif defined(VAGG) 
#define OLD_MODEL (3)
#endif

#ifdef VAGG
    //--------------------------------------------------------------------
    // 1. init a local LinearModel structure 
    // and get the weight vector from temp state
    //--------------------------------------------------------------------
    ArrayType *wparray = (ArrayType *) PG_GETARG_RAW_VARLENA_P(0);
    double *wp;
    int wpLen = my_parse_array_no_copy((struct varlena*) wparray, 
            sizeof(float8), (char **) &wp);
    struct LinearModel modelBuffer;
    struct LinearModel *ptrModel;
    // beginning of an epoch
    if (wpLen == 1) {
        // use arg[4] to retrieve serialized model w+
        ArrayType *initwparray = (ArrayType *) PG_GETARG_RAW_VARLENA_P(OLD_MODEL);
        double *initwp;
        int initwpLen = my_parse_array_no_copy((struct varlena*) initwparray, 
                sizeof(float8), (char **) &initwp);
        wparray = my_construct_array(initwpLen, sizeof(float8), FLOAT8OID);
        wpLen = my_parse_array_no_copy((struct varlena *)wparray, 
                sizeof(float8), (char **)&wp);
		memcpy(wp, initwp, initwpLen * sizeof(float8));
		assert(wp[6] == 0);
    }
    // local copy
    ptrModel = &modelBuffer;
    // init hyper parameters
    LinearModel_init(ptrModel, (int) wp[0], (int) wp[1], (int) wp[2], 
            wp[3], wp[4], wp[5]);
	// point to the weight vector and update in place
    ptrModel->w = wp + META_LEN;
    int wLen = (wpLen - META_LEN)/2;
    ptrModel->temp_v = wp + META_LEN + wLen;
    // count
    wp[6] ++;
    // elog(WARNING, "grad: count: %lf, nDims %d", ptrModel->w[ptrModel->nDims], ptrModel->nDims);
#else
    //--------------------------------------------------------------------
    // 1. get the LinearModel structure from the shared memory
    //    using mid (arg[0])
    //--------------------------------------------------------------------
    int32 mid = PG_GETARG_INT32(0);
    // model
    struct LinearModel modelBuffer;
    struct LinearModel* ptrModel = &modelBuffer;
    static struct LinearModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct LinearModel*)get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
	*ptrModel = (*ptrSharedModel);
	ptrModel->w = (double *)(&(ptrSharedModel->w) + 1);
    ptrModel->temp_v = (double *)(&(ptrSharedModel->temp_v) + 1);
#endif

    //--------------------------------------------------------------------
    // 2. parse the args (k, v, y) or (v, y)
    //--------------------------------------------------------------------
#if defined(SPARSE)
    // some decoding of the binary format has been done before arg passing
    // k
    int32 *k;
    int len1 = my_parse_array_no_copy((struct varlena*) PG_GETARG_RAW_VARLENA_P(1), 
            sizeof(int32), (char **)&k);
    // v
    float8 *v;
    int len2 = my_parse_array_no_copy((struct varlena*) PG_GETARG_RAW_VARLENA_P(2),
               sizeof(float8), (char **)&v);
    // length check
    //assert(len1 == len2);
    // y
    int32 y = PG_GETARG_INT32(3);
#else
    // some decoding of the binary format has been done before arg passing
    // v
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(1);
    float8* v;
    int len = my_parse_array_no_copy(v2, sizeof(float8), (char **)&v);
    // assert(len == N_DIMS);
    // y
    int32 y = PG_GETARG_INT32(2);
#endif
	// elog(WARNING, "grad: 2");

    //--------------------------------------------------------------------
    // 3. performing the gradient 
    //--------------------------------------------------------------------
#if !defined(VAGG) && defined(VLOCK)
	while (compare_and_swap(&(ptrSharedModel->token), 0, 1) == 0) {}
#endif

#ifdef SPARSE
    sparse_logit_grad(ptrModel, len1, k, v, y);
#else    
    dense_logit_grad(ptrModel, v, y);
#endif

#if !defined(VAGG) && defined(VLOCK)
	ptrSharedModel->token = 0;
#endif

#ifdef VAGG
	// return array for agg
    PG_RETURN_ARRAYTYPE_P(wparray);
#else
	// return null
    PG_RETURN_NULL();
#endif
}

/**
 * pre function
 */
Datum
pre(PG_FUNCTION_ARGS) {
#ifdef VAGG
    //--------------------------------------------------------------------
    // 1. average the weights and keep the count
    //--------------------------------------------------------------------
    // get state 0
    // elog(WARNING, "inside pre 0");
    ArrayType *wparray = (ArrayType *) PG_GETARG_RAW_VARLENA_P(0);
    double *wp;
    int wpLen = my_parse_array_no_copy((struct varlena*) wparray, 
            sizeof(float8), (char **) &wp);
    // elog(WARNING, "inside pre 1");
    // get state 1
    ArrayType *wparray1 = (ArrayType *) PG_GETARG_RAW_VARLENA_P(1);
    double *wp1;
    int wpLen1 = my_parse_array_no_copy((struct varlena*) wparray1, 
            sizeof(float8), (char **) &wp1);
    // elog(WARNING, "inside pre 2: wLen0: %d, wLen1: %d", wLen, wLen1);
    if (wpLen == 1) {
        PG_RETURN_ARRAYTYPE_P(wparray1);
    }
	if (wpLen1 == 1) {
        PG_RETURN_ARRAYTYPE_P(wparray);
    }
    // the count
    int count0 = wp[6];
    // elog(WARNING, "inside pre 3");
    int count1 = wp1[6];
    // elog(WARNING, "inside pre 4");
    // elog(WARNING, "count0: %d, count1: %d", count0, count1);
    int count = count0 + count1;
    int vLen = (wpLen - META_LEN)/2;
    // add 1 to 0 in place
    int i;
    for (i = META_LEN; i < wpLen - vLen; i ++) {
        wp[i] = (count0 * 1.0 / count) * wp[i] + (count1 * 1.0 / count) * wp1[i];
    }
    wp[6] = count;

    PG_RETURN_ARRAYTYPE_P(wparray);
#else
    //--------------------------------------------------------------------
    // 1. get the LinearModel structure from the shared memory
    //    using mid (arg[0])
    //--------------------------------------------------------------------
    int32 mid = PG_GETARG_INT32(0);
    // model
    struct LinearModel* ptrSharedModel = (struct LinearModel*) get_model_by_mid(mid);

    //--------------------------------------------------------------------
    // 2. update step size
    //--------------------------------------------------------------------
	LinearModel_take_step(ptrSharedModel);
    
    // return null
    PG_RETURN_NULL();
#endif
}

/**
 * final function
 */
Datum
final(PG_FUNCTION_ARGS) {
    ArrayType *warray;
    double *w;    //weight
    int wLen;
    double *temp_v;
    int vLen; 
#ifdef VAGG
    //--------------------------------------------------------------------
    // 1. cut the last count and return the state
    //--------------------------------------------------------------------
    // get state
    // elog(WARNING, "final: 0");
    ArrayType *wparray = (ArrayType *) PG_GETARG_RAW_VARLENA_P(0);
	double *wp;
    int wpLen = my_parse_array_no_copy((struct varlena*) wparray, 
            sizeof(float8), (char **) &wp);
    // sanity checking
    assert(wpLen == ((int) wp[1]) + (int) wp[1] + META_LEN);
    assert(((int) wp[2]) == ((int) wp[6]));
    //--------------------------------------------------------------------
    // 2. get rid of count when outputing
    //--------------------------------------------------------------------
	warray = my_construct_array((wpLen - META_LEN)/2, sizeof(float8), FLOAT8OID);
    ArrayType *varray = my_construct_array((wpLen - META_LEN)/2, sizeof(float8), FLOAT8OID);
	wLen = my_parse_array_no_copy((struct varlena *)warray, 
			sizeof(float8), (char **)&w);
    vLen = my_parse_array_no_copy((struct varlena *)varray, 
        sizeof(float8), (char **)&temp_v);

	memcpy(w, wp + META_LEN, (wpLen - META_LEN)/2 * sizeof(float8));
    memcpy(temp_v, wp + META_LEN + wLen, (wpLen - META_LEN)/2 * sizeof(float8));
#else
    //--------------------------------------------------------------------
    // 1. get model from shared memory
    //--------------------------------------------------------------------
    int32 mid = PG_GETARG_INT32(0);
    // model
    struct LinearModel* ptrSharedModel = (struct LinearModel*) get_model_by_mid(mid);

    //--------------------------------------------------------------------
    // 2. construct a PG array to return and delete the shared memory
    //--------------------------------------------------------------------
	wLen = ptrSharedModel->nDims;
	warray = my_construct_array(wLen, sizeof(float8), FLOAT8OID);
	wLen = my_parse_array_no_copy((struct varlena *)warray, 
			sizeof(float8), (char **)&w);
	memcpy(w, &(ptrSharedModel->w) + 1, wLen * sizeof(float8));

    vLen = ptrSharedModel->nDims;
    ArrayType *varray = my_construct_array(vLen, sizeof(float8), FLOAT8OID);
    vLen = my_parse_array_no_copy((struct varlena *)varray, 
            sizeof(float8), (char **)&temp_v);
    memcpy(temp_v, &(ptrSharedModel->temp_v) + 1, vLen * sizeof(float8));

	// delete the shared memory
	int shmid = shmget(ftok("/", mid), 0, SHM_R | SHM_W);
	if (shmid == -1) {	elog(ERROR, "In final, shmget failed!\n"); }
	struct shmid_ds shm_buf;
	if (shmctl(shmid, IPC_RMID, &shm_buf) == -1) {
		elog(ERROR, "shmctl failed in final()");
	}
#endif
    PG_RETURN_ARRAYTYPE_P(warray);
}

/**
 * loss function
 */
Datum
loss(PG_FUNCTION_ARGS) {
#ifdef VAGG
    //--------------------------------------------------------------------
    // 1. init a local LinearModel structure 
    // and get the weight vector from temp state
    //--------------------------------------------------------------------
    ArrayType *wparray = (ArrayType *) PG_GETARG_RAW_VARLENA_P(0);
    double *wp;
    int wpLen = my_parse_array_no_copy((struct varlena*) wparray, 
            sizeof(float8), (char **) &wp);
    struct LinearModel modelBuffer;
    struct LinearModel *ptrModel;
    // local copy
    ptrModel = &modelBuffer;
    // init hyper parameters
    LinearModel_init(ptrModel, (int) wp[0], (int) wp[1], (int) wp[2], 
            wp[3], wp[4], wp[5]);
	// point to the weight vector
    int wLen = (wpLen - META_LEN)/2;
    ptrModel->w = wp + META_LEN;
    ptrModel->temp_v = wp + META_LEN + wLen;
    // count
    wp[6] ++;
    // elog(WARNING, "grad: count: %lf, nDims %d", ptrModel->w[ptrModel->nDims], ptrModel->nDims);
#else
    //--------------------------------------------------------------------
    // 1. get the LinearModel structure from the shared memory
    //    using mid (arg[0])
    //--------------------------------------------------------------------
    int32 mid = PG_GETARG_INT32(0);
    // model
    struct LinearModel modelBuffer;
    struct LinearModel* ptrModel = &modelBuffer;
    static struct LinearModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct LinearModel*) get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
	*ptrModel = (*ptrSharedModel);
	ptrModel->w = (double *) (&(ptrSharedModel->w) + 1);
    ptrModel->temp_v = (double *) (&(ptrSharedModel->temp_v) + 1);

#endif

    //--------------------------------------------------------------------
    // 2. parse the args (k, v, y) or (v, y)
    //--------------------------------------------------------------------
#ifdef SPARSE
    // some decoding of the binary format has been done before arg passing
    // k
    struct varlena* v1 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(1);
    int32* k;
    int len1 = my_parse_array_no_copy(v1, sizeof(int32), (char **)&k);
    // v
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(2);
    float8* v;
    int len2 = my_parse_array_no_copy(v2, sizeof(float8), (char **)&v);
    // length check
    //assert(len1 == len2);
    // y
    int32 y = PG_GETARG_INT32(3);
#else
    // some decoding of the binary format has been done before arg passing
    // v
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(1);
    float8* v;
    int len = my_parse_array_no_copy(v2, sizeof(float8), (char **)&v);
    // assert(len == N_DIMS);
    // y
    int32 y = PG_GETARG_INT32(2);
#endif

    //--------------------------------------------------------------------
    // 3. computing loss
    //--------------------------------------------------------------------
    double err = -1;
#ifdef SPARSE
    err = sparse_logit_loss(ptrModel, len1, k, v, y);
#else
    err = dense_logit_loss(ptrModel, v, y);
#endif

    PG_RETURN_FLOAT8(err);
}


/**
 * predict function
 */
Datum
pred(PG_FUNCTION_ARGS) {
#ifdef VAGG
    //--------------------------------------------------------------------
    // 1. init a local LinearModel structure 
    // and get the weight vector from temp state
    //--------------------------------------------------------------------
    ArrayType *wparray = (ArrayType *) PG_GETARG_RAW_VARLENA_P(0);
    double *wp;
    int wpLen = my_parse_array_no_copy((struct varlena*) wparray, 
            sizeof(float8), (char **) &wp);
    struct LinearModel modelBuffer;
    struct LinearModel *ptrModel;
    // local copy
    ptrModel = &modelBuffer;
    // init hyper parameters
    LinearModel_init(ptrModel, (int) wp[0], (int) wp[1], (int) wp[2], 
            wp[3], wp[4], wp[5]);
	// point to the weight vector

    int wLen = (wpLen - META_LEN)/2;

    ptrModel->w = wp + META_LEN;
    ptrModel->temp_v = wp + META_LEN + wLen;
    // count
    wp[6] ++;
    // elog(WARNING, "grad: count: %lf, nDims %d", ptrModel->w[ptrModel->nDims], ptrModel->nDims);
#else
    //--------------------------------------------------------------------
    // 1. get the LinearModel structure from the shared memory
    //    using mid (arg[0])
    //--------------------------------------------------------------------
    int32 mid = PG_GETARG_INT32(0);
    // model
    struct LinearModel modelBuffer;
    struct LinearModel* ptrModel = &modelBuffer;
    static struct LinearModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct LinearModel*) get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
	*ptrModel = (*ptrSharedModel);
	ptrModel->w = (double *) (&(ptrSharedModel->w) + 1);
    ptrModel->temp_v = (double *) (&(ptrSharedModel->temp_v) + 1);
    #endif //We always read the model into shmem for prediction

    //--------------------------------------------------------------------
    // 2. parse the args (k, v) or (v)
    //--------------------------------------------------------------------
#ifdef SPARSE
    // some decoding of the binary format has been done before arg passing
    // k
    struct varlena* v1 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(1);
    int32* k;
    int len1 = my_parse_array_no_copy(v1, sizeof(int32), (char **)&k);
    // v
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(2);
    float8* v;
    int len2 = my_parse_array_no_copy(v2, sizeof(float8), (char **)&v);
    // length check
    //assert(len1 == len2);
#else
    // some decoding of the binary format has been done before arg passing
    // v
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(1);
    float8* v;
    int len = my_parse_array_no_copy(v2, sizeof(float8), (char **)&v);
    // assert(len == N_DIMS);
#endif

    //--------------------------------------------------------------------
    // 3. computing prob
    //--------------------------------------------------------------------
    double pred = -1;
#ifdef SPARSE
    pred = sparse_logit_pred(ptrModel, len1, k, v);
#else
    pred = dense_logit_pred(ptrModel, v);
#endif

    PG_RETURN_FLOAT8(pred);
}


