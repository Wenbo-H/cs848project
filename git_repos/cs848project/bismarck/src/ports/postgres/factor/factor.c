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
#include "modules/factor/factor_model.h"

/* the proof of postgresql version 1 C UDF */
PG_FUNCTION_INFO_V1(init);
PG_FUNCTION_INFO_V1(grad);
PG_FUNCTION_INFO_V1(pre);
PG_FUNCTION_INFO_V1(final);
PG_FUNCTION_INFO_V1(loss);

/**
 * init for a new model instance
 */
Datum
init(PG_FUNCTION_ARGS) {
    // -------------------------------------------------------------------
    // 0. parse factor_model row type into local variables
    // -------------------------------------------------------------------
    HeapTupleHeader modelTuple = PG_GETARG_HEAPTUPLEHEADER(0);
    bool isnull;
    // meta data
    int mid = DatumGetInt32(GetAttributeByNum(modelTuple, 1, &isnull));
    int nrows = DatumGetInt32(GetAttributeByNum(modelTuple, 2, &isnull));
    int ncols = DatumGetInt32(GetAttributeByNum(modelTuple, 3, &isnull));
    int maxrank = DatumGetInt32(GetAttributeByNum(modelTuple, 4, &isnull));
    int ndims = DatumGetInt32(GetAttributeByNum(modelTuple, 5, &isnull));
    int ntuples = DatumGetInt32(GetAttributeByNum(modelTuple, 6, &isnull));
    double B = DatumGetFloat8(GetAttributeByNum(modelTuple, 7, &isnull));
    double stepsize = DatumGetFloat8(GetAttributeByNum(modelTuple, 8, &isnull));
    double decay = DatumGetFloat8(GetAttributeByNum(modelTuple, 9, &isnull));
    // weight vector
    ArrayType *warray = (ArrayType *) GetAttributeByNum(modelTuple, 10, &isnull);
    double *w;
    int wLen = my_parse_array_no_copy((struct varlena*) warray, 
            sizeof(float8), (char **) &w);
    // dimension sanity check
    assert(wLen == ndims);
    assert((nrows + ncols) * maxrank == ndims);

#ifdef VAGG
    // -------------------------------------------------------------------
    // 1. allocate w+
    // -------------------------------------------------------------------
    double *wp;
    ArrayType *wparray = my_construct_array(wLen + META_LEN, sizeof(float8),
            FLOAT8OID);
    int wpLen = my_parse_array_no_copy((struct varlena *) wparray, 
            sizeof(float8), (char **) &wp);

    // -------------------------------------------------------------------
    // 2. serialize meta data into w+
    // -------------------------------------------------------------------
    wp[0] = mid;
    wp[1] = nrows;
    wp[2] = ncols;
    wp[3] = maxrank;
    wp[4] = ndims;
    wp[5] = ntuples;
    wp[6] = B;
    wp[7] = stepsize;
    wp[8] = decay;
    wp[9] = 0; // count of tuple seen

    // -------------------------------------------------------------------
    // 3. copy weight vector into w+
    // -------------------------------------------------------------------
    memcpy(wp + META_LEN, w, sizeof(double) * wLen);

    // return
    PG_RETURN_ARRAYTYPE_P(wparray);
#else
    //--------------------------------------------------------------------
    // 1. create a shared memory region for the FactorModel structure
    //    using mid as key
    //--------------------------------------------------------------------
    struct FactorModel* ptrModel;
    int size = sizeof(struct FactorModel) + sizeof(double) * (ndims) * 2;
    // open the shared memory
    int shmid = shmget(ftok("/", mid), size, SHM_R | SHM_W | IPC_CREAT);
    if (shmid == -1) { elog(ERROR, "In init, shmget failed!\n"); }
    // elog(WARNING, "init: after shmget\n");
    // attach the memory region
    ptrModel = (struct FactorModel*) shmat(shmid, NULL, 0);
    // elog(WARNING, "init: after shmat, model: %x\n", ptrModel);

    //--------------------------------------------------------------------
    // 2. init meta data
    //--------------------------------------------------------------------
    // constructor
    FactorModel_init(ptrModel, mid, nrows, ncols, maxrank, ntuples, 
			B, stepsize, decay);

    // -------------------------------------------------------------------
    // 3. copy weight vector into shared memory
    // -------------------------------------------------------------------
    memcpy(ptrModel->L, w, sizeof(double) * wLen);
	ptrModel->R = ptrModel->L + ptrModel->nRows * ptrModel->maxRank;

    PG_RETURN_NULL();
#endif
}

/**
 * gradient function
 */
Datum
grad(PG_FUNCTION_ARGS) {
#ifdef VAGG
    //--------------------------------------------------------------------
    // 1. init a local FactorModel structure 
    // and get the weight vector from temp state
    //--------------------------------------------------------------------
    ArrayType *wparray = (ArrayType *) PG_GETARG_RAW_VARLENA_P(0);
    double *wp;
    int wpLen = my_parse_array_no_copy((struct varlena*) wparray, 
            sizeof(float8), (char **) &wp);
    struct FactorModel modelBuffer;
    struct FactorModel *ptrModel;
    // beginning of an epoch
    if (wpLen == 1) {
        // use arg[4] to retrieve serialized model w+
        ArrayType *initwparray = (ArrayType *) PG_GETARG_RAW_VARLENA_P(4);
        double *initwp;
        int initwpLen = my_parse_array_no_copy((struct varlena*) initwparray, 
                sizeof(float8), (char **) &initwp);
        wparray = my_construct_array(initwpLen, sizeof(float8), FLOAT8OID);
        wpLen = my_parse_array_no_copy((struct varlena *)wparray, 
                sizeof(float8), (char **)&wp);
		memcpy(wp, initwp, initwpLen * sizeof(float8));
		assert(wp[9] == 0);
    }
    // local copy
    ptrModel = &modelBuffer;
    // get hard-coded hyper parameters
    FactorModel_init(ptrModel, (int) wp[0], (int) wp[1], (int) wp[2], (int) wp[3], (int) wp[5], 
            wp[6], wp[7], wp[8]);
    ptrModel->L = wp + META_LEN;
	ptrModel->R = ptrModel->L + ptrModel->nRows * ptrModel->maxRank;
    // count
    wp[9] ++;
    // elog(WARNING, "grad: count: %lf, nDims %d", ptrModel->w[ptrModel->nDims], ptrModel->nDims);
#else
    //--------------------------------------------------------------------
    // 1. get the FactorModel structure from the shared memory
    //    using mid (arg[0])
    //--------------------------------------------------------------------
    int32 mid = PG_GETARG_INT32(0);
    // model
    struct FactorModel modelBuffer;
    struct FactorModel* ptrModel = &modelBuffer;
    static struct FactorModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct FactorModel*)get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
	*ptrModel = (*ptrSharedModel);
	ptrModel->L = (double *)(&(ptrSharedModel->L) + 1);
	ptrModel->R = ptrModel->L + ptrModel->nRows * ptrModel->maxRank;
#endif

    //--------------------------------------------------------------------
    // 2. parse the args (row, col, rating)
    //--------------------------------------------------------------------
    int32 row = PG_GETARG_INT32(1);
    int32 col = PG_GETARG_INT32(2);
    float8 rating = PG_GETARG_FLOAT8(3);
	int i = row - 1;
	int j = col - 1;

    //--------------------------------------------------------------------
    // 3. performing the gradient 
    //--------------------------------------------------------------------
#if !defined(VAGG) && defined(VLOCK)
	while (compare_and_swap(&(ptrSharedModel->token), 0, 1) == 0) {}
#endif

    FactorModel_grad(ptrModel, i, j, rating);

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
    int count0 = wp[9];
    // elog(WARNING, "inside pre 3");
    int count1 = wp1[9];
    // elog(WARNING, "inside pre 4");
    // elog(WARNING, "count0: %d, count1: %d", count0, count1);
    int count = count0 + count1;
    // add 1 to 0 in place
    int i;
    for (i = META_LEN; i < wpLen; i ++) {
        wp[i] = (count0 * 1.0 / count) * wp[i] + (count1 * 1.0 / count) * wp1[i];
    }
    wp[9] = count;

    PG_RETURN_ARRAYTYPE_P(wparray);
#else
    //--------------------------------------------------------------------
    // 1. get the FactorModel structure from the shared memory
    //    using mid (arg[0])
    //--------------------------------------------------------------------
    int32 mid = PG_GETARG_INT32(0);
    // model
    struct FactorModel* ptrSharedModel = (struct FactorModel*) get_model_by_mid(mid);

    //--------------------------------------------------------------------
    // 2. update step size
    //--------------------------------------------------------------------
	FactorModel_take_step(ptrSharedModel);
    
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
    double *w;
    int wLen; 
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
    assert(wpLen == ((int) wp[4]) + META_LEN);
    assert(((int) wp[5]) == ((int) wp[9]));
    //--------------------------------------------------------------------
    // 2. get rid of count when outputing
    //--------------------------------------------------------------------
	warray = my_construct_array(wpLen - META_LEN, sizeof(float8), FLOAT8OID);
	wLen = my_parse_array_no_copy((struct varlena *)warray, 
			sizeof(float8), (char **)&w);
	memcpy(w, wp + META_LEN, (wpLen - META_LEN) * sizeof(float8));
#else
    //--------------------------------------------------------------------
    // 1. get model from shared memory
    //--------------------------------------------------------------------
    int32 mid = PG_GETARG_INT32(0);
    // model
    struct FactorModel* ptrSharedModel = 
		(struct FactorModel*) get_model_by_mid(mid);

    //--------------------------------------------------------------------
    // 2. construct a PG array to return and delete the shared memory
    //--------------------------------------------------------------------
	wLen = ptrSharedModel->nDims;
	warray = my_construct_array(wLen, sizeof(float8), FLOAT8OID);
	wLen = my_parse_array_no_copy((struct varlena *)warray, 
			sizeof(float8), (char **)&w);
	memcpy(w, &(ptrSharedModel->L) + 1, wLen * sizeof(float8));
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
    // 1. init a local FactorModel structure 
    // and get the weight vector from temp state
    //--------------------------------------------------------------------
    ArrayType *wparray = (ArrayType *) PG_GETARG_RAW_VARLENA_P(0);
    double *wp;
    int wpLen = my_parse_array_no_copy((struct varlena*) wparray, 
            sizeof(float8), (char **) &wp);
    struct FactorModel modelBuffer;
    struct FactorModel *ptrModel;
    // local copy
    ptrModel = &modelBuffer;
    // get hard-coded hyper parameters
    FactorModel_init(ptrModel, (int) wp[0], (int) wp[1], (int) wp[2], (int) wp[3], (int) wp[5], 
            wp[6], wp[7], wp[8]);
    ptrModel->L = wp + META_LEN;
	ptrModel->R = ptrModel->L + ptrModel->nRows * ptrModel->maxRank;
    // count
    wp[9] ++;
    // elog(WARNING, "grad: count: %lf, nDims %d", ptrModel->w[ptrModel->nDims], ptrModel->nDims);
#else
    //--------------------------------------------------------------------
    // 1. get the FactorModel structure from the shared memory
    //    using mid (arg[0])
    //--------------------------------------------------------------------
    int32 mid = PG_GETARG_INT32(0);
    // model
    struct FactorModel modelBuffer;
    struct FactorModel* ptrModel = &modelBuffer;
    static struct FactorModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct FactorModel*)get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
	*ptrModel = (*ptrSharedModel);
	ptrModel->L = (double *)(&(ptrSharedModel->L) + 1);
	ptrModel->R = ptrModel->L + ptrModel->nRows * ptrModel->maxRank;
#endif

    //--------------------------------------------------------------------
    // 2. parse the args (row, col, rating)
    //--------------------------------------------------------------------
    int32 row = PG_GETARG_INT32(1);
    int32 col = PG_GETARG_INT32(2);
    float8 rating = PG_GETARG_FLOAT8(3);
	int i = row - 1;
	int j = col - 1;

    //--------------------------------------------------------------------
    // 3. computing loss
    //--------------------------------------------------------------------
    double err = FactorModel_loss(ptrModel, i, j, rating);

    PG_RETURN_FLOAT8(err*err);
}

/**
 * pred function
 */
Datum
pred(PG_FUNCTION_ARGS) {
#ifdef VAGG
    //--------------------------------------------------------------------
    // 1. init a local FactorModel structure 
    // and get the weight vector from temp state
    //--------------------------------------------------------------------
    ArrayType *wparray = (ArrayType *) PG_GETARG_RAW_VARLENA_P(0);
    double *wp;
    int wpLen = my_parse_array_no_copy((struct varlena*) wparray, 
            sizeof(float8), (char **) &wp);
    struct FactorModel modelBuffer;
    struct FactorModel *ptrModel;
    // local copy
    ptrModel = &modelBuffer;
    // get hard-coded hyper parameters
    FactorModel_init(ptrModel, (int) wp[0], (int) wp[1], (int) wp[2], (int) wp[3], (int) wp[5], 
            wp[6], wp[7], wp[8]);
    ptrModel->L = wp + META_LEN;
	ptrModel->R = ptrModel->L + ptrModel->nRows * ptrModel->maxRank;
    // count
    wp[9] ++;
    // elog(WARNING, "grad: count: %lf, nDims %d", ptrModel->w[ptrModel->nDims], ptrModel->nDims);
#else
    //--------------------------------------------------------------------
    // 1. get the FactorModel structure from the shared memory
    //    using mid (arg[0])
    //--------------------------------------------------------------------
    int32 mid = PG_GETARG_INT32(0);
    // model
    struct FactorModel modelBuffer;
    struct FactorModel* ptrModel = &modelBuffer;
    static struct FactorModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct FactorModel*)get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
	*ptrModel = (*ptrSharedModel);
	ptrModel->L = (double *)(&(ptrSharedModel->L) + 1);
	ptrModel->R = ptrModel->L + ptrModel->nRows * ptrModel->maxRank;
#endif

    //--------------------------------------------------------------------
    // 2. parse the args (row, col, rating)
    //--------------------------------------------------------------------
    int32 row = PG_GETARG_INT32(1);
    int32 col = PG_GETARG_INT32(2);
	int i = row - 1;
	int j = col - 1;

    //--------------------------------------------------------------------
    // 3. computing loss
    //--------------------------------------------------------------------
    double pred = FactorModel_loss(ptrModel, i, j, 0);

    PG_RETURN_FLOAT8(pred);
}

