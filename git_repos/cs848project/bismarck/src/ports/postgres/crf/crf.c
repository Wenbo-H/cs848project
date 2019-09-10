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
#include "modules/crf/crf_model.h"

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
    // 0. parse crf_model row type into local variables
    // -------------------------------------------------------------------
    HeapTupleHeader modelTuple = PG_GETARG_HEAPTUPLEHEADER(0);
    bool isnull;
    // meta data
    int mid = DatumGetInt32(GetAttributeByNum(modelTuple, 1, &isnull));
    int nlabels = DatumGetInt32(GetAttributeByNum(modelTuple, 2, &isnull));
    int ntuples = DatumGetInt32(GetAttributeByNum(modelTuple, 3, &isnull));
    int ndims = DatumGetInt32(GetAttributeByNum(modelTuple, 4, &isnull));
    int nulines = DatumGetInt32(GetAttributeByNum(modelTuple, 5, &isnull));
    int nblines = DatumGetInt32(GetAttributeByNum(modelTuple, 6, &isnull));
    double mu = DatumGetFloat8(GetAttributeByNum(modelTuple, 7, &isnull));
    double stepsize = DatumGetFloat8(GetAttributeByNum(modelTuple, 8, &isnull));
    double decay = DatumGetFloat8(GetAttributeByNum(modelTuple, 9, &isnull));
    // weight vector
    ArrayType *warray = (ArrayType *) GetAttributeByNum(modelTuple, 10, &isnull);
    double *w;
    int wLen = my_parse_array_no_copy((struct varlena*) warray, 
            sizeof(float8), (char **) &w);
    // dimension sanity check
    assert(wLen == ndims);

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
    wp[1] = nlabels;
    wp[2] = ntuples;
    wp[3] = ndims;
    wp[4] = nulines;
    wp[5] = nblines;
    wp[6] = mu;
    wp[7] = stepsize;
    wp[8] = decay;
    wp[9] = 1.0; // wscale
    wp[10] = 0; // count of tuple seen

    // -------------------------------------------------------------------
    // 3. copy weight vector into w+
    // -------------------------------------------------------------------
    memcpy(wp + META_LEN, w, sizeof(double) * wLen);

    // return
    PG_RETURN_ARRAYTYPE_P(wparray);
#else
    //--------------------------------------------------------------------
    // 1. create a shared memory region for the CRFModel structure
    //    using mid as key
    //--------------------------------------------------------------------
    struct CRFModel* ptrModel;
    int size = sizeof(struct CRFModel) + sizeof(double) * (ndims) * 2;
    // open the shared memory
    int shmid = shmget(ftok("/", mid), size, SHM_R | SHM_W | IPC_CREAT);
    if (shmid == -1) { elog(ERROR, "In init, shmget failed!\n"); }
    // elog(WARNING, "init: after shmget\n");
    // attach the memory region
    ptrModel = (struct CRFModel*) shmat(shmid, NULL, 0);
    // elog(WARNING, "init: after shmat, model: %x\n", ptrModel);

    //--------------------------------------------------------------------
    // 2. init meta data
    //--------------------------------------------------------------------
    // constructor
    CRFModel_init(ptrModel, mid, nlabels, ntuples, ndims, nulines, nblines,
            mu, stepsize, decay);

    // -------------------------------------------------------------------
    // 3. copy weight vector into shared memory
    // -------------------------------------------------------------------
    memcpy(ptrModel->w, w, sizeof(double) * wLen);

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
    // 1. init a local CRFModel structure 
    // and get the weight vector from temp state
    //--------------------------------------------------------------------
    ArrayType *wparray = (ArrayType *) PG_GETARG_RAW_VARLENA_P(0);
    double *wp;
    int wpLen = my_parse_array_no_copy((struct varlena*) wparray, 
            sizeof(float8), (char **) &wp);
    struct CRFModel modelBuffer;
    struct CRFModel *ptrModel;
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
		assert(wp[9] == 1.0);
		assert(wp[10] == 0);
    }
    // local copy
    ptrModel = &modelBuffer;
    // get hyper parameters
    CRFModel_init(ptrModel, (int) wp[0], (int) wp[1], (int) wp[2], (int) wp[3], 
            (int) wp[4], (int) wp[5], wp[6], wp[7], wp[8]);
    ptrModel->w = wp + META_LEN;
    // count
    wp[10] ++;
    // elog(WARNING, "grad: count: %lf, nDims %d", ptrModel->w[ptrModel->nDims], ptrModel->nDims);
#else
    //--------------------------------------------------------------------
    // 1. get the CRFModel structure from the shared memory
    //    using mid (arg[0])
    //--------------------------------------------------------------------
    int32 mid = PG_GETARG_INT32(0);
    // model
    struct CRFModel modelBuffer;
    struct CRFModel* ptrModel = &modelBuffer;
    static struct CRFModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct CRFModel*)get_model_by_mid(mid);
        // elog(WARNING, "grad: NO");
    }
    modelBuffer = (*ptrSharedModel);
    modelBuffer.w = (double *)(&(ptrSharedModel->w) + 1);
#endif

    //--------------------------------------------------------------------
    // 2. parse the args (uObs, bObs, labels) and construct example document
    //--------------------------------------------------------------------
    // uObs int[]
    int32* uObs;
    struct varlena* v1 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(1);
    my_parse_array_no_copy(v1, sizeof(int32), (char **)&uObs);
    // bObs int[]
    int32* bObs;
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(2);
    my_parse_array_no_copy(v2, sizeof(int32), (char **)&bObs);
    // labels int[]
    int32* labels;
    struct varlena* v3 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(3);
    int len = my_parse_array_no_copy(v3, sizeof(int32), (char **)&labels);
    struct Example d = {len, labels, uObs, bObs};

    //--------------------------------------------------------------------
    // 3. performing the gradient 
    //--------------------------------------------------------------------
#if !defined(VAGG) && defined(VLOCK)
    while (compare_and_swap(&(ptrSharedModel->token), 0, 1) == 0) {}
#endif

    CRFModel_grad(ptrModel, &d);

    // regularization
    CRFModel_regularize(ptrModel);
#ifdef VAGG
    wp[9] = ptrModel->wscale;
#else
	ptrSharedModel->wscale = ptrModel->wscale;
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
    int wpLen = my_parse_array_no_copy((struct varlena*)wparray, 
            sizeof(float8), (char **)&wp);
    // elog(WARNING, "inside pre 1");
    // get state 1
    ArrayType *wparray1 = (ArrayType *) PG_GETARG_RAW_VARLENA_P(1);
    double *wp1;
    int wpLen1 = my_parse_array_no_copy((struct varlena*)wparray1, 
            sizeof(float8), (char **)&wp1);
    if (wpLen == 1) {
        PG_RETURN_ARRAYTYPE_P(wparray1);
    }
    if (wpLen1 == 1) {
        PG_RETURN_ARRAYTYPE_P(wparray);
    }
    // the count
    int count0 = wp[10];
    // elog(WARNING, "inside pre 3");
    int count1 = wp1[10];
    // elog(WARNING, "inside pre 4");
    // elog(WARNING, "count0: %d, count1: %d", count0, count1);
    int count = count0 + count1;
    // add 1 to 0 in place
    int i;
    for (i = META_LEN; i < wpLen; i ++) {
        wp[i] = (count0 * 1.0 / count) * wp[i] * wp[9] 
			+ (count1 * 1.0 / count) * wp1[i] * wp1[9];
    }
	wp[9] = 1.0; // reset wscale
    wp[10] = count;

    PG_RETURN_ARRAYTYPE_P(wparray);
#else
    //--------------------------------------------------------------------
    // 1. get the CRFModel structure from the shared memory
    //    using mid (arg[0])
    //--------------------------------------------------------------------
    int32 mid = PG_GETARG_INT32(0);
    // model
    struct CRFModel* ptrSharedModel = (struct CRFModel*) get_model_by_mid(mid);

    //--------------------------------------------------------------------
    // 2. update step size
    //--------------------------------------------------------------------
	CRFModel_take_step(ptrSharedModel);
    
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
    // count checking
    assert(wpLen == wp[3] + META_LEN);
    assert(((int) wp[2]) == ((int) wp[10]));
	// wscale
	if (wp[9] != 1.0) {
		scale_i(wp + META_LEN, wp[3], wp[9]);
		wp[9] = 1.0;
	}
	// step size
	wp[7] *= wp[8];

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
    struct CRFModel* ptrSharedModel = (struct CRFModel*) get_model_by_mid(mid);

    //--------------------------------------------------------------------
    // 2. construct a PG array to return and delete the shared memory
    //--------------------------------------------------------------------
	wLen = ptrSharedModel->nDims;
	warray = my_construct_array(wLen, sizeof(float8), FLOAT8OID);
	wLen = my_parse_array_no_copy((struct varlena *)warray, 
			sizeof(float8), (char **)&w);
	memcpy(w, &(ptrSharedModel->w) + 1, wLen * sizeof(float8));
    // delete the shared memory
    int shmid = shmget(ftok("/", mid), 0, SHM_R | SHM_W);
    if (shmid == -1) { elog(ERROR, "In final, shmget failed!\n"); }
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
    // 1. init a local CRFModel structure 
    // and get the weight vector from temp state
    //--------------------------------------------------------------------
    ArrayType *wparray = (ArrayType *) PG_GETARG_RAW_VARLENA_P(0);
    double *wp;
    int wpLen = my_parse_array_no_copy((struct varlena*)wparray, 
            sizeof(float8), (char **)&wp);
    struct CRFModel modelBuffer;
    struct CRFModel *ptrModel;
    // local copy
    ptrModel = &modelBuffer;
    // get hyper parameters
    CRFModel_init(ptrModel, (int) wp[0], (int) wp[1], (int) wp[2], (int) wp[3], 
            (int) wp[4], (int) wp[5], wp[6], wp[7], wp[8]);
    ptrModel->w = wp + META_LEN;
    // count
    wp[10] ++;
    // elog(WARNING, "grad: count: %lf, nDims %d", ptrModel->w[ptrModel->nDims], ptrModel->nDims);
#else
    //--------------------------------------------------------------------
    // 1. get the CRFModel structure from the shared memory
    //    using mid (arg[0])
    //--------------------------------------------------------------------
    int32 mid = PG_GETARG_INT32(0);
    // model
    struct CRFModel modelBuffer;
    struct CRFModel* ptrModel = &modelBuffer;
    static struct CRFModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct CRFModel*)get_model_by_mid(mid);
        // elog(WARNING, "loss: NO");
    }
    modelBuffer = (*ptrSharedModel);
    modelBuffer.w = (double *)(&(ptrSharedModel->w) + 1);
#endif

    //--------------------------------------------------------------------
    // 2. parse the args (uObs, bObs, labels) and construct example document
    //--------------------------------------------------------------------
    // uObs int[]
    int32* uObs;
    struct varlena* v1 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(1);
    my_parse_array_no_copy(v1, sizeof(int32), (char **)&uObs);
    // bObs int[]
    int32* bObs;
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(2);
    my_parse_array_no_copy(v2, sizeof(int32), (char **)&bObs);
    // labels int[]
    int32* labels;
    struct varlena* v3 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(3);
    int len = my_parse_array_no_copy(v3, sizeof(int32), (char **)&labels);
    struct Example d = {len, labels, uObs, bObs};

    //--------------------------------------------------------------------
    // 3. loss computation
    //--------------------------------------------------------------------
    double loss = CRFModel_loss(ptrModel, &d);

    PG_RETURN_FLOAT8(loss);
}

/**
 * inference function
 */
Datum
pred(PG_FUNCTION_ARGS) {
#ifdef VAGG
    //--------------------------------------------------------------------
    // 1. init a local CRFModel structure 
    // and get the weight vector from temp state
    //--------------------------------------------------------------------
    ArrayType *wparray = (ArrayType *) PG_GETARG_RAW_VARLENA_P(0);
    double *wp;
    int wpLen = my_parse_array_no_copy((struct varlena*)wparray, 
            sizeof(float8), (char **)&wp);
    struct CRFModel modelBuffer;
    struct CRFModel *ptrModel;
    // local copy
    ptrModel = &modelBuffer;
    // get hyper parameters
    CRFModel_init(ptrModel, (int) wp[0], (int) wp[1], (int) wp[2], (int) wp[3], 
            (int) wp[4], (int) wp[5], wp[6], wp[7], wp[8]);
    ptrModel->w = wp + META_LEN;
    // count
    wp[10] ++;
    // elog(WARNING, "grad: count: %lf, nDims %d", ptrModel->w[ptrModel->nDims], ptrModel->nDims);
#else
    //--------------------------------------------------------------------
    // 1. get the CRFModel structure from the shared memory
    //    using mid (arg[0])
    //--------------------------------------------------------------------
    int32 mid = PG_GETARG_INT32(0);
    // model
    struct CRFModel modelBuffer;
    struct CRFModel* ptrModel = &modelBuffer;
    static struct CRFModel* ptrSharedModel = NULL;
    if (ptrSharedModel == NULL || ptrSharedModel->mid != mid) {
        ptrSharedModel = (struct CRFModel*)get_model_by_mid(mid);
        // elog(WARNING, "pred: NO");
    }
    modelBuffer = (*ptrSharedModel);
    modelBuffer.w = (double *)(&(ptrSharedModel->w) + 1);
#endif

    //--------------------------------------------------------------------
    // 2. parse the args (uObs, bObs, labels) and construct example document
    //--------------------------------------------------------------------
    // uObs int[]
    int32* uObs;
    struct varlena* v1 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(1);
    int uObsLen = my_parse_array_no_copy(v1, sizeof(int32), (char **)&uObs);
    // bObs int[]
    int32* bObs;
    struct varlena* v2 = (struct varlena*) PG_GETARG_RAW_VARLENA_P(2);
    int bObsLen = my_parse_array_no_copy(v2, sizeof(int32), (char **)&bObs);
	// compute len using obs because we don't have labels as input
	assert(uObsLen / ptrModel->nULines == bObsLen / ptrModel->nBLines);
    int len = bObsLen / ptrModel->nBLines;
    struct Example d = {len, NULL, uObs, bObs};

    //--------------------------------------------------------------------
    // 3. prediction
    //--------------------------------------------------------------------
    int *labels = palloc(sizeof(int) * len);
	CRFModel_pred(ptrModel, &d, labels);
	ArrayType *retarray = my_construct_array(len, sizeof(int32), INT4OID);
	int *ret;
	my_parse_array_no_copy((struct varlena *) retarray, 
			sizeof(int32), (char **) &ret);
	memcpy(ret, labels, len * sizeof(int));

    PG_RETURN_ARRAYTYPE_P(retarray);
}

