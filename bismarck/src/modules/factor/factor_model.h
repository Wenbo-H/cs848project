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

#ifndef Factor_MODEL_H
#define Factor_MODEL_H

#define META_LEN (10)

/** a structure for model parameters and meta data */
struct FactorModel {
    int mid;
    int token;
	// meta data
	int nRows;
	int nCols;
	int maxRank;
	int nDims;
	int nTuples;
	// regularization hyper-parameters
	double B;
	double B2;
	// step size hyper-parameters
	int nSteps;
	double initStepSize;
	double stepsize;
	double decay;
	// weight vectors
	double *R;
	double *L;
};

/**
 * assign initial values to the model,
 * should go in a constructor if written in C++
 */
inline void 
FactorModel_init(struct FactorModel *ptrModel, int mid, int nRows, int nCols, int r, int nTuples, 
		double B, double step, double decay) {
    ptrModel->mid = mid;
    ptrModel->token = 0;

	// meta data
    ptrModel->maxRank = r;
    ptrModel->nRows = nRows;
    ptrModel->nCols = nCols;
    ptrModel->nDims = (nRows + nCols) * r;
	ptrModel->nTuples = nTuples;
	
	// regularization hyper-parameters
    ptrModel->B = B;
    ptrModel->B2 = B * B;
	
	// step size hyper-parameters
    ptrModel->initStepSize = step;
    ptrModel->stepsize = step;
	ptrModel->nSteps = 0;
	ptrModel->decay = decay;
	
	// weight vectors
	ptrModel->L = (double *)(&(ptrModel->L) + 1);
    ptrModel->R = ptrModel->L + nRows * r;	//R is serialized immed after L
	/*
    int nElems = (nCols + nRows) * r;	//sum of sizes of L and R
	int x, i, j;
	for (x = 0; x < nElems; x ++) {		//both L and R initialized simult
		ptrModel->L[x] = w * gaussrand();
	}
	for (i = 0; i < nRows; i++) {
		double *Li = ptrModel->L + r * i;
		ball_project(Li, r, B, ptrModel->B2);	//L is in row-major order
	}
	for (j = 0; j < nCols; j++) {
		double *Rj = ptrModel->R + r * j;
		ball_project(Rj, r, B, ptrModel->B2);	//R is in col-major order
	}
	*/
}

/**
 * take step
 */
inline void
FactorModel_take_step(struct FactorModel *ptrModel) {
	ptrModel->stepsize *= ptrModel->decay;
}

inline void
FactorModel_grad(struct FactorModel *ptrModel, const int i, const int j, const double rating) {
	//L is row-major, so get ith row directly; R is col-major, so get jth col directly
	//L is nrowsx20, R is 20xncols
	double *Li = ptrModel->L + i * ptrModel->maxRank;	//offset the dbl ptr of L
	double *Rj = ptrModel->L + ptrModel->nRows * ptrModel->maxRank + j * ptrModel->maxRank;
	double err = dot(Li, Rj, ptrModel->maxRank) - rating;
	double e = -(ptrModel->stepsize * err);
	double *tempLi = malloc(sizeof(double) * ptrModel->maxRank);
	//no need for set_L etc., since we update model in place!
	memcpy(tempLi, Li, ptrModel->maxRank * sizeof(double));
	add_and_scale(tempLi, ptrModel->maxRank, Rj, e);
	add_and_scale(Rj, ptrModel->maxRank, Li, e);
	memcpy(Li, tempLi, ptrModel->maxRank * sizeof(double));
	free(tempLi);
	// regularization
	ball_project(Li, ptrModel->maxRank, ptrModel->B, ptrModel->B2);
	ball_project(Rj, ptrModel->maxRank, ptrModel->B, ptrModel->B2);
}

inline double
FactorModel_loss(struct FactorModel *ptrModel, const int i, const int j, const double rating) {
	double *Li = ptrModel->L + i * ptrModel->maxRank;
	double *Rj = ptrModel->L + ptrModel->nRows * ptrModel->maxRank + j * ptrModel->maxRank;
	return dot(Li, Rj, ptrModel->maxRank) - rating;
}

#endif
