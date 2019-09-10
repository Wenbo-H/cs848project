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

#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

#define META_LEN (9)

/** a structure for model parameters and meta data */
struct LinearModel {
    int mid;
    int token;
	// meta data
	int nDims;
	int nTuples;
	// regularization hyper-parameters
	double mu;
	// step size hyper-parameters
	int nSteps;
	double initStepSize;
	double stepsize;
	double decay;
	// weight vector
	double *w;
	double *temp_v;  
};

/**
 * assign initial values to the model,
 * should go in a constructor if written in C++
 */
inline void 
LinearModel_init(struct LinearModel *ptrModel, int mid, int nDims, int nTuples, 
		double mu, double stepsize, double decay) {
    ptrModel->mid = mid;
    ptrModel->token = 0;

	// meta data
    ptrModel->nDims = nDims;
    ptrModel->nTuples = nTuples;
	
	// regularization hyper-parameters
    ptrModel->mu = mu;
	
	// step size hyper-parameters
	ptrModel->nSteps = 0;
    ptrModel->initStepSize = stepsize;
    ptrModel->stepsize = stepsize;
    ptrModel->decay = decay;

	// weight vector
	ptrModel->w = (double *)(&(ptrModel->w) + 1);  // ?????
	// ? temp_v value
	ptrModel->temp_v = (double *)(&(ptrModel->temp_v) + 1);
}

/**
 * take one step
 * should go in a constructor if written in C++
 */
inline void
LinearModel_take_step(struct LinearModel *ptrModel) {
	ptrModel->stepsize *= ptrModel->decay;
}

#endif
