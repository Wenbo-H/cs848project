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

#ifndef crf_model_h
#define crf_model_h

#define META_LEN (11)

struct Example {
	int  len;
	int *labels;
	int *uObs;
    int *bObs;
};

/** a structure for model parameters and meta data */
struct CRFModel {
    int mid;
    int token;
	// meta data
	int nLabels;
	int nTuples;
	int nDims;
	int nULines;
	int nBLines;
	// regularization hyper-parameters
	double mu;
	// step size hyper-parameters
	int nSteps;
	double initStepSize;
	double stepsize;
	double decay;
	// data
	double wscale;
	double *w;
};

/**
 * assign initial values to the model,
 * should go in a constructor if written in C++
 */
inline void 
CRFModel_init(struct CRFModel *ptrModel, int mid,
		int nLabels, int nTuples, int nDims, int nULines, int nBLines,
		double mu, double step, double decay) {
    ptrModel->mid = mid;
    ptrModel->token = 0;
	// meta data
    ptrModel->nLabels = nLabels;
    ptrModel->nTuples = nTuples;
    ptrModel->nDims = nDims;
    ptrModel->nULines = nULines;
    ptrModel->nBLines = nBLines;
	// regularization hyper-parameters
    ptrModel->mu = mu;
	// step size hyper-parameters
    ptrModel->initStepSize = step;
    ptrModel->stepsize = step;
    ptrModel->decay = decay;
	ptrModel->nSteps = 0;
	// data
	ptrModel->wscale = 1;
	ptrModel->w = (double *)(&(ptrModel->w) + 1);
}

/**
 * step size diminishing and count number of steps
 */
inline void
CRFModel_take_step(struct CRFModel *ptrModel) {
	// step size
	ptrModel->stepsize *= ptrModel->decay;
}

/**
 * force scaling factor w to be 1
 */
inline void
CRFModel_scale(struct CRFModel *ptrModel) {
	scale_i(ptrModel->w, ptrModel->nDims, ptrModel->wscale);
	ptrModel->wscale = 1;
}

/**
 * regularization by scaling w
 */
inline void
CRFModel_regularize(struct CRFModel *ptrModel) {
	// regularization
	ptrModel->wscale *= 1 - ptrModel->mu * ptrModel->stepsize;
	if (ptrModel->wscale < 1e-9) {
		scale_i(ptrModel->w, ptrModel->nDims, ptrModel->wscale);
		ptrModel->wscale = 1;
	}
}

static void
CRFModel_compute_psi(	const struct CRFModel   *ptrModel,	// model
						const struct Example 	*ptrDoc,	// document
						double 					*vpsi) {	// output
	// fixed number of unigram and bigram observations of each position
	// bounds
	const int 	  U 			= ptrModel->nULines;
	const int 	  B 			= ptrModel->nBLines;
	const int 	  Y 			= ptrModel->nLabels;
	const int 	  T 			= ptrDoc->len;
	// arrays (in)
	const double *w 			= ptrModel->w;
	const double wscale			= ptrModel->wscale;
	const int 	(*uObs)[T][U] 	= (void *)ptrDoc->uObs;
	const int 	(*bObs)[T][B] 	= (void *)ptrDoc->bObs;
	// array (out)
	double 		(*psi)[T][Y][Y] = (void *)vpsi;
	// unigram features
	int t, y, yp, n, d;
	for (t = 0; t < T; t++) {
		for (y = 0; y < Y; y++) {
			double sum = 0.0;
			for (n = 0; n < U; n++) {
				const int o = (*uObs)[t][n];
				sum += w[o + y];
			}
			for (yp = 0; yp < Y; yp++)
				(*psi)[t][yp][y] = sum;
		}
	}
	// bigram features, starting from 1 instead of 0
	for (t = 1; t < T; t++) {
		for (yp = 0, d = 0; yp < Y; yp++) {
			for (y = 0; y < Y; y++, d++) {
				double sum = 0.0;
				for (n = 0; n < B; n++) {
					const int o = (*bObs)[t][n];
					sum += w[o + d];
				}
				(*psi)[t][yp][y] += sum;
			}
		}
	}
	// scaling
	for (t = 0; t < T; t++) {
		for (yp = 0; yp < Y; yp++) {
			for (y = 0; y < Y; y++) {
				(*psi)[t][yp][y] *= wscale;
			}
		}
	}
//	// map to exponential and done
//	for (t = 0; t < T; t++) {
//		for (yp = 0; yp < Y; yp++) {
//			for (y = 0; y < Y; y++) {
//				(*psi)[t][yp][y] = exp((*psi)[t][yp][y]);
//			}
//		}
//	}
}

static double
CRFModel_fwd_bwd(	const struct CRFModel   *ptrModel,	// model
					const struct Example 	*ptrDoc,	// document
					const double			*vpsi,		// transition scores, trellis
					double 					*valpha,	// output 1
					double 					*vbeta) {	// output 2
	// bounds
	const int 	  Y 			= ptrModel->nLabels;
	const int 	  T 			= ptrDoc->len;
	// arrays (in)
	const double(*psi)[T][Y][Y] = (void *)vpsi;
	// array and z(out)
	double 		(*alpha)[T][Y] 	= (void *)valpha;
	double 		(*beta )[T][Y] 	= (void *)vbeta;
	double  	  z				= 0.0;
	// iters
	int y, yp, t;
	// forward
	for (y = 0; y < Y; y++) {
		(*alpha)[0][y] = (*psi)[0][0][y];
	}
	for (t = 1; t < T; t++) {
		for (y = 0; y < Y; y++) {
			double sum = (*alpha)[t - 1][0] + (*psi)[t][0][y];
			for (yp = 1; yp < Y; yp++)
				sum = log_sum((*alpha)[t - 1][yp] + (*psi)[t][yp][y], sum);
			(*alpha)[t][y] = sum;
		}
	}
	// backward
	for (yp = 0; yp < Y; yp++) {
		(*beta)[T - 1][yp] = 0;
	}
	for (t = T - 1; t > 0; t--) {
		for (yp = 0; yp < Y; yp++) {
			double sum = (*beta)[t][0] + (*psi)[t][yp][0];
			for (y = 1; y < Y; y++)
				sum = log_sum((*beta)[t][y] + (*psi)[t][yp][y], sum);
			(*beta)[t - 1][yp] = sum;
		}
	}
	// log(z)
	z += (*alpha)[T - 1][0];
	for (y = 1; y < Y; y++)
		z = log_sum(z, (*alpha)[T - 1][y]);
	// assert consistency between alpha and beta
//	double checkz = (*beta)[0][0];
//	 for (y = 1; y < Y; y++)
//	 	checkz = log_sum(checkz, (*beta)[0][y]);
//	//assert(z == checkz);
//	if (z != checkz) {
//		elog(WARNING, "z: %lf, checkz: %lf", z, checkz);
//	}
	return z;
}

static double
CRFModel_log_score(	const struct CRFModel   *ptrModel,	// model
					const struct Example 	*ptrDoc,	// document
					const double			*vpsi) {	// transition scores, trellis
	// fixed number of unigram and bigram observations of each position
	// bounds
	const int 	  Y 			= ptrModel->nLabels;
	const int 	  T 			= ptrDoc->len;
	// arrays (in)
	const int 	 *labels		= (void *)ptrDoc->labels;
	const double(*psi)[T][Y][Y] = (void *)vpsi;
	// score (out)
	double 		  score			= 0.0;
	// iter
	int t;
	// sum up transitions
	score = (*psi)[0][0][labels[0]];
	for (t = 1; t < T; t++) {
		score += (*psi)[t][labels[t-1]][labels[t]];
	}
	return score;
}

static void
CRFModel_do_grad(	struct CRFModel   		*ptrModel,	// model
					const struct Example 	*ptrDoc,	// document
					const double			*vpsi,		// transition scores, trellis
					const double			*valpha,	// forward scores
					const double			*vbeta,		// backward scores
					const double			 z) {		// normalization factor
	// fixed number of unigram and bigram observations of each position
	// bounds
	const int 	  U 			= ptrModel->nULines;
	const int 	  B 			= ptrModel->nBLines;
	const int 	  Y 			= ptrModel->nLabels;
	const int 	  T 			= ptrDoc->len;
	// arrays and parameters (in)
	const int 	 *labels		= (void *)ptrDoc->labels;
	const int 	(*uObs)[T][U] 	= (void *)ptrDoc->uObs;
	const int 	(*bObs)[T][B] 	= (void *)ptrDoc->bObs;
	const double(*psi)[T][Y][Y] = (void *)vpsi;
	const double(*alpha)[T][Y] 	= (void *)valpha;
	const double(*beta )[T][Y] 	= (void *)vbeta;
	const double  stepsize		= ptrModel->stepsize;
	const double  gain	 		= 
	   	1.0 / ptrModel->wscale / (1 - ptrModel->mu * ptrModel->stepsize);
	// array (out)
	double *w 					= ptrModel->w;
	// iters
	int t, yp, y, n, d;

	// -------------------------------------------------------------------
	// update by the expectation part
	// -------------------------------------------------------------------
	// unigram
	for (t = 0; t < T; t ++) {
		for (y = 0; y < Y; y ++) {
			double e = exp((*alpha)[t][y] + (*beta)[t][y] - z);
			for (n = 0; n < U; n++) {
				int o = (*uObs)[t][n];
				w[o + y] -= stepsize * e * gain;
			}
		}
	}
	// bigram
	for (t = 1; t < T; t++) {
		for (yp = 0, d = 0; yp < Y; yp++) {
			for (y = 0; y < Y; y++, d++) {
				// expectation is equal to probability,
				// because only one is nonzero
				double e = exp((*alpha)[t - 1][yp] + (*beta)[t][y] + (*psi)[t][yp][y] - z);
				for (n = 0; n < B; n++) {
					int o = (*bObs)[t][n];
					w[o + d] -= stepsize * e * gain;
				}
			}
		}
	}

	// -------------------------------------------------------------------
	// update by the observation part
	// -------------------------------------------------------------------
	// unigram
	for (t = 0; t < T; t ++) {
		for (n = 0; n < U; n++) {
			int o = (*uObs)[t][n];
			w[o + labels[t]] += stepsize * gain;
		}
	}
	// bigram
	for (t = 1; t < T; t ++) {
		d = labels[t-1] * Y + labels[t];
		for (n = 0; n < B; n++) {
			int o = (*bObs)[t][n];
			w[o + d] += stepsize * gain;
		}
	}
}

/**
 * everything is in the log domain, but no log_sum is needed,
 * as in log_score
 */
static void
CRFModel_viterbi(	const struct CRFModel   *ptrModel,	// model
					const struct Example 	*ptrDoc,	// document
					const double			*vpsi,		// transition scores, trellis
					int						*labels) {	// output best label path
	// bounds
	const int 	  Y 				  = ptrModel->nLabels;
	const int 	  T 				  = ptrDoc->len;
	// arrays (in)
	const double(*psi)[T][Y][Y] 	  = (void *)vpsi;
	// arrays (interm)
	int 		 *vbackpointers 	  = malloc(sizeof(int) * T * Y);
	int 		(*backpointers)[T][Y] = (void *)vbackpointers;
	double 		 *pathscores 		  = malloc(sizeof(double) * Y);
	double 		 *prepathscores		  = malloc(sizeof(double) * Y);
	double		 *swap;
	// final best score
	double 		  finalscore;
	// iters
	int y, yp, t;
	// forward
	for (y = 0; y < Y; y++) {
		prepathscores[y] = (*psi)[0][0][y];
		(*backpointers)[0][y] = -1;
	}
	for (t = 1; t < T; t++) {
		for (y = 0; y < Y; y ++) {
			double maxscore = prepathscores[0] + (*psi)[t][0][y];
			(*backpointers)[t][y] = 0;
			for (yp = 1; yp < Y; yp ++) {
				double tmpscore = prepathscores[yp] + (*psi)[t][yp][y];
				if (maxscore < tmpscore) {
					maxscore = tmpscore;
					(*backpointers)[t][y] = yp;
				}
			}
			pathscores[y] = maxscore;
		}
		// swap
		swap = prepathscores;
		prepathscores = pathscores;
		pathscores = swap;
	}
	// backward
	finalscore = prepathscores[0];
	labels[T - 1] = 0;
	for (y = 1; y < Y; y++) {
		if (finalscore < prepathscores[y]) {
			finalscore = prepathscores[y];
			labels[T - 1] = y;
		}
	}
	for (t = T - 2; t >= 0; t --) {
		labels[t] = (*backpointers)[t + 1][labels[t + 1]];
	}
	// free
	free(backpointers);
	free(pathscores);
	free(prepathscores);
}

inline void
CRFModel_grad(	struct CRFModel   		*ptrModel,	// model
				const struct Example 	*ptrDoc) {	// document
	int T = ptrDoc->len;
	int Y = ptrModel->nLabels;
	// alloc shared space
	double *psi, *alpha, *beta;
	psi = malloc(sizeof(double) * T * Y * Y);
	alpha = malloc(sizeof(double) * T * Y);
	beta = malloc(sizeof(double) * T * Y);
	// some dynamic programming
	CRFModel_compute_psi(ptrModel, ptrDoc, psi);
	double z = CRFModel_fwd_bwd(ptrModel, ptrDoc, psi, alpha, beta);
	CRFModel_do_grad(ptrModel, ptrDoc, psi, alpha, beta, z);
	// free
	free(psi);
	free(alpha);
	free(beta);
}

inline double
CRFModel_loss(	const struct CRFModel	*ptrModel,	// model
				const struct Example 	*ptrDoc) {	// document
	const int T = ptrDoc->len;
	const int Y = ptrModel->nLabels;
	// alloc shared space
	double *psi = malloc(sizeof(double) * T * Y * Y);
	double *alpha = malloc(sizeof(double) * T * Y);
	double *beta = malloc(sizeof(double) * T * Y);
	// some dynamic programming
	CRFModel_compute_psi(ptrModel, ptrDoc, psi);
	double z = CRFModel_fwd_bwd(ptrModel, ptrDoc, psi, alpha, beta);
	double logscore = CRFModel_log_score(ptrModel, ptrDoc, psi);
	double L = logscore - z;
	// free
	free(psi);
	free(alpha);
	free(beta);
	return -L;
}

inline void
CRFModel_pred(	const struct CRFModel	*ptrModel,	// model
				const struct Example 	*ptrDoc,
				int						*labels) {	// document
	const int T = ptrDoc->len;
	const int Y = ptrModel->nLabels;
	// alloc shared space
	double *psi = malloc(sizeof(double) * T * Y * Y);
	// some dynamic programming
	CRFModel_compute_psi(ptrModel, ptrDoc, psi);
	CRFModel_viterbi(ptrModel, ptrDoc, psi, labels);
	// free
	free(psi);
}

#endif

