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

#ifndef LOGIT_H
#define LOGIT_H

inline void
sparse_logit_grad(struct LinearModel *ptrModel, const int len, const int *k, const double *v, const int y) {
    // grad
    double wx = dot_dss(ptrModel->w, k, v, len);
    scale_dot_dss(ptrModel->temp_v, k, 0.9, len);
    double sig = sigma(-wx * y);
    double *temp;
    memcpy(temp, v, ptrModel->nDims * sizeof(float8)); 
    scale_dot_dss(temp, k, y*sig, len);
    scale_dot_dss(temp, k, 0.1, len);
    add_vector_dss(ptrModel->temp_v, k, temp, len);
    add_and_scale_dss(ptrModel->w, k, ptrModel->temp_v, len, -1*ptrModel->stepsize);

    //double c = ptrModel->stepsize * y * sig; // scale factor
    //add_and_scale_dss(ptrModel->w, k, v, len, c);
    
    // regularization
    double u = ptrModel->mu * ptrModel->stepsize;
    l1_shrink_mask(ptrModel->w, u, k, len);
}

inline void
dense_logit_grad(struct LinearModel *ptrModel, const double *v, const int y) {
    // read and prepare
    double wx = dot(ptrModel->w, v, ptrModel->nDims);
    scale_dot(ptrModel->temp_v, 0.9, ptrModel->nDims); // beta * v_dw
    double sig = sigma(-wx * y);
    //double c = ptrModel->stepsize * y * sig; // scale factor
    //add_and_scale(ptrModel->w, ptrModel->nDims, v, c);
    double temp[ptrModel->nDims];
    memcpy(&temp, v, ptrModel->nDims * sizeof(float8));   // v is feature value, not v_dw
    scale_dot(temp, y*sig, ptrModel->nDims); 
    scale_dot(temp, -0.1, ptrModel->nDims);
    add_vectors(ptrModel->temp_v, temp, ptrModel->nDims);
    add_and_scale(ptrModel->w, ptrModel->nDims, ptrModel->temp_v, -1*ptrModel->stepsize);
    // regularization
    double u = ptrModel->mu * ptrModel->stepsize;
    l1_shrink_mask_d(ptrModel->w, u, ptrModel->nDims);
}

inline double
sparse_logit_loss(struct LinearModel *ptrModel, const int len, const int *k, const double *v, const int y) {
    double wx = dot_dss(ptrModel->w, k, v, len);
    return log(1 + exp(-y * wx));
}

inline double
dense_logit_loss(struct LinearModel *ptrModel, const double *v, const int y) {
    double wx = dot(ptrModel->w, v, ptrModel->nDims);
    return log(1 + exp(-y * wx));
}

inline double
sparse_logit_pred(struct LinearModel *ptrModel, const int len, const int *k, const double *v) {
    double wx = dot_dss(ptrModel->w, k, v, len);
    return 1. / (1. + exp(-1 * wx));
}

inline double
dense_logit_pred(struct LinearModel *ptrModel, const double *v) {
    double wx = dot(ptrModel->w, v, ptrModel->nDims);
    return 1. / (1. + exp(-1 * wx));
}

#endif
