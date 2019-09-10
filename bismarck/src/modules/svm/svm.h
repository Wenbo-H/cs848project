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

#ifndef SVM_H
#define SVM_H

void
sparse_svm_grad(struct LinearModel *ptrModel, int len, int *k, double *v, int y) {
    // read and prepare
    double wx = dot_dss(ptrModel->w, k, v, len);
    double c = ptrModel->stepsize * y;
    // writes
    if(1 - y * wx > 0) {
        add_and_scale_dss(ptrModel->w, k, v, len, c);
    }
    // regularization
    double u = ptrModel->mu * ptrModel->stepsize;
    l1_shrink_mask(ptrModel->w, u, k, len);
}

void
dense_svm_grad(struct LinearModel *ptrModel, double *v, int y) {
    // read and prepare
    double wx = dot(ptrModel->w, v, ptrModel->nDims);
    double c = ptrModel->stepsize * y;
    // writes
    if(1 - y * wx > 0) {
        add_and_scale(ptrModel->w, ptrModel->nDims, v, c);
    }
    // regularization
    double u = ptrModel->mu * ptrModel->stepsize;
    l1_shrink_mask_d(ptrModel->w, u, ptrModel->nDims);
}

double
sparse_svm_loss(struct LinearModel *ptrModel, int len, int *k, double *v, int y) {
    double wx = dot_dss(ptrModel->w, k, v, len);
    double loss = 1 - y * wx;
    return (loss > 0) ? loss : 0;
}

double
dense_svm_loss(struct LinearModel *ptrModel, double *v, int y) {
    double wx = dot(ptrModel->w, v, ptrModel->nDims);
    double loss = 1 - y * wx;
    return (loss > 0) ? loss : 0;
}

double
sparse_svm_pred(struct LinearModel *ptrModel, int len, int *k, double *v) {
    double wx = dot_dss(ptrModel->w, k, v, len);
    double loss = 1 - wx;
    return (loss > 0) ? 1 : -1;
}

double
dense_svm_pred(struct LinearModel *ptrModel, double *v) {
    double wx = dot(ptrModel->w, v, ptrModel->nDims);
    double loss = 1 - wx;
    return (loss > 0) ? 1 : -1;
}

#endif

