
#ifndef LINEAR_REG_H
#define LINEAR_REG_H

inline void
linear_reg_grad(struct LinearModel *ptrModel, const double *v, const int y) {
    // read and prepare
    double wx = dot(ptrModel->w, v, ptrModel->nDims);
    double c = ptrModel->stepsize * (wx - y) * 2; // scale factor
    add_and_scale(ptrModel->w, ptrModel->nDims, v, c);
    // regularization
    double u = ptrModel->mu * ptrModel->stepsize;
    l1_shrink_mask_d(ptrModel->w, u, ptrModel->nDims);
}


inline double
linear_reg_loss(struct LinearModel *ptrModel, const double *v, const int y) {
    double wx = dot(ptrModel->w, v, ptrModel->nDims);
    double los = (wx - y) * (wx - y);
    return los;
}

inline double
linear_reg_pred(struct LinearModel *ptrModel, const double *v) {
    double wx = dot(ptrModel->w, v, ptrModel->nDims);
    return wx;
}
#endif
