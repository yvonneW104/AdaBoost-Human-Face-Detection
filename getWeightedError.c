#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mex.h"
#include "mat.h"
#include "matrix.h"

static const int N_pos = 11838;
static const int N_neg = 29669;
static const int N = 41507;
static const int feature_num = 9168;
static const int threshold_step = 200;


double find_min (double* features) {
    int i;
    double min_value = 100;
    for (i = 0; i < N; i=i+1) {
        min_value = (features[i] < min_value) ? features[i] : min_value;  
    }
    return min_value;
}

double find_max (double* features) {
    int i;
    double max_value = -100;
    for (i = 0; i < N; i=i+1) {
        max_value = (max_value < features[i]) ? features[i] : max_value;
    }
    return max_value;
}

void getWeightedError(double* features, double* y, double* weight, double* weighted_err, double* threshold, double* polarity){
    double FS, BG;
    double min_temp[N];
    double a, b;
    int init_index[N];

    double max_value = find_max(features);
    double min_value = find_min(features);

    double step = (max_value - min_value) / threshold_step;
    step = (step < 0) ? -1 * step : step;

    int i, j;
    double temp_threshold = min_value + step;
    double err_sum = 0;
    double min_weighted_err = 10;
    double temp_weighted_err, temp_h;
    int temp_polarity;
    //int polarity = 1;
    for (i = 0; i < threshold_step; i=i+1) {
        err_sum = 0;
        for (j = 0; j < N; j=j+1) {
            temp_h = (features[j] < temp_threshold) ? -1 : 1;
            if (temp_h != y[j]) {
                err_sum = err_sum + weight[j];
            }
        }

        temp_polarity = (err_sum < 0.5) ? 1 : -1;
        temp_weighted_err = (err_sum < 0.5) ? err_sum : (1 - err_sum);

        
        //threshold[0] = (temp_weighted_err < min_weighted_err) ? temp_threshold : threshold[0];
        //polarity = (temp_weighted_err < min_weighted_err) ? temp_polarity : polarity;
        //min_weighted_err = (temp_weighted_err < min_weighted_err) ? temp_weighted_err : min_weighted_err;

        if (temp_weighted_err < min_weighted_err) {
            threshold[0] = temp_threshold;
            polarity[0] = temp_polarity;
            min_weighted_err = temp_weighted_err;
        }

        temp_threshold = temp_threshold + step;
    }
    
    weighted_err[0] = min_weighted_err;

    /*
    for (i = 0; i < N; i=i+1) {
        h_x[i] = (features[i] < threshold[0]) ? -1 * polarity : 1 * polarity ;
    }
    */
}

/*
    nlhs: # of expected output mxArray
    plhs: Array of pointers to the expected output mxArrays
    nrhs: # of intput mxArray
    prhs: Array of pointers to the input mxArrays
    mxGexScalar()   get the value of scalar input
    mxGetPr()       create a pointer to the real data in the input matrix
    mxGetN()        get the size of the matrix
    mxCreateDoubleMatrix(1,ncols,mxREAL)    create the output matrix
*/
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[]) { 
    double* features;
    double* y;
    double* weight;

    //double* h_x;
    double* weighted_err;
    double* threshold;
    double* polarity;

    // check the number of input arguments
    if(nrhs!=3) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Four inputs required.");
    }

    // check the number of output arguments
    if(nlhs!=3) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    }

    // make sure the first input argument is type double
    /*
    if(!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Input matrix must be type double.");
    }
    
    if(mxGetM(prhs[0])!=9168 && mxGetN(prhs[0])!=57194) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","dimension of feature is not met");
    }

    if(mxGetM(prhs[1])!=1 && mxGetN(prhs[1])!=57194) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","dimension of y is not met");
    }

    if(mxGetM(prhs[2])!=1 && mxGetN(prhs[2])!=57194) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","dimension of weight is not met");
    }
    */

    // create a pointer to the real data in the input matrix
    features = mxGetPr(prhs[0]);
    y = mxGetPr(prhs[1]);
    weight = mxGetPr(prhs[2]);

    //create matrix
    plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    plhs[2] = mxCreateDoubleMatrix(1, 1, mxREAL);
    
    // get a pointer to the real data in the output matrix
    weighted_err = mxGetPr(plhs[0]);
    threshold = mxGetPr(plhs[1]);
    polarity = mxGetPr(plhs[2]);

    getWeightedError(features, y, weight, weighted_err, threshold, polarity);
}
