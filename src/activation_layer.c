#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"


// Run an activation layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = f(x)
matrix forward_activation_layer(layer l, matrix x)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(x);

    ACTIVATION a = l.activation;
    matrix y = copy_matrix(x);

    // TODO: 2.1
    // apply the activation function to matrix y
    // logistic(x) = 1/(1+e^(-x))
    // relu(x)     = x if x > 0 else 0
    // lrelu(x)    = x if x > 0 else .01 * x
    // softmax(x)  = e^{x_i} / sum(e^{x_j}) for all x_j in the same row 
    if (a == LOGISTIC || a == RELU || a == LRELU) {
        for (int i = 0; i < x.rows; ++i) {
            for (int j = 0; j < x.cols; ++j) {
                if (a == LOGISTIC) {
                    y.data[i * x.cols + j] = 1.0/(1.0 + exp(-x.data[i * x.cols + j]));
                } else if (a == RELU) {
                    float temp = x.data[i * x.cols + j];
                    if (temp > 0) {
                        y.data[i * x.cols + j] = temp;
                    } else {
                        y.data[i * x.cols + j] = 0;
                    }
                } else { // a == LRELU
                    float temp = x.data[i * x.cols + j];
                    if (temp > 0) {
                        y.data[i * x.cols + j] = temp;
                    } else {
                        y.data[i * x.cols + j] = 0.01 * temp;
                    }
                }
            }
        }       
    } else  {  // a == SOFTMAX
        float* rowSums = (float*) calloc(x.cols, sizeof(float));
        for (int i = 0; i < x.rows; ++i) {
            for (int j = 0; j < x.cols; ++j) {
                rowSums[i] += exp(x.data[i*x.cols + j]);
            }
        }
        for (int i = 0; i < x.rows; ++i) {
            for (int j = 0; j < x.cols; ++ j) {
                y.data[i*x.cols + j] = exp(x.data[i*x.cols + j]) / rowSums[i];
            }
        }
    }
    return y;
}

// Run an activation layer on input
// layer l: pointer to layer to run
// matrix dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
matrix backward_activation_layer(layer l, matrix dy)
{
    matrix x = *l.x;
    matrix dx = copy_matrix(dy);
    ACTIVATION a = l.activation;

    // TODO: 2.2
    // calculate dL/dx = f'(x) * dL/dy
    // assume for this part that f'(x) = 1 for softmax because we will only use
    // it with cross-entropy loss for classification and include it in the loss
    // calculations
    // d/dx logistic(x) = logistic(x) * (1 - logistic(x))
    // d/dx relu(x)     = 1 if x > 0 else 0
    // d/dx lrelu(x)    = 1 if x > 0 else 0.01
    // d/dx softmax(x)  = 1

    for (int i = 0; i < x.rows; ++i) {
        for (int j = 0; j < x.cols; ++j) {
            float temp_dx = 0;
            if (a == LOGISTIC) {
                float x_ij = x.data[i*x.cols + j];
                temp_dx = x_ij * (1.0 - x_ij);
            } else if (a == RELU) {
                if (x.data[i*x.cols + j] > 0) {
                    temp_dx = 1;
                }
            } else if (a == LRELU) {
                if (x.data[i*x.cols + j] > 0) {
                    temp_dx = 1;
                } else {
                    temp_dx = 0.01;
                }
            } else { // a == SOFTMAX
                temp_dx = 1;
            }
            dx.data[i*x.cols + j] = dy.data[i*x.cols + j] * temp_dx;
        }
    }

    return dx;
}

// Update activation layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_activation_layer(layer l, float rate, float momentum, float decay){}

layer make_activation_layer(ACTIVATION a)
{
    layer l = {0};
    l.activation = a;
    l.x = calloc(1, sizeof(matrix));
    l.forward = forward_activation_layer;
    l.backward = backward_activation_layer;
    l.update = update_activation_layer;
    return l;
}
