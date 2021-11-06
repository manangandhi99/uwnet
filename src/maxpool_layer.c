#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    // each row is a seperate image
    // flattened image: [channel 1, channel 2, etc..]
    // each channel: [row 1, row 2, etc..]
    
    // for each image
    for (int currImage = 0; currImage < in.rows; currImage++) {
        int imageStart = currImage * in.cols;
        int poolStart = currImage * out.cols;

        for (int currChannel = 0; currChannel < l.channels; currChannel++) {
            int imageChannelStart = l.width * l.height * currChannel;
            int poolChannelStart = outw * outh * currChannel;

            // go through the layer
            int squareCount = 0;
            for (int row = 0; row < l.height; row += l.stride) {
                for (int col = 0; col < l.width; col += l.stride) {

                    // for each "square"
                    float currMax = FLT_MIN;
                    
                    int evenOffset;
                    if (l.size % 2 == 0) {
                        evenOffset = 1;
                    } else {
                        evenOffset = 0;
                    }
                    int kernel_diff = l.size / 2;

                    for (int i = row - kernel_diff + evenOffset; i < row + kernel_diff + 1; i++) {
                        for (int j = col - kernel_diff + evenOffset; j < col + kernel_diff + 1; j++) {
                            
                            if (i >= 0 && j >= 0 && i < l.channels * l.height && j < l.width) {
                                float currVal = in.data[imageStart + imageChannelStart + i * l.width + j];
                                if (currVal > currMax) {
                                    currMax = currVal;
                                }
                            }
                        }
                    }

                    out.data[poolStart + poolChannelStart + squareCount] = currMax;
                    squareCount++;
                } 
            }
        }
    }

    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

    // for each image
    for (int currImage = 0; currImage < in.rows; currImage++) {
        int imageStart = currImage * in.cols;
        int deltaXStart = currImage * dx.cols;
        int deltaYStart = currImage * dy.cols;

        for (int currChannel = 0; currChannel < l.channels; currChannel++) {
            int imageChannelStart = l.width * l.height * currChannel;
            int deltaXChannelStart = outw * outh * currChannel;
            int deltaYChannelStart = outw * outh * currChannel;

            // go through the layer
            int squareCount = 0;
            for (int row = 0; row < l.height; row += l.stride) {
                for (int col = 0; col < l.width; col += l.stride) {

                    // for each "square"
                    float currMax = FLT_MIN;
                    int maxRow = -1;
                    int maxCol = -1;

                    int evenOffset;
                    if (l.size % 2 == 0) {
                        evenOffset = 1;
                    } else {
                        evenOffset = 0;
                    }
                    int kernel_diff = l.size / 2;

                    for (int i = row - kernel_diff + evenOffset; i < row + kernel_diff + 1; i++) {
                        for (int j = col - kernel_diff + evenOffset; j < col + kernel_diff + 1; j++) {
                            
                            if (i >= 0 && j >= 0 && i < l.channels * l.height && j < l.width) {
                                float currVal = in.data[imageStart + imageChannelStart + i * l.width + j];
                                if (currVal > currMax) {
                                    currMax = currVal;
                                    maxRow = i;
                                    maxCol = j;
                                }
                            }
                        }
                    }

                    // find corresponding error
                    float deltaVal = dy.data[deltaYStart + deltaYChannelStart + squareCount];
                    
                    // fill in delta
                    dx.data[imageChannelStart + deltaXStart + maxRow * l.width + maxCol] += deltaVal;
                    squareCount++;
                } 
            }
        }
    }


    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

