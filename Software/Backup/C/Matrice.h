#ifndef MATRICE_H
#define MATRICE_H
#include <stdio.h>
#include <stdlib.h>

#define MatValue(Mat, _row, _col) *(Mat.values + (_row) * Mat.col + (_col))

typedef struct
{
    int row,col;
    float* values;
}Matrix;

Matrix initMatrix(int row, int col);

void setMatrixValues(Matrix* matrix, float* values, int valuesLength);

Matrix convolution2D(Matrix matrix, Matrix filter);

void ReLU(Matrix* matrix);

Matrix MaxPooling(Matrix matrix, int kernelSize, int stride);

Matrix MatAdd(Matrix mat1, Matrix mat2);

void elemAdd(Matrix* mat, float elem);

void destroyMatrix(Matrix mat);

void printMatrix(Matrix matrix);
#endif