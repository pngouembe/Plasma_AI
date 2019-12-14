#ifndef MATRICE_H
#define MATRICE_H
#include <stdio.h>
//#include <stdlib.h>

#define MAX_ROW     28
#define MAX_COL     28
#define MatValue(Mat, _row, _col)   *(Mat.values + (_row) * Mat.col + (_col))        

extern int FIXED_POINT_FACTOR;
extern int FIXED_POINT_FRACTIONNAL_BITS;

typedef struct
{
    int row,col;
    int values[MAX_COL*MAX_ROW];
}Matrix;

Matrix initMatrix(int row, int col);

void setMatrixValues(Matrix* matrix, int* values, int valuesLength);

Matrix convolution2D(Matrix matrix, Matrix filter);

void ReLU(Matrix* matrix);

Matrix MaxPooling(Matrix matrix, int kernelSize, int stride);

Matrix MatAdd(Matrix mat1, Matrix mat2);

void elemAdd(Matrix* mat, int elem);

void destroyMatrix(Matrix mat);

void printMatrix(Matrix matrix);
#endif