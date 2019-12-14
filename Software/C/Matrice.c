#include "Matrice.h"

Matrix initMatrix(int row, int col)
{
    Matrix mat;
    mat.row = row;
    mat.col = col;
    mat.values[row*col];
    for (int i = 0; i < row*col; i++)
    {
        *(mat.values+i) = 0;
    }
    return mat;
}

void setMatrixValues(Matrix* matrix, int* values, int valuesLength)
{
    if(valuesLength != matrix->row * matrix->col)
        return;//printf("erreur de taille\n values lenght = %d and row*col = %d\n", valuesLength, matrix->row*matrix->col);
    else
    {
        for (int row = 0; row < matrix->row; row++)
        {
            for (int col = 0; col < matrix->col; col++)
            {
                *(matrix->values + row * matrix->col + col)  = *(values + row * matrix->col + col);
            }
            
        }
    }
}

// on part du principe que le filtre utilisé est de taille 5X5
// et qu'on applique le zero padding necessaire
Matrix convolution2D(Matrix matrix, Matrix filter)  
{ 
    Matrix out = initMatrix(matrix.row, matrix.col);
    int offset = filter.row/2;
    for (int row = 0; row < matrix.row ; row++)
    {
        for (int col = 0; col < matrix.col; col++)
        {
            //printf("m(%d,%d) = ", row, col);
            for (int filterRow = 0; filterRow < filter.row; filterRow++)
            {
                for (int filterCol = 0; filterCol < filter.col; filterCol++)
                {
                    if (row + filterRow - offset < 0 || col + filterCol - offset < 0 || row + filterRow - offset > matrix.row-1 || col + filterCol - offset > matrix.col - 1)
                    {
                        MatValue(out, row, col) += 0;
                    }
                    else
                    {
                        //printf("%.2f(m(%d,%d)) * %.2f(f(%d,%d)) + ", MatValue(matrix, row + filterRow - offset, col + filterCol - offset), row + filterRow - offset, col + filterCol - offset, MatValue(filter, filterRow, filterCol),filterRow ,filterCol);
                        
                        MatValue(out, row, col) += (MatValue(matrix, row + filterRow - offset, (col + filterCol - offset)) * MatValue(filter, filterRow, filterCol))>>FIXED_POINT_FRACTIONNAL_BITS;
                    }
                } 
            }
        }  
    } 
    return out;
}

void ReLU(Matrix* matrix)
{
    for (int i = 0; i < matrix->col*matrix->row; i++)
    {
        if(*(matrix->values + i) < 0)
        {
            *(matrix->values + i) = 0;
        }   
    }   
}

Matrix MaxPooling(Matrix matrix, int kernelSize, int stride)
{
    Matrix out = initMatrix((int)((1+matrix.row-(kernelSize-1)-1)/stride)+1,
                            (int)((1+matrix.col-(kernelSize-1)-1)/stride)+1);
    int max = 0;
    for (int row = 0; row < out.row; row++)
    {
        for (int col = 0; col < out.col; col++)
        {
            for (int kRow = 0; kRow < kernelSize; kRow++)
            {
                for (int kCol = 0; kCol < kernelSize; kCol++)
                {
                    if (MatValue(matrix, stride*row + kRow, stride*col + kCol) > max)
                    {
                        max = MatValue(matrix, stride*row + kRow, stride*col + kCol);
                    }
                    
                }
                
            }
            MatValue(out, row, col) = max;
            max = 0;
        }
    }
    return out;
}

Matrix MatAdd(Matrix mat1, Matrix mat2)
{
    Matrix tmp = initMatrix(mat1.row, mat1.col);
    if (mat1.row == mat2.row && mat1.col == mat2.col)
    {
        for (int i = 0; i < mat1.col*mat1.row; i++)
        {
            *(tmp.values + i) = *(mat1.values + i) + *(mat2.values + i);
        }
    }
    else
    {
        //printf("erreur sur les dimension de matrices (mat1(%d,%d) + mat2(%d,%d))\n", mat1.row, mat1.col, mat2.row, mat2.col);
    }
    
    return tmp;
}

void elemAdd(Matrix* mat, int elem)
{
    for (int i = 0; i < mat->col*mat->row; i++)
    {
        *(mat->values + i) = *(mat->values + i) + elem;
    }    
}


void destroyMatrix(Matrix mat)
{
    //free(mat.values);
}


void printMatrix(Matrix matrix)
{
    //printf("%f", *(matrix.values+1));
    for (int j = 0; j < matrix.row; j++)
    {
        printf("[");
        for (int i = 0; i < matrix.col; i++)
        {
            printf("%d\t",MatValue(matrix, j, i));
        }
        printf("]\n");
    }
    
}