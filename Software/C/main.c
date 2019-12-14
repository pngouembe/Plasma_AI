#include "Matrice.h"
#include "model.h"
#define SIZE_X      5
#define SIZE_Y      5
#define SIZE_VALUE  SIZE_Y*SIZE_X
int main()
{
    Matrix input = initMatrix(28,28);
    int nbFautes = 0;
    for (int iter = 0; iter < NB_IMG; iter++)
    {
        setMatrixValues(&input, testValues[iter], 28*28);
        // layer 1

        Matrix kernels1[16];
        for (int i = 0; i < 16; i++)
        {
            kernels1[i] = initMatrix(5,5);
            setMatrixValues(&kernels1[i], weight1[i], 5*5);
        }
        
        Matrix outL1[16];
        Matrix tmp;
        for (int i = 0; i < 16; i++)
        {
            tmp = convolution2D(input,kernels1[i]);        
            elemAdd(&tmp, bias1[i]);
            ReLU(&tmp);
            outL1[i] = MaxPooling(tmp,2,2);
        }
        /*printf("outL1[0] : \n");
        printMatrix(outL1[0]);*/
        destroyMatrix(tmp);
        for (int i = 0; i < 16; i++)
        {
            destroyMatrix(kernels1[i]);
        }
        destroyMatrix(input);

        //layer 2
        Matrix kernels2[32][16];
        for (int i = 0; i < 32; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                kernels2[i][j] = initMatrix(5,5); 
                setMatrixValues(&kernels2[i][j], weight2[i][j], 5*5);
            }
        }
        Matrix outL2[32];
        Matrix tmp1 = initMatrix(outL1[0].row, outL1[0].col);
        Matrix tmp2 = initMatrix(outL1[0].row, outL1[0].col);
        for (int i = 0; i < 32; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                tmp1 = convolution2D(outL1[j],kernels2[i][j]);
                elemAdd(&tmp1, bias2[i]);
                tmp2 = MatAdd(tmp2, tmp1);
            }
            ReLU(&tmp2);
            outL2[i] = MaxPooling(tmp2,2,2);
        }
        /*printf("outL2[0] : \n");
        printMatrix(outL2[0]);*/

        destroyMatrix(tmp1);
        destroyMatrix(tmp2);
        for (int i = 0; i < 32; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                destroyMatrix(kernels2[i][j]);
            }
        }
        for (int i = 0; i < 16; i++)
        {
            destroyMatrix(outL1[i]);
        }


        //Layer3
        int res[10];
        int max = -10;
        int num = 0;
        for (int i = 0; i < 10; i++)
        {   
            res[i] = 0;
            for (int j = 0; j < 32; j++)
            {
                for (int k = 0; k < outL2[j].col*outL2[j].row ; k++)
                {
                    res[i] += (*(outL2[j].values + k)*weight3[i][j*outL2[j].col*outL2[j].row + k])>>FIXED_POINT_FRACTIONNAL_BITS;
                }
            }
            res[i] += bias3[i];
            if (res[i] > max)   
            {
                max = res[i];
                num = i;
            }
        }
        for (int i = 0; i < 32; i++)
        {
            destroyMatrix(outL2[i]);
        }
        /* 
        printf("res = {");
        for (int i = 0; i < 10; i++)
        {
            printf("%d, ",res[i]);
        }
        printf("};\n");
        */
       /* printf("res converti = {");
        for (int i = 0; i < 10; i++)
        {
            printf("%f, ",(float)(res[i])/FIXED_POINT_FACTOR);
        }
        printf("};\n");*/
        
        printf("numero : %d\tLabel : %d\n", num, testLabel[iter]);
        if(num != testLabel[iter])
            nbFautes++;
        if(iter % 100 == 0)
            printf("Image #%d traitée\n", iter);
        //printf("test conv : %d * %d = %d --> %f",weight1[0][0],weight1[0][1],weight1[0][0]*weight1[0][1],(float)((weight1[0][0]*weight1[0][1])>>FIXED_POINT_FRACTIONNAL_BITS)/FIXED_POINT_FACTOR);
    }
    printf("\n\n\n nbFautes = %d/%d\n", nbFautes, NB_IMG);
    return 0;
}