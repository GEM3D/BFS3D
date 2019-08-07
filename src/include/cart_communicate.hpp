#ifndef _CART_COMMUNICATE_H_
#define _CART_COMMUNICATE_H_
#include "mpi.h"


class cart_communicate
{
private:
MPI_Comm com_cart;
int my_rank;
int comsize;
int npx;
int npy;
int ndims;
int dims[2];
int periods[2];
// type is defined to send/reciev strided info
int coord[2];


public:
cart_communicate(int npi, int npj,  int argc, char *argv[] );
cart_communicate(int npi, int npj,int myrank,int comsize);
void cart();
void shift(int direction, int disp,int *rank_source,int * rank_dest);
void commit(int x, int py);
~cart_communicate();

friend class pencilDcmp;

};


#endif
