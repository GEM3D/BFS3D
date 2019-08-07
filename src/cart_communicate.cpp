#include "mpi.h"
#include "cart_communicate.hpp"
#include <iostream>

using namespace std;

cart_communicate::cart_communicate(int npi, int npj, int argc, char *argv[]) {
  npx = npi;
  npy = npj;
  ndims = 2;

  // MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comsize);

  dims[0] = npx;
  dims[1] = npy;

  cart();
}

cart_communicate::cart_communicate(int npi, int npj,int myrank,int comsize) {
  npx = npi;
  npy = npj;
  ndims = 2;

  dims[0] = npx;
  dims[1] = npy;

  cart();
}

void cart_communicate::shift(int direction, int disp, int *rank_source,
                        int *rank_dest) {

  if (direction == 0) {
    direction = 1;
  } else if (direction == 1) {
    direction = 0;
  }

  MPI_Cart_shift(com_cart, direction, disp, rank_source, rank_dest);
  // cout<<my_rank<<" "<<(*rank_dest)<<" "<<*rank_source <<endl;
}

 cart_communicate::~cart_communicate() {
}

void cart_communicate::cart() {  

  int reorder = 0;

  periods[0] = 0;
  periods[1] = 0;

  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &com_cart);
}

