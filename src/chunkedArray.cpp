#include "chunkedArray.hpp"
#include "params.h"
#include <stdio.h>
#include <stdlib.h>

void ChunkedArray::moveHostToDevice()
{
#if ( PITTPACKACC )
#pragma acc update device( P [0:arraySize] )
#endif
}

double ChunkedArray::getValue(int id, int i, int j, int k,int index) const
{
  //return P[2 * ( id * chunkSize / 2 + i + nx * j + nx * ny * k )+index ];
   return ( P[2 * ( id * chunkSize / 2 + i + nx * j + nx * ny * k ) + index] ); //0 specified as default direction
//return P[2 * ( i + nx * j + nx * ny * k ) + index] ; 
}

void ChunkedArray::setValue(double *F)
{

//   for ( int id = 0; id < nChunk; id++ )
 //  {
      int id = 0;
      for ( int k = 0; k < nz; k++ )
      {       
        for ( int j = 0; j < ny; j++ )
        {                                        
            for ( int i = 0; i <nx; i++ )
            {
                P[2 * ( id * chunkSize / 2 + i + nx * j + nx * ny * k ) ]   =   F[2 * ( id * chunkSize / 2 + i + nx * j + nx * ny * k ) ];
                P[2 * ( id * chunkSize / 2 + i + nx * j + nx * ny * k )+1 ] =   F[2 * ( id * chunkSize / 2 + i + nx * j + nx * ny * k )+1 ];
//                P[2 * ( i + nx * j + nx * ny * k ) + 0] =F[2 * ( i + nx * j + nx * ny * k ) + 0];   
  //              P[2 * ( i + nx * j + nx * ny * k ) + 1] =F[2 * ( i + nx * j + nx * ny * k ) + 1];   
            }

        }
      }
 //  }
} 

ChunkedArray ChunkedArray:: operator+(const ChunkedArray& b) {

        ChunkedArray box;
        int n[3];
        n[0]=nx;      n[1]=ny;      n[2]=nz;
//    cout<<"Debug allocate: Nx="<<Nx<<"Ny="<<Ny<<"Nz="<<Nz<<"nx="<<nx<<"ny="<<ny<<"nz="<<nz<<"nChunk="<<nChunk<<"\n"; 
                box.allocate(n,nChunk);
//   for ( int id = 0; id < nChunk; id++ )
 //  {
     int id = 0;
     for ( int k = 0; k < nz; k++ )
     {       
        for ( int j = 0; j < ny; j++ )
        {                                        
            for ( int i = 0; i < nx; i++ )
            {
              box.P[2 * ( id * chunkSize / 2 + i + nx * j + nx * ny * k ) ] =  (*this)(id,0,i,j,k,0) + b.getValue(id,i,j,k,0);
              box.P[2 * ( id * chunkSize / 2 + i + nx * j + nx * ny * k )+1 ]  = (*this)(id,0,i,j,k,1) + b.getValue(id,i,j,k,1);
             // box.P[2 * ( i + nx * j + nx * ny * k )] = (*this)(i,j,k,0) + b.getValue(i,j,k,0);
            //  box.P[2 * ( i + nx * j + nx * ny * k ) + 1] =  (*this)(i,j,k,1) + b.getValue(i,j,k,1);         
            }

        }
      }
 //  }
         return box;
  }


PittPackResult ChunkedArray::allocate( int *n, int nbl )
{
    Nx = n[0];
    Ny = n[1];
    Nz = n[2];

    arraySize = 2 * Nx * Ny * Nz;

    // cout << " allocated P with size " << arraySize << endl;
    P = new ( std::nothrow ) double[arraySize];

    if ( nbl <= 0 )
    {
        cout << " Exit Code " << BLOCK_NUMBER_FAIL << endl;
        cout << BLUE << PittPackGetErrorEnum( BLOCK_NUMBER_FAIL ) << RESET << endl;
        exit( 1 );
    }

//    cout << " allocated P with size " << arraySize << endl;

    for ( int i = 0; i < arraySize; i++ )
    {
        P[i] = 0.0;
    }
    nChunk    = nbl;
    chunkSize = arraySize / nbl; /*! chunk size refers to the size of the chunks, that is the whole
                                    array divided by nchunks, this is to get rid of multiplication by two */

#if ( PITTPACKACC )
#pragma acc enter data create( this [0:1] ) async( 2 )
#pragma acc update device( this )
#pragma acc enter data create( P [0:arraySize] ) async( 3 )
#endif

#if ( PITTPACKACC )
    acc_async_wait_all();
#endif

    if ( P == NULL )
    {
        return ( ALLOCATION_FAIL );
    }
    else
    {
        return ( SUCCESS );
    }
}

int  ChunkedArray::getChunkSize() { return ( chunkSize ); }
void ChunkedArray::setCoords( double *X )
{
    Xa = X[0];
    Xb = X[1];
    Ya = X[2];
    Yb = X[3];
    Za = X[4];
    Zb = X[5];
}

void ChunkedArray::setDirection( int dir )
{
    orientation = dir;

    double coeff[3]    = {1.0, 1.0, 1.0};
    coeff[orientation] = 1. / nChunk;

    nx = coeff[0] * Nx;
    ny = coeff[1] * Ny;
    nz = coeff[2] * Nz;
#if ( PITTPACKACC )
#pragma acc update device( nx, ny, nz )
#endif
 //   cout << "each chunk dims=" << nx << " " << ny << " " << nz << endl;
}

#if ( PITTPACKACC )
#pragma acc routine seq
#endif
int ChunkedArray::size() { return ( arraySize ); }

void ChunkedArray::moveDeviceToHost()
{
#if ( PITTPACKACC )
#pragma acc update self( P [0:arraySize] )
#endif
}

#if ( PITTPACKACC )
#pragma acc routine seq
// index 0 retrieves real and indez zero retrieves
inline double &ChunkedArray::operator()( int i, int j, int k, int index ) { return P[2 * ( i + nx * j + nx * ny * k ) + index]; }
#else
double &ChunkedArray::operator()( int i, int j, int k, int index ) { return P[2 * ( i + nx * j + nx * ny * k ) + index]; }
#endif

#if ( PITTPACKACC )
#pragma acc routine seq
inline double &ChunkedArray::operator()( int i, int j, int k )
// double &ChunkedArray::operator()( int i, int j, int k )
{
    return P[2 * ( i + nx * j + nx * ny * k )];
}

#else
double &ChunkedArray::operator()( int i, int j, int k ) { return P[2 * ( i + nx * j + nx * ny * k )]; }

#endif

#if ( PITTPACKACC )
#pragma acc routine seq
inline double &ChunkedArray::operator()( int i, int j, int k, int dir, int index )
{
    if ( dir == 0 )
    {
        return ( P[2 * ( i + nx * j + nx * ny * k ) + index] );
    }
    else if ( dir == 1 )
    {
        return ( P[2 * ( j + ny * i + nx * ny * k ) + index] );
    }
    else
    {
        return ( P[index] );
    }
}
#else

double &ChunkedArray::operator()( int i, int j, int k, int dir, int index )
{
    if ( dir == 0 )
    {
        return ( P[2 * ( i + nx * j + nx * ny * k ) + index] );
    }
    else if ( dir == 1 )
    {
        return ( P[2 * ( j + ny * i + nx * ny * k ) + index] );
    }
    else
    {
        cout << "incorrect index in ChunkedArray operator " << endl;
        exit( 1 );
        return ( P[index] );
    }
}

#endif

#if ( PITTPACKACC )
#pragma acc routine seq
inline double &ChunkedArray::operator()( int chunkId, int dir, int i, int j, int k, int index )
{
    /*! brief: rearrangement is consistent with the planes where (1,0,0) (0,1,0) and (0,0,1) around which rotation occurs */

    if ( dir == 0 )
    {
        return ( P[2 * ( chunkId * chunkSize / 2 + i + nx * j + nx * ny * k ) + index] );
    }
    else if ( dir == 1 )
    {
        // return( P[2 * ( chunkId*chunkSize/2 + j + ny * k + nx * ny * i ) + index]);
        return ( P[2 * ( chunkId * chunkSize / 2 + j + ny * i + nx * ny * k ) + index] );
    }
    else if ( dir == 2 )
    {
        // return( P[2 * ( chunkId*chunkSize/2 + k + ny * i + ny * nz * j ) + index]);
        return ( P[2 * ( chunkId * chunkSize / 2 + k + nz * j + nz * ny * i ) + index] );
    }
    /* this one is added to help solve the Thomas in place and while in y- major direction, y , x and i and j replace each other*/

    else if ( dir == 3 )
    {
        return ( P[2 * ( chunkId * chunkSize / 2 + i + ny * j + nx * ny * k ) + index] );
    }
    // note that at rotated to y-dir, ny corresponds to i, nx corresponds to j
    else if ( dir == 4 )
    {
        return ( P[2 * ( chunkId * chunkSize / 2 + k + nz * j + nz * ny * i ) + index] );
    }
    else
    {
        return ( P[index] );
    }
}
#else
double &ChunkedArray::operator()( int chunkId, int dir, int i, int j, int k, int index )
{
    /*! brief: rearrangement is consistent with the planes where (1,0,0) (0,1,0) and (0,0,1) around which rotation occurs */

    if ( dir == 0 )
    {
        return ( P[2 * ( chunkId * chunkSize / 2 + i + nx * j + nx * ny * k ) + index] );
    }
    else if ( dir == 1 )
    {
        // return( P[2 * ( chunkId*chunkSize/2 + j + ny * k + nx * ny * i ) + index]);
        return ( P[2 * ( chunkId * chunkSize / 2 + j + ny * i + nx * ny * k ) + index] );
    }
    else if ( dir == 2 )
    {
        // return( P[2 * ( chunkId*chunkSize/2 + k + ny * i + ny * nz * j ) + index]);
        return ( P[2 * ( chunkId * chunkSize / 2 + k + nz * j + nz * ny * i ) + index] );
    }
    /* this one is added to help solve the Thomas in place and while in y- major direction, y , x and i and j replace each other*/

    else if ( dir == 3 )
    {
        return ( P[2 * ( chunkId * chunkSize / 2 + i + ny * j + nx * ny * k ) + index] );
    }
    // note that at rotated to y-dir, ny corresponds to i, nx corresponds to j
    else if ( dir == 4 )
    {
        return ( P[2 * ( chunkId * chunkSize / 2 + k + nz * j + nz * ny * i ) + index] );
    }
    else
    {
        cout << "incorrect index in ChunkedArray operator " << endl;
        exit( 1 );
        return ( P[index] );
    }
}

#endif

// intel's compiler complaining about inlining
#if ( PITTPACKACC )
#pragma acc routine seq
inline double &ChunkedArray::operator()( int i )
{
    /*
       if(i>=arraySize)
    {
      printf("index bigger than array size]n");

    //  exit(0);
    }
    */
    return P[i];
}
#else
double &ChunkedArray::operator()( int i )
{
    /*
       if(i>=arraySize)
    {
      printf("index bigger than array size]n");

    //  exit(0);
    }
    */
    return P[i];
}

#endif
/*!< the order is not important */
ChunkedArray::~ChunkedArray()
{
    if ( P != NULL )
    {
#if ( PITTPACKACC )
#pragma acc exit data delete ( P [0:arraySize] )
#endif
        delete[] P;
    }
#if ( PITTPACKACC )
#pragma acc exit data delete ( this )
#endif
}

void ChunkedArray::getAddress( double *rt )
{
    rt = P;
    cout << "pointer address " << rt << endl;
}
