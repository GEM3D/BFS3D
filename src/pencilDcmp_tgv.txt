#include "pencilDcmp.hpp"
#include "definitions.h"
#include "mathFunction.hpp"
#include "mpi.h"
#include "params.h"
#include <cmath>
#include <iostream>
#include <string.h>

#define SIZEMG 0
#define OFFS 0

using namespace std;

typedef void ( *exitroutinetype )( char *err_msg );
extern void acc_set_error_routine( exitroutinetype callback_routine );

void handle_gpu_errors( char *err_msg )
{
    printf( "GPU Error: %s", err_msg );
    printf( "Exiting...\n\n" );
    //  MPI_Abort(MPI_COMM_WORLD, 1);
    exit( -1 );
}

PencilDcmp::PencilDcmp( int n0, int n1, int n2, int px, int py )
{
    nx = n0;
    ny = n1;
    nz = n2;

    p0 = px;
    p1 = py;

    // set scale=1.0
    scale = 1.0;
    // this poisson solver will be used in a flow solver
    // itertively so our start and end are the same

    nxChunk = nx / p0;
    nyChunk = ny / p0;
    nzChunk = nz / p0;

    //   acc_set_error_routine(&handle_gpu_errors);
    //    cout<<" "<<nxChunk<<" "<<nyChunk<<" "<<" "<<nzChunk<<endl;

    MPIStartUp();

    p0 = sqrt( comSize );

    p1 = p0;

    nChunk = p0;

#if ( !DISABLE_CHECKS )
    PittPackResult result = checkInput();

    if ( result != SUCCESS )
    {
        cout << BLUE << "Exit Code : " << result << endl;
        cout << BLUE << PittPackGetErrorEnum( result ) << RESET << endl;
        exit( 1 );
    }
#endif

    coords = new double[6];
    dxyz   = new double[3];
    tmp    = new double[2 * nxChunk * nyChunk * nzChunk];

#if ( MULTIGRIDON )
    tmpMGReal = new double[nz];
    tmpMGImag = new double[nz];
#endif

    gangTri = MIN( nyChunk, TRI_NUM_GANG );

    if ( SOLUTIONMETHOD == 0 )
    {
        gangTri = 2 * gangTri;
    }

#if ( THOM_FULL_BATCH == 1 )
    x1 = new double[nxChunk * nyChunk * nz];
    x2 = new double[nxChunk * nyChunk * nz];
    x3 = new double[nxChunk * nyChunk * nz];
#else
    x1 = new double[gangTri * nz];
    x2 = new double[gangTri * nz];
    x3 = new double[gangTri * nz];
#endif

    if ( SOLUTIONMETHOD != 0 )
    {
        //    crpcr_lower=new double[gangTri*nz];
        crpcr_upper = new double[gangTri * nz];
    }

    subDiag = new double[3];
    supDiag = new double[3];
    length  = new int[3];
    omega   = new double[3];
    tags    = new int[3];
    freqs   = new double[2];
    bc      = new char[6];
    num     = new double[2];
    denum   = new double[2];

    subDiag[0] = 0.0;
    subDiag[1] = 1.0;
    subDiag[2] = 1.0;

    supDiag[0] = 1.0;
    supDiag[1] = 1.0;
    supDiag[2] = 0.0;

    dl = new double[nz];
    ds = new double[nz];
    du = new double[nz];

    dl[0]      = 0.0;
    du[nz - 1] = 0.0;

    for ( int i = 1; i < nz; i++ )
    {
        dl[i]     = 1.0;
        du[i - 1] = 1.0;
    }

    Xbox = new double[6];

    faceTag = new sint[6];

    indices = new int[p0];

// allocate some indices for help swap the data in-place rather than allocating a whole matrix
#if ( PITTPACKACC )
#pragma acc enter data create( this [0:1] ) async( 1 )
#pragma acc update device( this )
#pragma acc enter data create( omega [0:3] ) async( 2 )
#pragma acc enter data create( dxyz [0:3] ) async( 3 )
#pragma acc enter data create( coords [0:6] ) async( 4 )
#pragma acc enter data create( tags [0:3] ) async( 5 )
#pragma acc enter data create( subDiag [0:3] ) async( 5 )
#pragma acc enter data create( freqs [0:2] ) async( 6 )
#pragma acc enter data create( supDiag [0:3] ) async( 7 )
#pragma acc enter data create( length [0:3] ) async( 8 )
#pragma acc enter data create( num [0:2] ) async( 9 )
#pragma acc enter data create( denum [0:2] ) async( 10 )
#pragma acc enter data create( bc [0:6] ) async( 11 )
#pragma acc enter data create( faceTag [0:6] ) async( 12 )
#if ( MULTIGRIDON )
#pragma acc enter data create( tmpMGReal [0:nz + SIZEMG] ) async( 13 )
#pragma acc enter data create( tmpMGImag [0:nz + SIZEMG] ) async( 14 )
#endif
#pragma acc enter data create( Xbox [0:6] ) async( 15 )
#pragma acc update device( subDiag [0:3] )
#pragma acc update device( supDiag [0:3] )
#pragma acc enter data create( dl [0:nz] ) async( 16 )
#pragma acc enter data create( ds [0:nz] ) async( 17 )
#pragma acc enter data create( du [0:nz] ) async( 18 )
#pragma acc update device( dl [0:nz] )
#pragma acc update device( du [0:nz] )
#if ( THOMAS_FULL_BATCH == 1 )
#pragma acc enter data create( x1 [0:nz * nxChunk * nyChunk] ) async( 20 )
#pragma acc enter data create( x2 [0:nz * nxChunk * nyChunk] ) async( 21 )
#pragma acc enter data create( x3 [0:nz * nxChunk * nyChunk] ) async( 22 )
#else
#pragma acc enter data create( x1 [0:nz * gangTri] ) async( 23 )
#pragma acc enter data create( x2 [0:nz * gangTri] ) async( 24 )
#pragma acc enter data create( x2 [0:nz * gangTri] ) async( 25 )
#endif

    if ( SOLUTIONMETHOD != 0 )
    {
//#pragma acc enter data create( crpcr_lower[0 : nz*gangTri] )
#pragma acc enter data create( crpcr_upper [0:nz * gangTri] )
    }

    cout << "amount of memory allocated " << acc_get_free_memory() << endl;

#endif
    /* impoartant note about nested classes is that we need to go from top to bottom, first have the highest level
         class copyin 'this' for that class and then once the address are set on the GPU, then copyin the lower level class
         in our case it is in this order 1. pencil 2. chunked array*/

    allocateChunks();
    constructShuffleVectorX();
    constructShuffleVectorY();
    Sig.copyin();
    T.setElems( nChunk, nzChunk, subDiag, supDiag );

    // Ghost cell required for iterative solve
    MG.construct( ( nz + SIZEMG ), INNERITER, INNERITER, OUTERITER, subDiag, supDiag );
    MGC.construct( ( nz + SIZEMG ), INNERITER, INNERITER, OUTERITER, subDiag, supDiag );

    send_request = (MPI_Request *)malloc( sizeof( MPI_Request ) * ( p0 ) );
    send_status  = (MPI_Status *)malloc( sizeof( MPI_Status ) * ( p0 ) );

    recv_request = (MPI_Request *)malloc( sizeof( MPI_Request ) * ( p0 ) );
    recv_status  = (MPI_Status *)malloc( sizeof( MPI_Status ) * ( p0 ) );

#if ( DEBUG )
    cout << "iaxsize " << iaxSize << endl;
    cout << "iaysize " << iaySize << endl;
#endif

#if ( PITTPACKACC )
    acc_async_wait_all();
#endif
}

PencilDcmp::PencilDcmp( int argcs, char *pArgs[], int n0, int n1, int n2 )
{
    nx = n0;
    ny = n1;
    nz = n2;

    // set scale=1.0
    scale = 1.0;
    // this poisson solver will be used in a flow solver
    // itertively so our start and end are the same

    //    cout<<" "<<nxChunk<<" "<<nyChunk<<" "<<" "<<nzChunk<<endl;

    // check if MPI has already initialized, if not initialize it here

    int initFlag;
    if ( MPI_Initialized( &initFlag ) != MPI_SUCCESS && myRank == 0 )
    {
        cout << "failure in checking if MPI has already initialize" << endl;
        exit( 1 );
    }

    // cout<<" init flag "<<initFlag<<endl;
    if ( initFlag == 0 )
    {
        if ( MPI_Init( &argcs, &pArgs ) != MPI_SUCCESS && myRank == 0 )
        {
            cout << " Exit Code : " << PittPackGetErrorEnum( MPI_INIT_FAIL ) << endl;
            exit( 1 );
        }
    }

    MPIStartUp();

    p0 = sqrt( comSize );

    p1 = p0;

    nChunk = p0;

    nxChunk = nx / p0;
    nyChunk = ny / p0;
    nzChunk = nz / p0;

#if ( !DISABLE_CHECKS )
    PittPackResult result = checkInput();

    if ( result != SUCCESS )
    {
        cout << BLUE << "Exit Code : " << result << endl;
        cout << BLUE << PittPackGetErrorEnum( result ) << RESET << endl;
        exit( 1 );
    }
#endif

    coords = new double[6];
    dxyz   = new double[3];
    tmp    = new double[2 * nxChunk * nyChunk * nzChunk];
#if ( MULTIGRIDON )
    tmpMGReal = new double[nz + SIZEMG];
    tmpMGImag = new double[nz + SIZEMG];
#endif

    gangTri = MIN( nyChunk, TRI_NUM_GANG );

    if ( SOLUTIONMETHOD == 0 )
    {
        gangTri = 2 * gangTri;
    }

#if ( THOM_FULL_BATCH == 1 )
    x1 = new double[nxChunk * nyChunk * nz];
    x2 = new double[nxChunk * nyChunk * nz];
    x3 = new double[nxChunk * nyChunk * nz];
#else
    x1 = new double[gangTri * nz];
    x2 = new double[gangTri * nz];
    x3 = new double[gangTri * nz];
#endif

    /*
        x1 = new double[gangTri * nz];
        x2 = new double[gangTri * nz];
    */
    if ( SOLUTIONMETHOD != 0 )
    {
        //    crpcr_lower=new double[gangTri*nz];
        crpcr_upper = new double[gangTri * nz];
    }
    subDiag = new double[3];
    supDiag = new double[3];
    length  = new int[3];
    omega   = new double[3];
    tags    = new int[3];
    freqs   = new double[2];
    bc      = new char[6];
    num     = new double[2];
    denum   = new double[2];

    subDiag[0] = 0.0;
    subDiag[1] = 1.0;
    subDiag[2] = 1.0;

    supDiag[0] = 1.0;
    supDiag[1] = 1.0;
    supDiag[2] = 0.0;
    Xbox       = new double[6];
    faceTag    = new sint[6];

    dl = new double[nz];
    ds = new double[nz];
    du = new double[nz];

    dl[0]      = 0.0;
    du[nz - 1] = 0.0;

    for ( int i = 1; i < nz; i++ )
    {
        dl[i]     = 1.0;
        du[i - 1] = 1.0;
    }

    indices = new int[p0];
// allocate some indices for help swap the data in-place rather than allocating a whole matrix
#if ( PITTPACKACC )
#pragma acc enter data create( this [0:1] ) async( 1 )
#pragma acc update device( this )
#pragma acc enter data create( omega [0:3] ) async( 2 )
#pragma acc enter data create( dxyz [0:3] ) async( 3 )
#pragma acc enter data create( coords [0:6] ) async( 4 )
#pragma acc enter data create( tags [0:3] ) async( 5 )
#pragma acc enter data create( subDiag [0:3] ) async( 6 )
#pragma acc enter data create( freqs [0:2] ) async( 7 )
#pragma acc enter data create( supDiag [0:3] ) async( 8 )
#pragma acc enter data create( length [0:3] ) async( 9 )
#pragma acc enter data create( num [0:2] ) async( 10 )
#pragma acc enter data create( denum [0:2] ) async( 11 )
#pragma acc enter data create( bc [0:6] ) async( 12 )
#pragma acc enter data create( faceTag [0:6] ) async( 13 )
#pragma acc enter data create( Xbox [0:6] ) async( 14 )

#if ( MULTIGRIDON )
#pragma acc enter data create( tmpMGReal [0:nz + SIZEMG] ) async( 15 )
#pragma acc enter data create( tmpMGImag [0:nz + SIZEMG] ) async( 16 )
#endif

#pragma acc update device( subDiag [0:3] )
#pragma acc update device( supDiag [0:3] )
#pragma acc enter data create( dl [0:nz] ) async( 17 )
#pragma acc enter data create( ds [0:nz] ) async( 18 )
#pragma acc enter data create( du [0:nz] ) async( 19 )
#pragma acc update device( du [0:nz] )
#pragma acc update device( dl [0:nz] )
#if ( THOM_FULL_BATCH == 1 )
#pragma acc enter data create( x1 [0:nz * nxChunk * nyChunk] ) async( 20 )
#pragma acc enter data create( x2 [0:nz * nxChunk * nyChunk] ) async( 21 )
#pragma acc enter data create( x3 [0:nz * nxChunk * nyChunk] ) async( 22 )
#else
#pragma acc enter data create( x1 [0:nz * gangTri] ) async( 23 )
#pragma acc enter data create( x2 [0:nz * gangTri] ) async( 24 )
#pragma acc enter data create( x3 [0:nz * gangTri] ) async( 25 )
#endif
    if ( SOLUTIONMETHOD != 0 )
    {
//#pragma acc enter data create( crpcr_lower[0 : nz*gangTri] )
#pragma acc enter data create( crpcr_upper [0:nz * gangTri] )
    }
    cout << "amount of free memeory " << acc_get_free_memory() / 1e9 << endl;
#endif
    /* impoartant note about nested classes is that we need to go from top to bottom, first have the highest level
         class copyin 'this' for that class and then once the address are set on the GPU, then copyin the lower level class
         in our case it is in this order 1. pencil 2. chunked array*/
    allocateChunks();
    constructShuffleVectorX();
    constructShuffleVectorY();
    //   Sig.copyin();
    T.setElems( nChunk, nzChunk, subDiag, supDiag );
    MG.construct( ( nz + SIZEMG ), INNERITER, INNERITER, OUTERITER, subDiag, supDiag );
    MGC.construct( ( nz + SIZEMG ), INNERITER, INNERITER, OUTERITER, subDiag, supDiag );

    //    acc_map_data(P.P,P.P,2*nxChunk*nyChunk*nz*sizeof(double) );
    send_request = (MPI_Request *)malloc( sizeof( MPI_Request ) * ( p0 ) );
    send_status  = (MPI_Status *)malloc( sizeof( MPI_Status ) * ( p0 ) );

    recv_request = (MPI_Request *)malloc( sizeof( MPI_Request ) * ( p0 ) );
    recv_status  = (MPI_Status *)malloc( sizeof( MPI_Status ) * ( p0 ) );

#if ( DEBUG )
    cout << "iaxsize " << iaxSize << endl;
    cout << "iaysize " << iaySize << endl;
#endif

#if ( PITTPACKACC )
    acc_async_wait_all();
#endif
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::setDiag( int i, int j )
{
    double eig = getEigenVal( i, j );

#if ( PITTPACKACC )
#pragma acc loop
#endif
    for ( int i = 0; i < nz; i++ )
    {
        ds[i] = eig;
    }
}

void PencilDcmp::setBox( double *X )
{
    for ( int i = 0; i < 6; i++ )
    {
        Xbox[i] = X[i];
    }

#if ( PITTPACKACC )
#pragma acc update device( Xbox [0:6] )
#endif
}

PittPackResult PencilDcmp::checkInput()
{
    PittPackResult result = SUCCESS;

    if ( p0 * p0 != comSize )
    {
        result = COMSIZE_FAIL;
    }

    return ( result );
}

void PencilDcmp::MPIStartUp()
{
    if ( MPI_SUCCESS != MPI_Comm_dup( MPI_COMM_WORLD, &Comm ) )
    {
        cout << BLUE << " Rank(" << myRank << ") > Exit Code : " << MPI_DUP_FAIL << RESET << endl;
        cout << BLUE << PittPackGetErrorEnum( MPI_DUP_FAIL ) << RESET << endl;
        exit( 1 );
    }

    if ( MPI_SUCCESS != MPI_Comm_rank( Comm, &myRank ) )
    {
        cout << BLUE << " Rank(" << myRank << ") > Exit Code : " << MPI_GET_RANK_FAIL << RESET << endl;
        cout << BLUE << PittPackGetErrorEnum( MPI_GET_RANK_FAIL ) << RESET << endl;
        exit( 1 );
    }

    if ( MPI_SUCCESS != MPI_Comm_size( Comm, &comSize ) )
    {
        cout << BLUE << " Rank(" << myRank << ") > Exit Code : " << MPI_COMSIZE_FAIL << RESET << endl;
        cout << BLUE << PittPackGetErrorEnum( MPI_COMSIZE_FAIL ) << RESET << endl;
        exit( 1 );
    }

    if ( MPI_ERROR_DISABLE == 1 )
    {
        if ( MPI_SUCCESS != MPI_Comm_set_errhandler( Comm, MPI_ERRORS_RETURN ) )
        {
            cout << BLUE << " Rank(" << myRank << ") > Exit Code : " << MPI_ERROR_HANDLE_FAIL << RESET << endl;
            cout << BLUE << PittPackGetErrorEnum( MPI_ERROR_HANDLE_FAIL ) << RESET << endl;
            exit( 1 );
        }
    }
}

PencilDcmp::~PencilDcmp()
{
    int initFlag;

    if ( Nbrs[0] != NULL )
    {
        delete[] Nbrs[0];
    }
    if ( Nbrs[1] != NULL )
    {
        delete[] Nbrs[1];
    }
    if ( jax != NULL )
    {
        delete[] jax;
    }

    if ( iax != NULL )
    {
        delete[] iax;
    }
    if ( jay != NULL )
    {
        delete[] jay;
    }

    if ( iay != NULL )
    {
        delete[] iay;
    }

    delete[] indices;
// freeing the resources should be in reverse order
#if ( PITTPACKACC )
#pragma acc exit data delete ( coords )
#pragma acc exit data delete ( tags )
#pragma acc exit data delete ( dxyz )
#pragma acc exit data delete ( omega )
#pragma acc exit data delete ( length )
#pragma acc exit data delete ( subDiag )
#pragma acc exit data delete ( supDiag )
#pragma acc exit data delete ( freqs )
#pragma acc exit data delete ( bc )
#pragma acc exit data delete ( faceTag )
#if ( USE_SHARED != 1 )
#pragma acc exit data delete ( tmpX )
#pragma acc exit data delete ( tmpY )
#endif
#pragma acc exit data delete ( Xbox )
#if ( MULTIGRIDON )
#pragma acc exit data delete ( tmpMGReal )
#pragma acc exit data delete ( tmpMGImag )
#endif
#pragma acc exit data delete ( du )
#pragma acc exit data delete ( dl )
#pragma acc exit data delete ( ds )
    if ( SOLUTIONMETHOD != 0 )
    {
//#pragma acc exit data delete ( crpcr_lower )
#pragma acc exit data delete ( crpcr_upper )
    }
#pragma acc exit data delete ( x1 )
#pragma acc exit data delete ( x2 )
#pragma acc exit data delete ( x3 )
#pragma acc exit data delete ( this )
#endif

    delete[] coords;
    delete[] dxyz;

    delete[] tmp;

#if ( MULTIGRIDON )
    delete[] tmpMGReal;
    delete[] tmpMGImag;
#endif
    delete[] subDiag;
    delete[] supDiag;
    delete[] length;
    delete[] omega;
    delete[] tags;
    delete[] num;
    delete[] denum;
    delete[] faceTag;
#if ( USE_SHARED != 1 )
    delete[] tmpX;
    delete[] tmpY;
#endif
    delete[] Xbox;

    delete[] dl;
    delete[] ds;
    delete[] du;
    delete[] x1;
    delete[] x2;
    delete[] x3;

    if ( SOLUTIONMETHOD != 0 )
    {
        //   delete [] crpcr_lower;
        delete[] crpcr_upper;
    }
    // free the duplicated communicator

    MPI_Comm_free( &Comm );

   /* if ( MPI_Finalized( &initFlag ) != MPI_SUCCESS )
    {
        cout << "failure in checking if MPI has already initialize" << endl;
        exit( 1 );
    }

    // cout<<" init flag "<<initFlag<<endl;
    if ( initFlag == 0 )
    {
        if ( MPI_Finalize() != MPI_SUCCESS )
        {
            cout << " Exit Code : " << PittPackGetErrorEnum( MPI_FINALIZE_FAIL ) << endl;
            exit( 1 );
        }
    }*/
    delete[] send_request;
    delete[] send_status;

    delete[] recv_request;
    delete[] recv_status;
}

//
// start with the zdirection and rotate and come back to it
//
void PencilDcmp::allocateChunks()
{
    int n[3]   = {nx / p0, ny / p0, nz};
    int nChunk = p0;

#if ( DISABLE_CHECKS )
    if ( n[0] * p0 != nx )
    {
        cout << BLUE << " Rank(" << myRank << ") > Exit Code : " << MESH_DIVISIBLE << RESET << endl;
        cout << BLUE << PittPackGetErrorEnum( MESH_DIVISIBLE ) << RESET << endl;
        exit( 1 );
    }
#endif
    if ( P.allocate( n, nChunk ) != SUCCESS )
    {
        cout << BLUE << " Rank(" << myRank << ") > Exit Code : " << ALLOCATION_FAIL << RESET << endl;
        cout << BLUE << PittPackGetErrorEnum( ALLOCATION_FAIL ) << RESET << endl;
        exit( 1 );
    }

    P.getChunkSize();

#if ( COMM_PATTERN != 0 )
    {
        if ( R.allocate( n, nChunk ) != SUCCESS )
        {
            cout << BLUE << " Rank(" << myRank << ") > Exit Code : " << ALLOCATION_FAIL << RESET << endl;
            cout << BLUE << PittPackGetErrorEnum( ALLOCATION_FAIL ) << RESET << endl;
            exit( 1 );
        }

#if ( 1 )
        cout << "regular neighborhood AlltoAll full recv nameAppendixfer is used" << endl;
        cout << " chunkSize" << P.getChunkSize() << endl;
#endif
    }
#else
    {
        if ( R.allocate( n, 1 ) != SUCCESS )
        {
            cout << BLUE << " Rank(" << myRank << ") > Exit Code : " << ALLOCATION_FAIL << RESET << endl;
            cout << BLUE << PittPackGetErrorEnum( ALLOCATION_FAIL ) << RESET << endl;
            exit( 1 );
        }

#if ( DEBUG )
        cout << "pairwise exchange-- only chunksize nameAppendixfer is used" << endl;
#endif
    }
#endif

#if ( DEBUG )
    cout << "Total chunkSize for P=" << P.getChunkSize() << endl;
    cout << "Total chunkSize for R=" << R.getChunkSize() << endl;
#endif
    P.setDirection( 2 );
    R.setDirection( 2 );
}

void PencilDcmp::getChunkSize()
{
    int size = nx * ny * nz;

    if ( size % ( p0 * p1 ) == 0 )
    {
        size = ( nx * ny * nz ) / ( p0 * p1 );

#if ( DEBUG )
        cout << "chunk size " << size << endl;
#endif
    }
    else
    {
        cout << "warning not divisible " << endl;
    }

    fullSize = size;

    chunkSize = size / p0;
#if ( PITTPACKACC )
#pragma acc update device( chunkSize )
#pragma acc update device( fullSize )
#endif
}

void PencilDcmp::setCoords( int dir )
{
    // input is the initial box

    double X[6];
    double dx = 0.0, dy = 0.0, dz = 0.0;

    // 0 means x direction
    // 1 means y direction
    // 2 means z direction
    // no stride use %
    // stride use /

    switch ( dir )
    {
        case 2:

            dx = ( Xbox[1] - Xbox[0] ) / double( p0 );
            dy = ( Xbox[3] - Xbox[2] ) / double( p0 );

            X[0] = Xbox[0] + ( myRank % p0 ) * dx;
            X[1] = X[0] + dx;
            X[2] = Xbox[2] + ( myRank / p0 ) * dy;
            X[3] = X[2] + dy;
            X[4] = Xbox[4];
            X[5] = Xbox[5];
            break;

        case 0:
            dz = ( Xbox[5] - Xbox[4] ) / double( p0 );
            dy = ( Xbox[3] - Xbox[2] ) / double( p0 );

            X[0] = Xbox[0];
            X[1] = Xbox[1];
            X[2] = Xbox[2] + ( myRank / p0 ) * dy;
            X[3] = X[2] + dy;
            X[4] = Xbox[4] + ( myRank % p0 ) * dz;
            X[5] = X[4] + dz;
            break;

        case 1:
            dx = ( Xbox[1] - Xbox[0] ) / double( p0 );
            dz = ( Xbox[5] - Xbox[4] ) / double( p0 );

            X[0] = Xbox[0] + (double)( myRank / p0 ) * dx;
            X[1] = X[0] + dx;
            X[2] = Xbox[2];
            X[3] = Xbox[3];
            X[4] = Xbox[4] + (double)( myRank % p0 ) * dz;
            X[5] = X[4] + dz;
            break;
    }
        // modifying this from previous to monitor move of the chunks for Debug

#if ( DEBUG )
    cout << GREEN " rank= " << myRank << " Xa= " << X[0] << " Xb= " << X[1] << " dx " << dx << endl;
    cout << " my rank= " << myRank << " Ya= " << X[2] << " Yb= " << X[3] << "dy " << dy << endl;
    cout << " my rank= " << myRank << " Za= " << X[4] << " Zb= " << X[5] << " dz " << dz << RESET << endl;
#endif
    for ( int i = 0; i < 6; i++ )
    {
        coords[i] = X[i];

#if ( DEBUG )
        cout << RED << " coords in dir " << dir << " " << coords[i] << RESET << endl;
#endif
    }

    P.setCoords( X );
    setDxyz( Xbox );

#if ( PITTPACKACC )
#pragma acc update device( dxyz [0:3] )
#pragma acc update device( coords [0:6] )
#pragma acc enter data copyin( X [0:6] )
#endif
}
// potential bug, need to incorporate directions
void PencilDcmp::setDxyz( double *X )
{
    dxyz[0] = -( X[0] - X[1] ) / ( nx );
    dxyz[1] = -( X[2] - X[3] ) / ( ny );
    dxyz[2] = -( X[4] - X[5] ) / ( nz );

    MG.setDelx( 1.0 );

#if ( DEBUG )
    for ( int i = 0; i < 3; i++ )
    {
        cout << "dxyz " << dxyz[i] << endl;
    }
#endif
}

void PencilDcmp::getNoOfChunk()
{
    // number of chunks that every processor has
    nChunk = p0;
#if ( PITTPACKACC )
#pragma acc update device( nChunk )
#endif
}

PittPackResult PencilDcmp::constructConnectivity()
{
    Nbrs[0] = new ( std::nothrow ) int[p0];
    Nbrs[1] = new ( std::nothrow ) int[p0];

    int id = myRank / p0;
    // int id1;
    // int counter = 0;

    for ( int i = 0; i < p0; i++ )
    {
        Nbrs[0][i] = id * p0 + i;
    }

#if ( DEBUG )
    for ( int i = 0; i < p0; i++ )
    {
        cout << "Nbr X-dir " << id << " myRank " << myRank << " nbrs " << Nbrs[0][i] << endl;
    }
#endif

    id = myRank % p0;

    for ( int i = 0; i < p0; i++ )
    {
        Nbrs[1][i] = id + i * p0;
    }

#if ( DEBUG )
    for ( int i = 0; i < p0; i++ )
    {
        cout << "Nbr-Ydir " << id << " myRank " << myRank << " nbrs with stride " << Nbrs[1][i] << endl;
    }
#endif
    if ( Nbrs[0] != NULL && Nbrs[1] != NULL )
    {
        return ( SUCCESS );
    }
    else
    {
        return ( CONNECTIVITY_CONSTRUCTION_FAIL );
    }
}

//#if ( GPUAWARE && COMM_PATTERN == 0 )
#if ( COMM_PATTERN == 0 )
void PencilDcmp::changeOwnershipPairwiseExchangeZX()
{
#if ( COMM_ON )
    MPI_Request request0, request1, request2, request3;
    MPI_Status  status;

#if ( DEBUG_COMM )
    std::string filename = "rank";
    filename.append( to_string( myRank ) );
    ofstream myfile;
    myfile.open( filename );

#endif

#if ( DEBUG )
    int dest[2] = {myRank + 1, myRank - 1};
    for ( int i = 0; i < 2; i++ )
    {
        cout << " myRank  " << myRank << " will send to  " << getPeriodicRank( dest[i] ) << endl;
    }
#endif
    int i = myRank;
#if ( DEBUG )
    cout << " loop= " << ( p0 + 1 ) / 2 << endl;
#endif

    int     start = 0;
    int     end   = 0;
    double *ptr   = P.P;
    for ( int j = 1; j < ( p0 ) / 2 + 1; j++ )
    {
        start = getPeriodicIndex( i + j ) * P.chunkSize;
        end   = P.chunkSize;
        ptr   = &P( 0 ) + start;
#if ( PITTPACKACC )
#pragma acc update self( ptr [0:end] ) async( 1 )
#endif
        //   cout<<"start "<<start<<" end "<<end<<" periodic index "<<  getPeriodicIndex( i + j ) <<" chunkSize "<<P.chunkSize<<" ptr "
        // <<ptr<<endl;

        MPI_Irecv( &R( 0 ), P.chunkSize, MPI_DOUBLE, getPeriodicRank( i - j ), getPeriodicRank( i - j ), Comm, &request1 );

#if ( PITTPACKACC )
#pragma acc wait( 1 )
#endif
        MPI_Isend( &P( 0 ) + P.chunkSize * getPeriodicIndex( i + j ), P.chunkSize, MPI_DOUBLE, getPeriodicRank( i + j ), myRank, Comm,
                   &request0 );

        /// once Isend completes move the chunk with (i-j) argument
        ptr = &P( 0 ) + getPeriodicIndex( i - j ) * P.chunkSize;
        end = P.chunkSize;

        MPI_Wait( &request0, &status );

// P.moveDeviceToHost( getPeriodicIndex( i - j ) );
#if ( PITTPACKACC )
#pragma acc update self( ptr [0:end] ) async( 2 )
#endif
        /// switch send/recv order

        MPI_Irecv( &P( 0 ) + P.chunkSize * getPeriodicIndex( i + j ), P.chunkSize, MPI_DOUBLE, getPeriodicRank( i + j ),
                   getPeriodicRank( i + j ), Comm, &request3 );

#if ( PITTPACKACC )
#pragma acc wait( 2 )
#endif
        MPI_Isend( &P( 0 ) + P.chunkSize * getPeriodicIndex( i - j ), P.chunkSize, MPI_DOUBLE, getPeriodicRank( i - j ), myRank, Comm,
                   &request2 );
        ///     R.moveHostToDevice( );

        //      P.moveHostToDevice( getPeriodicIndex( i + j ) );
        ptr = &P( 0 ) + getPeriodicIndex( i + j ) * P.chunkSize;
        end = P.chunkSize;

        MPI_Wait( &request3, &status );

#if ( PITTPACKACC )
#pragma acc update device( ptr [0:end] ) async( 3 )
#endif

        MPI_Wait( &request2, &status );
#if ( DEBUG_COMM )
        dicIndex( i - j ) << endl myfile << "**********************************" << endl;
        myfile << " ZX myRank " << myRank << " destination " << getPeriodicRank( i - j ) << " tag and chunk id "
               << getPeriodicIndex( i - j ) << endl;
        myfile << " ZX myRank " << myRank << " source " << getPeriodicRank( i + j ) << " chunk ID " << getPeriodicIndex( i + j ) << endl;
        myfile << "**********************************" << endl;
#endif

        MPI_Wait( &request1, &status );

#if ( R_COPY == 0 )
        // old version where we assigned on CPU and updated on GPU
        for ( int k = 0; k < P.chunkSize; k++ )
        {
            P( k + P.chunkSize * getPeriodicIndex( i - j ) ) = R( k );
        }

        //          P.moveHostToDevice( getPeriodicIndex( i - j ) );

        ptr = &P( 0 ) + getPeriodicIndex( i - j ) * P.chunkSize;
        end = P.chunkSize;

#if ( PITTPACKACC )
#pragma acc update device( ptr [0:end] )
#endif

#else
#pragma acc update device( R.P [0:end] )
#pragma acc data present( P.P, R.P )
#pragma acc parallel loop
        for ( int k = 0; k < P.chunkSize; k++ )
        {
            P( k + P.chunkSize * getPeriodicIndex( i - j ) ) = R( k );
        }

#endif
#if ( PITTPACKACC )
#pragma acc wait( 3 )
#endif
    }

    //        #pragma acc wait(4)

#if ( DEBUG )
    // cout<<"SUCCESS " <<endl;
    for ( int i = 0; i < nChunk; i++ )
    {
        cout << "ZX my rank " << myRank << " " << P( 0 + P.chunkSize * i ) << endl;
    }
#endif

#if ( DEBUG_COMM )
    myfile.close();
#endif

#endif
}
#endif

#if ( COMM_PATTERN == 0 )
void PencilDcmp::changeOwnershipPairwiseExchangeXY()
{
#if ( COMM_ON )
    MPI_Request request0, request1, request2, request3;
    MPI_Status  status;
    int         stride = p0;

    // uses too much memory, use pairwise exhchange instead
    //
    // network congestion

#if ( DEBUG_COMM )
    std::string filename = "rank";
    filename.append( to_string( myRank ) );
    ofstream myfile;
    myfile.open( filename );

#endif

    //  cout << " faster method Solving " << endl;

    int counter = 0;
    // MPI_Isend( &P(0), P.chunkSize, MPI_DOUBLE, getDestinationPeriodic(myRank),
    // myRank, Comm, &request1[0] );
    int i = myRank;
#if ( DEBUG )
    cout << " loop= " << ( p0 + 1 ) / 2 << endl;
#endif

    int     start = 0;
    int     end   = 0;
    double *ptr   = P.P;

    for ( int j = 1; j < ( p0 ) / 2 + 1; j++ )
    {
        start = getPeriodicIndexStride( j ) * P.chunkSize;
        end   = P.chunkSize;
        ptr   = &P( 0 ) + start;
#if ( PITTPACKACC )
#pragma acc update self( ptr [0:end] ) async( 1 )
#endif
        MPI_Irecv( &R( 0 ), P.chunkSize, MPI_DOUBLE, getPeriodicRankStride( -j ), getPeriodicRankStride( -j ), Comm, &request1 );

#if ( PITTPACKACC )
#pragma acc wait( 1 )
#endif

        MPI_Isend( &P( 0 ) + P.chunkSize * getPeriodicIndexStride( j ), P.chunkSize, MPI_DOUBLE, getPeriodicRankStride( j ), myRank, Comm,
                   &request0 );
        MPI_Wait( &request0, &status );

#if ( DEBUG_COMM )
        myfile << "**********************************" << endl;
        myfile << " j " << j << " myRank " << myRank << " destination " << getPeriodicRankStride( j ) << " chunk id "
               << getPeriodicIndexStride( j ) << " tag " << myRank << endl;
        myfile << "**********************************" << endl;
        myfile << " j " << j << " myRank " << myRank << " source " << getPeriodicRankStride( -j ) << " chunk id "
               << getPeriodicIndexStride( -j ) << " tag " << getPeriodicRankStride( -j ) << endl;
#endif
        // write directly to the chunk that was previously send

        counter = 0;

        ptr = &P( 0 ) + getPeriodicIndexStride( -j ) * P.chunkSize;
        end = P.chunkSize;
        // P.moveDeviceToHost( getPeriodicIndex( i - j ) );

#if ( PITTPACKACC )
#pragma acc update self( ptr [0:end] ) async( 2 )
#endif
        MPI_Irecv( &P( 0 ) + P.chunkSize * getPeriodicIndexStride( j ), P.chunkSize, MPI_DOUBLE, getPeriodicRankStride( j ),
                   getPeriodicRankStride( j ), Comm, &request3 );
#if ( PITTPACKACC )
#pragma acc wait( 2 )
#endif

        MPI_Isend( &P( 0 ) + P.chunkSize * getPeriodicIndexStride( -j ), P.chunkSize, MPI_DOUBLE, getPeriodicRankStride( -j ), myRank, Comm,
                   &request2 );

        ptr = &P( 0 ) + getPeriodicIndexStride( j ) * P.chunkSize;
        end = P.chunkSize;

        MPI_Wait( &request2, &status );
        MPI_Wait( &request3, &status );

#if ( PITTPACKACC )
#pragma acc update device( ptr [0:end] ) async( 3 )
#endif
        // transfer from container to the desired location
        // cout<<nChunk<<endl;

        MPI_Wait( &request1, &status );

#if ( R_COPY == 0 )
        for ( int k = 0; k < P.chunkSize; k++ )
        {
            P( k + P.chunkSize * getPeriodicIndexStride( -j ) ) = R( k );
        }
        ptr = &P( 0 ) + getPeriodicIndexStride( -j ) * P.chunkSize;
        end = P.chunkSize;
#if ( PITTPACKACC )
#pragma acc update device( ptr [0:end] ) async( 4 )
#endif
#else
        ptr = &R( 0 );
        end = R.chunkSize;
#pragma acc update device( R.P [0:end] ) async( 5 )

#if ( PITTPACKACC )
#pragma acc wait( 4, 5 )
#endif

//#pragma acc data present(P.P,R.P)
#pragma acc parallel loop
        for ( int k = 0; k < P.chunkSize; k++ )
        {
            P( k + P.chunkSize * getPeriodicIndexStride( -j ) ) = R( k );
        }

#endif
#if ( PITTPACKACC )
#pragma acc wait( 3 )
#endif

#if ( DEBUG_COMM )
        cout << " myRank " << myRank << " i " << i << " source " << getPeriodicRankStride( -j ) << " chunk ID "
             << getPeriodicIndexStride( -j ) << endl;
        cout << "**********************************" << endl;

        cout << " myRank " << myRank << " i " << i << " source " << getPeriodicRankStride( j ) << " chunk ID "
             << getPeriodicIndexStride( j ) << endl;
        cout << "**********************************" << endl;
#endif
    }
#endif

#if ( DEBUG_COMM )
    myfile.close();
#endif
}

#endif

#if ( PITTPACKACC )
#pragma acc routine
#endif
inline int PencilDcmp::getPeriodicRank( int index )
{
    //
    // return the rank in a periodic way
    // 0,1,2,3,4 as the ranks, if the index is 5, 0,1,2,3,4,0,1,2,3
    //

    adjustIndex( index );

    //  cout<<" myrank "<<myRank<<" new index "<<rank<<" nbr[index]"<<Nbrs[0][rank]<<endl;
    return ( ( myRank / p0 ) * p0 + index );
    // return ( Nbrs[0][index] );
}

inline int PencilDcmp::getIndex( int rank )
{
    int count = 0;

    count = rank / p0;

    return ( count );
}

inline void PencilDcmp::adjustIndex( int &index )
{
    int qt = index / p0;

    if ( index >= p0 )
    {
        index = index - qt * p0;
    }
    else if ( index < 0 )
    {
        qt    = ( index + 1 ) / p0;
        index = index + ( -qt + 1 ) * p0;
    }
}

inline int PencilDcmp::getPeriodicRankStride( int index )
{
    //
    // return the rank in a periodic way
    // 0,4,8,12 as the ranks, if the index is 0,4,8,12,0,4,8,12,0,4,8,12
    //

    // get the index according to myRank for strided neighbors

    int location = getIndex( myRank );

    index = location + index;

    adjustIndex( index );
#if ( DEBUG )
    cout << " stride myrank " << myRank << " new index " << index << " nbr[index]" << myRank % p0 + p0 * index << endl;
#endif
    // return ( Nbrs[stride][rank] );
    return ( myRank % p0 + p0 * index );
}

int PencilDcmp::getPeriodicIndexStride( int index )
{
    //
    // return the rank in a periodic way
    // 0,1,2,3,4 as the ranks, if the index is 5, 0,1,2,3,4,0,1,2,3
    //
    int location = getIndex( myRank );
    index        = location + index;

    // note that the index grows with a stride here
    // int qt=index/comSize;

    adjustIndex( index );
    // cout<<" myrank "<<myRank<<" new index "<<rank<<endl;

    return ( index );
}

int PencilDcmp::getPeriodicIndex( int index )
{
    //
    // return the rank in a periodic way
    // 0,1,2,3,4 as the ranks, if the index is 5, 0,1,2,3,4,0,1,2,3
    //

    // int qt=index/comSize;
    adjustIndex( index );
    // cout<<" myrank "<<myRank<<" new index "<<rank<<" nbr[index]
    // "<<nbrs[rank]<<endl;
    return ( index );
}
void PencilDcmp::graphCreate() /*!Two different communicators are required due
                                  to the presence of stride in X to Y to
                                  rotation */
{
    int indegree = p0;
    // symmetric communications and hence outdegree and indegree are the same
    // int outdegree = p0;
    int reorder = 0;
    int ierr;

    ierr = MPI_Dist_graph_create_adjacent( Comm, indegree, Nbrs[0], MPI_UNWEIGHTED, indegree, Nbrs[0], MPI_UNWEIGHTED, MPI_INFO_NULL,
                                           reorder, &nbrComm[0] );

    if ( ierr != MPI_SUCCESS )
    {
        cout << " Exit Code: " << MPI_DIST_GTAPH_FAIL_X << endl;
        exit( 1 );
    }

    ierr = MPI_Dist_graph_create_adjacent( Comm, indegree, Nbrs[1], MPI_UNWEIGHTED, indegree, Nbrs[1], MPI_UNWEIGHTED, MPI_INFO_NULL,
                                           reorder, &nbrComm[1] );

    if ( ierr != MPI_SUCCESS )
    {
        cout << " Exit Code: " << MPI_DIST_GTAPH_FAIL_Y << endl;
        exit( 1 );
    }

#if ( DEBUG )
    checkGraph( 0 );
    checkGraph( 1 );
#endif
}

void PencilDcmp::checkGraph( int index )
{
    // int nneighbors = 0;
    int indegree  = 0;
    int outdegree = 0;

    if ( index > 1 )
    {
        printf( "unacceptable value for i, it is either zero or one" );
        exit( 0 );
    }

    int weight;

    MPI_Dist_graph_neighbors_count( nbrComm[index], &indegree, &outdegree, &weight );

    int *neighbors = new int[indegree];
    // nneighbors     = indegree;

#if ( DEBUG )
    cout << " nneighbors is = " << nneighbors << endl;
    if ( nneighbors != p0 )
    {
        cout << " nneighbors is = " << nneighbors << endl;
        exit( 0 );
    }
#endif
    int *sources       = new int[indegree];
    int *sourceweights = new int[indegree];
    int *destinations  = new int[outdegree];
    int *destweights   = new int[outdegree];

    MPI_Dist_graph_neighbors( nbrComm[index], indegree, sources, sourceweights, outdegree, destinations, destweights );

#if ( DEBUG )
    if ( myRank == 9 )
    {
        for ( int i = 0; i < nneighbors; i++ )
        {
            cout << " myRank " << myRank << " sources " << sources[i] << endl;
            if ( Nbrs[index][i] != sources[i] )
            {
                cout << " error in neighborhood " << endl;
                exit( 0 );
            }
        }
    }
#endif
    delete[] neighbors;
    delete[] sources;
    delete[] sourceweights;
    delete[] destinations;
    delete[] destweights;
}

void PencilDcmp::nbrAllToAllZX()
{
    // for All to allV

#if ( PITTPACKACC )
#pragma acc parallel loop num_gangs( 1024 )
#endif
    for ( int l = 0; l < nChunk * R.chunkSize; l++ )
    {
        R( l ) = P( l );
    }

#if ( DEBUG )
    int *sndCnts = new int[p0];
    int *disp    = new int[p0];

    for ( int i = 0; i < p0; i++ )
    {
        if ( Nbrs[0][i] != myRank )
        {
            sndCnts[i] = P.chunkSize;
        }
        else
        {
            sndCnts[i] = 0;
        }
    }

    for ( int i = 0; i < p0; i++ )
    {
        cout << " myrank " << myRank << " sendcounts " << sndCnts[i] << endl;
    }
#endif
    double *ptr = NULL;
    int     end = P.chunkSize;
    for ( int i = 0; i < p0; i++ )
    {
        ptr = &R( 0 ) + end * i;
#if ( PITTPACKACC )
#pragma acc update self( ptr [0:end] ) async( i )
#endif
    }

    MPI_Request request;
    MPI_Status  status;
    int         ierr;
// wait for all async copies to finish
#if ( PITTPACKACC )
    acc_wait_all();
#endif
    /*
       for ( int i = 0; i < p0; i++ )
        {
    #pragma acc wait(i)
        }
    */

    // ierr = MPI_Ineighbor_alltoall( &P( 0 ), P.chunkSize, MPI_DOUBLE, &R( 0 ), P.chunkSize, MPI_DOUBLE, nbrComm[0], &request );
    ierr = MPI_Ineighbor_alltoall( &R( 0 ), P.chunkSize, MPI_DOUBLE, &P( 0 ), P.chunkSize, MPI_DOUBLE, nbrComm[0], &request );
    // blocking fails   ierr = MPI_Neighbor_alltoall( &P( 0 ), P.chunkSize, MPI_DOUBLE, MPI_IN_PLACE, P.chunkSize, MPI_DOUBLE, nbrComm[0] );
    //  nonblocking also fails   ierr = MPI_Ineighbor_alltoall( &P( 0 ), P.chunkSize, MPI_DOUBLE, MPI_IN_PLACE, P.chunkSize, MPI_DOUBLE,
    // nbrComm[0], &request );
    //   cout<<RED<<" neighborhood collective "<<RESET<<endl;

    if ( ierr != MPI_SUCCESS )
    {
        cout << " Exit Code: " << MPI_INEIGHBOR_FAIL_ZX << endl;
        exit( 1 );
    }

    MPI_Wait( &request, &status );
    /*
     for ( int i = 0; i < nChunk * P.chunkSize; i++ )
     {
         P( i ) = R( i );
     }
 */
    for ( int i = 0; i < p0; i++ )
    {
        ptr = &P( 0 ) + end * i;

#if ( PITTPACKACC )
#pragma acc update device( ptr [0:end] ) async( i )
#endif
    }
    /*
       for ( int i = 0; i < p0; i++ )
        {
    #pragma acc wait(i)
        }
    */

#if ( PITTPACKACC )
    acc_wait_all();
#endif

#if ( 0 )
    for ( int i = 0; i < nChunk; i++ )
    {
        cout << "ZX rank " << myRank << " after transformation " << R( 0 + P.chunkSize * i ) << " " << P( 0 + P.chunkSize * i ) << endl;
    }
#endif
}

void PencilDcmp::nbrAllToAllZXOverlap()
{
    // for All to allV

    double *ptr = NULL;
    int     end = P.chunkSize;
    // int     ierr;

    //
    // post recieves
    //

    for ( int i = 0; i < ( p0 ); i++ )
    {
        // need to pull from device to host before sending
        // some update self directives ,

        ptr = &P( 0 ) + P.chunkSize * i;
#if ( PITTPACKACC )
#pragma acc update self( ptr [0:end] ) async( i + 1 )
#endif
        MPI_Irecv( &R( 0 ) + R.chunkSize * i, R.chunkSize, MPI_DOUBLE, Nbrs[0][i], Nbrs[0][i], Comm, &( recv_request[i] ) );

        // cout<<" Myrank  "<<myRank<<"recieving message to  Nbrs  "<<i <<" "<< Nbrs[0][i]<<" P0 "<<p0<<endl;
    }

    // count=0;

    for ( int i = 0; i < p0; i++ )
    {
#if ( PITTPACKACC )
        acc_async_wait( i + 1 );
#endif

        MPI_Isend( &P( 0 ) + P.chunkSize * i, P.chunkSize, MPI_DOUBLE, Nbrs[0][i], myRank, Comm, &( send_request[i] ) );

        // cout<<" Myrank  "<<myRank<<"sending message from  Nbrs  "<<i <<" "<< Nbrs[0][i]<<" P0 "<<p0<<endl;
    }
#if ( 1 )
    int indx;
    for ( int i = 0; i < p0; i++ )
    {
        MPI_Waitany( p0, recv_request, &indx, recv_status );
        // cout<<"my Rank "<<indx<<endl; ;
        indices[i] = indx;
        ptr        = &R( 0 ) + R.chunkSize * indx;
#if ( PITTPACKACC )
#pragma acc update device( ptr [0:end] ) async( p0 + i + 1 )
#endif
    }
#endif
    int id;
#if ( 1 )
    for ( int i = 0; i < p0; i++ )
    {
        id   = indices[i] + p0 + 1;
        indx = indices[i];
#if ( PITTPACKACC )
        acc_async_wait( id );
#endif
        MPI_Wait( &( send_request[indx] ), &( send_status[indx] ) );
#if ( PITTPACKACC )
#pragma acc data present( P.P, R.P, this ) copyin( indx )
#pragma acc parallel loop num_gangs( 1024 )
#endif
        for ( int l = 0; l < P.chunkSize; l++ )
        {
            P( P.chunkSize * indx + l ) = R( R.chunkSize * indx + l );
        }
    }
//    MPI_Waitall( p0, send_request, send_status );
#else

    MPI_Waitall( p0, send_request, send_status );
#if ( PITTPACKACC )
    acc_wait_all();
#endif
#endif
    /*
    #pragma acc data present( P.P, R.P, this )
    #pragma acc parallel loop num_gangs( 1024 )
        for ( int l = 0; l < nChunk * P.chunkSize; l++ )
        {
            P( l ) = R( l );
        }
    */
    // update device
    //

    // send
    //
    // if recieved push to device
#if ( 0 )
    for ( int i = 0; i < nChunk; i++ )
    {
        cout << "ZX rank " << myRank << " after transformation " << R( 0 + P.chunkSize * i ) << " " << P( 0 + P.chunkSize * i ) << endl;
    }
#endif
}

void PencilDcmp::nbrAllToAllXY()
{
// for All to allV
#if ( DEBUG )
    int *sndCnts = new int[p0];
    int *disp    = new int[p0];

    for ( int i = 0; i < p0; i++ )
    {
        if ( Nbrs[0][i] != myRank )
        {
            sndCnts[i] = P.chunkSize;
        }
        else
        {
            sndCnts[i] = 0;
        }
    }

    for ( int i = 0; i < p0; i++ )
    {
        cout << " myrank " << myRank << " sendcounts " << sndCnts[i] << endl;
    }
#endif

    for ( int i = 0; i < nChunk; i++ )
    {
        //      cout << " second my rank " << myRank << " before transformation " << R( 0 + P.chunkSize * i ) << " " << P( 0 + P.chunkSize *
        // i )
        //         << endl;
    }

#if ( PITTPACKACC )
#pragma acc parallel loop
#endif
    for ( int l = 0; l < nChunk * R.chunkSize; l++ )
    {
        R( l ) = P( l );
    }

    double *ptr = NULL;
    int     end = P.chunkSize;
    for ( int i = 0; i < p0; i++ )
    {
        // ptr=&P(0)+end*i;
        ptr = &R( 0 ) + end * i;
#if ( PITTPACKACC )
#pragma acc update self( ptr [0:end] ) async( i )
#endif
    }

    // MPI_Neighbor_alltoallv(&P(0), sndCnts, int sdispls[], MPI_DOUBLE, recvnameAppendix,
    // sndCnts, int rdispls[], MPI_DOUBLE ,nbrComm0, MPI_Request *request);
    // in the present algorithm for simplicity ever processor sends message to itself, making a blocking allto all might deadlock

    MPI_Request request;
    MPI_Status  status;
    int         ierr;
    // wait for all async copies to finish

#if ( PITTPACKACC )
    acc_wait_all();
#endif
    /*
       for ( int i = 0; i < p0; i++ )
        {
    #pragma acc wait(i)
        }
    */

    // ierr = MPI_Ineighbor_alltoall( &P( 0 ), P.chunkSize, MPI_DOUBLE, &R( 0 ), P.chunkSize, MPI_DOUBLE, nbrComm[1], &request );
    ierr = MPI_Ineighbor_alltoall( &R( 0 ), P.chunkSize, MPI_DOUBLE, &P( 0 ), P.chunkSize, MPI_DOUBLE, nbrComm[1], &request );

    if ( ierr != MPI_SUCCESS )
    {
        cout << " Exit Code: " << MPI_INEIGHBOR_FAIL_XY << endl;
        exit( 1 );
    }

    MPI_Wait( &request, &status );
    /*
        for ( int i = 0; i < P.chunkSize * nChunk; i++ )
        {
            P( i ) = R( i );
        }
    */
    for ( int i = 0; i < p0; i++ )
    {
        ptr = &P( 0 ) + end * i;
#if ( PITTPACKACC )
#pragma acc update device( ptr [0:end] ) async( i )
#endif
    }

#if ( PITTPACKACC )
    acc_wait_all();
#endif
    //#pragma acc_wait_all( );
    /*
       for ( int i = 0; i < p0; i++ )
        {
    #pragma acc wait(i)
        }
    */

#if ( 0 )
    for ( int i = 0; i < nChunk; i++ )
    {
        cout << " XY " << myRank << " after transformation " << R( 0 + P.chunkSize * i ) << " " << P( 0 + P.chunkSize * i ) << endl;
    }
#endif
}

#if ( COMM_PATTERN == 1 )

void PencilDcmp::changeOwnershipPairwiseExchangeZX()
{
    //  cout << GREEN << " nbrAllToAll ZX " << RESET << endl;
    nbrAllToAllZX();
}

void PencilDcmp::changeOwnershipPairwiseExchangeXY()
{
    // cout << GREEN << " nbrAllToAll XY " << RESET << endl;
    nbrAllToAllXY();
}

#endif

#if ( COMM_PATTERN == 2 )

void PencilDcmp::changeOwnershipPairwiseExchangeZX()
{
    //  cout << GREEN << " nbrAllToAll ZX " << RESET << endl;
    nbrAllToAllZXOverlap();
    // nbrAllToAllZX();
}
// keep the same one for now
void PencilDcmp::changeOwnershipPairwiseExchangeXY()
{
    // cout << GREEN << " nbrAllToAll XY " << RESET << endl;
    // nbrAllToAllXY();
    nbrAllToAllXYOverlap();
}

#endif

void PencilDcmp::nbrAllToAllXYOverlap()
{
    // for All to all

    //    int *indices=(int *) new int[p0];

#if ( 1 )

    double *ptr = NULL;
    int     end = P.chunkSize;
    // int     ierr;

    // part A
    for ( int i = 0; i < ( p0 ); i++ )
    {
        // pull from GPU to CPU,

        ptr = &P( 0 ) + P.chunkSize * i;

#if ( PITTPACKACC )
#pragma acc update self( ptr [0:end] ) async( i + 1 )
#endif
        // post recieves

        MPI_Irecv( &R( 0 ) + R.chunkSize * i, R.chunkSize, MPI_DOUBLE, Nbrs[1][i], Nbrs[1][i], Comm, &( recv_request[i] ) );

        // cout<<" Myrank  "<<myRank<<"recieving message to  Nbrs  "<<i <<" "<< Nbrs[0][i]<<" P0 "<<p0<<endl;
    }

    // count=0;

    for ( int i = 0; i < p0; i++ )
    {
// make sure copy to GPU is complete
#if ( PITTPACKACC )
        acc_async_wait( i + 1 );
#endif
        // send immediately
        MPI_Isend( &P( 0 ) + P.chunkSize * i, P.chunkSize, MPI_DOUBLE, Nbrs[1][i], myRank, Comm, &( send_request[i] ) );

        // cout<<" Myrank  "<<myRank<<"sending message from  Nbrs  "<<i <<" "<< Nbrs[0][i]<<" P0 "<<p0<<endl;
    }

// part B
#if ( 1 )
    // MPI_Waitall();
    int indx;
    for ( int i = 0; i < p0; i++ )
    {
        // check and see if any of the messages have arrived
        MPI_Waitany( p0, recv_request, &indx, recv_status );
        indices[i] = indx;
        ptr        = &R( 0 ) + R.chunkSize * indx;
// update the device immediately
#if ( PITTPACKACC )
#pragma acc update device( ptr [0:end] ) async( p0 + i + 1 )
#endif
    }
#endif

    int id;

// at the previous loop, I save the messages according to their arrival order
// assuming that the stream which started first can complete first
//  we wait for the message in the same order to assign to P
#if ( 1 )
    for ( int i = 0; i < p0; i++ )
    {
        //        MPI_Waitany( p0, send_request, &indx, send_status );

        id   = p0 + indices[i] + 1;
        indx = indices[i];
#if ( PITTPACKACC )
        acc_async_wait( id );
#endif
        // CPU version would have problems if we do not make sure send operation is completed before assigning it to R
        MPI_Wait( &( send_request[indx] ), &( send_status[indx] ) );

#if ( PITTPACKACC )
#pragma acc data present( P.P, R.P, this ) copyin( indx )
#pragma acc parallel loop num_gangs( 1024 )
#endif
        for ( int l = 0; l < P.chunkSize; l++ )
        {
            P( P.chunkSize * indx + l ) = R( R.chunkSize * indx + l );
        }
    }

//    MPI_Waitall( p0, send_request, send_status );
#else
    MPI_Waitall( p0, send_request, send_status );

#if ( PITTPACKACC )
    acc_wait_all();
#endif

#endif

#endif

    // update device
    //
    // send
    //
    // if recieved push to device
#if ( 0 )
    for ( int i = 0; i < nChunk; i++ )
    {
        cout << "ZX rank " << myRank << " after transformation " << R( 0 + P.chunkSize * i ) << " " << P( 0 + P.chunkSize * i ) << endl;
    }
#endif
}

void PencilDcmp::IO( int app, int dir, int aligndir )
{
    Phdf5 IO;

    IO.writeMultiBlockCellCenter( P, app, dir, aligndir );
}

#if ( PITTPACKACC )
#pragma acc routine seq
#endif
static double exactValue( double x, int tg )
{
    double ans = 0.0;

    if ( tg == 0 )
    {
        ans = ( sine( x ) );
    }
    else if ( tg == 1 )
    {
        ans = ( cosine( x ) );
    }
    else if ( tg == -1 )
    {
        ans = ( sine( x ) );
    }
    else if ( tg == -2 )
    {
        ans = ( cosine( x ) );
    }

    // cout<< "exact value "<<tg <<endl;
    return ( ans );
}


#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void CFlowVariable::initializeZero()
{
    #if ( PITTPACKACC )
#pragma acc loop seq
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
           
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 0; i < nxChunk; i++ )

                {
                    //          P( id, 1, i, j, k, 0 ) = 1.0;
                    P( id, 1, i, j, k, 0 ) = 0;
                    P( id, 1, i, j, k, 1 ) = 0;
                }
            }
        }
    }
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::initializeTrigonometric()
{
    double pi = 4. * arctan( 1.0 );
    double x, y, z;

    double Xa = coords[0];
    double Ya = coords[2];
    double Za = coords[4];
#if ( DEBUG )
    cout << BLUE << "rank " << myRank << " Xa " << Xa << " Ya " << Ya << " Za " << Za << RESET << endl;
#endif
    double c1, c2, c3;
    double shift = SHIFT;

    int Nx = nxChunk;
    int Ny = nyChunk;
    int Nz = nz;

    if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( Nx + 1. );
        c2 = ( coords[3] - coords[2] ) / ( Ny + 1. );
        c3 = ( coords[5] - coords[4] ) / ( Nz + 1. );
#if ( DEBUG )
        cout << " c1  " << c1 << " " << c2 << " " << c3 << endl;
#endif
    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }
    //   c3 = dxyz[2];

#if ( DEBUG )
    cout << " rank " << myRank << " c1  " << c1 << " " << c2 << " " << c3 << endl;
    cout << " rank " << myRank << " N:  " << Nx << " " << Ny << " " << Nz << endl;
    cout << "xmin = " << Xa + 0 * c1 + shift * c1 * .5 << " xmax " << Xa + ( Nx - 1 ) * c1 + shift * c1 * .5 << endl;
    ;
#endif
    double omega[3] = {COEFF0 * pi, COEFF1 * pi, COEFF2 * pi};

//   cout<<" omegas "<< omega[0]<<" "<<omega[1]<<" "<<omega[2]<<endl;
#if ( 1 )

#if ( PITTPACKACC )
#pragma acc loop gang private( x, y, z )
#endif
    for ( int k = 0; k < Nz; k++ )
    {
        if ( !SHIFT )
        {
            z = Za + ( k + 1 ) * c3;
        }
        else
        {
            z = Za + k * c3 + shift * c3 * 0.5;
        }

#if ( PITTPACKACC )
#pragma acc loop worker private( y )
#endif
        for ( int j = 0; j < Ny; j++ )
        {
            if ( !SHIFT )
            {
                y = Ya + ( j + 1 ) * c2;
            }
            else
            {
                y = Ya + j * c2 + shift * c2 * 0.5;
            }

#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int i = 0; i < Nx; i++ )
            {
                if ( !SHIFT )
                {
                    x = Xa + ( i + 1 ) * c1;
                }
                else
                {
                    x = Xa + i * c1 + shift * c1 * .5;
                }

                // cout << " rank "<<myRank <<" xyz  " << x << " " << y << " " << z << endl;

#if ( !EXACT )

                //                    P( i, j, k )=(i+1)*(i+1);
                /*
                                    P( i, j, k )
                                    = ( ( omega[0] * omega[0] ) * exactValue( omega[0] * x, tags[0] ) * exactValue( omega[1] * y, tags[1] )
                                        * exactValue( omega[2] * z, tags[2] ) + ( omega[1] * omega[1] ) * exactValue( omega[0] * x, tags[0]
                   )
                                                                                * exactValue( omega[1] * y, tags[1] ) * exactValue( omega[2]
                   * z, tags[2] )
                                        + ( omega[2] * omega[2] ) * exactValue( omega[0] * x, tags[0] ) * exactValue( omega[1] * y, tags[1]
                   )
                                          * exactValue( omega[2] * z, tags[2] ) ) * c3 * c3;

                                    P( i, j, k, 1 ) = 0.0;
                */

                //  initialize periodic in x and y and dirichlet in z all with same frequency, pay attention to satisfy periodicity and
                // set omega as multiples of 2*pi
                // PPPPXX

                if ( bc[0] == 'P' || bc[2] == 'P' )
                {
                    P( i, j, k, 0 ) = -3. * omega[1] * omega[1] * ( sine( omega[1] * z ) * cosine( omega[1] * ( x + y ) ) ) * c3 * c3;
                    P( i, j, k, 1 ) = -3. * omega[1] * omega[1] * ( sine( omega[1] * z ) * sine( omega[1] * ( x + y ) ) ) * c3 * c3;
                }
                else
                {
                    //  For big mesh sizes calling class operator calls problems
                    P( i, j, k, 0 ) = -( ( omega[0] * omega[0] ) * exactValue( omega[0] * x, tags[0] ) * exactValue( omega[1] * y, tags[1] )
                                         * exactValue( omega[2] * z, tags[2] )
                                         + ( omega[1] * omega[1] ) * exactValue( omega[0] * x, tags[0] )
                                           * exactValue( omega[1] * y, tags[1] ) * exactValue( omega[2] * z, tags[2] )
                                         + ( omega[2] * omega[2] ) * exactValue( omega[0] * x, tags[0] )
                                           * exactValue( omega[1] * y, tags[1] ) * exactValue( omega[2] * z, tags[2] ) )
                                      * c3 * c3;

                    P( i, j, k, 1 ) = 0.0;
#if ( DEBUG )
                    cout << P( i, j, k ) << '\t';
#endif
                }
                //   P(i,j,k,0)=(x*y*z);
                //   P(i,j,k,1)=(x*y*z);

                // all peridic meaning PPPPP

                if ( bc[0] == 'P' && bc[2] == 'P' && bc[4] == 'P' )
                {
                    P( i, j, k, 0 ) = -3. * omega[1] * omega[1] * ( cosine( omega[1] * ( x + y + z ) ) ) * c3 * c3;
                    P( i, j, k, 1 ) = -3. * omega[1] * omega[1] * ( sine( omega[1] * ( x + y + z ) ) ) * c3 * c3;
                }

#else
                //           P(i,j,k,0)=(sin(omega[1]*z)*cos(omega[1]*(x+y)));
                //           P(i,j,k,1)=(sin(omega[1]*z)*sin(omega[1]*(x+y)));

                //                  P( i, j, k, 0 ) = cos( omega[1] * ( x + y + z ) );
                //                  P( i, j, k, 1 ) = sin( omega[1] * ( x + y + z ) );
                P( i, j, k )
                = -exactValue( omega[0] * x, tags[0] ) * exactValue( omega[1] * y, tags[1] ) * exactValue( omega[2] * z, tags[2] );

                // all periodic

#endif
            }
#if ( DEBUG )
            cout << endl;
#endif
        }
    }
    // P.moveHostToDevice();

#endif
    /*
        if ( bc[0] != 'P' && bc[2] != 'P' )
        {
           // cout << RED << " enforcing Dirichlet " << endl;
         //   modifyRhsDirichlet();
        }
    */
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
double PencilDcmp::getError()
{
    double x, y, z;
    double Xa = coords[0];
    double Ya = coords[2];
    double Za = coords[4];
#if ( DEBUG )
    cout << BLUE << "rank " << myRank << " Xa " << Xa << " Ya " << Ya << " Za " << Za << RESET << endl;
#endif
    double c1, c2, c3;
    double shift = SHIFT;

    int Nx = nx / p0;
    int Ny = ny / p0;
    int Nz = nz;

    int unitSize = nxChunk * nyChunk * nzChunk * nChunk;
    if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( Nx + 1. );
        c2 = ( coords[3] - coords[2] ) / ( Ny + 1. );
        c3 = ( coords[5] - coords[4] ) / ( Nz + 1. );
#if ( DEBUG )
        cout << " c1  " << c1 << " " << c2 << " " << c3 << endl;
#endif
    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }

#if ( DEBUG )
    cout << " rank " << myRank << " c1  " << c1 << " " << c2 << " " << c3 << endl;
    cout << " rank " << myRank << " N:  " << Nx << " " << Ny << " " << Nz << endl;
#endif
    double omega[3] = {COEFF0 * pi, COEFF1 * pi, COEFF2 * pi};

    double val  = 0.0;
    double val1 = 0.0;
#if ( 1 )
#if ( PITTPACKACC )
#pragma acc loop gang private( x, y, z, val, val1 )
#endif
    for ( int k = 0; k < Nz; k++ )
    {
        if ( !SHIFT )
        {
            z = Za + ( k + 1 ) * c3;
        }
        else
        {
            z = Za + k * c3 + shift * c3 * 0.5;
        }

#if ( PITTPACKACC )
#pragma acc loop worker
#endif
        for ( int j = 0; j < Ny; j++ )
        {
            if ( !SHIFT )
            {
                y = Ya + ( j + 1 ) * c2;
            }
            else
            {
                y = Ya + j * c2 + shift * c2 * 0.5;
            }

#if ( PITTPACKACC )
#pragma acc loop vector private( val, val1 )
#endif
            for ( int i = 0; i < Nx; i++ )
            {
                if ( !SHIFT )
                {
                    x = Xa + ( i + 1 ) * c1;
                }
                else
                {
                    x = Xa + i * c1 + shift * c1 * .5;
                }

                // Sn( i, j, k ) = - exactValue( omega[0] * x,tags[0] ) * exactValue( omega[1] * y,tags[1] ) * exactValue( omega[2] *
                // z,tags[2] );

                /*
                                                 val=P(i,j,k,0)-(sin(omega[1]*z)*cos(omega[1]*(x+y)));
                                                 val1=P(i,j,k,1)-(sin(omega[1]*z)*sin(omega[1]*(x+y)));
                  */
                // P(i,j,k)+=omega[1]*omega[1]*(sin(omega[1]*z)*sin(omega[1]*(x+y)));
                if ( bc[0] == 'P' || bc[2] == 'P' )
                {
                    val  = P( i, j, k, 0 ) - ( sine( omega[1] * z ) * cosine( omega[1] * ( x + y ) ) );
                    val1 = P( i, j, k, 1 ) - ( sine( omega[1] * z ) * sine( omega[1] * ( x + y ) ) );
                }
                else
                {
                    val = P( i, j, k )
                          - exactValue( omega[0] * x, tags[0] ) * exactValue( omega[1] * y, tags[1] ) * exactValue( omega[2] * z, tags[2] );

                    val1 = 0.0;
                }

                if ( bc[0] == 'P' && bc[2] == 'P' && bc[4] == 'P' )
                {
                    val  = P( i, j, k, 0 ) - cosine( omega[1] * ( x + y + z ) );
                    val1 = P( i, j, k, 1 ) - sine( omega[1] * ( x + y + z ) );

#if ( 0 )
                    //#if ( DEBUG )
                    if ( ( k == ( Nz - 1 ) ) && ( i == 0 && j == 0 ) )
                    {
                        cout << "( i, " << i << " j, " << j << " k " << k << ") " << P( i, j, k, 0 ) << " + " << P( i, j, k, 1 ) << endl;
                        cout << "( i, " << i << " j, " << j << " k " << k << ") " << cosine( omega[1] * ( x + y + z ) ) << " + "
                             << sine( omega[1] * ( x + y + z ) ) << endl;
                        cout << "( i, " << i << " j, " << j << " k " << k << ") " << x << " , " << y << " y " << y << " " << z << endl;
                    }
#endif
                }
                //                  cout<<"omega "<<omega[1]<<endl;

                // cout<<"val 1 "<<val1<<endl;
                //   cout << RED << " myRank " << myRank << " x= " << x << " y= " << y << " z= " << z << " P= " << P( i, j, k ) << RESET
                //

                P( i, j, k ) = ( val * val + val1 * val1 );

                //        << endl;
                //  P( i, j, k ) = ( exactValue( pi * x, tags[0] ) );
                //   cout<<" P "<<( exactValue( pi * x, tags[0] ) )<<endl;
            }
        }
    }

    double err = 0.0;

#if ( PITTPACKACC )
#pragma acc loop vector firstprivate( err ) reduction( + : err )
#endif
    for ( int i = 0; i < unitSize; i++ )
    {
        err = err + P( 2 * i );
    }
    err = err / ( unitSize );
    err = squareRoot( err );

    /*
    double finalErr = 0.0;
        MPI_Allreduce( &err, &finalErr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

        cout << YELLOW << "Error  (" << myRank << ") =" << err << " " << finalErr << RESET << endl;
       cout << YELLOW << "Error Per processor"
           << " " << finalErr / p0 / p0 << RESET << endl;
    */
    ////   finalErr=finalErr/p0/p0;
    return ( err );
#endif
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::debug()
{
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
    for ( int i = 0; i < 6; i++ )
    {
        P( i ) = scale;
    }
}

#if ( REVTRSP == 0 )
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::rearrangeX2Y()
{
    double tmp[2 * NXCHUNK1 * NYCHUNK1];

    /*
    #if ( PITTPACKACC )
        double tmp[2 * NXCHUNK * NYCHUNK];
    #else

        double tmp[2 * nxChunk * nyChunk];
    #endif
    */
    // rearranges x to y

#if ( PITTPACKACC )
#pragma acc loop seq
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
            assignTempX2Y( id, k, tmp );
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 0; i < nxChunk; i++ )

                {
                    //          P( id, 1, i, j, k, 0 ) = 1.0;
                    P( id, 1, i, j, k, 0 ) = tmp[2 * ( nxChunk * j + i )];
                    P( id, 1, i, j, k, 1 ) = tmp[2 * ( nxChunk * j + i ) + 1];
                }
            }
        }
    }
}
//#endif

#else

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::rearrangeX2Y()
{
#if ( PITTPACKACC )
#pragma acc loop seq
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
        assignTempX2Yv1( id );
        assignBackTempX2Y( id );
    }
}

#endif

#if ( PITTPACKACC )
#pragma acc routine worker
#endif
void PencilDcmp::assignTempX2Y( const int chunkId, const int k, double *tmp )
{
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
    for ( int j = 0; j < nyChunk; j++ )
    {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
        for ( int i = 0; i < nxChunk; i++ )
        {
            tmp[2 * ( nxChunk * j + i )]     = P( chunkId, 0, i, j, k, 0 );
            tmp[2 * ( nxChunk * j + i ) + 1] = P( chunkId, 0, i, j, k, 1 );
        }
    }
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::assignTempX2Yv1( const int chunkId )
{
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
    for ( int k = 0; k < nzChunk; k++ )
    {
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
        for ( int j = 0; j < nyChunk; j++ )
        {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int i = 0; i < nxChunk; i++ )
            {
                //   tmp[2 * ( nxChunk * j + i )] = P( chunkId, 0, i, j, k, 0 );
                //   tmp[2 * ( nxChunk * j + i ) + 1] = P( chunkId, 0, i, j, k, 1 );
                R( i, j, k, 0 ) = P( chunkId, 0, i, j, k, 0 );
                R( i, j, k, 1 ) = P( chunkId, 0, i, j, k, 1 );

                //    R( 2*(nxChunk*j+i) ) = P( chunkId, 0, i, j, k, 0 );
                //    R( 2*(nxChunk*j+i)+1 ) = P( chunkId, 0, i, j, k, 1 );

                //   tmp[2 * ( nxChunk * j + i ) + 1] = P( chunkId, 0, i, j, k, 1 );
            }
        }
    }
}
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::assignBackTempX2Y( const int chunkId )
{
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
    for ( int k = 0; k < nzChunk; k++ )
    {
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
        for ( int j = 0; j < nyChunk; j++ )
        {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int i = 0; i < nxChunk; i++ )
            {
                //          P( chunkId, 1, i, j, k, 0 )=i+3*j;

                P( chunkId, 1, i, j, k, 0 ) = R( i, j, k, 0 );
                P( chunkId, 1, i, j, k, 1 ) = R( i, j, k, 1 );

                //                P( chunkId, 1, i, j, k, 0 )  =R(2 * ( nxChunk * j + i+ nxChunk*nyChunk*k ));
                //                 P( chunkId, 1, i, j, k, 1 )  =R(2 * ( nxChunk * j + i+nxChunk*nyChunk*k )+1);
                //   P(i,j,k,0)= 0;
                //    R(i,j,k,1)= 0;
                //   tmp[2 * ( nxChunk * j + i ) + 1] = P( chunkId, 0, i, j, k, 1 );
            }
        }
    }
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::assignTempY2X( const int chunkId )
{
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
    for ( int k = 0; k < nzChunk; k++ )
    {
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
        for ( int i = 0; i < nxChunk; i++ )
        {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
                //   tmp[2 * ( nxChunk * j + i )] = P( chunkId, 0, i, j, k, 0 );
                //   tmp[2 * ( nxChunk * j + i ) + 1] = P( chunkId, 0, i, j, k, 1 );
                R( i, j, k, 1, 0 ) = P( chunkId, 1, i, j, k, 0 );
                R( i, j, k, 1, 1 ) = P( chunkId, 1, i, j, k, 1 );

                // R( 2*(nxChunk*j+i) ) = P( chunkId, 0, i, j, k, 0 );
                //  R( 2*(nxChunk*j+i)+1 ) = P( chunkId, 0, i, j, k, 1 );

                //   tmp[2 * ( nxChunk * j + i ) + 1] = P( chunkId, 0, i, j, k, 1 );
            }
        }
    }

    /*
    #if ( PITTPACKACC )
    #pragma acc loop gang
    #endif
        for ( int i = 0; i < 2*nxChunk*nyChunk*nzChunk; i++ )
        {
            R( i ) = P( chunkId*chunkSize+ i );
        }
    */
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::assignBackTempY2X( const int chunkId )
{
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
    for ( int k = 0; k < nzChunk; k++ )
    {
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
        for ( int i = 0; i < nxChunk; i++ )
        {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
                // P( chunkId, 0, i, j, k, 0 ) = R( 0, 1, i, j, k, 0 );
                // P( chunkId, 0, i, j, k, 1 ) = R( 0, 1, i, j, k, 1 );
                // P( chunkId, 0, i, j, k, 0 ) = R( 2*(i*nyChunk+j+nxChunk*nyChunk*k) );
                // P( chunkId, 0, i, j, k, 1 ) =  R( 2*(i*nyChunk+j+nxChunk*nyChunk*k)+1 );
                P( chunkId, 0, i, j, k, 0 ) = R( i, j, k, 1, 0 );
                P( chunkId, 0, i, j, k, 1 ) = R( i, j, k, 1, 1 );
            }
        }
    }
}

#if ( REVTRSP == 0 )
//#if (  0 )
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::rearrangeX2YInverse()
{
#if ( PITTPACKACC )
    double tmp[2 * NXCHUNK1 * NYCHUNK1];
#endif
    // rearranges x to y
#if ( PITTPACKACC )
//#pragma acc loop seq
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] ) collapse( 2 )
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
//#pragma acc loop gang  private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
            assignTempX2Y( id, k, tmp );
#if ( PITTPACKACC )
#pragma acc loop worker
#endif

            for ( int i = 0; i < nxChunk; i++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int j = 0; j < nyChunk; j++ )
                {
                    P( id, 0, i, j, k, 0 ) = tmp[2 * ( j + nyChunk * i )];
                    P( id, 0, i, j, k, 1 ) = tmp[2 * ( j + nyChunk * i ) + 1];
                }
            }
        }
    }
}
#else

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::rearrangeX2YInverse()
{
// rearranges x to y
#if ( PITTPACKACC )
#pragma acc loop seq
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
        assignTempY2X( id );

        assignBackTempY2X( id );
    }
}

#endif

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::assignTempY2Z( const int chunkId )
{
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
    for ( int k = 0; k < nzChunk; k++ )
    {
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
        for ( int j = 0; j < nyChunk; j++ )
        {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int i = 0; i < nxChunk; i++ )
            {
                //   tmp[2 * ( nxChunk * j + i )] = P( chunkId, 0, i, j, k, 0 );
                //   tmp[2 * ( nxChunk * j + i ) + 1] = P( chunkId, 0, i, j, k, 1 );
                R( i, j, k, 0 ) = P( chunkId, 1, i, j, k, 0 );
                R( i, j, k, 1 ) = P( chunkId, 1, i, j, k, 1 );
                //   P(i,j,k,0)= 0;
                //    R(i,j,k,1)= 0;
                //   tmp[2 * ( nxChunk * j + i ) + 1] = P( chunkId, 0, i, j, k, 1 );
            }
        }
    }
}

// basically assigns the array with chunksizeto a tmeporary array which
// same function of X2Y rotation is used
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::rearrangeY2Z()
{
    for ( int id = 0; id < nChunk; id++ )
    {
        assignTempY2Z( id );
        assignBackTempY2Z( id );
    }
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::rearrangeY2ZInverse()
{
    for ( int id = 0; id < nChunk; id++ )
    {
        assignTempZ2Y( id );
        assignBackTempZ2Y( id );
    }
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::assignBackTempY2Z( const int chunkId )
{
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
    for ( int k = 0; k < nzChunk; k++ )
    {
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
        for ( int j = 0; j < nyChunk; j++ )
        {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int i = 0; i < nxChunk; i++ )
            {
                // P( chunkId, 4, i, j, k, 0 ) = R( 2*(j+ny*i+nx*ny*k) );
                //  P( chunkId, 4, i, j, k, 1 ) = R( 2*(j+ny*i+nx*ny*k)+1);

                P( chunkId, 4, i, j, k, 1 ) = R( i, j, k, 1, 0 );
                //
                P( chunkId, 4, i, j, k, 1 ) = R( i, j, k, 1, 1 );

                // some changes
                //                P(2*( i*(ny*nz) + j* nz  +k) ) = R( i, j, k, 0 );
                //                P(2*(i*(ny*nz) + j* nz  +k)+1 ) = R( i, j, k, 1 );
                //   P(i,j,k,0)= 0;
                //    R(i,j,k,1)= 0;
                //   tmp[2 * ( nxChunk * j + i ) + 1] = P( chunkId, 0, i, j, k, 1 );
            }
        }
    }
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::assignTempZ2Y( const int chunkId )
{
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
    for ( int k = 0; k < nzChunk; k++ )
    {
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
        for ( int j = 0; j < nyChunk; j++ )
        {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int i = 0; i < nxChunk; i++ )
            {
                //   tmp[2 * ( nxChunk * j + i )] = P( chunkId, 0, i, j, k, 0 );
                //   tmp[2 * ( nxChunk * j + i ) + 1] = P( chunkId, 0, i, j, k, 1 );

                R( i, j, k, 1, 0 ) = P( chunkId, 4, i, j, k, 0 );
                R( i, j, k, 1, 1 ) = P( chunkId, 4, i, j, k, 1 );
                // some changes here
                //                R(2*( i*(ny*nz)+ j* nz +k) ) = P( chunkId, 4, i, j, k, 0 );
                //                R(2*( i*(ny*nz)+ j* nz +k)+1 ) = P( chunkId, 4, i, j, k, 0 );
                //   P(i,j,k,0)= 0;
                //    R(i,j,k,1)= 0;
                //   tmp[2 * ( nxChunk * j + i ) + 1] = P( chunkId, 0, i, j, k, 1 );
            }
        }
    }
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::assignBackTempZ2Y( const int chunkId )
{
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
    for ( int k = 0; k < nzChunk; k++ )
    {
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
        for ( int j = 0; j < nyChunk; j++ )
        {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int i = 0; i < nxChunk; i++ )
            {
                //          P( chunkId, 1, i, j, k, 0 )=i+3*j;
                P( chunkId, 1, i, j, k, 0 ) = R( i, j, k, 1, 0 );
                P( chunkId, 1, i, j, k, 1 ) = R( i, j, k, 1, 1 );
                //   P(i,j,k,0)= 0;
                //    R(i,j,k,1)= 0;
                //   tmp[2 * ( nxChunk * j + i ) + 1] = P( chunkId, 0, i, j, k, 1 );
            }
        }
    }
}

inline int PencilDcmp::getTransposeRank()
{
    int i = myRank % p0;
    int j = myRank / p0;
    return ( p0 * i + j );
}

void PencilDcmp::constructShuffleVectorX()
{
    sint size = nChunk * nyChunk * nzChunk;

    char *bol = new char[size];

    memset( bol, '1', size * sizeof( char ) );

    // sint i0, j0, k0, chunkId0,index0;
    // sint i1, j1, k1, chunkId1;
    sint id, dest;

    // need to save this due to parallelization issue,
    // might lead to race condition if done on GPU
    shuffle0         b;
    vector<shuffle0> a;
    sint             count = 0;

    for ( sint i = 1; i < size - 1; i++ )
    {
        //  for (int i = size-1; i > 1; i--) {
        if ( bol[i] == '0' )
        {
            continue;
        }
        else if ( i != getDestinationLocX( i ) )
        {
            a.push_back( b );
            a[count].id = i;
            count++;
        }

        id = i;

        while ( bol[id] != '0' )
        {
            dest    = getDestinationLocX( id );
            bol[id] = '0';

            if ( id != dest )
            {
                id = dest;

                count--;
                if ( id != i )
                {
                    a[count].nbr.push_back( dest );
                }
                count++;
            }
        }
    }

    sint totalSize = 0;

    for ( sint i = 0; i < a.size(); i++ )
    {
        //    cout << " id " << a[i].id << endl;
        totalSize = totalSize + a[i].nbr.size();

        /*
                for ( sint j = 0; j < a[i].nbr.size(); j++ )
                {
                    cout << " " << a[i].nbr[j] << '\n';
                }
                cout << endl;
        */
    }

//    cout << "totalSize= " << totalSize << endl;

    setUpShuffleArraysX( a );

    delete[] bol;
}
// note that for x-direction every calculation should be independent of nx
// This is only one on the host once and passed onto the device
void PencilDcmp::setUpShuffleArraysX( vector<shuffle0> &a )
{
    iaxSize    = a.size();
    sint count = 0;

    for ( sint i = 0; i < a.size(); i++ )
    {
        count = count + a[i].nbr.size();
    }

    sint finalsize = count + a.size();
#if ( 1 )
//    cout << "FinalSize= " << finalsize << endl;
#endif
    jax = new sint[finalsize];

    count = 0;
    for ( sint i = 0; i < a.size(); i++ )
    {
        jax[count] = a[i].id;
        count++;
        for ( sint j = 0; j < a[i].nbr.size(); j++ )
        {
            jax[count] = a[i].nbr.at( j );
            //  cout << " jax["<< count << "]" << ja[count] << endl;
            count++;
        }
    }

    iax    = new sint[a.size() + 1];
    iax[0] = 0;

    jaxSize = finalsize;

    for ( sint i = 0; i < a.size(); i++ )
    {
        iax[i + 1] = iax[i] + 1 + a[i].nbr.size();
    }
#if ( DEBUG )
    //#if ( 1 )
    for ( sint i = 0; i < a.size() + 1; i++ )
    {
        cout << " iax " << iax[i] << endl;
    }

    for ( int i = 0; i < finalsize; i++ )
    {
        cout << " jax[" << i << "] = " << jax[i] << endl;
    }
#endif
    /*
        for ( sint i = 0; i < a.size() + 1; i++ )
        {
            cout << " iax " << iax[i] << endl;
        }
    */

    int xSize = 2 * a.size() * nxChunk + 1;
#if ( USE_SHARED != 1 )
    tmpX = new double[xSize];
#endif
    cout << " xSize  " << xSize << " iaxSize  " << endl;

#if ( PITTPACKACC )
#pragma acc update device( iaxSize )
#pragma acc update device( jaxSize )

#pragma acc enter data create( iax [0:iaxSize + 1] )
#pragma acc update device( iax [0:iaxSize + 1] )

#pragma acc enter data create( jax [0:jaxSize] )
#pragma acc update device( jax [0:jaxSize] )

#if ( USE_SHARED != 1 )
#pragma acc enter data create( tmpX [0:xSize] )
#pragma acc update device( tmpX [0:xSize] )
#endif
#endif
}

// this is designed for first transform
//  I will generalize this for any direction
// trying to convert local to global to get the global one
#if ( PITTPACKACC )
#pragma acc routine seq
#endif
inline sint PencilDcmp::oldIndexX( const sint chunkId, const sint j, const sint k )
{
    return ( chunkId * nyChunk * nzChunk + nyChunk * k + j );
}
#if ( PITTPACKACC )
#pragma acc routine seq
#endif
inline sint PencilDcmp::getDestinationLocX( const sint id )
{
    sint chunkId1, j1, k1, index0;

    decomposeOldIndexX( id, chunkId1, j1, k1 );

    // cout << " chunkId  " << chunkId1 << " j1  " << j1 << " k1 " << k1 << endl;

    index0 = chunkId1 + j1 * nChunk + ( nyChunk * nChunk ) * k1;

    return ( index0 );
}

#if ( PITTPACKACC )
#pragma acc routine seq
#endif
inline void PencilDcmp::decomposeOldIndexX( const sint id, sint &chunkId, sint &j, sint &k )
{
    chunkId = id / nyChunk / nzChunk;
    k       = ( id % ( nyChunk * nzChunk ) ) / nyChunk;
    j       = ( id % ( nyChunk * nzChunk ) ) % nyChunk;
}

// note that for y-direction every calculation should be independent of ny
void PencilDcmp::constructShuffleVectorY()
{
    sint size = nChunk * nxChunk * nzChunk;

    char *bol = new char[size];

    memset( bol, '1', size * sizeof( char ) );

    // sint i0, j0, k0, chunkId0;
    // sint i1, j1, k1, chunkId1,index0;
    sint id, dest;

    // need to save this due to parallelization issue,
    // might lead to race condition if done on GPU
    shuffle0         b;
    vector<shuffle0> a;
    sint             count = 0;

    for ( sint i = 1; i < size - 1; i++ )
    {
        //  for (int i = size-1; i > 1; i--) {
        if ( bol[i] == '0' )
        {
            continue;
        }
        else if ( i != getDestinationLocY( i ) )
        {
            a.push_back( b );
            a[count].id = i;
            count++;
        }

        id = i;

        while ( bol[id] != '0' )
        {
            dest    = getDestinationLocY( id );
            bol[id] = '0';

            if ( id != dest )
            {
                id = dest;

                count--;
                if ( id != i )
                {
                    a[count].nbr.push_back( dest );
                }
                count++;
            }
        }
    }

    sint totalSize = 0;

    for ( sint i = 0; i < a.size(); i++ )
    {
#if ( DEBUG )
        cout << " id " << a[i].id << endl;
#endif
        totalSize = totalSize + a[i].nbr.size();
#if ( DEBUG )
        for ( sint j = 0; j < a[i].nbr.size(); j++ )
        {
            cout << " " << a[i].nbr[j] << '\n';
        }
        cout << endl;
#endif
    }

#if ( DEBUG )
//    cout << "totalSize= " << totalSize << endl;
#endif
    setUpShuffleArraysY( a );

    delete[] bol;
}

void PencilDcmp::setUpShuffleArraysY( vector<shuffle0> &a )
{
    sint count = 0;

    for ( sint i = 0; i < a.size(); i++ )
    {
        count = count + a[i].nbr.size();
        //       cout << " idx " << idx[i] << endl;
    }

    iaySize = a.size();

    int finalsize = count + a.size();

    jay = new sint[finalsize];

    jaySize = finalsize;

    count = 0;
    for ( sint i = 0; i < a.size(); i++ )
    {
        jay[count] = a[i].id;
        count++;
        for ( sint j = 0; j < a[i].nbr.size(); j++ )
        {
            jay[count] = a[i].nbr.at( j );
            //            cout << " ja " << jax[count] << endl;
            count++;
        }
    }
#if ( DEBUG )
    for ( int i = 0; i < finalsize; i++ )
    {
        cout << " jay[" << i << "] = " << jay[i] << endl;
    }
#endif
    iay    = new sint[a.size() + 1];
    iay[0] = 0;
    for ( sint i = 0; i < a.size(); i++ )
    {
        iay[i + 1] = iay[i] + 1 + a[i].nbr.size();
    }

#if ( DEBUG )
    for ( sint i = 0; i < a.size() + 1; i++ )
    {
        cout << " iay " << iay[i] << endl;
    }
#endif

    // I added one to prvent seg fault for allocation of size 0 while deleting
    int ySize = 2 * a.size() * nyChunk + 1;
#if ( USE_SHARED != 1 )
    tmpY = new double[ySize];
#endif
#if ( PITTPACKACC )
#pragma acc update device( iaySize )
#pragma acc update device( jaySize )

#pragma acc enter data create( iay [0:iaySize + 1] )
#pragma acc update device( iay [0:iaySize + 1] )

#pragma acc enter data create( jay [0:jaySize] )
#pragma acc update device( jay [0:jaySize] )

#if ( USE_SHARED != 1 )
#pragma acc enter data create( tmpY [0:ySize] )
#pragma acc update device( tmpY [0:ySize] )
#endif

#endif
}

#if ( PITTPACKACC )
#pragma acc routine seq
#endif
inline sint PencilDcmp::oldIndexY( const sint chunkId, const sint i, const sint k )
{
    return ( chunkId * nxChunk * nzChunk + nxChunk * k + i );
}
#if ( PITTPACKACC )
#pragma acc routine seq
#endif
inline sint PencilDcmp::getDestinationLocY( const sint id )
{
    sint chunkId1, i1, k1, index0;

    decomposeOldIndexY( id, chunkId1, i1, k1 );

    index0 = chunkId1 + i1 * nChunk + ( nxChunk * nChunk ) * k1;

    return ( index0 );
}

#if ( PITTPACKACC )
#pragma acc routine seq
#endif
inline void PencilDcmp::decomposeOldIndexY( const sint id, sint &chunkId, sint &i, sint &k )
{
    chunkId = id / ( nxChunk * nzChunk );
    k       = ( id % ( nxChunk * nzChunk ) ) / nxChunk;
    i       = ( id % ( nxChunk * nzChunk ) ) % nxChunk;
}

#if ( PITTPACKACC )
#pragma acc routine vector
#endif
void PencilDcmp::saveToTmp( const sint id, double *tmp, sint dir )
{
    int N;
    if ( dir == 0 )
    {
        N = nxChunk;
    }
    else if ( dir == 1 )
    {
        N = nyChunk;
    }
    else if ( dir == 2 )
    {
        N = nzChunk;
    }
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
    for ( int i = 0; i < 2 * N; i++ )
    {
        tmp[i] = P( 2 * N * id + i );
    }
}

#if ( PITTPACKACC )
#pragma acc routine vector
#endif
void PencilDcmp::restoreTmp( const sint id, double *tmp, sint dir )
{
    int N;
    if ( dir == 0 )
    {
        N = nxChunk;
    }
    else if ( dir == 1 )
    {
        N = nyChunk;
    }
    else if ( dir == 2 )
    {
        N = nzChunk;
    }

#if ( PITTPACKACC )
#pragma acc loop vector
#endif
    for ( sint i = 0; i < 2 * N; i++ )
    {
        P( 2 * N * id + i ) = tmp[i];
    }
}

#if ( USE_SHARED != 1 )
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::changeLocationX()
{
    double *tm = NULL;

#if ( PITTPACKACC )
//#pragma acc loop worker private( tm[1] )
#pragma acc loop private( tm[1] )
#endif
    for ( sint i = 0; i < iaxSize; i++ )
    {
        tm = tmpX + 2 * nxChunk * i;

        saveToTmp( jax[iax[i + 1] - 1], tm, 0 );

        //       cout<<" iax "<< iax[i]<<" iax+1  "<<iax[i+1]<<endl;
        //#if(PITTPACKACC)
        //#pragma acc loop worker
        //#endif
        for ( sint j = iax[i + 1] - 1; j > iax[i]; j-- )
        {
            // set the last one to tmp
            saveToDest( jax[j - 1], jax[j], 0 );
        }

        saveTmpToDest( tm, jax[iax[i]], 0 );
    }
}
#else

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::changeLocationX()
{
    double tm[2 * TMPSIZE];

#if ( PITTPACKACC )
//#pragma acc loop worker private( tm[1] )
#pragma acc loop private( tm[2 * TMPSIZE] )
#endif
    for ( sint i = 0; i < iaxSize; i++ )
    {
        //    tm = tmpX + 2 * nxChunk * i;

        saveToTmp( jax[iax[i + 1] - 1], tm, 0 );

        //       cout<<" iax "<< iax[i]<<" iax+1  "<<iax[i+1]<<endl;
        //#if(PITTPACKACC)
        //#pragma acc loop worker
        //#endif
        for ( sint j = iax[i + 1] - 1; j > iax[i]; j-- )
        {
            // set the last one to tmp
            saveToDest( jax[j - 1], jax[j], 0 );
        }

        saveTmpToDest( tm, jax[iax[i]], 0 );
    }
}

#endif

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::changeLocationXOverlap()
{
#if ( PITTPACKACC )
#pragma acc loop seq
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 0; i < nxChunk; i++ )
                {
                    P( 2 * ( k * nxChunk * nyChunk * nChunk + nChunk * nxChunk * j + nxChunk * id + i ) )     = R( id, 0, i, j, k, 0 );
                    P( 2 * ( k * nxChunk * nyChunk * nChunk + nChunk * nxChunk * j + nxChunk * id + i ) + 1 ) = R( id, 0, i, j, k, 1 );
                }
            }
        }
    }
}

// not verified yet
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::changeLocationYOverlap()
{
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
    for ( int k = 0; k < nChunk * P.chunkSize; k++ )
    {
        R( k ) = P( k );
    }

#if ( PITTPACKACC )
#pragma acc loop seq
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nxChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 0; i < nyChunk; i++ )
                {
                    P( 2 * ( k * nxChunk * nyChunk * nChunk + nChunk * nyChunk * j + nyChunk * id + i ) )     = R( id, 3, i, j, k, 0 );
                    P( 2 * ( k * nxChunk * nyChunk * nChunk + nChunk * nyChunk * j + nyChunk * id + i ) + 1 ) = R( id, 3, i, j, k, 1 );
                }
            }
        }
    }
}

#if ( USE_SHARED != 1 )
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::changeLocationY()
{
    double *tm = NULL;

#if ( PITTPACKACC )
#pragma acc loop gang private( tm[1] )
#endif
    for ( sint i = 0; i < iaySize; i++ )
    {
        tm = tmpY + 2 * nyChunk * i;
        saveToTmp( jay[iay[i + 1] - 1], tm, 1 );
        for ( sint j = iay[i + 1] - 1; j > iay[i]; j-- )
        {
            saveToDest( jay[j - 1], jay[j], 1 );
        }
        saveTmpToDest( tm, jay[iay[i]], 1 );
    }
}
#else

void PencilDcmp::changeLocationY()
{
    double tm[2 * TMPSIZE];
#if ( PITTPACKACC )
#pragma acc loop gang private( tm[2 * TMPSIZE] )
#endif
    for ( sint i = 0; i < iaySize; i++ )
    {
        //        tm = tmpY + 2 * nyChunk * i;
        saveToTmp( jay[iay[i + 1] - 1], tm, 1 );
        for ( sint j = iay[i + 1] - 1; j > iay[i]; j-- )
        {
            saveToDest( jay[j - 1], jay[j], 1 );
        }
        saveTmpToDest( tm, jay[iay[i]], 1 );
    }
}

#endif

#if ( USE_SHARED != 1 )
#if ( PITTPACKACC )
#pragma acc routine gang
//#pragma acc routine gang
#endif
void PencilDcmp::restoreLocationY()
{
    double *tm = NULL;

#if ( PITTPACKACC )
#pragma acc loop gang private( tm[1] )
#endif
    for ( sint i = 0; i < iaySize; i++ )
    {
        tm = tmpY + 2 * nyChunk * i;
        saveToTmp( jay[iay[i]], tm, 1 );

        for ( sint j = iay[i]; j < iay[i + 1] - 1; j++ )
        {
            // set the last one to tmp
            saveToDest( jay[j + 1], jay[j], 1 );
        }
        saveTmpToDest( tm, jay[iay[i + 1] - 1], 1 );
    }
}
#else

void PencilDcmp::restoreLocationY()
{
    double tm[2 * TMPSIZE];
#if ( PITTPACKACC )
#pragma acc loop gang private( tm[2 * TMPSIZE] )
#endif
    for ( sint i = 0; i < iaySize; i++ )
    {
        //       tm = tmpY + 2 * nyChunk * i;
        saveToTmp( jay[iay[i]], tm, 1 );

        for ( sint j = iay[i]; j < iay[i + 1] - 1; j++ )
        {
            // set the last one to tmp
            saveToDest( jay[j + 1], jay[j], 1 );
        }
        saveTmpToDest( tm, jay[iay[i + 1] - 1], 1 );
    }
}

#endif

#if ( USE_SHARED != 1 )
#if ( PITTPACKACC )
#pragma acc routine gang
//#pragma acc routine vector
#endif
void PencilDcmp::restoreLocationX()
{
    double *tm = NULL;

#if ( PITTPACKACC )
#pragma acc loop gang private( tm[1] )
#endif
    for ( sint i = 0; i < iaxSize; i++ )
    {
        tm = tmpX + 2 * i * nxChunk;

        saveToTmp( jax[iax[i]], tm, 0 );

        for ( sint j = iax[i]; j < iax[i + 1] - 1; j++ )
        {
            saveToDest( jax[j + 1], jax[j], 0 );
        }
        saveTmpToDest( tm, jax[iax[i + 1] - 1], 0 );
    }
}
#else

void PencilDcmp::restoreLocationX()
{
    double tm[2 * TMPSIZE];

#if ( PITTPACKACC )
#pragma acc loop gang private( tm[2 * TMPSIZE] )
#endif
    for ( sint i = 0; i < iaxSize; i++ )
    {
        //        tm = tmpX + 2 * i * nxChunk;

        saveToTmp( jax[iax[i]], tm, 0 );

        for ( sint j = iax[i]; j < iax[i + 1] - 1; j++ )
        {
            saveToDest( jax[j + 1], jax[j], 0 );
        }
        saveTmpToDest( tm, jax[iax[i + 1] - 1], 0 );
    }
}

#endif

#if ( PITTPACKACC )
#pragma acc routine vector
#endif
void PencilDcmp::saveToDest( const sint source, const sint dest, sint dir )
{
    int N;

    if ( dir == 0 )
    {
        N = nxChunk;
    }
    else if ( dir == 1 )
    {
        N = nyChunk;
    }
    else if ( dir == 2 )
    {
        N = nzChunk;
    }

#if ( PITTPACKACC )
#pragma acc loop vector
#endif
    for ( sint i = 0; i < 2 * N; i++ )
    {
        P( 2 * dest * N + i ) = P( 2 * source * N + i );
    }
}

#if ( PITTPACKACC )
#pragma acc routine vector
#endif
void PencilDcmp::saveTmpToDest( const double *tmp, const sint dest, sint dir )
{
    int N;
    if ( dir == 0 )
    {
        N = nxChunk;
    }
    else if ( dir == 1 )
    {
        N = nyChunk;
    }
    else if ( dir == 2 )
    {
        N = nzChunk;
    }
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
    for ( sint i = 0; i < 2 * N; i++ )
    {
        P( 2 * dest * N + i ) = tmp[i];
    }
}
void PencilDcmp::printX( ofstream &myfile )
{
    myfile << endl;
    myfile << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;
    for ( int i = 0; i < nChunk * nyChunk * nzChunk; i++ )
    {
        for ( int j = 0; j < nxChunk; j++ )
        {
            //   myfile << setprecision( 4 ) << "( " << P( 2 * ( nxChunk * i + j ) ) << " , " << P( 2 * ( nxChunk * i + j ) + 1 ) << " ) "
            // << '\t';
            myfile << setprecision( 6 ) << " " << P( 2 * ( nxChunk * i + j ) ) << " + " << P( 2 * ( nxChunk * i + j ) + 1 ) << "i " << '\t';
            // myfile << setprecision( 6 ) << " " << P(  ( nxChunk * i + j ) ) <<'\t';
            //         myfile << setprecision( 4 ) << P( 2 * ( nxChunk * i + j ) ) << '\t';
        }
        // cout << endl;
        myfile << endl;
    }
    myfile << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;
    myfile << endl;
}

void PencilDcmp::printY( ofstream &myfile )
{
    myfile << endl;
    myfile << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;
    for ( int i = 0; i < nChunk * nxChunk * nzChunk; i++ )
    {
        for ( int j = 0; j < nyChunk; j++ )
        {
            //   myfile << setprecision( 4 ) << "( " << P( 2 * ( nxChunk * i + j ) ) << " , " << P( 2 * ( nxChunk * i + j ) + 1 ) << " ) "
            // << '\t';
            myfile << setprecision( 6 ) << " " << P( 2 * ( nyChunk * i + j ) ) << " + " << P( 2 * ( nyChunk * i + j ) + 1 ) << "i " << '\t';
            //         myfile << setprecision( 4 ) << P( 2 * ( nxChunk * i + j ) ) << '\t';
        }
        // cout << endl;
        myfile << endl;
    }
    myfile << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;
    myfile << endl;
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::rescale()
{
#if ( 0 )
#pragma acc loop gang collapse( 2 )
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int i = 0; i < nxChunk; i++ )
        {
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int k = 0; k < nzChunk; k++ )
                {
                    P( id, 0, i, j, k, 0 ) = 1. / (scale)*P( id, 0, i, j, k, 0 );
                    P( id, 0, i, j, k, 1 ) = 1. / (scale)*P( id, 0, i, j, k, 1 );
                }
            }
        }
    }
}

void PencilDcmp::setEigenVal()
{
    freqs[0] = pi;
    freqs[1] = pi;

    if ( tags[0] == 0 )
    {
        num[0]   = 1.0;
        denum[0] = 0.0;
    }
    else if ( tags[0] == 1 )
    {
        num[0]   = 0.0;
        denum[0] = 0.0;
    }
    else if ( tags[0] == 2 )
    {
        num[0]   = 0.0;
        denum[0] = 0.0;
        freqs[0] = 2. * freqs[0];
    }

    if ( tags[1] == 0 )
    {
        num[1]   = 1.0;
        denum[1] = 0.0;
    }
    else if ( tags[1] == 1 )
    {
        num[1]   = 0.0;
        denum[1] = 0.0;
    }
    else if ( tags[1] == 2 )
    {
        num[1]   = 0.0;
        denum[1] = 0.0;
        freqs[1] = 2. * freqs[1];
    }
#if ( PITTPACKACC )
#pragma acc update device( num [0:2] )
#pragma acc update device( denum [0:2] )
#pragma acc update device( freqs [0:2] )
#endif
}

void PencilDcmp::assignBoundary( char *boundary )
{
    for ( int i = 0; i < 6; i++ )
    {
        bc[i] = boundary[i];
        //  cout<<" BC "<<bc[i]<<endl;
        faceTag[i] = 0;
    }

    // tag in X direction

    if ( bc[0] == 'D' && myRank % p0 == 0 )
    {
        faceTag[0] = 1;
    }
    if ( bc[1] == 'D' && myRank % p0 == ( p0 - 1 ) )
    {
        faceTag[1] = 1;
    }
    if ( bc[2] == 'D' && myRank / p0 == 0 )
    {
        faceTag[2] = 1;
    }
    if ( bc[3] == 'D' && myRank / p0 == ( p0 - 1 ) )
    {
        faceTag[3] = 1;
    }

    for ( int i = 4; i < 6; i++ )
    {
        if ( bc[i] == 'D' )
        {
            faceTag[i] = 1;
        }
    }

    T.assignBC( bc );

    for ( int i = 0; i < 6 && myRank == 0; i++ )
    {
        cout << " myRank " << myRank << " Dirichlet tags " << faceTag[i] << endl;
    }

//#pragma acc enter data create( bc[0 : 6] )
#if ( PITTPACKACC )
#pragma acc update device( bc [0:6] )
#pragma acc update device( faceTag [0:6] )
#endif
    detectTransforms();
    extractTag();
    setEigenVal();
    setScale();

#if ( PITTPACKACC )
#pragma acc update device( tags [0:3] )
#endif
}

void PencilDcmp::extractTag()
{
    for ( int i = 0; i < 3; i++ )
    {
        if ( transform[i] == 'S' )
        {
            tags[i] = 0;
        }
        else if ( transform[i] == 'C' )
        {
            tags[i] = 1;
        }
        else if ( transform[i] == 'F' )
        {
            tags[i] = 2;
        }
        else
        {
            tags[i] = -1;
        }
    }

    if ( bc[4] == 'D' )
    {
        tags[2] = -1;
    }
    else if ( bc[4] == 'N' )
    {
        tags[2] = -2;
    }

    std::cout << "tags " << tags[0] << " " << tags[1] << " " << tags[2] << std::endl;
}

void PencilDcmp::setScale()
{
    int sc[2];

    // new method ROX all scales are N
    sc[0] = nxChunk * nChunk;
    sc[1] = nyChunk * nChunk;
    scale = sc[0] * sc[1];

    std::cout << "scale " << scale << std::endl;
#if ( PITTPACKACC )
#pragma acc update device( scale )
#endif
}

void PencilDcmp::detectTransforms()
{
    for ( int i = 0; i < 3; i++ )
    {
        transform[i] = 'U';
    }

    for ( int i = 0; i < 3; i++ )
    {
        if ( bc[2 * i] == 'D' && bc[2 * i + 1] == 'D' )
        {
            transform[i] = 'S';
        }
        else if ( bc[2 * i] == 'N' && bc[2 * i + 1] == 'N' )
        {
            transform[i] = 'C';
        }
        else if ( bc[2 * i] == 'P' && bc[2 * i + 1] == 'P' )
        {
            transform[i] = 'F';
            //            cout<<" here in periodic" <<endl;
        }
    }

    // do not make transform in all 3 directions, as we would like to solve a tridiag

    int count = 0;

    for ( int i = 0; i < 3; i++ )
    {
        if ( transform[i] == 'U' )
        {
            count++;
        }
    }

    if ( count == 0 )
    {
        transform[2] = 'U';
    }

    for ( int i = 0; i < 3; i++ )
    {
        std::cout << " transform[ " << i << " ]= " << transform[i] << std::endl;
    }

    // detect the number of di
}
#if ( PITTPACKACC )
#pragma acc routine seq
#endif
int PencilDcmp::thomas( int i, int j, int dir, int index )
{
    // double bet;

    // the enterior is always set according to eigenvalues
    // the two ends decided by BC
    int result = SUCCESS;

    double onDiag[3];
    onDiag[1] = getEigenVal( i, j );

    // int this_rank = 1;

    // assign bc
    // bc 0 >> dirichlet, value known at ghost point
    if ( bc[4] == 'D' )
    {
        onDiag[0] = onDiag[1] - 1.0;
    }
    else if ( bc[4] == 'N' )
    {
        onDiag[0]                   = -1.0;
        P( 0, dir, i, j, 0, index ) = 0.0;
    }
    else if ( bc[4] == 'P' )
    {
        // peroiodic thomas (sherman-morrison) is implemented as a stand alone function
    }

    if ( bc[5] == 'D' )
    {
        onDiag[2] = onDiag[1] - 1.0;
    }
    else if ( bc[5] == 'N' )
    {
        onDiag[2]                                      = -1.0;
        P( nChunk - 1, dir, i, j, nzChunk - 1, index ) = 0.0;
    }
    else if ( bc[5] == 'P' )
    {
        // peroiodic thomas (sherman-morrison) is implemented as a stand alone function
    }
    /*
       cout<<" onDiag  " << onDiag[0]<<" "<<onDiag[1]<<" "<<onDiag[2]<< endl;
       cout<<" subDiag  " << subDiag[0]<<" "<<subDiag[1]<<" "<<subDiag[2]<< endl;
       cout<<" supDiag  " << supDiag[0]<<" "<<supDiag[1]<<" "<<supDiag[2]<< endl;


       cout<<" before inside thomas function "<<endl;

       for(int k=0;k<nz;k++)
           {
            cout<<P(0, dir,i,j,k,index)<<endl;
           }
    */

    if ( bc[0] == 'P' || bc[2] == 'P' )
    {
        //     if(  T.thomas( P, onDiag, i, j, dir, index )!=SUCCESS) /*!slower */
        {
            //      result=THOMAS_FAIL;
        }
        T.thomas( P, onDiag, i, j, dir, index );
    }
    else // it is real
    {
        result = T.thomasReal( P, onDiag, i, j, dir ); /*!slower */
                                                       /*
if(T.thomasReal( P, onDiag, i, j, dir )!=SUCCESS) !slower
{
result=THOMAS_FAIL;
}
*/
        //         T.thomas( P, onDiag, i, j, dir, index );
    }
    /*
   cout<<" ----- solved --------- "<<endl;

          for(int k=0;k<nz;k++)
          {
           cout<<P(0, dir,i,j,k,index)<<endl;
          }
   */

    return ( result );
}

#if ( PITTPACKACC )
#pragma acc routine seq
#endif
void PencilDcmp::thomasSingleBlock( int i, int j, int dir, int index )
{
    /*
    double bet;
    int    n;
    */

    double onDiag[3];
    onDiag[1] = getEigenVal( i, j );

    // int this_rank = 0;

    // assign bc
    // bc 0 >> dirichlet, value known at ghost point
    if ( bc[4] == 'D' )
    {
        onDiag[0] = onDiag[1] - 1.0;
    }
    else if ( bc[4] == 'N' )
    {
        onDiag[0]                   = -1.0;
        P( 0, dir, i, j, 0, index ) = 0.0;
    }
    else if ( bc[4] == 'P' )
    {
        // calls shemran morrision in thi way yet
    }

    if ( bc[5] == 'D' )
    {
        onDiag[2] = onDiag[1] - 1.0;
    }
    else if ( bc[5] == 'N' )
    {
        onDiag[2]                             = -1.0;
        P( 0, dir, i, j, nzChunk - 1, index ) = 0.0;
    }
    else if ( bc[5] == 'P' )
    {
        // call shemran morrision in thi way yet
    }

    T.thomasSingleBlock( P, onDiag, i, j, dir, index );
}

#if ( PITTPACKACC )
#pragma acc routine seq
#endif
void PencilDcmp::thomasPeriodic( int i, int j, int dir, int index )
{
    /*
        double bet;

        // the enterior is always set according to eigenvalues
        // the two ends decided by BC


        int    n;
        double alpha = 1.0;
        double beta  = 1.0;
    */
    double onDiag[3];
    onDiag[1] = getEigenVal( i, j );

    if ( bc[0] == 'P' || bc[2] == 'P' )
    {
        T.thomasPeriodic( P, onDiag, i, j, dir, index );
    }
    else
    {
        T.thomasPeriodicReal( P, onDiag, i, j, dir );
    }
}

#if ( 0 )
// this is an in-place solve for thomas in the blocks
// wastes bandwidth and is not prefered for GPU
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
int PencilDcmp::solve()
// void PencilDcmp::solve()
{
    int num = 1;

    if ( bc[0] == 'P' || bc[2] == 'P' )
    {
        num = 2;
    }

    // PittPackResult result=SUCCESS;
    int result = 0;

#if ( PITTPACKACC )
#pragma acc loop gang
#endif
    for ( int j = 0; j < nxChunk; j++ )
    {
#if ( PITTPACKACC )
//#pragma acc loop worker  firstprivate(result) reduction(+:result)
#pragma acc loop worker
#endif
        for ( int i = 0; i < nyChunk; i++ )
        {
#if ( 1 )
            //   if ( p0 != 1 )
            if ( 1 )
            {
                if ( bc[4] == 'P' && bc[5] == 'P' )
                {
                    //#if ( PITTPACKACC )
                    // this loop is by nature serial, do not try to parallalize it
                    //#endif
                    for ( int k = 0; k < num; k++ )
                    {
                        thomasPeriodic( i, j, 3, k );
                        //  thomasPeriodic( i, j, 3, 1 );
                    }
                }
                else
                {
                    //#if ( !PITTPACKACC )
                    // this loop is by nature serial, do not try to parallalize it
                    //#endif
                    for ( int k = 0; k < num; k++ )
                    {
                        result = thomas( i, j, 3, k );
                        //                     result=10;
                        /*
                                            if(thomas( i, j, 3, k ))
                                             {
                                              result=THOMAS_FAIL;
                                             }

                        */
                        //  cout<<" This one is solving "<<endl;
                    }
                }
            }
            else
            {
                cout << " single block solving " << endl;
                thomasSingleBlock( i, j, 3, 0 );
                //       thomasSingleBlock( i, j, 3, 1 );
            }
#endif
            /*
             for(int k=0;k<nz;k++)
                   {
                    cout<<P(0, 3,i,j,k,0)<<endl;
                   }
              cout<<" -------------- "<<endl;
            */
        }
    }

    // returnVal=result;
    //#pragma acc update self(returnVal)
    // return(result);
    //  return ( result );
}
#endif

// continue form here

#if ( 1 )
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::fillInArrayContig( const int i, const int j, int index )
{
//    int dir = 3;
/*
    double *tm[2];

    tm[0] = tmpMGReal;
    tm[1] = tmpMGImag;
*/
/*
if(index==0)
{
tm=tmpMGReal;
}
else
{
tm=tmpMGImag;
}
*/

// tmpMGReal[0]=0.0;
// tmpMGReal[16-1]=0.0;

//    cout<<"before inside fill in"<<endl;
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
            if ( index == 0 )
            {
                tmpMGReal[id * nzChunk + k + OFFS] = P( id, 3, i, j, k, index );
            }
            else
            {
                tmpMGImag[id * nzChunk + k + OFFS] = P( id, 3, i, j, k, index );
            }
            //      tmpMGReal[id * nzChunk + k + OFFS] =4.*scale*k;
            // cout<<" ( " << i <<" , "<< j <<" , "<< k<<" ) "<<tmpMGReal[k]<<endl;
            //     cout<<P( id, dir, i, j, k, index )<<endl;
        }
    }
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::solveMG()
{
    // not that rhs already scaled for initialization for Thomas so here we need to rescale
    //  back as MultiGrid has its own scaling
    //
    //    MultiGrid  MG;
    //    double diag[3];
    double eig;
    int    index = 0;
    // cout<< RED<<"multiGrid" <<RESET<<endl;
    // int count = 0;

    int bol = MG.gridSizePowerTwoPlusOne;

    //    int i=2;
    //    int j=1;

#if ( PITTPACKACC )
#pragma acc loop seq private( eig )
#endif
    for ( int k = 0; k < nxChunk * nyChunk; k++ )
    //   for ( int j = 0; j < nxChunk; j++ )
    {
        //#pragma acc loop seq
        //        for ( int i = 0; i < nyChunk; i++ )
        {
            int j = k / nyChunk;
            int i = k % nyChunk;
// debugged work fine
#if ( 0 )
            MG.fillInArrayContig( P, nChunk, nzChunk, i, j, index );
            eig = getEigenVal( i, j );
            MG.setDiag( eig );
            MG.solveMono( eig );
            MG.fillInArrayBack( P, nChunk, nzChunk, i, j, index );
            MG.reset();
#else
            //            eig=2.0;
            /*
                        for ( int k = 0; k < 3; k++ )
                        {
                            diag[k] = eig;
                            //       cout<<" main diagonal "<<diag[k]<<endl;
                        }

            //             diag[0]=eig;
            //             diag[1]=eig;
            //             diag[2]=eig;
            */

#if ( 1 )
            eig = getEigenVal( i, j );
            MG.setDiag( eig );
            //          MG.setDiag( eig );

            // int i= 2;
            //  fillInArrayContig( i, j, index );
            MG.fillInArrayContig( P, nChunk, nzChunk, i, j, index );

            copySourceToMG( index );
            //            MG.reset();
            //            MG.assign( tmpMGReal );
            // if(0)
            // for debug            MG.redBlackMono();
            if ( eig < -2.1 )
            {
                // printf("EIGGGGGGGGGGGGGGGi =%lf  \n",eig);
                MG.solveMulti( tmpMGReal );
            }
            else
            {
                MG.solveMono( eig );
                //   printf("eig= %lf\n",eig);
                //  count++;
            }

            copyAnsFromMG( index );

            //            MG.reFill();
            MG.fillInArrayBack( P, nChunk, nzChunk, i, j, index );
            //            fillInArrayBack( i, j, index );

            MG.reset();
#endif
#endif
        }
    }
    // printf("number without MG %d\n",count );
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::solveMGC()
{
    double eig;
    // int    count = 0;
    int index = 1;
    int bol   = MGC.gridSizePowerTwoPlusOne;

#if ( PITTPACKACC )
#pragma acc loop seq private( eig )
#endif
    for ( int k = 0; k < nxChunk * nyChunk; k++ )
    {
        int j = k / nyChunk;
        int i = k % nyChunk;
// debugged work fine
#if ( 0 )
        MGC.fillInArrayContig( P, nChunk, nzChunk, i, j, index );
        eig = getEigenVal( i, j );
        MGC.setDiag( eig );
        MGC.solveMono( eig );
        MGC.fillInArrayBack( P, nChunk, nzChunk, i, j, index );
        MGC.reset();
#else
        //            eig=2.0;
        /*
                    for ( int k = 0; k < 3; k++ )
                    {
                        diag[k] = eig;
                        //       cout<<" main diagonal "<<diag[k]<<endl;
                    }

                 //    diag[0]=eig;
                 //    diag[1]=eig;
                 //    diag[2]=eig;
        */
        eig = getEigenVal( i, j );
        MGC.setDiag( eig );
        //          MG.setDiag( eig );

        // int i= 2;
        //  fillInArrayContig( i, j, index );
        MGC.fillInArrayContig( P, nChunk, nzChunk, i, j, index );

        copySourceToMG( index );
//            MG.reset();
//            MG.assign( tmpMGReal );

// for debug            MG.redBlackMono();
#if ( 1 )
        if ( eig < -2.1 && bol == 1 )
        //           if( 0 )
        {
            MGC.solveMulti( tmpMGImag );
        }
        else
        {
            MGC.solveMono( eig );
            //   printf("eig= %lf\n",eig);
            //  count++;
        }
#endif
        copyAnsFromMG( index );

        //            MG.reFill();
        MGC.fillInArrayBack( P, nChunk, nzChunk, i, j, index );
        //            fillInArrayBack( i, j, index );

        MGC.reset();
#endif
    }
    // printf("number without MG %d\n",count );
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::fillInArrayBack( const int i, const int j, const int index )
{
//     cout<<"after"<<endl;
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
            if ( index == 0 )
            {
                P( id, 3, i, j, k, index ) = tmpMGReal[id * nzChunk + k + OFFS]; //      cout<<k<<"   "<<tmpMG[i]<<endl;
            }
            else
            {
                P( id, 3, i, j, k, index ) = tmpMGImag[id * nzChunk + k + OFFS]; //      cout<<k<<"   "<<tmpMG[i]<<endl;
            }
            //    P( id, 3, i, j, k, index ) =5.*scale; //      cout<<k<<"   "<<tmpMG[i]<<endl;
            //        P(id, dir, i, j, k, 1 )  = 0; //      cout<<k<<"   "<<tmpMGReal[i]<<endl;
            //      cout<<tmpMGReal[k]<<endl;
        }
    }
    //      cout<<"-------------------"<<endl;
}
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::copyAnsFromMG( int index )
{
    if ( index == 0 )
    {
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int i = 0; i < nz; i++ )
        {
            tmpMGReal[i] = MG.u[i];
            //    cout<< " answer "<< out[i]<<"delx[0] "<<delx[0]<<endl;
        }
    }
    else
    {
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int i = 0; i < nz; i++ )
        {
            tmpMGImag[i] = MGC.u[i];
            //    cout<< " answer "<< out[i]<<"delx[0] "<<delx[0]<<endl;
        }
    }
}
// do not forget the negative sign
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::copySourceToMG( int index )
{
    if ( index == 0 )
    {
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int i = 0; i < nz; i++ )
        {
            // MG.rhs[i] = -tmpMGReal[i];
            MG.u[i] = -tmpMGReal[i];
            //    cout<< " answer "<< out[i]<<"delx[0] "<<delx[0]<<endl;
        }
    }
    else
    {
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int i = 0; i < nz; i++ )
        {
            MGC.u[i] = -tmpMGImag[i];
        }
    }
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
int PencilDcmp::solveThm( const int index )
{
    // not that rhs already scaled for initialization for Thomas so here we need to rescale
    //  back as MultiGrid has its own scaling

    // double diag[3];
    double  eig;
    double *tm[2];

    tm[0] = tmpMGReal;
    tm[1] = tmpMGImag;

    // cout<< RED<<"multiGrid" <<RESET<<endl;

    // int count = 0;

#if ( 1 )

#if ( PITTPACKACC )
#pragma acc loop seq
#endif
    for ( int j = 0; j < nxChunk; j++ )
    {
#if ( PITTPACKACC )
#pragma acc loop seq
#endif
        for ( int i = 0; i < nyChunk; i++ )
        {
#if ( 1 )
            eig = getEigenVal( i, j );
            /*
            #pragma acc loop vector
                 for(int k=0;k<3;k++)
                  {
                   diag[k]=eig;
            //       cout<<" main diagonal "<<diag[k]<<endl;
                  }

                MG.setDiag(diag);
            */
            fillInArrayContig( i, j, index );

            MG.assign( tm[index] );

            // solves Thomas
            MG.thomasLowMem( tm[index], eig, index );
            MG.thomasPutBack( tm[index], index );

            fillInArrayBack( i, j, index );
            MG.reset();
#endif
        }
    }
// printf("number without MG %d\n",count );
#endif
    // int dir=3;
    // int index=0;
    return ( 0 );
}

#endif

void PencilDcmp::eigenVal( ofstream &myfile )
{
    // note that we perform solve when the data is aligned in y direction

    // int    ioffset;
    // int    joffset;
    // double onDiag;
    myfile << endl;
    myfile << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;

    for ( int j = 0; j < nxChunk; j++ )
    {
        for ( int i = 0; i < nyChunk; i++ )
        {
            myfile << setprecision( 4 ) << getEigenVal( i, j ) << '\t';
        }
        myfile << endl;
    }
    myfile << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;

    myfile << endl;
}

#if ( PITTPACKACC )
#pragma acc routine
#endif
PittPackReal PencilDcmp::getEigenVal( int i, int j )
{
    // note that we perform solve when the data is aligned in y direction
    // i and j are goind to be swapped in the thomas call

    int    ioffset;
    int    joffset;
    double onDiag;

    ioffset = myRank % p0 * nyChunk;
    joffset = myRank / p0 * nxChunk;

    double Cx = ( dxyz[2] / dxyz[0] ) * ( dxyz[2] / dxyz[0] );
    double Cy = ( dxyz[2] / dxyz[1] ) * ( dxyz[2] / dxyz[1] );

    //    double a[3] = {0.0, 0.0, 0.0};

    // modifyEigForDirichlet( i, j, a );
    // we multiply both sides bu del_z as the solution is performed in the z-direction

    onDiag = ( -2.0 + ( -2.0 + 2. * cosine( ( j + joffset + num[0] ) * freqs[0] / ( nx + denum[0] ) ) ) * Cx
               + ( -2.0 + 2. * cosine( ( i + ioffset + num[1] ) * freqs[1] / ( ny + denum[1] ) ) ) * Cy );

    return ( onDiag );
}



#if ( 0 )
// This is not used look for independent init function
PittPackResult PoissonGPU::initializeAndBind()
{
    PittPackResult result = SUCCESS;
#if ( PITTPACKACC )

    int nodalComSize;

    acc_init( acc_device_nvidia ); // OpenACC call

    const int numDev = acc_get_num_devices( acc_device_nvidia ); // #GPUs

    if ( numDev == 0 )
    {
        cout << "NO GPUs to run "
             << "\n";
    }

    nodalComSize = createNodalCommunicator();

    if ( nodalComSize == numDev )
    {
        const int devId = myRank % numDev;
        //  const int devId = 0;
        acc_set_device_num( devId, acc_device_nvidia ); // assign GPU to one MPI process
#if ( 1 )
        cout << "MPI process " << myRank << "  is assigned to GPU " << devId << "\n";
#endif
    }
    else
    {
        result = GPU_INIT_FAILURE;
    }

#endif

    return ( result );
}
#endif

//#pragma acc routine worker
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::preprocessSignalAccordingly( const int direction, const int counter )
{
    int size;
    int limit0;
    if ( direction == 0 )
    {
        limit0 = nyChunk;
        size   = nxChunk * nChunk;
    }
    else if ( direction == 1 )
    {
        limit0 = nxChunk;
        size   = nyChunk * nChunk;
    }

    if ( tags[counter] == 0 )
    {
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int k = 0; k < nzChunk * limit0; k++ )
        {
            Sig.preprocessSignalDST10( P, size, k, direction );
        }
    }
    else if ( tags[counter] == 1 )
    {
//#pragma acc loop worker
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int k = 0; k < nzChunk * limit0; k++ )
        {
            Sig.preprocessSignalDCT10( P, size, k, direction );
        }
    }
}
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::postprocessSignalAccordingly( const int direction, const int counter )
{
    int limit0;
    int size;
    if ( direction == 0 )
    {
        limit0 = nyChunk;
        size   = nChunk * nxChunk;
    }
    else if ( direction == 1 )
    {
        limit0 = nxChunk;
        size   = nChunk * nyChunk;
    }

    if ( tags[counter] == 0 )
    {
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int k = 0; k < limit0 * nzChunk; k++ )
        {
            Sig.postprocessSignalDST10( P, size, k, direction );
        }
    }
    else if ( tags[counter] == 1 )
    {
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int k = 0; k < nzChunk * limit0; k++ )
        {
            Sig.postprocessSignalDCT10( P, size, k, direction );
        }
    }
}
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::postprocessSignalAccordinglyReverse( const int direction, const int counter )
{
    int limit0;
    int size;
    if ( direction == 0 )
    {
        limit0 = nyChunk;
        size   = nChunk * nxChunk;
    }
    else if ( direction == 1 )
    {
        limit0 = nxChunk;
        size   = nChunk * nyChunk;
    }

    if ( tags[counter] == 0 )
    {
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int k = 0; k < nzChunk * limit0; k++ )
        {
            Sig.postprocessSignalDST01( P, size, k, direction );
        }
    }
    else if ( tags[counter] == 1 )
    {
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int k = 0; k < nzChunk * limit0; k++ )
        {
            Sig.postprocessSignalDCT01( P, size, k, direction );
        }
    }
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::preprocessSignalAccordinglyReverse( const int direction, const int counter )
{
    int limit0;
    int size;
    if ( direction == 0 )
    {
        limit0 = nyChunk;

        size = nChunk * nxChunk;
    }
    else if ( direction == 1 )
    {
        limit0 = nxChunk;
        size   = nChunk * nyChunk;
    }
    if ( tags[counter] == 0 )
    {
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int k = 0; k < nzChunk * limit0; k++ )
        {
            Sig.preprocessSignalDST01( P, size, k, direction );
        }
    }
    else if ( tags[counter] == 1 )
    {
#if ( PITTPACKACC )
#pragma acc loop gang
#endif
        for ( int k = 0; k < nzChunk * limit0; k++ )
        {
            Sig.preprocessSignalDCT01( P, size, k, direction );
        }
    }
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::modifyRhsDirichlet()
{
    double pi = 4. * arctan( 1.0 );
    double x, y, z;

    double Xa = coords[0];
    double Ya = coords[2];
    double Za = coords[4];
#if ( DEBUG )
    cout << BLUE << "rank " << myRank << " Xa " << Xa << " Ya " << Ya << " Za " << Za << RESET << endl;
#endif
    double c1, c2, c3;
    double shift = SHIFT;

    int Nx = nx / p0;
    int Ny = ny / p0;
    int Nz = nz;

    if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( Nx + 1. );
        c2 = ( coords[3] - coords[2] ) / ( Ny + 1. );
        c3 = ( coords[5] - coords[4] ) / ( Nz + 1. );
#if ( DEBUG )
        cout << " c1  " << c1 << " " << c2 << " " << c3 << endl;
#endif
    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }

#if ( DEBUG )
    cout << " rank " << myRank << " c1  " << c1 << " " << c2 << " " << c3 << endl;
    cout << " rank " << myRank << " N:  " << Nx << " " << Ny << " " << Nz << endl;
    cout << "xmin = " << Xa + 0 * c1 + shift * c1 * .5 << " xmax " << Xa + ( Nx - 1 ) * c1 + shift * c1 * .5 << endl;
    ;
#endif
    double omega[3] = {pi, pi, pi};
    // double val;
    // z direction, acting on facetags 5 and 6 with Zmin and Zmax
    if ( faceTag[4] == 1 )
    {
        z = Za;
        //      cout << YELLOW << " Zmin " << z << RESET << endl;
        int k = 0;
#if ( PITTPACKACC )
#pragma acc loop gang private( x, y )
#endif
        for ( int i = 0; i < nxChunk; i++ )
        {
            x = Xa + i * c1 + shift * c1 * .5;
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
                y = Ya + j * c2 + shift * c2 * 0.5;

                P( i, j, k )
                = P( i, j, k )
                  + 2 * exactValue( omega[0] * x, tags[0] ) * exactValue( omega[1] * y, tags[1] ) * exactValue( omega[2] * z, tags[2] );
            }
        }
    }

    if ( faceTag[5] == 1 )
    {
        z = Za + ( Nz - 1 ) * c3 + c3;

        //        cout << YELLOW << " Zmax " << z << RESET << endl;

        int k = Nz - 1;
#if ( PITTPACKACC )
#pragma acc loop gang private( x, y )
#endif
        for ( int i = 0; i < nxChunk; i++ )
        {
            x = Xa + i * c1 + shift * c1 * .5;
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
                y = Ya + j * c2 + shift * c2 * 0.5;

                P( i, j, k )
                = P( i, j, k )
                  + 2 * exactValue( omega[0] * x, tags[0] ) * exactValue( omega[1] * y, tags[1] ) * exactValue( omega[2] * z, tags[2] );
            }
        }
    }

    if ( faceTag[0] == 1 )
    {
        x     = Xa;
        int i = 0;
//        cout << YELLOW << " Xmin " << x << RESET << endl;
#if ( PITTPACKACC )
#pragma acc loop gang private( z, y )
#endif
        for ( int k = 0; k < nzChunk * nChunk; k++ )
        {
            z = Za + k * c3 + shift * c3 * .5;
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
                y = Ya + j * c2 + shift * c2 * 0.5;

                P( i, j, k )
                = P( i, j, k )
                  + 2 * exactValue( omega[0] * x, tags[0] ) * exactValue( omega[1] * y, tags[1] ) * exactValue( omega[2] * z, tags[2] );
            }
        }
    }

    if ( faceTag[1] == 1 )
    {
        int i = nxChunk - 1;
        x     = Xa + ( i + 1 ) * c1;
        //  cout << YELLOW << " Xmax " << x << RESET << endl;

#if ( PITTPACKACC )
#pragma acc loop gang private( z, y )
#endif
        for ( int k = 0; k < nzChunk * nChunk; k++ )
        {
            z = Za + k * c3 + shift * c3 * .5;
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
                y = Ya + j * c2 + shift * c2 * 0.5;
                P( i, j, k )
                = P( i, j, k )
                  + 2 * exactValue( omega[0] * x, tags[0] ) * exactValue( omega[1] * y, tags[1] ) * exactValue( omega[2] * z, tags[2] );
            }
        }
    }

    if ( faceTag[2] == 1 )
    {
        y     = Ya;
        int j = 0;

//    cout << YELLOW << " Ymin " << y << RESET << endl;
#if ( PITTPACKACC )
#pragma acc loop gang private( z, x )
#endif
        for ( int k = 0; k < nzChunk * nChunk; k++ )
        {
            z = Za + k * c3 + shift * c3 * .5;
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int i = 0; i < nxChunk; i++ )
            {
                x = Xa + i * c1 + shift * c1 * 0.5;
                P( i, j, k )
                = P( i, j, k )
                  + 2 * exactValue( omega[0] * x, tags[0] ) * exactValue( omega[1] * y, tags[1] ) * exactValue( omega[2] * z, tags[2] );
            }
        }
    }

    if ( faceTag[3] == 1 )
    {
        int j = nyChunk - 1;
        y     = Ya + ( j + 1 ) * c2;
// cout << YELLOW << " Ymax " << y << RESET << endl;
#if ( PITTPACKACC )
#pragma acc loop gang private( z, x )
#endif
        for ( int k = 0; k < nzChunk * nChunk; k++ )
        {
            z = Za + k * c3 + shift * c3 * .5;
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
            for ( int i = 0; i < nxChunk; i++ )
            {
                x = Xa + i * c1 + shift * c1 * 0.5;
                P( i, j, k )
                = P( i, j, k )
                  + 2 * exactValue( omega[0] * x, tags[0] ) * exactValue( omega[1] * y, tags[1] ) * exactValue( omega[2] * z, tags[2] );
            }
        }
    }
}

static int     createNodalCommunicator();
PittPackResult OPENACC_Init( int &my_rank, int &com_size ) /*!relates initializes and carries out CPU-GPU binding for PoissonGPU class*/
{
    PittPackResult result = SUCCESS;

#if ( PITTPACKACC )
    int nodalComSize;

    acc_init( acc_device_nvidia ); // OpenACC call

    const int num_dev = acc_get_num_devices( acc_device_nvidia ); // #GPUs

    nodalComSize = createNodalCommunicator();

    if ( nodalComSize <= num_dev && num_dev != 0 )
    {
        const int dev_id = my_rank % num_dev;
        acc_set_device_num( dev_id, acc_device_nvidia ); // assign GPU to one MPI process
                                                         //#if ( DEBUG )
        cout << "MPI process " << my_rank << "  is assigned to GPU " << dev_id << "\n";
        //#endif
    }
    else
    {
        cout << " number of GPU " << num_dev << " come_size " << com_size << endl;
        result = GPU_INIT_FAILURE;
    }

#endif
    return ( result );
}

const char *PittPackGetErrorEnum( PittPackResult error )
{
    switch ( error )
    {
        case SUCCESS:
            return "SUCCESS";
        case GPU_INIT_FAILURE:
            return ( "\t GPU_INIT_FAILURE: \n Number of GPUs is not equal to number of CPUs on every node,\n algorithm is designed for a "
                     "one-on-one binding" );
        case SOLVER_ARRAY_INCONSISTENT:
            return ( "\n Error in input, ZSIZE is not big enough to accomodate Thomas algorithm\n" );
        case MPI_INIT_FAIL:
            return ( "\n Error in MPI initialization\n " );
        case MPI_DUP_FAIL:
            return ( "\n Error in MPI Comm Duplication\n " );
        case COMSIZE_FAIL:
            return ( "\n Comsize sould be i*i, where 'i' is an integer\n " );
        case MPI_GET_RANK_FAIL:
            return ( "\n MPI_Comm_rank in PencilDcmp::MPIStartUp failed  \n " );
        case MPI_COMSIZE_FAIL:
            return ( "\n   MPI_Comm_size in PencilDcmp::MPIStartUp failed   \n " );
        case BLOCK_NUMBER_FAIL:
            return ( "\n   nChunk needs to be >=0   \n " );
        case ALLOCATION_FAIL:
            return ( "\n   failed to allocate memory \n " );

        case MPI_ERROR_HANDLE_FAIL:
            return ( "\n   failed to in setting error handler for MPI \n " );
        case THOMAS_FAIL:
            return ( "\n   Thomas algorithm Failed \n " );

        case INPUT_ARGS:
            return ( "\n   number of input arguments are not correct \n " );
        case CUFFT_FAIL_X:
            return ( "\n   CUFFT failed in X-transform  \n " );
        case CUFFT_FAIL_Y:
            return ( "\n   CUFFT failed in Y-transform \n " );
        case CUFFT_FAIL_INV_X:
            return ( "\n   CUFFT failed in inverse X-transform  \n " );
        case CUFFT_FAIL_INV_Y:
            return ( "\n   CUFFT failed in inverse Y-transform \n " );
    }

    return "<unknown>";
}

void PencilDcmp::currentDateTime()
{
    time_t    now = time( 0 );
    struct tm tstruct;
    // char       nameAppendix[80];
    tstruct = *localtime( &now );
    strftime( nameAppendix, sizeof( nameAppendix ), "%Y_%m_%d_%X", &tstruct );
    // return nameAppendix;
}

void PencilDcmp::runInfo()
{
    currentDateTime();

    long long int tmp0 = nxChunk;
    long long int tmp1 = nyChunk;
    long long int tmp2 = nzChunk;
    long long int tmp  = tmp0 * tmp1 * tmp2;

    long long int meshSize = nChunk * nChunk * nChunk * tmp;

    //    double meshSize = nChunk * nChunk * nChunk * nxChunk * nyChunk * nzChunk;

    if ( myRank == 10000 ) //cheng
    {
        ofstream    PittOut;
        std::string filename = "PittPack_";
        filename.append( nameAppendix );
        filename.append( "_out" );

        PittOut.open( filename );
        PittOut << "*********************************************************\n" << endl;
        PittOut << "              PittPack: " << nameAppendix << "        \n" << endl;
        PittOut << "*********************************************************\n" << endl;
#if ( PITTPACKACC )
        PittOut << "                     GPU-Solution                          \n" << endl;
#else
        PittOut << "                     CPU-Solution                          \n" << endl;
#endif
        PittOut << "---------------------------------------------------------\n" << endl;
        PittOut << "                     Parameters                        \n" << endl;
        PittOut << "---------------------------------------------------------\n" << endl;
        PittOut << "# Boundary Conditions : " << bc[0] << bc[1] << "-" << bc[2] << bc[3] << "-" << bc[4] << bc[5] << endl;
        PittOut << "# nxChunk = " << nxChunk << endl;
        PittOut << "# nyChunk = " << nyChunk << endl;
        PittOut << "# nzChunk = " << nzChunk << endl;
        PittOut << "# number of Chunks = " << nChunk << endl;
        PittOut << "# total number of cells  = " << FormatWithCommas( meshSize ) << " ~ " << setprecision( 2 ) << std::fixed
                << meshSize / ( 1e6 ) << " (M) " << endl;
        PittOut << "# number of cells per processor  = " << meshSize / p0 / p0 / ( 1.e6 ) << " (M) " << endl;
        PittOut << "# theoretical memory size  = " << meshSize * 16 / ( 1.e9 ) << " GB " << endl;
        PittOut << "# theoretical memory size per processor  = " << meshSize * 16 / ( 1.e9 * p0 * p0 ) << " (GB) " << endl;
        PittOut << "# theoretical memory size per processor plus extra container = "
                << ( meshSize + meshSize / p0 ) * 16 / ( 1.e9 * p0 * p0 ) << " (GB) " << endl;

        PittOut << "# frequencies for MMS  = [" << COEFF0 << "*pi, "
                << " " << COEFF1 << "*pi, "
                << " " << COEFF2 << "*pi]" << endl;

        if ( SOLUTIONMETHOD == 0 )
        {
            PittOut << "# Direct Solution Method in Z-direction  = "
                    << " Thomas " << endl;
        }
        else if ( SOLUTIONMETHOD == 1 )
        {
            PittOut << "# Direct Solution Method in Z-direction  = "
                    << " PCR  " << endl;
        }
        else if ( SOLUTIONMETHOD == 2 )
        {
            PittOut << "# Solution Method in Z-direction  = "
                    << " CR-P " << endl;
        }

#if ( PITTPACKACC )
        PittOut << "# number of GPUs = " << comSize << endl;
#else
        PittOut << "# number of processors = " << comSize << endl;
#endif
        PittOut << "# Solver Tag = " << SOLUTIONMETHOD << endl;

        PittOut << "# ZSIZE = " << ZSIZE << endl;
        PittOut << "# iaxSize = " << iaxSize << endl;
        PittOut << "# jaxSize = " << jaxSize << endl;
        PittOut << "# iaySize = " << iaySize << endl;
        PittOut << "# jaySize = " << jaySize << endl;
#if ( COMM_PATTERN == 0 )
        PittOut << "# COMM Pattern = Pairwise Exchange" << endl;
#elif ( COMM_PATTERN == 1 )
        PittOut << "# COMM Pattern = Neighborhood Collective " << endl;
#else
        PittOut << "# COMM Pattern = Waitany " << endl;
#endif
        PittOut << "# Code used short/int for CRS matrices =  " << SHORT_ << endl;
        if ( I_O == 0 )
        {
            PittOut << "# I/O = OFF"
                    << "\n"
                    << endl;
        }
        else
        {
            PittOut << "# I/O =  ON"
                    << "\n"
                    << endl;
        }

        if ( INCLUDE_ERROE_CAL_IN_TIMING == 1 )
        {
            PittOut << "---------------------------------------------------------\n" << endl;
            PittOut << "               Global Error  = " << setprecision( 12 ) << finalErr << "\n" << endl;
        }
        PittOut << "---------------------------------------------------------\n" << endl;
        PittOut << "                Wall-Time per Iteration  = " << runTime << " (s) "
                << "\n"
                << endl;
        PittOut << "                Number of Iteration  = " << NITER << "\n" << endl;

        PittOut << "---------------------------------------------------------\n" << endl;
        cout << "Grid size= [ " << nxChunk * nChunk << " * " << nyChunk * nChunk << " * " << nzChunk * nChunk << "], error = " << finalErr
             << ", time = " << runTime << " (s) " << endl;
    }
}


// clear the tem x3 vector in agang rather than in the thread inside sheramn morrison
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::clear( double *x )
{
#if ( THOM_FULL_BATCH == 1 )
    int size = nxChunk * nyChunk * nz;
#else
    int size = gangTri * nz;
#endif

#if ( PITTPACKACC )
#pragma acc loop
#endif
    for ( int i = 0; i < size; i++ )
    {
        x[i] = 0.0;
    }
}

#if ( THOM_FULL_BATCH == 0 )
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
int PencilDcmp::solveThmBatch( const int index )
{
    double eig;

    for ( int j = 0; j < nxChunk; j++ )
    {
/*
    if(bc[4]=='P')
    {
    clear(x3);
    }
*/
// calculate i and j
#if ( PITTPACKACC )
#pragma acc loop private( eig )
#endif
        for ( int i = 0; i < nyChunk; i++ )
        {
            eig = getEigenVal( i, j );

            fillInArrayContig( i, j, index, x1 + i * nz );

            //     cout<< RED<<"solveThmBatch" <<RESET<<endl;
            if ( bc[4] != 'P' )
            {
                T.thomasLowMem( x2 + i * nz, x1 + nz * i, eig, index );
                // cout<< RED<<"solveThmBatch 0" <<RESET<<endl;
            }
            else
            {
//                T.shermanMorrisonThomasV1( x2 + i * nz, x1 + nz * i, x3 + i * nz, eig, 0.0, 1.0, index );
                T.shermanMorrisonThomas( x2 + i * nz, x1 + nz * i, x3 + i * nz, eig, 1.0, 1.0, index );
                // cout<< RED<<"solveThmBatch 0 N" <<RESET<<endl;
            }
            fillInArrayBack( i, j, index, x1 + nz * i );
        }
    }

    return ( 0 );
}
#else
#if ( PITTPACKACC )
#pragma acc routine gang
#endif
int PencilDcmp::solveThmBatch( const int index )
{
    double eig;

    /*
        if(bc[4]=='P')
        {
        clear(x3);
        }
    */

    //    cout<< RED<<"solveThmBatch" <<RESET<<endl;

    //    clear(x2);

    int count = 0;
    int i, j;
#if ( PITTPACKACC )
#pragma acc loop private( eig, i, j )
#endif
    for ( int l = 0; l < nyChunk * nxChunk; l++ )
    {
        i = l % nyChunk;
        j = l / nyChunk;
        eig = getEigenVal( i, j );
        fillInArrayContig( i, j, index, x1 + l * nz );
        // T.thomasLowMem( x2 + l * nz, x1 + nz * l, eig, index );

        //     cout<< RED<<"solveThmBatch" <<RESET<<endl;
        if ( bc[4] != 'P' )
        {
            T.thomasLowMem( x2 + l * nz, x1 + nz * l, eig, index );
        }
        else
        {
            //  cout<< RED<<"solve full batch N" <<RESET<<endl;
            //   this is too strong for enforcing boundries.
            //    T.shermanMorrisonThomas( x2 + l * nz, x1 + nz * l, x3+l*nz ,eig, 0.0 ,-eig, index );
           // T.shermanMorrisonThomasV1( x2 + l * nz, x1 + nz * l, x3 + l * nz, eig, 0.0, 1.0, index );
             T.shermanMorrisonThomas( x2 + l * nz, x1 + nz * l, x3 + l * nz, eig, 1.0, 1.0, index );
        }

        fillInArrayBack( i, j, index, x1 + nz * l );
    }
    return ( 0 );
}

#endif

#if ( PITTPACKACC )
#pragma acc routine worker
#endif
void PencilDcmp::fillInArrayContig( const int i, const int j, int index, double *container )
{
    //    cout<<"before inside fill in"<<endl;

#if ( PITTPACKACC )
#pragma acc loop worker
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
            if ( index == 0 )
            {
                container[id * nzChunk + k] = P( id, 3, i, j, k, index );
            }
            else
            {
                container[id * nzChunk + k] = P( id, 3, i, j, k, index );
            }
        }
    }
    // this is not correct if you are trying to enforce at the cell-face
    // if not you need to use this.
    /*
       container[0] =0.0;
       container[nz-1] =0.0;
    */
    //  a) no modificatoin is needed for the Dirichlet BC

    //  b) sets the righ hand side to zero to enforce Neumann BC
    // letf hand side modification is done inside thomasLowMem
    /*
        if( bc[4] == 'N' )
        {
           container[0] =0.0;
        }
        if( bc[5] == 'N' )
        {
           container[nz-1] =0.0;
        }
    */
}

#if ( PITTPACKACC )
#pragma acc routine worker
#endif
void PencilDcmp::fillInArrayBack( const int i, const int j, const int index, double *container )
{
//     cout<<"after"<<endl;
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
            if ( index == 0 )
            {
                P( id, 3, i, j, k, index ) = container[id * nzChunk + k];
            }
            else
            {
                P( id, 3, i, j, k, index ) = container[id * nzChunk + k];
            }
        }
    }
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::solvePCR( const int index )
{
    double eig;
    // int    count = 0;
    // int    i, j;
#if ( PITTPACKACC )
#pragma acc loop seq
#endif
    for ( int j = 0; j < nxChunk; j++ )
    {
// calculate i and j
#if ( PITTPACKACC )
#pragma acc loop private( eig )
#endif
        for ( int i = 0; i < nyChunk; i++ )
        {
            eig = getEigenVal( i, j );

            fillInArrayContigNormalize( i, j, index, x1 + nz * i, eig );

            imposeBoundaryonContainer( i, j, index, eig, x1 + nz * i );

            offDiagNormalize( i, j, x2 + nz * i, crpcr_upper + nz * i, eig );

            imposeBoundaryonOffDiag( eig, x2 + i * nz, crpcr_upper + nz * i );

            T.pcr( nz, x2 + i * nz, crpcr_upper + nz * i, x1 + nz * i );

            fillInArrayBack( i, j, x1 + nz * i, index );
        }
    }
}

#if ( PITTPACKACC )
#pragma acc routine gang
#endif
void PencilDcmp::solveCRP( const int index )
{
    double eig;

    // int    count = 0;
    double offdiag;

    // maximum level is set to log2(16384)+1=15
    double tmpA[15];
    double tmpC[15];

    double ThmB[3]   = {1.0, 1.0, 1.0};
    double ThmA[3]   = {0.0, 0.0, 0.0};
    double ThmC[3]   = {0.0, 0.0, 0.0};
    double gam1[3]   = {0.0, 0.0, 0.0};
    double tmpRHS[3] = {0.0, 0.0, 0.0};

#if ( PITTPACKACC )
#pragma acc loop seq
#endif
    for ( int j = 0; j < nxChunk; j++ )
    {
// calculate i and j
#if ( PITTPACKACC )
#pragma acc loop private( offdiag, eig )
#endif
        for ( int i = 0; i < nyChunk; i++ )
        {
            eig = getEigenVal( i, j );

            offdiag = 1. / eig;

            fillInArrayContigNormalize( i, j, index, x1 + nz * i, eig );

            imposeBoundaryonCRPTmp( i, j, index, eig, x1 + nz * i );
            T.crp( nz, offdiag, tmpA, tmpC, tmpRHS, ThmA, ThmC, ThmB, gam1, x1 + nz * i );

            fillInArrayBack( i, j, x1 + nz * i, index );
        }
    }
}

#if ( PITTPACKACC )
#pragma acc routine worker
#endif
void PencilDcmp::fillInArrayContigNormalize( const int i, const int j, int index, double *container, double eig )
{
    double eigInv = 1. / eig;

#if ( PITTPACKACC )
#pragma acc loop worker firstprivate( eigInv )
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
            if ( index == 0 && ( ( id * nzChunk + k ) != ( nz - 1 ) ) && ( ( id * nzChunk + k ) != 0 ) )
            {
                container[id * nzChunk + k] = P( id, 3, i, j, k, index ) * eigInv;
            }
            else if ( index == 1 && ( ( id * nzChunk + k ) != ( nz - 1 ) ) && ( ( id * nzChunk + k ) != 0 ) )
            {
                container[id * nzChunk + k] = P( id, 3, i, j, k, index ) * eigInv;
            }
        }
    }
    //        container[0] = P( 0, 3, i, j, 0, index ) / ( eig - 1. );
    //        container[nz - 1] = P( nChunk - 1, 3, i, j, nz - 1, index ) / ( eig - 1. );

    //   imposeBoundaryonContainer( i, j,index,eig,container);
}

#if ( PITTPACKACC )
#pragma acc routine seq
#endif
void PencilDcmp::imposeBoundaryonCRPTmp( int i, int j, int index, double eig, double *container )
{
    // Dirichlet BC
    // for now only workd for Dirichlet with ghost cell
    if ( bc[4] == 'D' )
    {
        container[0] = P( 0, 3, i, j, 0, index ) / ( eig );
    }
    if ( bc[5] == 'D' )
    {
        container[nz - 1] = P( nChunk - 1, 3, i, j, nzChunk - 1, index ) / ( eig );
    }

    // Neuman BC
    // to be implemented

    /*
            container[0] = P( 0, 3, i, j, 0, index ) / ( eig + 1. );
            container[nz - 1] = P( nChunk - 1, 3, i, j, nzChunk - 1, index ) / ( eig + 1. );
    */
}

#if ( PITTPACKACC )
#pragma acc routine seq
#endif
void PencilDcmp::imposeBoundaryonContainer( int i, int j, int index, double eig, double *container )
{
    // Dirichlet BC
    if ( bc[4] == 'D' )
    {
        container[0] = P( 0, 3, i, j, 0, index ) / ( eig - 1. );
    }
    if ( bc[5] == 'D' )
    {
        container[nz - 1] = P( nChunk - 1, 3, i, j, nzChunk - 1, index ) / ( eig - 1. );
    }

    // Neuman BC

    if ( bc[4] == 'N' )
    {
        container[0] = P( 0, 3, i, j, 0, index ) / ( eig + 1. );
    }

    if ( bc[5] == 'N' )
    {
        container[nz - 1] = P( nChunk - 1, 3, i, j, nzChunk - 1, index ) / ( eig + 1. );
    }

    /*
            container[0] = P( 0, 3, i, j, 0, index ) / ( eig + 1. );
            container[nz - 1] = P( nChunk - 1, 3, i, j, nzChunk - 1, index ) / ( eig + 1. );
    */
}

#if ( PITTPACKACC )
#pragma acc routine vector
#endif
void PencilDcmp::offDiagNormalize( const int i, const int j, double *lower, double *upper, double eig )
{
    double eigInv = 1. / eig;
    // dirichlet

#if ( PITTPACKACC )
#pragma acc loop vector
#endif
    for ( int k = 1; k < nz; k++ )
    {
        lower[k] = eigInv;
    }
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
    for ( int k = 0; k < nz - 1; k++ )
    {
        upper[k] = eigInv;
    }

    //    imposeBoundaryonOffDiag(eig,lower,upper);

    /*
         upper[0] = 1. / ( eig - 1. );
         lower[nz - 1] = 1. / ( eig - 1. );
    */
    // boundary conditions affects the normalization
    //#pragma acc loop seq firstprivate(eig)
    // for(int i=0;i<1;i++)
    //{
    /*
        if ( bc[4] == 'D' )
        {
            upper[0] = 1. / ( eig - 1. );
        }
        if ( bc[5] == 'D' )
        {
            lower[nz - 1] = 1. / ( eig - 1. );
        }

        if ( bc[4] == 'N' )
        {
            upper[0] = 1. / ( eig + 1. );
        }
        if ( bc[5] == 'N' )
        {
            lower[nz - 1] = 1. / ( eig + 1. );
        }
    */
    //}
}

#if ( PITTPACKACC )
#pragma acc routine seq
#endif
void PencilDcmp::imposeBoundaryonOffDiag( double eig, double *lower, double *upper )
{
    lower[0]      = 0.0;
    upper[nz - 1] = 0.0;

    if ( bc[4] == 'D' )
    {
        upper[0] = 1. / ( eig - 1. );
    }
    if ( bc[5] == 'D' )
    {
        lower[nz - 1] = 1. / ( eig - 1. );
    }

    if ( bc[4] == 'N' )
    {
        upper[0] = 1. / ( eig + 1. );
    }
    if ( bc[5] == 'N' )
    {
        lower[nz - 1] = 1. / ( eig + 1. );
    }
}

#if ( PITTPACKACC )
#pragma acc routine worker
#endif
void PencilDcmp::fillInArrayBack( const int i, const int j, double *container, const int index )
{
//     cout<<"after"<<endl;
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
            if ( index == 0 )
            {
                P( id, 3, i, j, k, index ) = container[id * nzChunk + k + OFFS];
            }
            else
            {
                P( id, 3, i, j, k, index ) = container[id * nzChunk + k + OFFS];
            }
        }
    }
}

static int extractIntegerWords( string str )
{
    std::string temp;
    int         number = 0;

    for ( unsigned int i = 0; i < str.size(); i++ )
    {
        if ( isdigit( str[i] ) )
        {
            for ( unsigned int a = i; a < str.size(); a++ )
            {
                temp += str[a];
            }
            break;
        }
    }
    std::istringstream stream( temp );
    stream >> number;

    return ( number );
}

/*! \brief Helper Function to
 *         check one-on-one mapping between CPU-GPU.
 *
 *  1. Processor name is obtained using MPI_Get_processor_name()
 *  2. Node numbers are extracted from the string
 *  3. Local nodal communicator is constricted using MPI_Comm_split()
 *  4. The size of this new communicator is returned to OpenACCINIT
 *  5. The size of the nodl communicator should be equal to number of GPU's
 *  6. If not, error code 1 is thrown
 */

static int createNodalCommunicator()
{
    MPI_Comm nodalComm;

    int my_rank, np, my_new_rank, new_np;
    int color, key, len;

    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank );
    MPI_Comm_size( MPI_COMM_WORLD, &np );

    char pname[256];

    MPI_Get_processor_name( pname, &len );

    // long int d;

    // some stream wizardry to extract node number from the proc name
    stringstream ss;
    string       s;
    ss << pname;
    ss >> s;
    //   cout<<s<<endl;
    color = extractIntegerWords( s );
    key   = my_rank;

    printf( "PROCESS %d of %d: Running on %s  color= %d len=%d\n", my_rank + 1, np, pname, color, len );

    MPI_Comm_split( MPI_COMM_WORLD, color, key, &nodalComm );
    MPI_Comm_rank( nodalComm, &my_new_rank );
    MPI_Comm_size( nodalComm, &new_np );

    printf( "Process %d (in COMM_WORLD) is process %d (in split communicator)\n", my_rank, my_new_rank );
    printf( "Process %d (in COMM_WORLD): COMM_WORLD has %d processes\n", my_rank, np );
    printf( "Process %d (in COMM_WORLD): split communicator has %d processes\n", my_rank, new_np );

    MPI_Comm_free( &nodalComm );

    return ( new_np );
}

void PencilDcmp::assignRhs( double *rhs )
{
    double c1, c2, c3;
    double shift = SHIFT;

    int Nx = nxChunk;
    int Ny = nyChunk;
    int Nz = nz;

    if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( Nx + 1. );
        c2 = ( coords[3] - coords[2] ) / ( Ny + 1. );
        c3 = ( coords[5] - coords[4] ) / ( Nz + 1. );
#if ( DEBUG )
        cout << " c1  " << c1 << " " << c2 << " " << c3 << endl;
#endif
    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }

#if ( PITTPACKACC )
#pragma acc data copy( rhs [0:Nx * Ny * Nz] ) copyin( Nz, Ny, Nx ) present( P.P [0:2 * Nx * Ny * Nz] )
#endif
    {
#if ( PITTPACKACC )
#pragma acc parallel loop gang
#endif
//    for(int id=0;id<nChunk;id++)
  //  {
        for ( int k = 0; k < Nz; k++ )
        {
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < Ny; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 0; i < Nx; i++ )
                {
                    P(i, j, k, 0 ) = rhs[i + Nx * j + Nx * Ny * k]* c3 * c3;
                    P(i, j, k, 1 ) = 0.0;
 //                   cout<<P(i,j,k)<<endl;;
                }
            }
        }
     }
    
}


void PencilDcmp::subtractMeanValue()
{
    int    Nx  = nxChunk;
    int    Ny  = nyChunk;
    int    Nz  = nz;
    double sum = 0.0;
    for ( int k = 0; k < Nz; k++ )
    {
        for ( int j = 0; j < Ny; j++ )
        {
            for ( int i = 0; i < Nx; i++ )
            {
                sum = sum + P( i, j, k, 0 );
            }
        }
    }
    mean = 0.0;

    MPI_Allreduce( &sum, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD );

    mean = mean / nxChunk / nyChunk / nzChunk / nChunk / nChunk / nChunk;

    cout << "mean value " << mean << endl;

    for ( int k = 0; k < Nz; k++ )
    {
        for ( int j = 0; j < Ny; j++ )
        {
            for ( int i = 0; i < Nx; i++ )
            {
                P( i, j, k, 0 ) -= mean;
            }
        }
    }
}

void PencilDcmp::fillTrigonometric( double *rhs )
{
    double pi = 4. * arctan( 1.0 );
    double x, y, z;

    double Xa = coords[0];
    double Ya = coords[2];
    double Za = coords[4];
    double c1, c2, c3;
    double shift = SHIFT;

    int Nx = nxChunk;
    int Ny = nyChunk;
    int Nz = nz;

    if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( Nx + 1. );
        c2 = ( coords[3] - coords[2] ) / ( Ny + 1. );
        c3 = ( coords[5] - coords[4] ) / ( Nz + 1. );
    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }

    double omega[3] = {COEFF0 * pi, COEFF1 * pi, COEFF2 * pi};

    for ( int k = 0; k < Nz; k++ )
    {
        if ( !SHIFT )
        {
            z = Za + ( k + 1 ) * c3;
        }
        else
        {
            z = Za + k * c3 + shift * c3 * 0.5;
        }

        for ( int j = 0; j < Ny; j++ )
        {
            if ( !SHIFT )
            {
                y = Ya + ( j + 1 ) * c2;
            }
            else
            {
                y = Ya + j * c2 + shift * c2 * 0.5;
            }

            for ( int i = 0; i < Nx; i++ )
            {
                if ( !SHIFT )
                {
                    x = Xa + ( i + 1 ) * c1;
                }
                else
                {
                    x = Xa + i * c1 + shift * c1 * .5;
                }

                rhs[i + j * Nx + Nx * Ny * k] = 4*pi*pi*sin(2*pi*x); /* ( ( omega[0] * omega[0] ) * exactValue( omega[0] * x, tags[0] )
                                                   * exactValue( omega[1] * y, tags[1] ) * exactValue( omega[2] * z, tags[2] )
                                                   + ( omega[1] * omega[1] ) * exactValue( omega[0] * x, tags[0] )
                                                     * exactValue( omega[1] * y, tags[1] ) * exactValue( omega[2] * z, tags[2] )
                                                   + ( omega[2] * omega[2] ) * exactValue( omega[0] * x, tags[0] )
                                                     * exactValue( omega[1] * y, tags[1] ) * exactValue( omega[2] * z, tags[2] ) )*/;
/*
            if(i==0 || i==(Nz-1))
            {
              cout<<x<<" "<<rhs[i + j * Nx + Nx * Ny * k]<<endl;
            }
*/

            }

        }
    }
}


double PencilDcmp::getValue(int id, int i, int j, int k,int index) 
{
  //return P[2 * ( id * chunkSize / 2 + i + nx * j + nx * ny * k )+index ];
   return  P(id,0,i,j,k,index ); //0 specified as default direction
//return P[2 * ( i + nx * j + nx * ny * k ) + index] ; 
}

void PencilDcmp::setValue(double v,int id, int i, int j, int k,int index) 
{
  //return P[2 * ( id * chunkSize / 2 + i + nx * j + nx * ny * k )+index ];
  P(id,0,i,j,k,index )=v; //0 specified as default direction
//return P[2 * ( i + nx * j + nx * ny * k ) + index] ; 
}

void PencilDcmp::print()
{
    int Nx = nxChunk;
    int Ny = nyChunk;
    int Nz = nz;

    cout << RED << " inside  " << RESET << endl;
    for ( int k = 0; k < Nz; k++ )
    {
        for ( int j = 0; j < Ny; j++ )
        {
            for ( int i = 0; i < Nx; i++ )
            {
                cout << P( i, j, k ) << '\t';
                ;
            }
            cout << endl;
        }
    }
}

/*PencilDcmp PencilDcmp::operator+(const PencilDcmp& b)
{
     PencilDcmp pResult(nx,ny,nz,p0,p0);
//     pResult.setBox(this->Xbox);
    
  //   int dir=2;
  //   pResult.setCoords(dir);

    for ( int id = 0; id < b.nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < b.nzChunk; k++ )
        {
//            assignTempX2Y( id, k, tmp );
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < b.nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 0; i < b.nxChunk; i++ )

                {
                    //          P( id, 1, i, j, k, 0 ) = 1.0;
                  //  pResult.P( id, 0, i, j, k, 0 ) =  P(id,0,i,j,k,0) + b.P.getValue(id,  i, j, k, 0);//default dir =0
                  //  pResult.P( id, 0, i, j, k, 1 ) =  P(id,0,i,j,k,1) + b.P.getValue(id,  i, j, k, 1);
                }
            }
        }
    }
 return pResult; 
} */


void CFlowVariable::initializeField()
{
    double c1, c2, c3;
    double shift = 0;
    double x,y,z;
    double pi=3.14159;
    int Nx = nxChunk;
    int Ny = nyChunk;
    int Nz = nz;

    double Xa = coords[0];
    double Ya = coords[2];
    double Za = coords[4];
    if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( Nx + 1. );
        c2 = ( coords[3] - coords[2] ) / ( Ny + 1. );
        c3 = ( coords[5] - coords[4] ) / ( Nz + 1. );
#if ( DEBUG )
        cout << " c1  " << c1 << " " << c2 << " " << c3 << endl;
#endif
    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }
      
#if ( PITTPACKACC )
#pragma acc data copy( rhs [0:Nx * Ny * Nz] ) copyin( Nz, Ny, Nx ) present( P.P [0:2 * Nx * Ny * Nz] )
#endif
    
  initializeZero();
}

CFlowVariable::CFlowVariable(PencilDcmp *Field)
{
      nx=Field->getnx();
      ny=Field->getny();
      nz=Field->getnz();
      p0=Field->getp0(); 

      nxChunk=Field->getnxChunk();
      nyChunk=Field->getnyChunk();
      nzChunk=Field->getnzChunk();
      ChunkedArray *Values=Field->getP();
      int i,j,k;

        for ( int k = 0; k < nzChunk; k++ )
        {
         
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
               
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 0; i < nxChunk; i++ )
                {
                 
                   P(i,j,k,0) = (*Values)(i,j,k,0);
                   P(i,j,k,1) = (*Values)(i,j,k,1);
                }
            }

        }
}

/*CFlowVariable CFlowVariable:: operator +(const CFlowVariable& A)
{
      int  nxAChunk=A.getnxChunk();
      int  nyAChunk=A.getnyChunk();
      int  nzAChunk=A.getnzChunk();


      if(this->nxChunk!=nxAChunk ||this->nyChunk!=nyAChunk|| this->nzChunk!=nzAChunk)
         throw "Fields of different sizes cannot be summed up!";
    

      
      ChunkedArray *ValueA=A.getP();
      
      CFlowVariable Result(&A);
      ChunkedArray ResultArray=*(Result.getP());
      int i,j,k;

        for ( int k = 0; k < this->nzChunk; k++ )
        {
         
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < this->nyChunk; j++ )
            {
               
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 0; i < this->nxChunk; i++ )
                {
                 
                   ResultArray(i,j,k,0) = (*ValueA)(i,j,k,0)+ this->P(i,j,k,0);
                   ResultArray(i,j,k,1) = (*ValueA)(i,j,k,1)+ this->P(i,j,k,1);
                }
            }

        }

    return Result;
}*/


void CFlowVariable::computeGradX()
{
    
    if(GradX==NULL)
    {
      GradX=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0);
      GradX->setBox(this->Xbox);
//    cout<<"Xbox elements ="<<GradX->Xbox[0]<<" "<<GradX->Xbox[1]<<" "<<GradX->Xbox[2]<<" "<<GradX->Xbox[3]<<" "<<GradX->Xbox[4]<<"\n"; 
       int dir=2;
       GradX->setCoords(dir);
    }
    int rank_x = myRank%p0;
    double c1=GradX->dxyz[0]; 
    updateGhostsEW();

//    cout<<"nChunk, x,y,k,dx="<<nChunk<<" "<<nxChunk<<" "<<nyChunk<<" "<<nzChunk<<" "<<GradX->dxyz[0]<<"\n"; 
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
//            assignTempX2Y( id, k, tmp );
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 1; i < nxChunk-1; i++ )//interior points
                {
                   GradX->P(id,0,i,j,k,0) =(this->P(id,0,i+1,j,k,0)-P(id,0,i-1,j,k,0))/(2*c1) ;
                   GradX->P(id,0,i,j,k,1) = 0 ;                 
                }

             //   if(rank_x!=p0-1) //right side processor interface not on the rightmost processor  
             //   {
                  GradX->P(id,0,nxChunk-1,j,k,0) = (EastGhosts[j + nyChunk * k+ id*nyChunk*nzChunk]-P(id,0,nxChunk-2,j,k,0))/(2*c1) ;
                  GradX->P(id,0,nxChunk-1,j,k,1) = 0 ;                 
              /*  }
                else //right side processor interface on the right most processor
                { //insert bc treatment here
                   GradX->P(id,0,nxChunk-1,j,k,0) = (EastBoundary[j + nyChunk * k+ id*nyChunk*nzChunk]-P(id,0,nxChunk-2,j,k,0))/(2*c1) ;
                   GradX->P(id,0,nxChunk-1,j,k,1) = 0 ;                
  //                 cout<<"East Boundary value = "<<  EastBoundary[j + nyChunk * k+ id*nyChunk*nzChunk]<<"\n ???????????????????????????\n"; 
                }*/
                  
             //   if(rank_x!=0) //left side processor interface not on the leftmost processor  
             //   {
                  GradX->P(id,0,0,j,k,0) = (P(id,0,1,j,k,0)  -WestGhosts[j + nyChunk * k+ id*nyChunk*nzChunk] )/(2*c1) ;
                  GradX->P(id,0,0,j,k,1) = 0 ;                 
              /*  }
                else
                {
                  GradX->P(id,0,0,j,k,0) = (P(id,0,1,j,k,0)  -WestBoundary[j + nyChunk * k+ id*nyChunk*nzChunk] )/(2*c1) ;
                  GradX->P(id,0,0,j,k,1) = 0 ;                 
//                  cout<<"West Boundary value = "<<  WestBoundary[j + nyChunk * k+ id*nyChunk*nzChunk]<<"\n ???????????????????????????\n"; 
                }*/
            }
        }
    }

}

void CFlowVariable::computedX2()
{
    
    if(Grad2X==NULL)
    {
      Grad2X=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0);
      Grad2X->setBox(this->Xbox);
//    cout<<"Xbox elements ="<<GradX->Xbox[0]<<" "<<GradX->Xbox[1]<<" "<<GradX->Xbox[2]<<" "<<GradX->Xbox[3]<<" "<<GradX->Xbox[4]<<"\n"; 
       int dir=2;
       Grad2X->setCoords(dir);
    }
    int rank_x = myRank%p0;
    double c1=Grad2X->dxyz[0]; 
    updateGhostsEW();

//    cout<<"nChunk, x,y,k,dx="<<nChunk<<" "<<nxChunk<<" "<<nyChunk<<" "<<nzChunk<<" "<<GradX->dxyz[0]<<"\n"; 
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
//            assignTempX2Y( id, k, tmp );
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 1; i < nxChunk-1; i++ )//interior points
                {
                   Grad2X->P(id,0,i,j,k,0) =(P(id,0,i+1,j,k,0)-2*P(id,0,i,j,k,0)+P(id,0,i-1,j,k,0) )/(c1*c1) ;
                   Grad2X->P(id,0,i,j,k,1) = 0 ;                 
                }

             //   if(rank_x!=p0-1) //right side processor interface not on the rightmost processor  
             //   {
                  Grad2X->P(id,0,nxChunk-1,j,k,0) = (EastGhosts[j + nyChunk * k+ id*nyChunk*nzChunk]-2*P(id,0,nxChunk-1,j,k,0) + P(id,0,nxChunk-2,j,k,0) )/(c1*c1) ;
                  Grad2X->P(id,0,nxChunk-1,j,k,1) = 0 ;                 
              /*  }
                else //right side processor interface on the right most processor
                { //insert bc treatment here
                   GradX->P(id,0,nxChunk-1,j,k,0) = (EastBoundary[j + nyChunk * k+ id*nyChunk*nzChunk]-P(id,0,nxChunk-2,j,k,0))/(2*c1) ;
                   GradX->P(id,0,nxChunk-1,j,k,1) = 0 ;                
  //                 cout<<"East Boundary value = "<<  EastBoundary[j + nyChunk * k+ id*nyChunk*nzChunk]<<"\n ???????????????????????????\n"; 
                }*/
                  
             //   if(rank_x!=0) //left side processor interface not on the leftmost processor  
             //   {
                  Grad2X->P(id,0,0,j,k,0) = (P(id,0,1,j,k,0) -2*P(id,0,0,j,k,0) +WestGhosts[j + nyChunk * k+ id*nyChunk*nzChunk] )/(c1*c1) ;
                  Grad2X->P(id,0,0,j,k,1) = 0 ;                 
              /*  }
                else
                {
                  GradX->P(id,0,0,j,k,0) = (P(id,0,1,j,k,0)  -WestBoundary[j + nyChunk * k+ id*nyChunk*nzChunk] )/(2*c1) ;
                  GradX->P(id,0,0,j,k,1) = 0 ;                 
//                  cout<<"West Boundary value = "<<  WestBoundary[j + nyChunk * k+ id*nyChunk*nzChunk]<<"\n ???????????????????????????\n"; 
                }*/
            }
        }
    }

}


void CFlowVariable::computeGradY()
{
     
      double x,y,z,u,v,w,c1,c2,c3;
     double *coords=getCoords();
     double *dxyz = getdxyz();
     double Xa = coords[0];
     double Ya = coords[2];
     double Za = coords[4];
     double pi=3.14159265358979;

   if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( nxChunk*p0 + 1. );
        c2 = ( coords[3] - coords[2] ) / ( nyChunk*p0 + 1. );
        c3 = ( coords[5] - coords[4] ) / ( nzChunk*p0 + 1. );

    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }
    if(GradY==NULL)
    {
      GradY=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0);
      GradY->setBox(this->Xbox);
//    cout<<"Xbox elements ="<<GradX->Xbox[0]<<" "<<GradX->Xbox[1]<<" "<<GradX->Xbox[2]<<" "<<GradX->Xbox[3]<<" "<<GradX->Xbox[4]<<"\n"; 
       int dir=2;
       GradY->setCoords(dir);
    }
    updateGhostsNS();
    int rank_y=myRank/p0;
//    double c2=GradY->dxyz[1]; 

//    cout<<"nChunk, x,y,k,dx="<<nChunk<<" "<<nxChunk<<" "<<nyChunk<<" "<<nzChunk<<" "<<GradX->dxyz[0]<<"\n"; 
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
//            assignTempX2Y( id, k, tmp );
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int i = 0; i < nxChunk; i++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int j = 1; j < nyChunk-1; j++ )//interior points
                {
                   y = Ya + (j+1) * c2 - SHIFT * c2 * .5;
                   GradY->P(id,0,i,j,k,0) =(this->P(id,0,i,j+1,k,0)-P(id,0,i,j-1,k,0))/(2*c2);//-2*pi*cos(2*pi*y) ;
                   GradY->P(id,0,i,j,k,1) = 0 ;                 
                }

             //   if(rank_y!=p0-1) //right side processor interface not on the rightmost processor  
              //  {
                  y = Ya + nyChunk * c2 - SHIFT * c2 * .5;
                  GradY->P(id,0,i,nyChunk-1,k,0) = (SouthGhosts[i + nxChunk * k + id*nxChunk*nzChunk]-P(id,0,i,nyChunk-2,k,0))/(2*c2);// -2*pi*cos(2*pi*y) ;
                  GradY->P(id,0,i,nyChunk-1,k,1) = 0 ;                 
              /*  }
                else //right side processor interface on the right most processor
                { //insert bc treatment here
                   y = Ya + nyChunk * c2 - SHIFT * c2 * .5;
                  GradY->P(id,0,i,nyChunk-1,k,0) = (SouthBoundary[i + nxChunk * k + id*nxChunk*nzChunk]-P(id,0,i,nyChunk-2,k,0))/(2*c2);// -2*pi*cos(2*pi*y) ;
                  GradY->P(id,0,i,nyChunk-1,k,1) = 0 ;                  
                }*/
                  
             //   if(rank_y!=0) //left side processor interface not on the leftmost processor  
             //   {
                  y = Ya + (0+1) * c2 - SHIFT * c2 * .5;
                  GradY->P(id,0,i,0,k,0) = (P(id,0,i,1,k,0)-NorthGhosts[i + nxChunk * k + id*nxChunk*nzChunk])/(2*c2);// -2*pi*cos(2*pi*y) ;
                  GradY->P(id,0,i,0,k,1) = 0 ;                 
              /*  }
                else
                {
                  y = Ya + (0+1) * c2 - SHIFT * c2 * .5;
                  GradY->P(id,0,i,0,k,0) = (P(id,0,i,1,k,0)-NorthBoundary[i + nxChunk * k + id*nxChunk*nzChunk])/(2*c2);//-2*pi*cos(2*pi*y)  ;
                  GradY->P(id,0,i,0,k,1) = 0 ;                
                }*/
            }
        }
    }
}


void CFlowVariable::computedY2()
{
     
      double x,y,z,u,v,w,c1,c2,c3;
     double *coords=getCoords();
     double *dxyz = getdxyz();
     double Xa = coords[0];
     double Ya = coords[2];
     double Za = coords[4];
     double pi=3.14159265358979;

   if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( nxChunk*p0 + 1. );
        c2 = ( coords[3] - coords[2] ) / ( nyChunk*p0 + 1. );
        c3 = ( coords[5] - coords[4] ) / ( nzChunk*p0 + 1. );

    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }
    if(Grad2Y==NULL)
    {
      Grad2Y=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0);
      Grad2Y->setBox(this->Xbox);
//    cout<<"Xbox elements ="<<GradX->Xbox[0]<<" "<<GradX->Xbox[1]<<" "<<GradX->Xbox[2]<<" "<<GradX->Xbox[3]<<" "<<GradX->Xbox[4]<<"\n"; 
       int dir=2;
       Grad2Y->setCoords(dir);
    }
    updateGhostsNS();
    int rank_y=myRank/p0;
//    double c2=GradY->dxyz[1]; 

//    cout<<"nChunk, x,y,k,dx="<<nChunk<<" "<<nxChunk<<" "<<nyChunk<<" "<<nzChunk<<" "<<GradX->dxyz[0]<<"\n"; 
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
//            assignTempX2Y( id, k, tmp );
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int i = 0; i < nxChunk; i++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int j = 1; j < nyChunk-1; j++ )//interior points
                {
                   y = Ya + (j+1) * c2 - SHIFT * c2 * .5;
                   Grad2Y->P(id,0,i,j,k,0) =(P(id,0,i,j+1,k,0) - 2*P(id,0,i,j,k,0) + P(id,0,i,j-1,k,0))/(c2*c2);//-2*pi*cos(2*pi*y) ;
                   Grad2Y->P(id,0,i,j,k,1) = 0 ;                 
                }

             //   if(rank_y!=p0-1) //right side processor interface not on the rightmost processor  
              //  {
                  y = Ya + nyChunk * c2 - SHIFT * c2 * .5;
                  Grad2Y->P(id,0,i,nyChunk-1,k,0) = (SouthGhosts[i + nxChunk * k + id*nxChunk*nzChunk] -2*P(id,0,i,nyChunk-1,k,0) + P(id,0,i,nyChunk-2,k,0))/(c2*c2);// -2*pi*cos(2*pi*y) ;
                  Grad2Y->P(id,0,i,nyChunk-1,k,1) = 0 ;                 
              /*  }
                else //right side processor interface on the right most processor
                { //insert bc treatment here
                   y = Ya + nyChunk * c2 - SHIFT * c2 * .5;
                  GradY->P(id,0,i,nyChunk-1,k,0) = (SouthBoundary[i + nxChunk * k + id*nxChunk*nzChunk]-P(id,0,i,nyChunk-2,k,0))/(2*c2);// -2*pi*cos(2*pi*y) ;
                  GradY->P(id,0,i,nyChunk-1,k,1) = 0 ;                  
                }*/
                  
             //   if(rank_y!=0) //left side processor interface not on the leftmost processor  
             //   {
                  y = Ya + (0+1) * c2 - SHIFT * c2 * .5;
                  Grad2Y->P(id,0,i,0,k,0) = (P(id,0,i,1,k,0) -2*P(id,0,i,0,k,0)+ NorthGhosts[i + nxChunk * k + id*nxChunk*nzChunk])/(c2*c2);// -2*pi*cos(2*pi*y) ;
                  Grad2Y->P(id,0,i,0,k,1) = 0 ;                 
              /*  }
                else
                {
                  y = Ya + (0+1) * c2 - SHIFT * c2 * .5;
                  GradY->P(id,0,i,0,k,0) = (P(id,0,i,1,k,0)-NorthBoundary[i + nxChunk * k + id*nxChunk*nzChunk])/(2*c2);//-2*pi*cos(2*pi*y)  ;
                  GradY->P(id,0,i,0,k,1) = 0 ;                
                }*/
            }
        }
    }
}



void CFlowVariable::computeGradZ()
{
    
    if(GradZ==NULL)
    {  
       GradZ=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0);
       GradZ->setBox(this->Xbox);
//    cout<<"Xbox elements ="<<GradX->Xbox[0]<<" "<<GradX->Xbox[1]<<" "<<GradX->Xbox[2]<<" "<<GradX->Xbox[3]<<" "<<GradX->Xbox[4]<<"\n"; 
       int dir=2;
       GradZ->setCoords(dir);
    }

//     setGradZTopBoundary();
  //   setGradZBottomBoundary();

//    int rank_y=myRank/p0;
    double c3=GradZ->dxyz[2]; 
    updateGhostBoundariesBT();
//    cout<<"nChunk, x,y,k,dx="<<nChunk<<" "<<nxChunk<<" "<<nyChunk<<" "<<nzChunk<<" "<<GradX->dxyz[0]<<"\n"; 
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int j = 0; j < nyChunk; j++ )
        {
//            assignTempX2Y( id, k, tmp );
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int i = 0; i < nxChunk; i++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int k = 1; k < nzChunk-1; k++ )//interior points
                {
                   GradZ->P(id,0,i,j,k,0) =(this->P(id,0,i,j,k+1,0)-P(id,0,i,j,k-1,0))/(2*c3) ;
                   GradZ->P(id,0,i,j,k,1) = 0 ;                 
                }

                if(id!=nChunk-1)
                {
                  GradZ->P(id,0,i,j,nzChunk-1,0) = (P(id+1,0,i,j,0,0)-P(id,0,i,j,nzChunk-2,0))/(2*c3) ;
                  GradZ->P(id,0,i,j,nzChunk-1,1) = 0 ;                                
                }                              
                else 
                {
                  GradZ->P(id,0,i,j,nzChunk-1,0) = (TopGhosts[i+j*nxChunk] - P(id,0,i,j,nzChunk-2,0))/(2*c3) ; ;
                  GradZ->P(id,0,i,j,nzChunk-1,1) = 0 ;                                
                }                                 
         
                if(id!=0)
                { 
                   GradZ->P(id,0,i,j,0,0) = (P(id,0,i,j,1,0)-P(id-1,0,i,j,nzChunk-1,0))/(2*c3) ;
                   GradZ->P(id,0,i,j,0,1) = 0 ;                 
                } 
                else
                {
                   GradZ->P(id,0,i,j,0,0) =(P(id,0,i,j,1,0) - BottomGhosts[i+j*nxChunk])/(2*c3) ;
                   GradZ->P(id,0,i,j,0,1) = 0 ;                 

                }                
               
            }
        }
    }
}



void CFlowVariable::computedZ2()
{
    
    if(Grad2Z==NULL)
    {  
       Grad2Z=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0);
       Grad2Z->setBox(this->Xbox);
//    cout<<"Xbox elements ="<<GradX->Xbox[0]<<" "<<GradX->Xbox[1]<<" "<<GradX->Xbox[2]<<" "<<GradX->Xbox[3]<<" "<<GradX->Xbox[4]<<"\n"; 
       int dir=2;
       Grad2Z->setCoords(dir);
    }

//     setGradZTopBoundary();
  //   setGradZBottomBoundary();

//    int rank_y=myRank/p0;
    double c3=Grad2Z->dxyz[2]; 
    updateGhostBoundariesBT();
//    cout<<"nChunk, x,y,k,dx="<<nChunk<<" "<<nxChunk<<" "<<nyChunk<<" "<<nzChunk<<" "<<GradX->dxyz[0]<<"\n"; 
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int j = 0; j < nyChunk; j++ )
        {
//            assignTempX2Y( id, k, tmp );
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int i = 0; i < nxChunk; i++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int k = 1; k < nzChunk-1; k++ )//interior points
                {
                   Grad2Z->P(id,0,i,j,k,0) =(P(id,0,i,j,k+1,0)- 2*P(id,0,i,j,k,0) +P(id,0,i,j,k-1,0))/(c3*c3) ;
                   Grad2Z->P(id,0,i,j,k,1) = 0 ;                 
                }

                if(id!=nChunk-1)
                {
                  Grad2Z->P(id,0,i,j,nzChunk-1,0) = (P(id+1,0,i,j,0,0)-2*P(id,0,i,j,nzChunk-1,0) +P(id,0,i,j,nzChunk-2,0))/(c3*c3) ;
                  Grad2Z->P(id,0,i,j,nzChunk-1,1) = 0 ;                                
                }                              
                else 
                {
                  Grad2Z->P(id,0,i,j,nzChunk-1,0) = (TopGhosts[i+j*nxChunk] -2*P(id,0,i,j,nzChunk-1,0)+ P(id,0,i,j,nzChunk-2,0))/(c3*c3) ; ;
                  Grad2Z->P(id,0,i,j,nzChunk-1,1) = 0 ;                                
                }                                 
         
                if(id!=0)
                { 
                   Grad2Z->P(id,0,i,j,0,0) = (P(id,0,i,j,1,0)- 2*P(id,0,i,j,0,0) +P(id-1,0,i,j,nzChunk-1,0))/(c3*c3) ;
                   Grad2Z->P(id,0,i,j,0,1) = 0 ;                 
                } 
                else
                {
                   Grad2Z->P(id,0,i,j,0,0) =(P(id,0,i,j,1,0) -2*P(id,0,i,j,0,0)+ BottomGhosts[i+j*nxChunk])/(c3*c3) ;
                   Grad2Z->P(id,0,i,j,0,1) = 0 ;                 

                }               
               
            }
        }
    }
}


void CFlowVariable::computeLaplacian()
{
    if(Laplacian==NULL)
    {  
       Laplacian=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0);
       Laplacian->setBox(this->Xbox);
//    cout<<"Xbox elements ="<<GradX->Xbox[0]<<" "<<GradX->Xbox[1]<<" "<<GradX->Xbox[2]<<" "<<GradX->Xbox[3]<<" "<<GradX->Xbox[4]<<"\n"; 
       int dir=2;
       Laplacian->setCoords(dir);
    }
    
     Laplacian->initializeZero(); 
    /*
     updateGhosts();       
     if(GradX==NULL)
        computeGradX();
     if(GradY==NULL)
        computeGradY();
     if(GradZ==NULL)
        computeGradZ();
   
     GradX->computeGradX();
     GradY->computeGradY();
     GradZ->computeGradZ();
     Laplacian->add(GradX->getGradX());
     Laplacian->add(GradY->getGradY());
     Laplacian->add(GradZ->getGradZ());  */ 

     updateGhosts();
     computedX2();
     computedY2();
     computedZ2();

     Laplacian->add(*Grad2X);
   Laplacian->add(*Grad2Y);
   Laplacian->add(*Grad2Z);
   
 
}

void CFlowVariable:: assignValues(PencilDcmp *v)
{

  //  CFlowVariableSingleBlock pResult(nx,ny,nz,p0);
   //  pResult.setBox(this->Xbox);
    
  //   int dir=2;
//     pResult.setCoords(dir);
//   cout<<"assigning values, nChunk = "<<nChunk<<" nzChunk = "<<nzChunk<<"nyChunk = "<<nyChunk<<"nxChunk = "<<nxChunk<<"\n";
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
//            assignTempX2Y( id, k, tmp );
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 0; i < nxChunk; i++ )

                {
                    //          P( id, 1, i, j, k, 0 ) = 1.0;
                    P( id, 0, i, j, k, 0 ) =  v->getValue(id, i, j, k, 0);//default dir =0
                    P( id, 0, i, j, k, 1 ) =  v->getValue(id, i, j, k, 1);
                   
                  //  cout<<"id ="<<id<<" i = "<<i<<" j = "<<j<<" k = "<<k<<"  value = "<<P(id,0,i,j,k,0)<<"  \n"; 
                }
            }
        }
    }
}

void CFlowVariable::add( PencilDcmp& b)
{
//     PencilDcmp pResult(nx,ny,nz,p0,p0);
  //   pResult.setBox(this->Xbox);
    
   //  int dir=2;
   //  pResult.setCoords(dir);

    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
//            assignTempX2Y( id, k, tmp );
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 0; i < nxChunk; i++ )

                {
                    //          P( id, 1, i, j, k, 0 ) = 1.0;
                     P(id,0,i,j,k,0)+= 1.0*b.getValue(id,i,j,k,0) ;//default dir =0
                     P(id,0,i,j,k,1)+= 1.0*b.getValue(id,i,j,k,1);
                }
            }
        }
    }
// return pResult; 
}

CFlowVariableSingleBlock CFlowVariableSingleBlock::operator+(const CFlowVariableSingleBlock& b)
{
     CFlowVariableSingleBlock pResult(nx,ny,nz,p0);
     pResult.setBox(this->Xbox);
    
     int dir=2;
     pResult.setCoords(dir);

    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
//            assignTempX2Y( id, k, tmp );
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 0; i < nxChunk; i++ )

                {
                    //          P( id, 1, i, j, k, 0 ) = 1.0;
                    pResult.P( id, 0, i, j, k, 0 ) = this->P(id,0,i,j,k,0)+b.P.getValue(id,i,j,k,0);//default dir =0
                    pResult.P( id, 0, i, j, k, 1 ) =  this->P(id,0,i,j,k,1)+b.P.getValue(id,i,j,k,1);
                }
            }
        }
    }
 return pResult; 
}

void CFlowVariableSingleBlock::start_exchangeEW(double* north_buf,double* south_buf,
   unsigned long layer_size, unsigned long nlayers,
   mpi_exchange* exch, int flags)
{
   //make sure buffer of exch is initialized
//   assert(data_dev != 0);
   //if (HCONSTANT_GPUCOUNT == 1)  return;

 //  if (flags & EXCH_FLAG_NOSYNC) {
  //   GTIME_BEG(GTIME_EXCHANGE, TIMER_CUDA_NOSYNC);
 //  } else {
 //    GTIME_BEG(GTIME_EXCHANGE, TIMER_CUDA_SYNC);
//   }

   int pid   = myRank;//HCONSTANT_DEVICE;//adapt as fit
   int nproc = p0;//HCONSTANT_GPUCOUNT;//adapt as fit
   
   int  north_send_offset = layer_size;  // offset by one layer
   int  south_send_offset = layer_size * nlayers;
   size_t layer_size_bytes = layer_size * sizeof(double);


  for(int chunk=0;chunk<2*nChunk;chunk++)
   {
     exch->send_request[chunk] = MPI_REQUEST_NULL;
   //  exch->send_request[1] = MPI_REQUEST_NULL;
     exch->recv_request[chunk] = MPI_REQUEST_NULL;
   //  exch->recv_request[1] = MPI_REQUEST_NULL;
   }

//   cout<<"nChunk = "<<nChunk<<"???????????? \n";

   double* hbufnr = exch->buffer + 0*layer_size;
   double* hbufsr = exch->buffer + 1*layer_size;
   double* hbufns = exch->buffer + 2*layer_size;
   double* hbufss = exch->buffer + 3*layer_size;

  int proc_row=pid%nproc;
   MPI_Status status[2];
   MPI_Status stat;
   MPI_Request req[2];

 
  
/*
  for (int chunk=0; chunk<nChunk;chunk++)
  {
//    Post receive for data from the north
   if (proc_row > 0){
      MPI_Irecv(hbufnr, layer_size, MPI_DOUBLE, pid-1, 2*chunk+1, MPI_COMM_WORLD, exch->recv_request+ 2*chunk+0);
   //   MPI_Wait(exch->recv_request + 2*chunk+0, &stat);
         }
   
  //    Match the receive: send data south
   if (proc_row < nproc-1) {
//      COPY_FROM_GPU(hbufss, data_dev+south_send_offset, layer_size_bytes, "MPI Exch: D2H BufSS");//adapt as fit: copy gpu, single layer only

      for(int i=0;i<layer_size;i++) hbufss[i] = south_buf[chunk*nyChunk*nzChunk + i];

      MPI_Isend(hbufss, layer_size, MPI_DOUBLE, pid+1, 2*chunk+1, MPI_COMM_WORLD, exch->send_request+ 2*chunk+1);
  //    MPI_Wait(exch->send_request + 2*chunk+1, &stat);
      cout<<"rank "<<myRank<<" sends chunk "<<chunk<<" to "<<pid+1<<" with Tag "<<2*chunk+1<<"address = "<<hbufss<<" with value0 = "<<hbufss[1]<<"  \n\n"; 
   }

   // Post receive for data from the south
   if (proc_row < nproc-1){
      MPI_Irecv(hbufsr, layer_size, MPI_DOUBLE, pid+1, 2*chunk+0, MPI_COMM_WORLD, exch->recv_request+ 2*chunk+1);
}
   //    MPI_Wait(exch->recv_request + 2*chunk+1, &stat);                         }
//      cout<<"rank "<<myRank<<" receives chunk "<<chunk<<" from "<<pid+1<<" with value0 = "<<hbufsr[1]<<"  \n\n"; }

//    Match the receive: send data north
   if (proc_row > 0) {
  //    COPY_FROM_GPU(hbufns, data_dev+north_send_offset, layer_size_bytes, "MPI Exch: D2H BufNS");
      
      for(int i=0;i<layer_size;i++) hbufns[i] = north_buf[chunk*nyChunk*nzChunk + i];

      MPI_Isend(hbufns, layer_size, MPI_DOUBLE, pid-1, 2*chunk+0, MPI_COMM_WORLD, exch->send_request+ 2*chunk+0);
 //      MPI_Wait(exch->send_request + 2*chunk+0, &stat);     
      cout<<"rank "<<myRank<<" sends chunk "<<chunk<<" to "<<pid-1<<" with Tag "<<2*chunk<<"address = "<<hbufns<<" with value0 = "<<hbufns[1]<<"  \n\n"; 
   }*/
  /* if(proc_row > 0 && proc_row < nproc-1) 
   {
     req[0] = exch->send_request[2*chunk];
     req[1] = exch->send_request[2*chunk+1];
     MPI_Waitall(2,req,status);   
      
   }
   else if(proc_row==0)
     MPI_Wait(send_request+2*chunk+1,MPI_STATUS_IGNORE);
   else if(proc_row == nproc-1)
     MPI_Wait(send_request+2*chunk,MPI_STATUS_IGNORE);
*/
    
// }

  double *temp_nbuf =new double[layer_size*nChunk];
  double *temp_sbuf =new double[layer_size*nChunk];
  for (int chunk=0; chunk<nChunk;chunk++)
  {
     for(int i=0;i<layer_size;i++)
      {
         temp_sbuf[chunk*nyChunk*nzChunk +i ] = south_buf[chunk*nyChunk*nzChunk + i];
         temp_nbuf[chunk*nyChunk*nzChunk +i ] = north_buf[chunk*nyChunk*nzChunk + i];
      }
  }


  int count = 0; 
  MPI_Request recv_req[2];
  int dest;
 for (int chunk=0; chunk<nChunk;chunk++)
  {
//    Post receive for data from the north
   if (proc_row > 0)
   {
      //MPI_Irecv(hbufnr, layer_size, MPI_DOUBLE, pid-1, 2*chunk+1, MPI_COMM_WORLD, exch->recv_request+ 2*chunk+0);
    dest =pid-1;
    if(dest<0) dest=MPI_PROC_NULL;
      MPI_Irecv(hbufnr, layer_size, MPI_DOUBLE, dest, 2*chunk+1, MPI_COMM_WORLD, &(recv_req[0]));   
      MPI_Wait(&(recv_req[0]),MPI_STATUS_IGNORE);
     for(int i=0;i<layer_size;i++)north_buf[chunk*nyChunk*nzChunk+i]=hbufnr[i];  
      count++;
 
    //  MPI_Recv(hbufnr, layer_size, MPI_DOUBLE, pid-1, 2*chunk+1, MPI_COMM_WORLD,&(recv_status[0]));
           
  //    fout<<" rank "<<myRank<< " recieving from "<<pid-1 <<endl;
  //    fout<< hbufnr  <<endl;
   //   fout<<" tag   "<<2*chunk+1 <<endl;
    }    
//    Match the receive: send data south
   if (proc_row < nproc-1) {
//      COPY_FROM_GPU(hbufss, data_dev+south_send_offset, layer_size_bytes, "MPI Exch: D2H BufSS");//adapt as fit: copy gpu, single layer only
      dest = pid+1;
      if(dest>nproc*nproc-1) dest= MPI_PROC_NULL;
      for(int i=0;i<layer_size;i++) hbufss[i] =  temp_sbuf[chunk*nyChunk*nzChunk + i];
     // MPI_Isend(hbufss, layer_size, MPI_DOUBLE, pid+1, 2*chunk+1, MPI_COMM_WORLD, &(send_req[0]));
      MPI_Send(hbufss, layer_size, MPI_DOUBLE, dest, 2*chunk+1, MPI_COMM_WORLD);
     
    //  fout<<"============================================="<<endl;
    //  fout<<" rank "<<myRank<< " sending to "<<pid+1 <<endl;
    //  fout<< hbufnr  <<endl;
    //  fout<<" tag   "<<2*chunk+1 <<endl;
   }

   // Post receive for data from the south
   if (proc_row < nproc-1)
    {  
       dest =pid+1;
      if(dest>nproc*nproc-1) dest= MPI_PROC_NULL;  
      MPI_Irecv(hbufsr, layer_size, MPI_DOUBLE, dest, 2*chunk+0, MPI_COMM_WORLD,&(recv_req[1]));
      MPI_Wait(&(recv_req[1]),MPI_STATUS_IGNORE);
      for(int i=0;i<layer_size;i++) south_buf[chunk*nyChunk*nzChunk+i]=hbufsr[i];
     }    
    //  fout<<" Ircv "<<myRank<< " recieving from "<<pid+1 <<endl;
   //   fout<< hbufnr  <<endl;
   //   fout<<" tag   "<< 2*chunk <<endl;

//    Match the receive: send data north
   if (proc_row > 0) {
  //    COPY_FROM_GPU(hbufns, data_dev+north_send_offset, layer_size_bytes, "MPI Exch: D2H BufNS");
     dest =pid-1;
    if(dest<0) dest=MPI_PROC_NULL;          
      for(int i=0;i<layer_size;i++) hbufns[i] = temp_nbuf[chunk*nyChunk*nzChunk + i];
      //MPI_Isend(hbufns, layer_size, MPI_DOUBLE, pid-1, 2*chunk+0, MPI_COMM_WORLD, &(send_req[0]));
      MPI_Send(hbufns, layer_size, MPI_DOUBLE, dest, 2*chunk+0, MPI_COMM_WORLD);

    }
   }

  delete temp_nbuf;
  delete temp_sbuf;  

}

void CFlowVariableSingleBlock:: finish_exchangeEW(double* north_buf,double* south_buf,
   unsigned long layer_size, unsigned long nlayers,mpi_exchange* exch, int flags)
{
//   assert(data_dev != 0);
   //if (HCONSTANT_GPUCOUNT == 1)  return;
   int pid   = myRank;
   int nproc = p0;

   int  north_recv_offset = 0;
   int  south_recv_offset = layer_size + (layer_size * nlayers);
   size_t layer_size_bytes = layer_size * sizeof(double);

   double* hbufnr = exch->buffer + 0*layer_size;
   double* hbufsr = exch->buffer + 1*layer_size;
   MPI_Status stat;

   #ifdef DEBUG_STATEMENTS_LEVEL09
   printf("[%d] Finish Exch: nlayers = %d layer_size = %d\n", pid, nlayers, layer_size);
   #endif // DEBUG_STATEMENTS_LEVEL09

//   GTIME_BEG(GTIME_EXCHANGE, TIMER_CUDA_NOSYNC);
   int proc_row=pid%nproc;
 //  cout<<"finish exch: nproc= "<<nproc<<" row number = "<<proc_row<<"layer size="<<layer_size<<" rank = "<<pid<<" \n";
   for(int chunk=0;chunk<nChunk;chunk++) 
{
   if (proc_row > 0) {
      MPI_Wait(exch->recv_request + 2*chunk+0, &stat);
   //   COPY_TO_GPU(data_dev+north_recv_offset, hbufnr, layer_size_bytes, "MPI Exch: H2D bufnr");
       for(int i=0;i<layer_size;i++)north_buf[chunk*nyChunk*nzChunk+i]=hbufnr[i];  
      if(hbufnr[0]!=chunk*16) 
       cout<<"rank= "<<myRank<<" receiving chunk = "<<chunk<<" from source "<<stat.MPI_SOURCE<<" with tag "<<stat.MPI_TAG<<"address = "<<hbufnr<<" left value0 = "<<hbufnr[0]<<"\n\n\n";
   }
   if (proc_row < nproc - 1) {
      MPI_Wait(exch->recv_request + 2*chunk+1, &stat);
    //  COPY_TO_GPU(data_dev+south_recv_offset, hbufsr, layer_size_bytes, "MPI Exch: H2D bufsr");
       for(int i=0;i<layer_size;i++) south_buf[chunk*nyChunk*nzChunk+i]=hbufsr[i];
       if(hbufsr[0]!=chunk*16)   
       cout<<"rank= "<<myRank<<" receiving chunk = "<<chunk<<" from source "<<stat.MPI_SOURCE<<" with tag "<<stat.MPI_TAG<<"address = "<<hbufsr<<" right value0 = "<<hbufsr[0]<<"\n\n\n";
   }

      MPI_Waitall(2*nChunk, exch->send_request, MPI_STATUSES_IGNORE);

  }

}


void CFlowVariableSingleBlock::exchange_scalarEW(double* north_buf, double* south_buf,unsigned long layer_size, unsigned long nlayers,
   mpi_exchange* exch, int flags)
{
   // Optimize away single GPU case.
//   if (HCONSTANT_GPUCOUNT == 1)return;

   start_exchangeEW(north_buf, south_buf,layer_size, nlayers, exch, flags);
 //  finish_exchangeEW(north_buf, south_buf,layer_size, nlayers, exch, flags);
}

void CFlowVariableSingleBlock::updateGhostsEW()
{
  mpi_exchange exch;
  exch.send_request = new MPI_Request[2*nChunk]; 
  exch.recv_request = new MPI_Request[2*nChunk]; 

  exch.buffer = new double[4*nzChunk*nyChunk]; 
//   cout<<"nChunk = "<<nChunk<<" ++++++++-----!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
    for(int chunk=0; chunk<nChunk;chunk++)
    {   
       for ( int k = 0; k < nzChunk; k++ )
        {
            int i_w = 0, i_e=nxChunk-1;
        
            for ( int j = 0; j < nyChunk; j++ )
            {
                                                 
                EastGhosts[chunk*nzChunk*nyChunk + nyChunk * k + j] = P(chunk,0,i_e,j,k,0);
                WestGhosts[chunk*nzChunk*nyChunk + nyChunk * k + j] = P(chunk,0,i_w,j,k,0);
               int index = j+nyChunk*k+chunk*nzChunk*nyChunk;
         //    if( fabs(WestGhosts[index]-EastGhosts[index])>1.0e-3 ) 
           //   cout<<"before exchange, inequal ghosts:  rank= "<<myRank<<" chunk = "<<chunk<<", j= "<<j<<", k = "<<k<<":West Ghost= "<<WestGhosts[index]<<" EastGhost= "<<EastGhosts[index]<<" \n";
          //   if(myRank%p0 >0)
          //    cout<<"before exchange: right rank= "<<myRank<<" chunk = "<<chunk<<", j= "<<j<<", "<<k<<":West Ghost= "<<WestGhosts[index]<<" EastGhost= "<<EastGhosts[index]<<" \n";
            }

      }
     }

  exchange_scalarEW(WestGhosts, EastGhosts, nyChunk*nzChunk,1,&exch,0 );
  updateGhostBoundariesEW();
  
  delete   exch.send_request;
  delete   exch.recv_request;
  delete   exch.buffer; 


for(int chunk=0;chunk<nChunk;chunk++)
{
  for ( int k = 0; k < nzChunk; k++ )
    {
           
        
            for ( int j = 0; j < nyChunk; j++ )
            {
               int index=j+nyChunk*k+chunk*nzChunk*nyChunk;
           //   if( fabs(WestGhosts[index]-EastGhosts[index])>1.0e-3   )
           //   cout<<"after exchange, inequal ghosts:  rank= "<<myRank<<" chunk = "<<chunk<<" j= "<<j<<", k= "<<k<<":West Ghost= "<<WestGhosts[index]<<" EastGhost= "<<EastGhosts[index]<<" \n";
            // if(myRank%p0 >0)
             // cout<<"after exchange: right rank= "<<myRank<<" chunk = "<<chunk<<" j= "<<j<<", "<<k<<":West Ghost= "<<WestGhosts[index]<<" EastGhost= "<<EastGhosts[index]<<" \n";
            }                                        

            }
}
      
}


void CFlowVariableSingleBlock::start_exchangeNS(double* north_buf,double* south_buf,
   unsigned long layer_size, unsigned long nlayers,
   mpi_exchange* exch, int flags)
{
   //make sure buffer of exch is initialized
//   assert(data_dev != 0);
   //if (HCONSTANT_GPUCOUNT == 1)  return;

 //  if (flags & EXCH_FLAG_NOSYNC) {
  //   GTIME_BEG(GTIME_EXCHANGE, TIMER_CUDA_NOSYNC);
 //  } else {
 //    GTIME_BEG(GTIME_EXCHANGE, TIMER_CUDA_SYNC);
//   }

   int pid   = myRank;//HCONSTANT_DEVICE;//adapt as fit
   int nproc = p0;//HCONSTANT_GPUCOUNT;//adapt as fit

   int  north_send_offset = layer_size;  // offset by one layer
   int  south_send_offset = layer_size * nlayers;
   size_t layer_size_bytes = layer_size * sizeof(double);

   for(int chunk=0;chunk<2*nChunk;chunk++)
   {
     exch->send_request[chunk] = MPI_REQUEST_NULL;
   //  exch->send_request[1] = MPI_REQUEST_NULL;
     exch->recv_request[chunk] = MPI_REQUEST_NULL;
   //  exch->recv_request[1] = MPI_REQUEST_NULL;
   }
   double* hbufnr = exch->buffer + 0*layer_size;
   double* hbufsr = exch->buffer + 1*layer_size;
   double* hbufns = exch->buffer + 2*layer_size;
   double* hbufss = exch->buffer + 3*layer_size;

 //   With non-stream exchanges, we will do synchronous memcpy's and MPI
//     asynchronous calls.

   #ifdef DEBUG_STATEMENTS_LEVEL09
   printf("[%d] Start Exch: nlayers = %d layer_size = %d\n", pid, nlayers, layer_size);
   #endif 
   int proc_row=pid/nproc;
//   cout<<"start exch: nproc= "<<nproc<<" row number = "<<proc_row<<"layer size= "<<layer_size<<" \n";
/*  for (int chunk=0; chunk<nChunk;chunk++)
  {
//    Post receive for data from the north
   if (proc_row > 0)
      MPI_Irecv(hbufnr, layer_size, MPI_DOUBLE, pid-1*nproc, DATA_NORTH*nChunk+chunk, MPI_COMM_WORLD, exch->recv_request + 2*chunk+0);

//    Match the receive: send data south
   if (proc_row < nproc-1) {
//      COPY_FROM_GPU(hbufss, data_dev+south_send_offset, layer_size_bytes, "MPI Exch: D2H BufSS");//adapt as fit: copy gpu, single layer only

      for(int i=0;i<layer_size;i++) hbufss[i] = south_buf[chunk*nxChunk*nzChunk+i];

      MPI_Isend(hbufss, layer_size, MPI_DOUBLE, pid+1*nproc, DATA_NORTH*nChunk+chunk, MPI_COMM_WORLD, exch->send_request + 2*chunk+1);
   }

   // Post receive for data from the south
   if (proc_row < nproc-1)
      MPI_Irecv(hbufsr, layer_size, MPI_DOUBLE, pid+1*nproc, DATA_SOUTH*nChunk+chunk, MPI_COMM_WORLD, exch->recv_request + 2*chunk+1);

//    Match the receive: send data north
   if (proc_row > 0) {
  //    COPY_FROM_GPU(hbufns, data_dev+north_send_offset, layer_size_bytes, "MPI Exch: D2H BufNS");
      
      for(int i=0;i<layer_size;i++) hbufns[i] = north_buf[chunk*nxChunk*nzChunk+i];

      MPI_Isend(hbufns, layer_size, MPI_DOUBLE, pid-1*nproc, DATA_SOUTH*nChunk+chunk, MPI_COMM_WORLD, exch->send_request + 2*chunk+0);
   }


  }
*/

  int count = 0; 
  MPI_Request recv_req[2];


  double *temp_nbuf =new double[layer_size*nChunk];
  double *temp_sbuf =new double[layer_size*nChunk];
  for (int chunk=0; chunk<nChunk;chunk++)
  {
     for(int i=0;i<layer_size;i++)
      {
         temp_sbuf[chunk*nxChunk*nzChunk +i ] = south_buf[chunk*nxChunk*nzChunk + i];
         temp_nbuf[chunk*nxChunk*nzChunk +i ] = north_buf[chunk*nxChunk*nzChunk + i];
      }
  }
  int dest;
 for (int chunk=0; chunk<nChunk;chunk++)
  {
//    Post receive for data from the north
   if (proc_row > 0)
   {
      dest = pid-1*nproc;
      if(dest<0) dest= MPI_PROC_NULL;
      //MPI_Irecv(hbufnr, layer_size, MPI_DOUBLE, pid-1, 2*chunk+1, MPI_COMM_WORLD, exch->recv_request+ 2*chunk+0);
      MPI_Irecv(hbufnr, layer_size, MPI_DOUBLE, dest, 2*chunk+1, MPI_COMM_WORLD, &(recv_req[0]));   
      MPI_Wait(&(recv_req[0]),MPI_STATUS_IGNORE);
      for(int i=0;i<layer_size;i++) north_buf[chunk*nyChunk*nzChunk+i]=hbufnr[i];
      count++;
 
    //  MPI_Recv(hbufnr, layer_size, MPI_DOUBLE, pid-1, 2*chunk+1, MPI_COMM_WORLD,&(recv_status[0]));
           
  //    fout<<" rank "<<myRank<< " recieving from "<<pid-1 <<endl;
  //    fout<< hbufnr  <<endl;
   //   fout<<" tag   "<<2*chunk+1 <<endl;
    }    
//    Match the receive: send data south
   if (proc_row < nproc-1) 
    {
      dest = pid+1*nproc;
      if (dest > nproc*nproc-1)
        dest = MPI_PROC_NULL;
//      COPY_FROM_GPU(hbufss, data_dev+south_send_offset, layer_size_bytes, "MPI Exch: D2H BufSS");//adapt as fit: copy gpu, single layer only
      for(int i=0;i<layer_size;i++) hbufss[i] = temp_sbuf[chunk*nxChunk*nzChunk + i];
     // MPI_Isend(hbufss, layer_size, MPI_DOUBLE, pid+1, 2*chunk+1, MPI_COMM_WORLD, &(send_req[0]));
      MPI_Send(hbufss, layer_size, MPI_DOUBLE, dest, 2*chunk+1, MPI_COMM_WORLD);
     
    //  fout<<"============================================="<<endl;
    //  fout<<" rank "<<myRank<< " sending to "<<pid+1 <<endl;
    //  fout<< hbufnr  <<endl;
    //  fout<<" tag   "<<2*chunk+1 <<endl;
   }

   // Post receive for data from the south
   if (proc_row < nproc-1)
    {
       dest = pid+1*nproc;
      if (dest > nproc*nproc-1)
        dest = MPI_PROC_NULL;   
      MPI_Irecv(hbufsr, layer_size, MPI_DOUBLE, dest, 2*chunk+0, MPI_COMM_WORLD,&(recv_req[1]));
      MPI_Wait(&(recv_req[1]),MPI_STATUS_IGNORE);
      for(int i=0;i<layer_size;i++) south_buf[chunk*nyChunk*nzChunk+i]=hbufsr[i];
     }    
    //  fout<<" Ircv "<<myRank<< " recieving from "<<pid+1 <<endl;
   //   fout<< hbufnr  <<endl;
   //   fout<<" tag   "<< 2*chunk <<endl;

//    Match the receive: send data north
   if (proc_row > 0) {
  //    COPY_FROM_GPU(hbufns, data_dev+north_send_offset, layer_size_bytes, "MPI Exch: D2H BufNS");
       dest = pid-1*nproc;
      if(dest<0) dest= MPI_PROC_NULL;          
      for(int i=0;i<layer_size;i++) hbufns[i] = temp_nbuf[chunk*nxChunk*nzChunk + i];
      //MPI_Isend(hbufns, layer_size, MPI_DOUBLE, pid-1, 2*chunk+0, MPI_COMM_WORLD, &(send_req[0]));
      MPI_Send(hbufns, layer_size, MPI_DOUBLE, dest, 2*chunk+0, MPI_COMM_WORLD);
    }
  } 
 
  delete temp_nbuf;
  delete temp_sbuf;  
}

void CFlowVariableSingleBlock:: finish_exchangeNS(double* north_buf,double* south_buf,
   unsigned long layer_size, unsigned long nlayers,mpi_exchange* exch, int flags)
{
//   assert(data_dev != 0);
   //if (HCONSTANT_GPUCOUNT == 1)  return;
   int pid   = myRank;
   int nproc = p0;

   int  north_recv_offset = 0;
   int  south_recv_offset = layer_size + (layer_size * nlayers);
   size_t layer_size_bytes = layer_size * sizeof(double);

   double* hbufnr = exch->buffer + 0*layer_size;
   double* hbufsr = exch->buffer + 1*layer_size;

   #ifdef DEBUG_STATEMENTS_LEVEL09
   printf("[%d] Finish Exch: nlayers = %d layer_size = %d\n", pid, nlayers, layer_size);
   #endif // DEBUG_STATEMENTS_LEVEL09

//   GTIME_BEG(GTIME_EXCHANGE, TIMER_CUDA_NOSYNC);
   int proc_row=pid/nproc;
//   cout<<"finish exch: nproc= "<<nproc<<" row number = "<<proc_row<<"layer size="<<layer_size<<" \n";
  for (int chunk=0; chunk<nChunk;chunk++)
  {
   if (proc_row > 0) {
      MPI_Wait(exch->recv_request+ 2*chunk+0, MPI_STATUS_IGNORE);
   //   COPY_TO_GPU(data_dev+north_recv_offset, hbufnr, layer_size_bytes, "MPI Exch: H2D bufnr");
       for(int i=0;i<layer_size;i++) north_buf[chunk*nxChunk*nzChunk+i]=hbufnr[i];
   }
   if (proc_row < nproc - 1) {
      MPI_Wait(exch->recv_request + 2*chunk+1, MPI_STATUS_IGNORE);
    //  COPY_TO_GPU(data_dev+south_recv_offset, hbufsr, layer_size_bytes, "MPI Exch: H2D bufsr");
       for(int i=0;i<layer_size;i++) south_buf[chunk*nxChunk*nzChunk+i]=hbufsr[i];
   }

//   if (flags & EXCH_FLAG_NOWAIT) {
      // Skip the MPI_Waitall.  You _must_ call it before reusing the buffer!
//   } else {
      MPI_Waitall(2, exch->send_request, MPI_STATUSES_IGNORE);
   }
  // GTIME_END(GTIME_EXCHANGE, TIMER_CUDA_NOSYNC);
}


void CFlowVariableSingleBlock::exchange_scalarNS(double* north_buf, double* south_buf,unsigned long layer_size, unsigned long nlayers,
   mpi_exchange* exch, int flags)
{
   // Optimize away single GPU case.
//   if (HCONSTANT_GPUCOUNT == 1)return;

   start_exchangeNS(north_buf, south_buf,layer_size, nlayers, exch, flags);
  // finish_exchangeNS(north_buf, south_buf,layer_size, nlayers, exch, flags);
}

void CFlowVariableSingleBlock::updateGhostsNS()
{
  mpi_exchange exch;
   exch.send_request = new MPI_Request[2*nChunk]; 
  exch.recv_request = new MPI_Request[2*nChunk];  
  exch.buffer = new double[4*nzChunk*nxChunk]; 
//   cout<<"nzChunk = "<<nzChunk<<" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
   for(int chunk=0; chunk<nChunk;chunk++)
    {   
  for ( int k = 0; k < nzChunk; k++ )
    {
            int j_n = 0, j_s=nyChunk-1;
        
            for ( int i = 0; i < nxChunk; i++ )
            {
                                                 
                NorthGhosts[i + nxChunk * k + chunk*nzChunk*nxChunk  ] = P(chunk,0,i,j_n,k,0);
                SouthGhosts[i + nxChunk * k + chunk*nzChunk*nxChunk  ] = P(chunk,0,i,j_s,k,0);
            // if(myRank<2)
            //  cout<<"before exchange: lower rank= "<<myRank<<" i= "<<i<<", "<<k<<": North Ghost= "<<NorthGhosts[i+nxChunk*k]<<" SouthGhost= "<<SouthGhosts[i+nxChunk*k]<<" \n";
           //  if(myRank>=2)
           //   cout<<"before exchange: upper rank= "<<myRank<<" i= "<<i<<", "<<k<<": North Ghost= "<<NorthGhosts[i+nxChunk*k]<<" SouthGhost= "<<SouthGhosts[i+nxChunk*k]<<" \n";
            }

      }
} 
    for ( int i = 0; i < nzChunk*nxChunk; i++ )
    {
          
           
//     cout<<"Rank= "<<myRank<<", "<<i<<": North Ghost= "<<NorthGhosts[i]<<" SouthGhost= "<<SouthGhosts[i]<<" \n";


      }   

  exchange_scalarNS(NorthGhosts, SouthGhosts, nxChunk*nzChunk,1,&exch,0 );
 // setGradYNorthBoundary();
//  setGradYSouthBoundary();
  updateGhostBoundariesNS();

  delete   exch.send_request;
  delete   exch.recv_request;
  delete   exch.buffer; 
/*  for ( int k = 0; k < nzChunk; k++ )
    {
           
        
            for ( int i = 0; i < nxChunk; i++ )
            {
                                                 
              //  NorthGhosts[i + nxChunk * k] = P(i,j_n,k,0);
            //    SouthGhosts[i + nxChunk * k] = P(i,j_s,k,0);
         //  if(myRank<2) 
         //    cout<<"after exchange: lower rank= "<<myRank<<", i= "<<i<<", "<<k<<": North Ghost= "<<NorthGhosts[i+nxChunk*k]<<" SouthGhost= "<<SouthGhosts[i+nxChunk*k]<<" \n";
         //  if(myRank>=2) 
           //  cout<<"after exchange: upper rank= "<<myRank<<", i= "<<i<<", "<<k<<": North Ghost= "<<NorthGhosts[i+nxChunk*k]<<" SouthGhost= "<<SouthGhosts[i+nxChunk*k]<<" \n";
            }

      }*/
}


void CFlowVariable::setGradXWestBoundary()
{
     int proc_column = myRank % p0;
     double Xa = coords[0];
     double Ya = coords[2];
     double Za = coords[4];


     double x,y,z,c1,c2,c3;
     double pi=3.14159265358979;

     if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( nxChunk*p0 + 1. );
        c2 = ( coords[3] - coords[2] ) / ( nyChunk*p0 + 1. );
        c3 = ( coords[5] - coords[4] ) / ( nzChunk*p0 + 1. );

    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }  

     if(proc_column == 0)
     { 
       for(int chunk=0; chunk<nChunk;chunk++)
       {   
         for ( int k = 0; k < nzChunk; k++ )
         {
            z = Za + ( k+1  ) * c3 + chunk*nzChunk*c3 - SHIFT*c3*0.5;      
            int  i_w=-1;
            x = Xa + ( i_w +1 ) * c1 - SHIFT*c1*0.5; 
            
            for ( int j = 0; j < nyChunk; j++ )
            {                             
                    
                WestGhosts[chunk*nzChunk*nyChunk + nyChunk * k + j] = P(chunk,0,nxChunk-1,j,k,0);
          //      WestBoundary[chunk*nzChunk*nyChunk + nyChunk * k + j] = P(chunk,0,nxChunk-1,j,k,0);
//1*2*pi*cos(2*pi*x) ;
//if(chunk==nChunk-1 && k==nzChunk-1 && j==nyChunk-1)
// cout<<"x coordinate is"<<x<<"West ghost value at top = "<<  WestGhosts[(nChunk-1)*nzChunk*nyChunk + nyChunk * (nzChunk-1) + nyChunk-1]<<"!!!!!!!!!!!!!!!!!!!\n";               
            }

      }
     }
 

   }

}

void CFlowVariable::setGradXEastBoundary()
{
     int proc_column = myRank % p0;
     double Xa = coords[0];
     double Ya = coords[2];
     double Za = coords[4];


     double x,y,z,c1,c2,c3;
         double pi=3.14159265358979;
    if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( nxChunk*p0 + 1. );
        c2 = ( coords[3] - coords[2] ) / ( nyChunk*p0 + 1. );
        c3 = ( coords[5] - coords[4] ) / ( nzChunk*p0 + 1. );

    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }

//   cout<<"set Grad East Boundary!!!! Xa = "<<Xa<<" 

     if(proc_column == p0-1)
     { 
       for(int chunk=0; chunk<nChunk;chunk++)
       {   
         for ( int k = 0; k < nzChunk; k++ )
         {
            z = Za + ( k + 1 ) * c3 -SHIFT*0.5*c3 + chunk*nzChunk*c3;      
            int  i_e = nxChunk;
            x = Xa + ( i_e +1 ) * c1 -SHIFT*0.5*c1; 
            
            for ( int j = 0; j < nyChunk; j++ )
            {                             
                    
                EastGhosts[chunk*nzChunk*nyChunk + nyChunk * k + j] = P(chunk,0,0,j,k,0);
           //     EastBoundary[chunk*nzChunk*nyChunk + nyChunk * k + j] = P(chunk,0,0,j,k,0);
//1*2*pi*cos(2*pi*x) ;
//      cout<<"east ghost value = "<<  P(chunk,0,0,j,k,0)<<"\n???????????????????????????\n";          
            }

      }
 
   }
  } 
    
    
   
    
}

void CFlowVariable::updateGhostBoundariesEW()//update boundary ghost values for periodic boundary conditions
{
      
//      for (int pid=0;pid<p0*p0;pid+=p0)
      int proc_column = myRank%p0; 

      MPI_Request send_req[2]; 
      MPI_Request recv_req[2];

      send_req[0]=MPI_REQUEST_NULL;
      send_req[1]=MPI_REQUEST_NULL;     
      recv_req[0]=MPI_REQUEST_NULL;
      recv_req[1]=MPI_REQUEST_NULL;    



      if(proc_column==p0-1)
      {
         MPI_Irecv(EastBoundary, nChunk*nzChunk*nyChunk, MPI_DOUBLE, myRank-(p0-1), 0, MPI_COMM_WORLD, &recv_req[0]);
         MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

  
      }

      if(proc_column==0)
      {


        MPI_Send(WestGhosts, nChunk*nzChunk*nyChunk, MPI_DOUBLE, myRank+(p0-1),0,MPI_COMM_WORLD);//, &send_req[1]);//send to right most processor
      }
     
     


      if (proc_column == 0) {
        MPI_Irecv(WestBoundary, nChunk*nzChunk*nyChunk, MPI_DOUBLE, myRank+(p0-1), 1, MPI_COMM_WORLD, &recv_req[1]);
        MPI_Wait(&recv_req[1], MPI_STATUS_IGNORE);

   }


      if (proc_column == p0-1) {
         MPI_Send(EastGhosts, nChunk*nzChunk*nyChunk, MPI_DOUBLE, myRank-(p0-1),1,MPI_COMM_WORLD);//, &send_req[0]);//send to right most processor

      }


    /*  if(proc_column==p0-1)
      {
         MPI_Irecv(EastBoundary, nChunk*nzChunk*nyChunk, MPI_DOUBLE, myRank-p0+1, 0, MPI_COMM_WORLD, &recv_req[0]);


         MPI_Isend(EastGhosts, nChunk*nzChunk*nyChunk, MPI_DOUBLE, myRank-p0+1,1,MPI_COMM_WORLD, &send_req[0]);//send to right most processor
  
      }

      if(proc_column==0)
      {
        MPI_Irecv(WestBoundary, nChunk*nzChunk*nyChunk, MPI_DOUBLE, myRank+p0-1, 1, MPI_COMM_WORLD, &recv_req[1]);


        MPI_Isend(WestGhosts, nChunk*nzChunk*nyChunk, MPI_DOUBLE, myRank+p0-1,0,MPI_COMM_WORLD, &send_req[1]);//send to right most processor
      }
     
      if (proc_column == p0-1) {
      MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);
      }


      if (proc_column == 0) {
      MPI_Wait(&recv_req[1], MPI_STATUS_IGNORE);

   }*/


   if(proc_column==p0-1)
   {
         for(int chunk=0; chunk<nChunk;chunk++)
       {   
         for ( int k = 0; k < nzChunk; k++ )
         {
    
            for ( int i = 0; i < nyChunk; i++ )
            {                             
                    
                EastGhosts[chunk*nzChunk*nyChunk + nyChunk * k + i] = EastBoundary[chunk*nzChunk*nyChunk + nyChunk * k + i] ;

               
            }

      }
     }
   }
   else if(proc_column==0)
   {
         for(int chunk=0; chunk<nChunk;chunk++)
       {   
         for ( int k = 0; k < nzChunk; k++ )
         {
                        
            for ( int i = 0; i < nyChunk; i++ )
            {                             
                    
                WestGhosts[chunk*nzChunk*nyChunk + nyChunk * k + i] = WestBoundary[chunk*nzChunk*nyChunk + nyChunk * k + i] ;

               
            }

      }
     }
   }
   

}


void CFlowVariable::updateGhostBoundariesNS()//update boundary ghost values for periodic boundary conditions
{
      
//      for (int pid=0;pid<p0*p0;pid+=p0)
      int proc_row = myRank/p0; 

      MPI_Request send_req[2]; 
      MPI_Request recv_req[2];

      send_req[0]=MPI_REQUEST_NULL;
      send_req[1]=MPI_REQUEST_NULL;     
      recv_req[0]=MPI_REQUEST_NULL;
      recv_req[1]=MPI_REQUEST_NULL;    





      if(proc_row==p0-1)
      {
         MPI_Irecv(SouthBoundary, nChunk*nzChunk*nxChunk, MPI_DOUBLE, myRank-(p0-1)*p0, 0, MPI_COMM_WORLD, &recv_req[0]);
         MPI_Wait(&recv_req[0], MPI_STATUS_IGNORE);

  
      }

      if(proc_row==0)
      {


        MPI_Send(NorthGhosts, nChunk*nzChunk*nxChunk, MPI_DOUBLE, myRank+(p0-1)*p0,0,MPI_COMM_WORLD);//, &send_req[1]);//send to right most processor
      }
     
     


      if (proc_row == 0) {
        MPI_Irecv(NorthBoundary, nChunk*nzChunk*nxChunk, MPI_DOUBLE, myRank+(p0-1)*p0, 1, MPI_COMM_WORLD, &recv_req[1]);
        MPI_Wait(&recv_req[1], MPI_STATUS_IGNORE);

   }


      if (proc_row == p0-1) {
         MPI_Send(SouthGhosts, nChunk*nzChunk*nxChunk, MPI_DOUBLE, myRank-(p0-1)*p0,1,MPI_COMM_WORLD);//, &send_req[0]);//send to right most processor

      }


   if(proc_row==p0-1)
   {
         for(int chunk=0; chunk<nChunk;chunk++)
       {   
         for ( int k = 0; k < nzChunk; k++ )
         {
    
            for ( int i = 0; i < nxChunk; i++ )
            {                             
                    
                SouthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i] = SouthBoundary[chunk*nzChunk*nxChunk + nxChunk * k + i] ;

               
            }

      }
     }
   }
   else if(proc_row==0)
   {
         for(int chunk=0; chunk<nChunk;chunk++)
       {   
         for ( int k = 0; k < nzChunk; k++ )
         {
                        
            for ( int i = 0; i < nxChunk; i++ )
            {                             
                    
                NorthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i] = NorthBoundary[chunk*nzChunk*nxChunk + nxChunk * k + i] ;

               
            }

      }
     }
   }
   

}


void CFlowVariable::RefreshGradient()
{
   GradX=NULL; GradY=NULL; GradZ=NULL;

}

void CFlowVariable::setGradYSouthBoundary()
{
     int proc_row = myRank / p0;
     double Xa = coords[0];
     double Ya = coords[2];
     double Za = coords[4];



     double x,y,z,c1,c2,c3;
    
    if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( nxChunk*p0 + 1. );
        c2 = ( coords[3] - coords[2] ) / ( nyChunk*p0 + 1. );
        c3 = ( coords[5] - coords[4] ) / ( nzChunk*p0 + 1. );

    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }

//   cout<<"set Grad East Boundary!!!! Xa = "<<Xa<<" 

     if(proc_row == p0-1)
     { 
       for(int chunk=0; chunk<nChunk;chunk++)
       {   
         for ( int k = 0; k < nzChunk; k++ )
         {
            z = Za + ( k + 1 ) * c3 -SHIFT*0.5*c3 + chunk*nzChunk*c3;      
            int  j_s = nyChunk;
            y = Ya + ( j_s +1 ) * c2 -SHIFT*0.5*c2; 
            
            for ( int i = 0; i < nxChunk; i++ )
            {                             
                    
                SouthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i] =  P(chunk,0,i,0,k,0);
                SouthBoundary[chunk*nzChunk*nxChunk + nxChunk * k + i] =  P(chunk,0,i,0,k,0);
//0*2*pi*cos(2*pi*y) ; ;
               
            }

      }
     }
   }
   
//  MPI_Bcast(SouthBoundary , Msg.count, ConvertType( getAbstractionDataType<Type>() ), Msg.sender, Com.mpiCom );
}

void CFlowVariable::setGradYNorthBoundary()
{
     int proc_row = myRank / p0;
     double Xa = coords[0];
     double Ya = coords[2];
     double Za = coords[4];



     double x,y,z,c1,c2,c3;
    
    if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( nxChunk*p0 + 1. );
        c2 = ( coords[3] - coords[2] ) / ( nyChunk*p0 + 1. );
        c3 = ( coords[5] - coords[4] ) / ( nzChunk*p0 + 1. );

    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }

//   cout<<"set Grad East Boundary!!!! Xa = "<<Xa<<" 

     if(proc_row == p0-1)
     { 
       for(int chunk=0; chunk<nChunk;chunk++)
       {   
         for ( int k = 0; k < nzChunk; k++ )
         {
            z = Za + ( k + 1 ) * c3 -SHIFT*0.5*c3 + chunk*nzChunk*c3;      
            int  j_n = -1;
            y = Ya + ( j_n +1 ) * c2 -SHIFT*0.5*c2; 
            
            for ( int i = 0; i < nxChunk; i++ )
            {                             
               NorthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i] =  //P(chunk,0,i,nyChunk-1,k,0);
0*2*pi*cos(2*pi*y) ; ;
               
            }

      }
     }
   }
}

void CFlowVariable::updateGhostBoundariesBT()
{
//     int proc_column = myRank % p0;
     double Xa = coords[0];
     double Ya = coords[2];
     double Za = coords[4];



     double x,y,z,c1,c2,c3;
    
    if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( nxChunk*p0 + 1. );
        c2 = ( coords[3] - coords[2] ) / ( nyChunk*p0 + 1. );
        c3 = ( coords[5] - coords[4] ) / ( nzChunk*p0 + 1. );

    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }

         for ( int j = 0; j < nyChunk; j++ )
         {
            int k_t = nzChunk;
            z = Za + ( k_t + 1 ) * c3 -SHIFT*0.5*c3 + (nChunk-1)*nzChunk*c3;      
           
            y = Ya + ( j +1 ) * c2 -SHIFT*0.5*c2; 
                 

            for ( int i = 0; i < nxChunk; i++ )
            {                             
                x = Xa + ( i +1 ) * c1 -SHIFT*0.5*c1;              
                TopGhosts[ nxChunk * j + i] =  P(0,0,i,j,0,0);
                BottomGhosts[ nxChunk * j + i] =  P(nChunk-1,0,i,j,nzChunk-1,0);

            }
     }
   
}



void CFlowVariable::scaleField(double s)
{
    #if ( PITTPACKACC )
#pragma acc loop seq
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
           
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 0; i < nxChunk; i++ )

                {
                    //          P( id, 1, i, j, k, 0 ) = 1.0;
                    P( id, 1, i, j, k, 0 )*= s;
                    P( id, 1, i, j, k, 1 )*= s;
                }
            }
        }
    }
}


void CFlowVariableSingleBlock::updateGhosts()
{
  updateGhostsEW();
  updateGhostsNS();
  updateGhostBoundariesBT();
}


void CVelocitySingleBlock::setBox( double *X )
{
    for ( int i = 0; i < 6; i++ )
    {
        Xbox[i] = X[i];
    }

#if ( PITTPACKACC )
#pragma acc update device( Xbox [0:6] )
#endif
}

void CVelocitySingleBlock::setVelocity(PencilDcmp &UU, PencilDcmp &VV, PencilDcmp &WW)
{
/*   U->setBox(UU.getXbox());
   V->setBox(VV.getXbox());
   W->setBox(WW.getXbox());

   int dir=2;
   U->setCoords(dir);
   V->setCoords(dir);
   W->setCoords(dir);*/

  #if ( PITTPACKACC )
#pragma acc loop seq
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
           
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 0; i < nxChunk; i++ )

                {
                    //          P( id, 1, i, j, k, 0 ) = 1.0;
                    U->setValue(UU.getValue( id,  i, j, k, 0 ),id,i,j,k,0) ;
                    V->setValue(VV.getValue( id,  i, j, k, 0 ),id,i,j,k,0) ;
                    W->setValue(WW.getValue( id,  i, j, k, 0 ),id,i,j,k,0) ;
                    U->setValue(UU.getValue( id,  i, j, k, 1 ),id,i,j,k,1) ;
                    V->setValue(VV.getValue( id,  i, j, k, 1 ),id,i,j,k,1) ;
                    W->setValue(WW.getValue( id,  i, j, k, 1 ),id,i,j,k,1) ;
                }
            }
        }
    } 
   
}


void CVelocitySingleBlock::initializeVelocity()
{



     double x,y,z,u,v,w,c1,c2,c3;
     double *coords=U->getCoords();
     double *dxyz = U->getdxyz();
     double Xa = coords[0];
     double Ya = coords[2];
     double Za = coords[4];
     double pi=3.14159265358979;

   if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( nxChunk*p_dir + 1. );
        c2 = ( coords[3] - coords[2] ) / ( nyChunk*p_dir + 1. );
        c3 = ( coords[5] - coords[4] ) / ( nzChunk*p_dir + 1. );

    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }
     cout<<"at velocity single block: rank = "<<U->getMyRank()<<", Xa= "<<Xa<<" c1= "<<c1<<" \n";
     #if ( PITTPACKACC )
#pragma acc loop seq
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
//            z = Za + ( k + 1 ) * c3 + id*nzChunk*c3;
            z = Za + (k+1) * c3 - SHIFT * c3 * .5 +id*nzChunk*c3;     
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif                    
                 y = Ya + (j+1) * c2 - SHIFT * c2 * .5;
               //  y = Ya + ( j + 1 ) * c2; 
                for ( int i = 0; i < nxChunk; i++ )

                {
                    x = Xa + (i+1) * c1 - SHIFT * c1 * .5;
                   // x = Xa + ( i + 1 ) * c1; 
                    u = sin(2*pi*x); 
                    w = sin(2*pi*z);
                    v = sin(2*pi*y); 

                    U->setValue(u, id ,i,j,k,0);
                    V->setValue(v, id ,i,j,k,0);
                    W->setValue(w, id ,i,j,k,0);

//                   if(i==1 && j==nyChunk-1&& k==nzChunk-1 && id==nChunk-1)
  //                         cout<<"Top left x coordinate equals "<<x<<"value is"<<U->getValue(nChunk-1,1,nyChunk-1,nzChunk-1,0)<<" !!!!!!!!!!!!!!!!!!!!!!!\n";
                }
            }
        }
    } 


// U->updateGhostsEW();
// V->updateGhostsNS();
//dW/dZ does not have ghosts along the z direction, but ghost update at the domain boundary needed 
// W->updateGhostBoundaryBT();

}


void CVelocitySingleBlock::Refresh()
{
   U->RefreshGradient();
   V->RefreshGradient();
   W->RefreshGradient();

}



void CVelocitySingleBlock::computeDivergence()
{
   
  /* if(U->getGradX()==NULL)
   {
    GradX=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0);
    GradX->setBox(this->Xbox);
//    cout<<"Xbox elements ="<<GradX->Xbox[0]<<" "<<GradX->Xbox[1]<<" "<<GradX->Xbox[2]<<" "<<GradX->Xbox[3]<<" "<<GradX->Xbox[4]<<"\n"; 
    int dir=2;
    GradX->setCoords(dir);
   }

   if(V->getGradX()==NULL) 
   {

     GradY=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0);
     GradY->setBox(this->Xbox);
//    cout<<"Xbox elements ="<<GradX->Xbox[0]<<" "<<GradX->Xbox[1]<<" "<<GradX->Xbox[2]<<" "<<GradX->Xbox[3]<<" "<<GradX->Xbox[4]<<"\n"; 
     int dir=2;
     GradY->setCoords(dir);
   }

  


   if(W->getGradZ()==NULL) 
   {
      GradZ=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0);
      GradZ->setBox(this->Xbox);
//    cout<<"Xbox elements ="<<GradX->Xbox[0]<<" "<<GradX->Xbox[1]<<" "<<GradX->Xbox[2]<<" "<<GradX->Xbox[3]<<" "<<GradX->Xbox[4]<<"\n"; 
      int dir=2;
      GradZ->setCoords(dir);
   }*/

 
    if(U!=NULL){ U->computeGradX();} // also check if ghosts are up to date!
   else throw "U not intiialized before div!\n";

       //also check if ghosts are up to date
    if(V!=NULL) {  V->computeGradY();}
    else throw "V not intiialized before div!\n";
  
   
    if(W!=NULL) W->computeGradZ();  
    else throw "W not intiialized before div!\n";    

    if(Divergence==NULL)
    {
      Divergence=new CFlowVariableSingleBlock(nxChunk*p_dir,nyChunk*p_dir,nzChunk*p_dir,p_dir);
      Divergence->setBox(U->getXbox());
//    cout<<"Xbox elements ="<<GradX->Xbox[0]<<" "<<GradX->Xbox[1]<<" "<<GradX->Xbox[2]<<" "<<GradX->Xbox[3]<<" "<<GradX->Xbox[4]<<"\n"; 
      int dir=2;
      Divergence->setCoords(dir);
    }
    
    Divergence->initializeZero(); 
       
     Divergence->add(U->getGradX());
     Divergence->add(V->getGradY());
     Divergence->add(W->getGradZ());


//    *Divergence = *(U->GradX) + *(V->GradY) + *(W->GradZ);

//   Divergence
}




void CVelocitySingleBlock::scaleDivergence( double s)
{
   
   Divergence->scaleField(s);
}


void CProjectionMomentumSingleBlock:: initialize()
{


     double x,y,z,u,v,w,c1,c2,c3;
     double *coords= Veln->getU().getCoords();
     double *dxyz =  Veln->getU().getdxyz();
     double Xa = coords[0];
     double Ya = coords[2];
     double Za = coords[4];
     double pi=3.14159265358979;

     int nChunk=p0; 
     int nxChunk=Nx/p0;
     int nyChunk=Ny/p0;
     int nzChunk=Nz/p0;


     CFlowVariableSingleBlock *U = new CFlowVariableSingleBlock(Nx,  Ny,  Nz,  p0 );
     CFlowVariableSingleBlock *V = new CFlowVariableSingleBlock(Nx,  Ny,  Nz,  p0 );
     CFlowVariableSingleBlock *W = new CFlowVariableSingleBlock(Nx,  Ny,  Nz,  p0 );
     PoissonSolver->setBox( Veln->getU().getXbox() );
    
     int dir=2;
     PoissonSolver->setCoords(dir);

     PoissonSolver->constructConnectivity();

     PoissonSolver->graphCreate();

   if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( Nx + 1. );
        c2 = ( coords[3] - coords[2] ) / ( Ny + 1. );
        c3 = ( coords[5] - coords[4] ) / ( Nz + 1. );

    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }
//     cout<<"at velocity single block: rank = "<<U->getMyRank()<<", Xa= "<<Xa<<" c1= "<<c1<<" \n";
     #if ( PITTPACKACC )
#pragma acc loop seq
#endif
    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        for ( int k = 0; k < nzChunk; k++ )
        {
//            z = Za + ( k + 1 ) * c3 + id*nzChunk*c3;
            z = Za + (k+1) * c3 - SHIFT * c3 * .5 +id*nzChunk*c3;     
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j = 0; j < nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif                    
                 y = Ya + (j+1) * c2 - SHIFT * c2 * .5;
               //  y = Ya + ( j + 1 ) * c2; 
                for ( int i = 0; i < nxChunk; i++ )

                {
                    x = Xa + (i+1) * c1 - SHIFT * c1 * .5;
                   // x = Xa + ( i + 1 ) * c1; 
                    u = 1.0*cos(4*pi*x)*sin(4*pi*y); 
                    w = 0;//-sin(6*pi*x)*cos(4*pi*y)*sin(2*pi*z); 
                    v = -1.0*sin(4*pi*x)*cos(4*pi*y);//*cos(2*pi*z);  

                    U->setValue(u, id ,i,j,k,0);
                    V->setValue(v, id ,i,j,k,0);
                    W->setValue(w, id ,i,j,k,0);

//                   if(i==1 && j==nyChunk-1&& k==nzChunk-1 && id==nChunk-1)
  //                         cout<<"Top left x coordinate equals "<<x<<"value is"<<U->getValue(nChunk-1,1,nyChunk-1,nzChunk-1,0)<<" !!!!!!!!!!!!!!!!!!!!!!!\n";
                }
            }
        }
    } 


   Veln->setVelocity(*U,*V,*W);
   Vel_predict->setVelocity(*U,*V,*W);


   Vel_old->setVelocity(*U,*V,*W);
   Vel_predict_old->setVelocity(*U,*V,*W);
}


CProjectionMomentumSingleBlock::CProjectionMomentumSingleBlock(double dt, int nx,int ny, int nz, int np, double nu,  double *Xb,char *bc)
{
   TimeStep = dt;
   Nx=nx;
   Ny=ny;
   Nz=nz;
   p0=np;
   Viscosity=nu;   

   PoissonSolver = new PoissonCPU( Nx, Ny, Nz,p0 );
   for(int i=0;i<6;i++)mybc[i]=bc[i];

   PoissonSolver->assignBoundary( mybc );
    





   Veln= new CVelocitySingleBlock(nx,ny,nz,np,Xb);
   Vel_old= new CVelocitySingleBlock(nx,ny,nz,np,Xb);  
   Vel_predict= new CVelocitySingleBlock(nx,ny,nz,np,Xb);
   Vel_predict_old= new CVelocitySingleBlock(nx,ny,nz,np,Xb);
   Pressure = new CFlowVariableSingleBlock(  nx,  ny,  nz,  np, Xb );
   Pressure->initializeZero();
   Massflow = new CFlowVariableSingleBlock(  nx,  ny,  nz,  np, Xb );
}


void CProjectionMomentumSingleBlock::predictVelocity()
{
   //currently use first order time stepping, may be unconditionally unstable in inviscid case for CD

     Vel_predict->setVelocity(Veln->getU(),Veln->getV(),Veln->getW());
        
     Vel_predict->getU().updateGhosts();
     Vel_predict->getV().updateGhosts(); 
     Vel_predict->getW().updateGhosts();

     Vel_predict_old->getU().updateGhosts();
     Vel_predict_old->getV().updateGhosts(); 
     Vel_predict_old->getW().updateGhosts();

     Veln->getU().updateGhosts();
     Veln->getV().updateGhosts(); 
     Veln->getW().updateGhosts();    

     Vel_old->getU().updateGhosts();
     Vel_old->getV().updateGhosts(); 
     Vel_old->getW().updateGhosts();    

     Pressure->updateGhosts();
     //Veln -grad p should be divergence free 
     cout<<"About to compute convective terms!!!+++++------++++++++----------------\n\n\n"<<endl;
     CFlowVariableSingleBlock *cvu = convectionTerm(*Vel_predict,Veln->getU());
//     delete cvu;     
     CFlowVariableSingleBlock *cvv = convectionTerm(*Vel_predict,Veln->getV());
 //    delete cvv;
     CFlowVariableSingleBlock *cvw = convectionTerm(*Vel_predict,Veln->getW());
 //    delete cvw;
     CFlowVariableSingleBlock *cvu_old = convectionTerm(*Vel_predict_old,Vel_old->getU());
   //  delete cvu_old;
     CFlowVariableSingleBlock *cvv_old = convectionTerm(*Vel_predict_old,Vel_old->getV());
  //   delete cvv_old;
     CFlowVariableSingleBlock *cvw_old = convectionTerm(*Vel_predict_old,Vel_old->getW());
  //   delete cvw_old;


     cvu->scaleField(1.5*TimeStep);
     cvv->scaleField(1.5*TimeStep);
     cvw->scaleField(1.5*TimeStep);

     cvu_old->scaleField(-0.5*TimeStep);
     cvv_old->scaleField(-0.5*TimeStep);
     cvw_old->scaleField(-0.5*TimeStep);


     Veln->getU().computeLaplacian();
     Veln->getU().getLaplacian().scaleField(0.5*TimeStep*Viscosity);


     Veln->getV().computeLaplacian();
     Veln->getV().getLaplacian().scaleField(0.5*TimeStep*Viscosity);
    

     Veln->getW().computeLaplacian();
     Veln->getW().getLaplacian().scaleField(0.5*TimeStep*Viscosity);


     Vel_old->getU().computeLaplacian();
     Vel_old->getU().getLaplacian().scaleField(0.5*TimeStep*Viscosity);


     Vel_old->getV().computeLaplacian();
     Vel_old->getV().getLaplacian().scaleField(0.5*TimeStep*Viscosity);
    

     Vel_old->getW().computeLaplacian();
     Vel_old->getW().getLaplacian().scaleField(0.5*TimeStep*Viscosity);

     Vel_predict_old->setVelocity(Vel_predict->getU(),Vel_predict->getV(),Vel_predict->getW());

     Vel_predict->getU().add(*cvu); 
     Vel_predict->getU().add(Veln->getU().getLaplacian()); 

     Vel_predict->getV().add(*cvv); 
     Vel_predict->getV().add(Veln->getV().getLaplacian()); 

     Vel_predict->getW().add(*cvw); 
     Vel_predict->getW().add(Veln->getW().getLaplacian()); 

     Vel_predict->getU().add(*cvu_old); 
     Vel_predict->getU().add(Vel_old->getU().getLaplacian()); 

     Vel_predict->getV().add(*cvv_old); 
     Vel_predict->getV().add(Vel_old->getV().getLaplacian()); 

     Vel_predict->getW().add(*cvw_old); 
     Vel_predict->getW().add(Vel_old->getW().getLaplacian()); 

 //    delete cvu, cvv, cvw, cvu_old,cvv_old,cvw_old;
     cvu->freeVariables(); cvv->freeVariables(); cvw->freeVariables();
     cvu_old->freeVariables(); cvv_old->freeVariables(); cvw_old->freeVariables();
     delete cvu;delete cvv; delete cvw;
     delete cvu_old ; delete cvv_old; delete cvw_old;
     
   
}



void CProjectionMomentumSingleBlock::projection()//solve Poisson equation
{

    Vel_predict->computeDivergence();
    Vel_predict->scaleDivergence(1.0/TimeStep);
   
    PoissonSolver->setRHS(Vel_predict->getDivergence());
    PoissonSolver->pittPack();
   
    Pressure->assignValues(PoissonSolver);

    //PoissonSolver->setRHS(Veln->getU());

}


void CProjectionMomentumSingleBlock::correction()//correct velocities based on new pressure
{
  // all comments temporary for debug purposes 
   Vel_old->setVelocity(Veln->getU(),Veln->getV(),Veln->getW());
   Veln->setVelocity(Vel_predict->getU(),Vel_predict->getV(),Vel_predict->getW());

   Pressure->computeGradX();
   Pressure->getGradX().scaleField(-1.0*TimeStep); 
   Pressure->computeGradY();
   Pressure->getGradY().scaleField(-1.0*TimeStep); 
   Pressure->computeGradZ();
   Pressure->getGradZ().scaleField(-1.0*TimeStep); 
   
   Veln->getU().add(Pressure->getGradX());
   Veln->getV().add(Pressure->getGradY());
   Veln->getW().add(Pressure->getGradZ());
  
}


void CProjectionMomentumSingleBlock::solve(int N)
{
  //      predictVelocity();
   for (int i=0;i<N;i++)
   {
        predictVelocity(); 
        projection();
        correction();
 
   }

}

CFlowVariableSingleBlock* CProjectionMomentumSingleBlock::convectionTerm(CVelocitySingleBlock &cvel, CFlowVariableSingleBlock &sc )//convection part using cell-centered velocity and scalar values
{
   //ideally cvel already divergence-free
   CFlowVariableSingleBlock *conv;
   conv=new CFlowVariableSingleBlock(sc.getnx(),sc.getny(),sc.getnz(),sc.getp0());

    double c1,c2,c3;
    int  rank_x=0, rank_y=0;
    int nChunk=sc.getp0(), nxChunk=sc.getnx()/sc.getp0(), nyChunk=sc.getny()/sc.getp0(), nzChunk=sc.getnz()/sc.getp0();
    conv->setBox(sc.getXbox());
//    cout<<"Xbox elements ="<<GradX->Xbox[0]<<" "<<GradX->Xbox[1]<<" "<<GradX->Xbox[2]<<" "<<GradX->Xbox[3]<<" "<<GradX->Xbox[4]<<"\n"; 
    int dir=2;
    conv->setCoords(dir);
    if(&(sc.getGradX())==NULL) sc.computeGradX(); 
    if(&(sc.getGradY())==NULL) sc.computeGradY();
    if(&(sc.getGradZ())==NULL) sc.computeGradZ();


     double *coords= Veln->getU().getCoords();
     double *dxyz =  Veln->getU().getdxyz();
     double Xa = coords[0];
     double Ya = coords[2];
     double Za = coords[4];

    if ( !SHIFT )
    {
        c1 = ( coords[1] - coords[0] ) / ( Nx + 1. );
        c2 = ( coords[3] - coords[2] ) / ( Ny + 1. );
        c3 = ( coords[5] - coords[4] ) / ( Nz + 1. );

    }
    else
    {
        c1 = dxyz[0];
        c2 = dxyz[1];
        c3 = dxyz[2];
    }
          
    double temp=0,tempe=0,tempw=0,temps=0,tempn=0,tempb=0,tempt=0;
    double mflow=0;

    for ( int id = 0; id < nChunk; id++ )
    {
#if ( PITTPACKACC )
#pragma acc loop gang private( tmp[2 * nxChunk * nyChunk] )
#endif
        
        for ( int k = 0; k < nzChunk; k++ )
        {
//            assignTempX2Y( id, k, tmp );
#if ( PITTPACKACC )
#pragma acc loop worker
#endif
            for ( int j =0; j < nyChunk; j++ )
            {
#if ( PITTPACKACC )
#pragma acc loop vector
#endif
                for ( int i = 0; i < nxChunk; i++ )//interior points
                {
                   temp = cvel.getU().getValue(id,i,j,k,0)*sc.getGradX().getValue(id,i,j,k,0) 
                       + cvel.getV().getValue(id,i,j,k,0)*sc.getGradY().getValue(id,i,j,k,0) 
                       + cvel.getW().getValue(id,i,j,k,0)*sc.getGradZ().getValue(id,i,j,k,0);
/*                
                   tempw = 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i-1,j,k,0))*
                           0.5*(cvel.getU().getValue(id,i-1,j,k,0) + cvel.getU().getValue(id,i,j,k,0) - TimeStep/c1*(Pressure->getValue(id,i,j,k,0)-Pressure->getValue(id,i-1,j,k,0))  );
                   
                   tempe = 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i+1,j,k,0))*
                           0.5*(cvel.getU().getValue(id,i+1,j,k,0) + cvel.getU().getValue(id,i,j,k,0) - TimeStep/c1*(Pressure->getValue(id,i+1,j,k,0) - Pressure->getValue(id,i,j,k,0)) );
                 
                   tempn = 0.5*(sc.getValue(id,i,j-1,k,0) + sc.getValue(id,i,j,k,0))*
                           0.5*(cvel.getV().getValue(id,i,j-1,k,0) + cvel.getV().getValue(id,i,j,k,0) - TimeStep/c2*(Pressure->getValue(id,i,j,k,0) - Pressure->getValue(id,i,j-1,k,0))  );
                 
                   temps = 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j+1,k,0))*
                           0.5*(cvel.getV().getValue(id,i,j,k,0) + cvel.getV().getValue(id,i,j+1,k,0) - TimeStep*(Pressure->getValue(id,i,j+1,k,0)- Pressure->getValue(id,i,j,k,0))  );
                 
                   tempb = 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j,k-1,0))*
                           0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().getValue(id,i,j,k-1,0) - TimeStep*(Pressure->getValue(id,i,j,k,0)- Pressure->getValue(id,i,j,k-1,0))   );

                   tempt = 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j,k+1,0))*
                           0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().getValue(id,i,j,k+1,0) - TimeStep*(Pressure->getValue(id,i,j,k+1,0) - Pressure->getValue(id,i,j,k,0))  );*/
                  //scale with 1/dx, 1/dy, 1/dz!
                   if(i>0)
                   tempw = 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i-1,j,k,0))*
                           (0.5*(cvel.getU().getValue(id,i-1,j,k,0) + cvel.getU().getValue(id,i,j,k,0)) - TimeStep/c1*(Pressure->getValue(id,i,j,k,0)-Pressure->getValue(id,i-1,j,k,0)))  ;
                   else 
                    tempw = 0.5*(sc.getValue(id,i,j,k,0) + sc.WestGhosts[j + nyChunk * k+ id*nyChunk*nzChunk])*
                           ( 0.5*(cvel.getU().WestGhosts[j + nyChunk * k+ id*nyChunk*nzChunk] + cvel.getU().getValue(id,i,j,k,0)) 
                              - TimeStep/c1*(Pressure->getValue(id,i,j,k,0)-Pressure->WestGhosts[j + nyChunk * k+ id*nyChunk*nzChunk] ))  ;                

                   if(i<nxChunk-1)
                     tempe = 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i+1,j,k,0))*
                          ( 0.5*(cvel.getU().getValue(id,i+1,j,k,0) + cvel.getU().getValue(id,i,j,k,0)) - TimeStep/c1*(Pressure->getValue(id,i+1,j,k,0) - Pressure->getValue(id,i,j,k,0))) ;
                   else 
                     tempe = 0.5*(sc.getValue(id,i,j,k,0) + sc.EastGhosts[j + nyChunk * k+ id*nyChunk*nzChunk])*
                          ( 0.5*(cvel.getU().EastGhosts[j + nyChunk * k+ id*nyChunk*nzChunk]  + cvel.getU().getValue(id,i,j,k,0)) 
                          - TimeStep/c1*(Pressure->EastGhosts[j + nyChunk * k+ id*nyChunk*nzChunk]  - Pressure->getValue(id,i,j,k,0))) ;                     
                   
                   if(j>0)
                   tempn = 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j-1,k,0))*
                           (0.5*(cvel.getV().getValue(id,i,j-1,k,0) + cvel.getV().getValue(id,i,j,k,0) ) - TimeStep/c2*(Pressure->getValue(id,i,j,k,0) - Pressure->getValue(id,i,j-1,k,0))) ;
                   else 
                      tempn =    0.5*(sc.getValue(id,i,j,k,0) + sc.NorthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk])*
                      (0.5*(cvel.getV().NorthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk] + cvel.getV().getValue(id,i,j,k,0))
                            - TimeStep/c2*( Pressure->getValue(id,i,j,k,0) - Pressure->NorthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk]   ) ) ;
                


                   if(j<nyChunk-1)
                   temps =  0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j+1,k,0))*
                           (0.5*(cvel.getV().getValue(id,i,j,k,0) + cvel.getV().getValue(id,i,j+1,k,0)) - TimeStep/c2*(Pressure->getValue(id,i,j+1,k,0)- Pressure->getValue(id,i,j,k,0)))  ;
                   else 
                      temps = 0.5*(sc.getValue(id,i,j,k,0) + sc.SouthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk])*
                          (0.5*(cvel.getV().SouthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk] + cvel.getV().getValue(id,i,j,k,0))
                            - TimeStep/c2*(Pressure->SouthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk] - Pressure->getValue(id,i,j,k,0)  ))  ;

                   if(k>0) 
                   tempb =  0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j,k-1,0))*
                           (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().getValue(id,i,j,k-1,0)) - TimeStep/c3*(Pressure->getValue(id,i,j,k,0)- Pressure->getValue(id,i,j,k-1,0)))   ;
                   else if(id==0) 
                   
                    tempb =   0.5*(sc.getValue(id,i,j,k,0) + sc.BottomGhosts[i+j*nxChunk])*
                           (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().BottomGhosts[i+j*nxChunk]) - TimeStep/c3*( Pressure->getValue(id,i,j,k,0) - Pressure->BottomGhosts[i+j*nxChunk]  ) )  ;
                   else if(id>0)
                     tempb =   0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id-1,i,j,nzChunk-1,0))*
                           (0.5*( cvel.getW().getValue(id,i,j,k,0) + cvel.getW().getValue(id-1,i,j,nzChunk-1,0) ) - TimeStep/c3*(Pressure->getValue(id,i,j,k,0)-Pressure->getValue(id-1,i,j,nzChunk-1,0)));

                   if(k<nzChunk-1)
                   tempt =  0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j,k+1,0))*
                           (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().getValue(id,i,j,k+1,0)) - TimeStep/c3*(Pressure->getValue(id,i,j,k+1,0) - Pressure->getValue(id,i,j,k,0))) ;
                   else if(id==nChunk-1) 
                        tempt =  0.5*(sc.getValue(id,i,j,k,0) + sc.TopGhosts[i+j*nxChunk])*
                       (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().TopGhosts[i+j*nxChunk]) - TimeStep/c3*(Pressure->TopGhosts[i+j*nxChunk] - Pressure->getValue(id,i,j,k,0)) )  ;

                   else if(id<nChunk-1)
                        tempt = 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id+1,i,j,0,0))*
                      (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().getValue(id+1,i,j,0,0)) - TimeStep/c3*(Pressure->getValue(id+1,i,j,0,0)- Pressure->getValue(id,i,j,k,0)))   ;

                


                   mflow =  (tempe-tempw)/(c1)+(temps-tempn)/(c2)+(tempt-tempb)/(c3);//scale with 1/dx, 1/dy, 1/dz
               
                   conv->setValue(mflow,id,i,j,k,0); 
                    if(i>0)
                   tempw = //0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i-1,j,k,0))*
                           (0.5*(cvel.getU().getValue(id,i-1,j,k,0) + cvel.getU().getValue(id,i,j,k,0)) - TimeStep/c1*(Pressure->getValue(id,i,j,k,0)-Pressure->getValue(id,i-1,j,k,0)))  ;
                   else 
                    tempw = //0.5*(sc.getValue(id,i,j,k,0) + sc.WestGhosts[j + nyChunk * k+ id*nyChunk*nzChunk])*
                           ( 0.5*(cvel.getU().WestGhosts[j + nyChunk * k+ id*nyChunk*nzChunk] + cvel.getU().getValue(id,i,j,k,0)) 
                              - TimeStep/c1*(Pressure->getValue(id,i,j,k,0)-Pressure->WestGhosts[j + nyChunk * k+ id*nyChunk*nzChunk] ))  ;                

                   if(i<nxChunk-1)
                     tempe = //0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i+1,j,k,0))*
                          ( 0.5*(cvel.getU().getValue(id,i+1,j,k,0) + cvel.getU().getValue(id,i,j,k,0)) - TimeStep/c1*(Pressure->getValue(id,i+1,j,k,0) - Pressure->getValue(id,i,j,k,0))) ;
                   else 
                     tempe = //0.5*(sc.getValue(id,i,j,k,0) + sc.EastGhosts[j + nyChunk * k+ id*nyChunk*nzChunk])*
                          ( 0.5*(cvel.getU().EastGhosts[j + nyChunk * k+ id*nyChunk*nzChunk]  + cvel.getU().getValue(id,i,j,k,0)) 
                          - TimeStep/c1*(Pressure->EastGhosts[j + nyChunk * k+ id*nyChunk*nzChunk]  - Pressure->getValue(id,i,j,k,0))) ;                     
                   
                   if(j>0)
                   tempn = //0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j-1,k,0))*
                           (0.5*(cvel.getV().getValue(id,i,j-1,k,0) + cvel.getV().getValue(id,i,j,k,0) ) - TimeStep/c2*(Pressure->getValue(id,i,j,k,0) - Pressure->getValue(id,i,j-1,k,0))) ;
                   else 
                      tempn =   // 0.5*(sc.getValue(id,i,j,k,0) + sc.NorthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk])*
                      (0.5*(cvel.getV().NorthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk] + cvel.getV().getValue(id,i,j,k,0))
                            - TimeStep/c2*( Pressure->getValue(id,i,j,k,0) - Pressure->NorthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk]   ) ) ;
                


                   if(j<nyChunk-1)
                   temps = // 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j+1,k,0))*
                           (0.5*(cvel.getV().getValue(id,i,j,k,0) + cvel.getV().getValue(id,i,j+1,k,0)) - TimeStep/c2*(Pressure->getValue(id,i,j+1,k,0)- Pressure->getValue(id,i,j,k,0)))  ;
                   else 
                      temps =// 0.5*(sc.getValue(id,i,j,k,0) + sc.SouthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk])*
                          (0.5*(cvel.getV().SouthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk] + cvel.getV().getValue(id,i,j,k,0))
                            - TimeStep/c2*(Pressure->SouthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk] - Pressure->getValue(id,i,j,k,0)  ))  ;

                   if(k>0) 
                   tempb =  //0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j,k-1,0))*
                           (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().getValue(id,i,j,k-1,0)) - TimeStep/c3*(Pressure->getValue(id,i,j,k,0)- Pressure->getValue(id,i,j,k-1,0)))   ;
                   else if(id==0) 
                   
                    tempb =  // 0.5*(sc.getValue(id,i,j,k,0) + sc.BottomGhosts[i+j*nxChunk])*
                           (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().BottomGhosts[i+j*nxChunk]) - TimeStep/c3*( Pressure->getValue(id,i,j,k,0) - Pressure->BottomGhosts[i+j*nxChunk]  ) )  ;
                   else if(id>0)
                     tempb =  // 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id-1,i,j,nzChunk-1,0))*
                           (0.5*( cvel.getW().getValue(id,i,j,k,0) + cvel.getW().getValue(id-1,i,j,nzChunk-1,0) ) - TimeStep/c3*(Pressure->getValue(id,i,j,k,0)-Pressure->getValue(id-1,i,j,nzChunk-1,0)));

                   if(k<nzChunk-1)
                   tempt = // 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j,k+1,0))*
                           (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().getValue(id,i,j,k+1,0)) - TimeStep/c3*(Pressure->getValue(id,i,j,k+1,0) - Pressure->getValue(id,i,j,k,0))) ;
                   else if(id==nChunk-1) 
                        tempt = // 0.5*(sc.getValue(id,i,j,k,0) + sc.TopGhosts[i+j*nxChunk])*
                       (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().TopGhosts[i+j*nxChunk]) - TimeStep/c3*(Pressure->TopGhosts[i+j*nxChunk] - Pressure->getValue(id,i,j,k,0)) )  ;

                   else if(id<nChunk-1)
                        tempt =// 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id+1,i,j,0,0))*
                      (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().getValue(id+1,i,j,0,0)) - TimeStep/c3*(Pressure->getValue(id+1,i,j,0,0)- Pressure->getValue(id,i,j,k,0)))   ;            

                   mflow =  (tempe-tempw)/c1+(temps-tempn)/c2+(tempt-tempb)/c3;
                   Massflow->setValue(mflow,id,i,j,k,0);

                }

            }
        }
    }
 
 return  conv;

}
