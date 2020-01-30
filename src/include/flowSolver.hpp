#ifndef _FLOWSOLVE_H_
#define _FLOWSOLVE_H_
#include "chunkedArray.hpp"
#include "definitions.h"
#include "mpi.h"
#include "phdf5.hpp"
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#if ( PITTPACKACC )
#include "cufft.h"
#endif
#include "multiGrid.hpp"
#include "signalProc.hpp"
#include "triDiag.hpp"

//#define restrict __restrict__

/*!    \class PencilDcmp
 *     \brief  This Class performs pencil decomposition and encapsulates all the necessary communications required for FFT transformation
 *
 *
 *
      \f{eqnarray*}{
        \Delta u &=& f  \f}
 *   NOTE: inlining the functions will cause the inherited class not to see the function and hence be undefined in CPU version
 * */

typedef unsigned int uint;



class CFlowVariable : public PencilDcmp
{
    protected:
      

       
       double *WestBoundary;
       double *EastBoundary;
       double *NorthBoundary;
       double *SouthBoundary;


       CFlowVariable *GradX;    
       CFlowVariable *GradY;   
       CFlowVariable *GradZ;

       CFlowVariable *Grad2X;    
       CFlowVariable *Grad2Y;   
       CFlowVariable *Grad2Z;
       CFlowVariable *Laplacian;
       int type;
    public:
    //   CFlowVariable operator+(const CFlowVariable& b);
       double *WestGhosts;
       double *EastGhosts; 
       double *NorthGhosts;
       double *SouthGhosts;
       double *TopGhosts;
       double *BottomGhosts;
       void initializeZero(); 
        
       void initializeField();  
       void scaleField(double s); 
       void assignValues(PencilDcmp *v);
       void add( PencilDcmp& b);      
    //   PencilDcmp getField(){return P;};  
         CFlowVariable( int nx, int ny, int nz, int p0,int t ) : PencilDcmp( nx, ny, nz, p0,p0 )//constructor without creating grid and coordinates
    {initializeField(); NorthGhosts = new double[nz*nx];  SouthGhosts = new double[nz*nx]; 
     type=t;
     EastGhosts = new double[nz*ny];  WestGhosts = new double[nz*ny]; TopGhosts= new double[nx*ny]; BottomGhosts= new double[nx*ny]; 
     EastBoundary = new double[nz*ny];  WestBoundary = new double[nz*ny];NorthBoundary = new double[nz*nx];  SouthBoundary = new double[nz*nx];  
     GradX=NULL; GradY=NULL;GradZ=NULL;  Grad2X=NULL; Grad2Y=NULL;Grad2Z=NULL; Laplacian=NULL;};
         
       CFlowVariable( int nx, int ny, int nz, int p0, double *Xb, int t ) : PencilDcmp( nx, ny, nz, p0,p0 )//constructor creating grid and coordinates
       {initializeField(); NorthGhosts = new double[nz*nx/p0];  SouthGhosts = new double[nz*nx/p0]; type=t;
       EastGhosts = new double[nz*ny/p0];  WestGhosts = new double[nz*ny/p0]; TopGhosts= new double[nx*ny]; BottomGhosts= new double[nx*ny]; 
       EastBoundary = new double[nz*ny];  WestBoundary = new double[nz*ny];NorthBoundary = new double[nz*nx];  SouthBoundary = new double[nz*nx];  
       GradX=NULL; GradY=NULL;GradZ=NULL;    Grad2X=NULL; Grad2Y=NULL;Grad2Z=NULL;Laplacian=NULL;
       setBox(Xb);
       int dir =2; 
       setCoords(dir);    };
      
       void freeVariables(){delete NorthGhosts; delete WestGhosts; delete SouthGhosts; delete EastGhosts; delete TopGhosts; delete BottomGhosts;
                                  delete EastBoundary; delete WestBoundary; delete NorthBoundary; delete SouthBoundary;  };


         void RefreshGradient();
         CFlowVariable(PencilDcmp *Field );
          
        // CFlowVariable* getField(){return P;};
         CFlowVariable& getGradX(){return *GradX;};
         CFlowVariable& getGradY(){return *GradY;};
         CFlowVariable& getGradZ(){return *GradZ;};
         CFlowVariable& getLaplacian(){return *Laplacian;};
         void computedX2();
         void computedY2();
         void computedZ2();
         void computeGradX();
         void computeGradY();
         void computeGradZ();
         void computeLaplacian();
         CFlowVariable* shiftStaggeredX();
         CFlowVariable* shiftStaggeredY();
         CFlowVariable* shiftStaggeredZ();
         virtual void updateGhostsNS()=0;
         virtual void updateGhostsEW()=0;
          void setGradXEastBoundary();
         void setGradXWestBoundary();
         virtual void updateGhosts()=0;
         void setGradYSouthBoundary();
         void setGradYNorthBoundary(); 

         void setGradZTopBoundary();
         void setGradZBottomBoundary(); 
         
         void updateGhostBoundariesEW(); 
         void updateGhostBoundariesNS();
         void updateGhostBoundariesBT();
};

class CFlowVariableSingleBlock : public CFlowVariable
{

    public:
   
    CFlowVariableSingleBlock operator+(const CFlowVariableSingleBlock& b);
    typedef struct mpi_exchange_s
    {  
      MPI_Request *send_request;
      MPI_Request *recv_request;
      double*       buffer;
    } mpi_exchange;
         void updateGhostsNS();
         void updateGhostsEW();
         void updateGhosts();
 
         CFlowVariableSingleBlock( int nx, int ny, int nz, int p0, int t ) : CFlowVariable( nx, ny, nz, p0,t  ){};
         CFlowVariableSingleBlock( int nx, int ny, int nz, int p0, double *Xb, int t ) : CFlowVariable( nx, ny, nz, p0, Xb,t ){};
    void start_exchangeNS(double* north_buf, double* south_buf,unsigned long layer_size, unsigned long nlayers,mpi_exchange* exch, int flags);

   void  finish_exchangeNS(double* north_buf, double* south_buf,unsigned long layer_size, unsigned long nlayers,mpi_exchange* exch, int flags);
   void exchange_scalarNS(double* north_buf, double* south_buf, unsigned long layer_size, unsigned long nlayers,
         mpi_exchange* exch, int flags);
    void start_exchangeEW(double* north_buf, double* south_buf,unsigned long layer_size, unsigned long nlayers,mpi_exchange* exch, int flags);

   void  finish_exchangeEW(double* north_buf, double* south_buf,unsigned long layer_size, unsigned long nlayers,mpi_exchange* exch, int flags);
   void exchange_scalarEW(double* north_buf, double* south_buf, unsigned long layer_size, unsigned long nlayers,
         mpi_exchange* exch, int flags);


    ~CFlowVariableSingleBlock(){}; 

};

class CVelocitySingleBlock
{
   private:
          CFlowVariableSingleBlock* U;
          CFlowVariableSingleBlock* V;
          CFlowVariableSingleBlock* W;

          int nxChunk,nyChunk, nzChunk,p_dir, nChunk;
          double Xbox[6];
          CFlowVariableSingleBlock* Divergence;          
   public:

         CFlowVariableSingleBlock* FaceVelocity;

         void scaleDivergence(double s); 
         void initializeVelocity();   
         void setBox(double *X);
         void setCoords(int dir);
         CVelocitySingleBlock(int nx,int ny,int nz, int p0,double *Xb)
         { U = new CFlowVariableSingleBlock (nx,ny,nz,p0,0);V = new CFlowVariableSingleBlock (nx,ny,nz,p0,1);W = new CFlowVariableSingleBlock (nx,ny,nz,p0,2);
           U->setBox(Xb);U->setCoords(2);  V->setBox(Xb);V->setCoords(2);  W->setBox(Xb);W->setCoords(2);   
           nxChunk=nx/p0; nyChunk=ny/p0; nzChunk=nz/p0; nChunk=p0;p_dir=p0;  Divergence = NULL;//nChunk set to p0
           for (int i=0;i<6;i++) Xbox[i]=Xb[i];   }
            
         void setVelocity(PencilDcmp &U, PencilDcmp &V, PencilDcmp &W); 
         void updateFaceVelocity();
         CFlowVariableSingleBlock& getU(){return *U;};
         CFlowVariableSingleBlock& getV(){return *V;};
         CFlowVariableSingleBlock& getW(){return *W;};
        
         CFlowVariable& getDivergence(){return *Divergence;}; 
          

         void Refresh();         
         void computeDivergence();

};


class CProjectionMomentumSingleBlock
{
   protected:




   CFlowVariableSingleBlock* UConvection;
   CFlowVariableSingleBlock* VConvection;
   CFlowVariableSingleBlock* WConvection;
   double Xbox[6];
   double TimeStep;  
   int Nx,Ny,Nz,p0;
   double Viscosity;

   char mybc[6];
   PoissonCPU *PoissonSolver;



   public:
   CFlowVariableSingleBlock* Pressure; 
   CFlowVariableSingleBlock *Massflow;
   CVelocitySingleBlock* Vel_old;
   CVelocitySingleBlock* Veln;
   CVelocitySingleBlock* Vel_predict;
   CVelocitySingleBlock* Vel_predict_old; 
   void initialize();
   void updateNextTimestep();
   void solve(int N);
   CProjectionMomentumSingleBlock(double dt, int X,int Y, int Z, int np, double nu,double *Xb,char *bc);
   CVelocitySingleBlock& getVelocity(){return *Veln; };
   CFlowVariableSingleBlock& getPressure(){return *Pressure; };
   double getTimeStep(){return TimeStep;};
   void predictVelocity();
   void projection();
   void correction();
   void solvePressure();
   void correctVelocity();
   void evaluateConvection();
   void evaluateDiffusion();
   CFlowVariableSingleBlock* convectionTerm(CVelocitySingleBlock &cvel, CFlowVariableSingleBlock &sc);
   CFlowVariableSingleBlock& getUConvection(){return *UConvection;}; 

};


#endif
