#include "pencilDcmp.hpp"
#include "flowSolver.hpp"
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
      GradX=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0,type);
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
      Grad2X=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0,type);
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
      GradY=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0,type);
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
                   GradY->P(id,0,i,j,k,0) = (this->P(id,0,i,j+1,k,0)-P(id,0,i,j-1,k,0))/(2*c2);//-2*pi*cos(2*pi*y) ;
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
      Grad2Y=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0,type);
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
       GradZ=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0,type);
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
       Grad2Z=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0,type);
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
       Laplacian=new CFlowVariableSingleBlock(nxChunk*p0,nyChunk*p0,nzChunk*p0,p0,type);
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
//   cout<<"assigning values, nChunk  = "<<nChunk<<" nzChunk  = "<<nzChunk<<"nyChunk = "<<nyChunk<<"nxChunk = "<<nxChunk<<"\n";
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
     CFlowVariableSingleBlock pResult(nx,ny,nz,p0,-1);
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
                                                 
                EastGhosts[chunk*nzChunk*nyChunk + nyChunk * k + j] = P(chunk,0,i_e,j,k,0);//periodic bc
                WestGhosts[chunk*nzChunk*nyChunk + nyChunk * k + j] = P(chunk,0,i_w,j,k,0);//periodic bc
             //   EastGhosts[chunk*nzChunk*nyChunk + nyChunk * k + j] = -P(chunk,0,i_w,j,k,0);//no-slip bc
             //   WestGhosts[chunk*nzChunk*nyChunk + nyChunk * k + j] = -P(chunk,0,i_e,j,k,0);//no-slip bc
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
//   cout<<"nzChunk  = "<<nzChunk<<" !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
   for(int chunk=0; chunk<nChunk;chunk++)
    {   
  for ( int k = 0; k < nzChunk; k++ )
    {
            int j_n = 0, j_s=nyChunk-1;
        
            for ( int i = 0; i < nxChunk; i++ )
            {
                                                 
                NorthGhosts[i + nxChunk * k + chunk*nzChunk*nxChunk  ] = P(chunk,0,i,j_n,k,0);//periodic bc
                SouthGhosts[i + nxChunk * k + chunk*nzChunk*nxChunk  ] = P(chunk,0,i,j_s,k,0);//periodic bc
             //   NorthGhosts[i + nxChunk * k + chunk*nzChunk*nxChunk  ] = -P(chunk,0,i,j_s,k,0);//no-slip bc
             //   SouthGhosts[i + nxChunk * k + chunk*nzChunk*nxChunk  ] = -P(chunk,0,i,j_n,k,0);//no-slip bc
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
                    
//               WestGhosts[chunk*nzChunk*nyChunk + nyChunk * k + j] = P(chunk,0,nxChunk-1,j,k,0);//periodic bc
                WestGhosts[chunk*nzChunk*nyChunk + nyChunk * k + j] = -P(chunk,0,0,j,k,0);//no-slip
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
                    
//                EastGhosts[chunk*nzChunk*nyChunk + nyChunk * k + j] = P(chunk,0,0,j,k,0);//periodic bc
                EastGhosts[chunk*nzChunk*nyChunk + nyChunk * k + j] = -P(chunk,0,nxChunk-1,j,k,0);//no-slip bc
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

   int i_w = 0, i_e=nxChunk-1;
   if(proc_column==p0-1)
   {
         for(int chunk=0; chunk<nChunk;chunk++)
       {   
         for ( int k = 0; k < nzChunk; k++ )
         {
    
            for ( int i = 0; i < nyChunk; i++ )
            {                             
                    
            //    EastGhosts[chunk*nzChunk*nyChunk + nyChunk * k + i] = EastBoundary[chunk*nzChunk*nyChunk + nyChunk * k + i] ;// periodic bc
            if(type!=3) 
              EastGhosts[chunk*nzChunk*nyChunk + nyChunk * k + i] =  -P(chunk,0,i_e,i,k,0) ; //no-slip bc for velocity
           else if(type==3)
               EastGhosts[chunk*nzChunk*nyChunk + nyChunk * k + i] =  P(chunk,0,i_e,i,k,0) ;  //Neumann bc for pressure            
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
                    
//                WestGhosts[chunk*nzChunk*nyChunk + nyChunk * k + i] = WestBoundary[chunk*nzChunk*nyChunk + nyChunk * k + i] ;
          if(type!=3)       
              WestGhosts[chunk*nzChunk*nyChunk + nyChunk * k + i] =  -P(chunk,0,i_w,i,k,0) ; //no-slip bc for velocity
          else if(type==3) 
              WestGhosts[chunk*nzChunk*nyChunk + nyChunk * k + i] =  P(chunk,0,i_w,i,k,0) ; //Neumann bc for pressure
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



      int j_n = 0, j_s=nyChunk-1;

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
                   
//                SouthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i] = SouthBoundary[chunk*nzChunk*nxChunk + nxChunk * k + i] ;//periodic bc
           
 //                SouthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i] = 0;//fixed velocity bc
            if(type!=3)
                 SouthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i  ] = -P(chunk,0,i,j_s,k,0);//no-slip bc for velocity
            else if(type==3)
                 SouthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i  ] = P(chunk,0,i,j_s,k,0);//Neumann bc for pressure              
               
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
                    
 //               NorthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i] = NorthBoundary[chunk*nzChunk*nxChunk + nxChunk * k + i] ; //periodic bc

                if(type!=0 &&type!=3)    //not u velocity component or pressure no-slip
                      NorthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i]  = -P(chunk,0,i,j_n,k,0);//no-slip velocity bc
                else if(type==0)
                     NorthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i]  =2-P(chunk,0,i,j_n,k,0);//fixed velocity bc
                else if(type==3)          
                     NorthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i  ] =  P(chunk,0,i,j_n,k,0);//Neumann bc for pressure                        
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
                    
     //          SouthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i] =  P(chunk,0,i,0,k,0);//periodic bc
                SouthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i] =  -P(chunk,0,i,nyChunk-1,k,0);//no-slip bc
//                SouthBoundary[chunk*nzChunk*nxChunk + nxChunk * k + i] =  P(chunk,0,i,0,k,0);
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
//               NorthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i] =  P(chunk,0,i,nyChunk-1,k,0);//periodic bc
                 NorthGhosts[chunk*nzChunk*nxChunk + nxChunk * k + i] =  -P(chunk,0,i,0,k,0);//no-slip
//0*2*pi*cos(2*pi*y) ; ;
               
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
                    w = 0*sin(2*pi*z);
                    v = 0*sin(2*pi*y); 

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
      Divergence=new CFlowVariableSingleBlock(nxChunk*p_dir,nyChunk*p_dir,nzChunk*p_dir,p_dir,-1);
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
     int nzChunk=Nz/p0;//nChunk set to p0


     CFlowVariableSingleBlock *U = new CFlowVariableSingleBlock(Nx,  Ny,  Nz,  p0,0 );
     CFlowVariableSingleBlock *V = new CFlowVariableSingleBlock(Nx,  Ny,  Nz,  p0,1 );
     CFlowVariableSingleBlock *W = new CFlowVariableSingleBlock(Nx,  Ny,  Nz,  p0,2 );
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
                    u = 0.0*cos(4*pi*x)*sin(4*pi*y); 
                    w = 0;//-sin(6*pi*x)*cos(4*pi*y)*sin(2*pi*z); 
                    v = -0.0*sin(4*pi*x)*cos(4*pi*y);//*cos(2*pi*z);  

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
   Pressure = new CFlowVariableSingleBlock(  nx,  ny,  nz,  np, Xb,3 );
   Pressure->initializeZero();
   Massflow = new CFlowVariableSingleBlock(  nx,  ny,  nz,  np, Xb,-1 );
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
//     cout<<"About to compute convective terms!!!+++++------++++++++----------------\n\n\n"<<endl;
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

//    PoissonSolver->setRHS(Veln->getU());

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
   conv=new CFlowVariableSingleBlock(sc.getnx(),sc.getny(),sc.getnz(),sc.getp0(),-1);

    double c1,c2,c3;
    int  rank_x=0, rank_y=0;
    int nChunk=sc.getp0(), nxChunk=sc.getnx()/sc.getp0(), nyChunk=sc.getny()/sc.getp0(), nzChunk=sc.getnz()/sc.getp0();//nChunk set to p0
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
/*                   temp = cvel.getU().getValue(id,i,j,k,0)*sc.getGradX().getValue(id,i,j,k,0) 
                       + cvel.getV().getValue(id,i,j,k,0)*sc.getGradY().getValue(id,i,j,k,0) 
                       + cvel.getW().getValue(id,i,j,k,0)*sc.getGradZ().getValue(id,i,j,k,0);
               
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
                           (0.5*(cvel.getU().getValue(id,i-1,j,k,0) + cvel.getU().getValue(id,i,j,k,0)) - (-1.0)*TimeStep/c1*(Pressure->getValue(id,i,j,k,0)-Pressure->getValue(id,i-1,j,k,0)))  ;
                   else 
                    tempw = 0.5*(sc.getValue(id,i,j,k,0) + sc.WestGhosts[j + nyChunk * k+ id*nyChunk*nzChunk])*
                           ( 0.5*(cvel.getU().WestGhosts[j + nyChunk * k+ id*nyChunk*nzChunk] + cvel.getU().getValue(id,i,j,k,0)) 
                              - (-1.0)*TimeStep/c1*(Pressure->getValue(id,i,j,k,0)-Pressure->WestGhosts[j + nyChunk * k+ id*nyChunk*nzChunk] ))  ;                

                   if(i<nxChunk-1)
                     tempe = 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i+1,j,k,0))*
                          ( 0.5*(cvel.getU().getValue(id,i+1,j,k,0) + cvel.getU().getValue(id,i,j,k,0)) - (-1.0)*TimeStep/c1*(Pressure->getValue(id,i+1,j,k,0) - Pressure->getValue(id,i,j,k,0))) ;
                   else 
                     tempe = 0.5*(sc.getValue(id,i,j,k,0) + sc.EastGhosts[j + nyChunk * k+ id*nyChunk*nzChunk])*
                          ( 0.5*(cvel.getU().EastGhosts[j + nyChunk * k+ id*nyChunk*nzChunk]  + cvel.getU().getValue(id,i,j,k,0)) 
                          - (-1.0)*TimeStep/c1*(Pressure->EastGhosts[j + nyChunk * k+ id*nyChunk*nzChunk]  - Pressure->getValue(id,i,j,k,0))) ;                     
                   
                   if(j>0)
                   tempn = 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j-1,k,0))*
                           (0.5*(cvel.getV().getValue(id,i,j-1,k,0) + cvel.getV().getValue(id,i,j,k,0) ) - (-1.0)*TimeStep/c2*(Pressure->getValue(id,i,j,k,0) - Pressure->getValue(id,i,j-1,k,0))) ;
                   else 
                      tempn =    0.5*(sc.getValue(id,i,j,k,0) + sc.NorthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk])*
                      (0.5*(cvel.getV().NorthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk] + cvel.getV().getValue(id,i,j,k,0))
                            - (-1.0)*TimeStep/c2*( Pressure->getValue(id,i,j,k,0) - Pressure->NorthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk]   ) ) ;
                


                   if(j<nyChunk-1)
                   temps =  0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j+1,k,0))*
                           (0.5*(cvel.getV().getValue(id,i,j,k,0) + cvel.getV().getValue(id,i,j+1,k,0)) - (-1.0)*TimeStep/c2*(Pressure->getValue(id,i,j+1,k,0)- Pressure->getValue(id,i,j,k,0)))  ;
                   else 
                      temps = 0.5*(sc.getValue(id,i,j,k,0) + sc.SouthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk])*
                          (0.5*(cvel.getV().SouthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk] + cvel.getV().getValue(id,i,j,k,0))
                            - (-1.0)*TimeStep/c2*(Pressure->SouthGhosts[i + nxChunk * k+ id*nxChunk*nzChunk] - Pressure->getValue(id,i,j,k,0)  ))  ;

                   if(k>0) 
                   tempb =  0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j,k-1,0))*
                           (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().getValue(id,i,j,k-1,0)) - (-1.0)*TimeStep/c3*(Pressure->getValue(id,i,j,k,0)- Pressure->getValue(id,i,j,k-1,0)))   ;
                   else if(id==0) 
                   
                    tempb =   0.5*(sc.getValue(id,i,j,k,0) + sc.BottomGhosts[i+j*nxChunk])*
                           (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().BottomGhosts[i+j*nxChunk]) - (-1.0)*TimeStep/c3*( Pressure->getValue(id,i,j,k,0) - Pressure->BottomGhosts[i+j*nxChunk]  ) )  ;
                   else if(id>0)
                     tempb =   0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id-1,i,j,nzChunk-1,0))*
                           (0.5*( cvel.getW().getValue(id,i,j,k,0) + cvel.getW().getValue(id-1,i,j,nzChunk-1,0) ) - (-1.0)*TimeStep/c3*(Pressure->getValue(id,i,j,k,0)-Pressure->getValue(id-1,i,j,nzChunk-1,0)));

                   if(k<nzChunk-1)
                   tempt =  0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id,i,j,k+1,0))*
                           (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().getValue(id,i,j,k+1,0)) - (-1.0)*TimeStep/c3*(Pressure->getValue(id,i,j,k+1,0) - Pressure->getValue(id,i,j,k,0))) ;
                   else if(id==nChunk-1) 
                        tempt =  0.5*(sc.getValue(id,i,j,k,0) + sc.TopGhosts[i+j*nxChunk])*
                       (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().TopGhosts[i+j*nxChunk]) - (-1.0)*TimeStep/c3*(Pressure->TopGhosts[i+j*nxChunk] - Pressure->getValue(id,i,j,k,0)) )  ;

                   else if(id<nChunk-1)
                        tempt = 0.5*(sc.getValue(id,i,j,k,0) + sc.getValue(id+1,i,j,0,0))*
                      (0.5*(cvel.getW().getValue(id,i,j,k,0) + cvel.getW().getValue(id+1,i,j,0,0)) - (-1.0)*TimeStep/c3*(Pressure->getValue(id+1,i,j,0,0)- Pressure->getValue(id,i,j,k,0)))   ;

                


                   mflow =  (tempe-tempw)/(c1)+(temps-tempn)/(c2)+(tempt-tempb)/(c3);//scale with 1/dx, 1/dy, 1/dz
               
                   conv->setValue(-mflow,id,i,j,k,0); 
             
             /*        if(i>0)
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
                   Massflow->setValue(mflow,id,i,j,k,0);*/

                }

            }
        }
    }
 
 return  conv;

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
