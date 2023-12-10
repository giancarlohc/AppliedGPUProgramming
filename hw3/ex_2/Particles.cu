#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */

__global__ void mover_PC_kernel(struct particles *part, struct EMfield* field, struct grid* grd, struct parameters* param) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < part->nop){
        
        // auxiliary variables
        FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
        FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
        FPpart omdtsq, denom, ut, vt, wt, udotb;

        // local (to the particle) electric and magnetic field
        FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
        
        // interpolation densities
        int ix,iy,iz;
        FPfield weight[2][2][2];
        FPfield xi[2], eta[2], zeta[2];
        
        // intermediate particle position and velocity
        FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

        xptilde = part->x[i];
        yptilde = part->y[i];
        zptilde = part->z[i];
        // calculate the average velocity iteratively
        for(int innter=0; innter < part->NiterMover; innter++){
            // interpolation G-->P
            ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
            iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
            iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
            
            // calculate weights
            xi[0]   = part->x[i] - grd->XN_flat[get_idx(ix - 1, iy, iz, grd->nyn, grd->nzn)];
            eta[0]  = part->y[i] - grd->YN_flat[get_idx(ix, iy - 1, iz, grd->nyn, grd->nzn)];
            zeta[0] = part->z[i] - grd->ZN_flat[get_idx(ix, iy, iz - 1, grd->nyn, grd->nzn)];
            xi[1]   = grd->XN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->x[i];
            eta[1]  = grd->YN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->y[i];
            zeta[1] = grd->ZN_flat[get_idx(ix, iy, iz, grd->nyn, grd->nzn)] - part->z[i];
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
            
            // set to zero local electric and magnetic field
            Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
            
            for (int ii=0; ii < 2; ii++)
                for (int jj=0; jj < 2; jj++)
                    for(int kk=0; kk < 2; kk++){
                        Exl += weight[ii][jj][kk]*field->Ex_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        Eyl += weight[ii][jj][kk]*field->Ey_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        Ezl += weight[ii][jj][kk]*field->Ez_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        Bxl += weight[ii][jj][kk]*field->Bxn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        Byl += weight[ii][jj][kk]*field->Byn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                        Bzl += weight[ii][jj][kk]*field->Bzn_flat[get_idx(ix-ii, iy-jj, iz-kk, grd->nyn, grd->nzn)];
                    }
            
            // end interpolation
            omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
            denom = 1.0/(1.0 + omdtsq);
            // solve the position equation
            ut= part->u[i] + qomdt2*Exl;
            vt= part->v[i] + qomdt2*Eyl;
            wt= part->w[i] + qomdt2*Ezl;
            udotb = ut*Bxl + vt*Byl + wt*Bzl;
            // solve the velocity equation
            uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
            vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
            wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
            // update position
            part->x[i] = xptilde + uptilde*dto2;
            part->y[i] = yptilde + vptilde*dto2;
            part->z[i] = zptilde + wptilde*dto2;
            
            
        } // end of iteration
        // update the final position and velocity
        part->u[i]= 2.0*uptilde - part->u[i];
        part->v[i]= 2.0*vptilde - part->v[i];
        part->w[i]= 2.0*wptilde - part->w[i];
        part->x[i] = xptilde + uptilde*dt_sub_cycling;
        part->y[i] = yptilde + vptilde*dt_sub_cycling;
        part->z[i] = zptilde + wptilde*dt_sub_cycling;
        
        
        //////////
        //////////
        ////////// BC
                                    
        // X-DIRECTION: BC particles
        if (part->x[i] > grd->Lx){
            if (param->PERIODICX==true){ // PERIODIC
                part->x[i] = part->x[i] - grd->Lx;
            } else { // REFLECTING BC
                part->u[i] = -part->u[i];
                part->x[i] = 2*grd->Lx - part->x[i];
            }
        }
                                                                    
        if (part->x[i] < 0){
            if (param->PERIODICX==true){ // PERIODIC
                part->x[i] = part->x[i] + grd->Lx;
            } else { // REFLECTING BC
                part->u[i] = -part->u[i];
                part->x[i] = -part->x[i];
            }
        }
            
        
        // Y-DIRECTION: BC particles
        if (part->y[i] > grd->Ly){
            if (param->PERIODICY==true){ // PERIODIC
                part->y[i] = part->y[i] - grd->Ly;
            } else { // REFLECTING BC
                part->v[i] = -part->v[i];
                part->y[i] = 2*grd->Ly - part->y[i];
            }
        }
                                                                    
        if (part->y[i] < 0){
            if (param->PERIODICY==true){ // PERIODIC
                part->y[i] = part->y[i] + grd->Ly;
            } else { // REFLECTING BC
                part->v[i] = -part->v[i];
                part->y[i] = -part->y[i];
            }
        }
                                                                    
        // Z-DIRECTION: BC particles
        if (part->z[i] > grd->Lz){
            if (param->PERIODICZ==true){ // PERIODIC
                part->z[i] = part->z[i] - grd->Lz;
            } else { // REFLECTING BC
                part->w[i] = -part->w[i];
                part->z[i] = 2*grd->Lz - part->z[i];
            }
        }
                                                                    
        if (part->z[i] < 0){
            if (param->PERIODICZ==true){ // PERIODIC
                part->z[i] = part->z[i] + grd->Lz;
            } else { // REFLECTING BC
                part->w[i] = -part->w[i];
                part->z[i] = -part->z[i];
            }
        }
    }
}

/** GPU particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    //prepping variables for solution
    FPpart* host_x; FPpart*  host_y; FPpart* host_z; FPpart* host_u; FPpart* host_v; FPpart* host_w;
    host_x = (FPpart*)malloc(part->npmax * sizeof(FPpart));
    host_y = (FPpart*)malloc(part->npmax * sizeof(FPpart));
    host_z = (FPpart*)malloc(part->npmax * sizeof(FPpart));
    host_u = (FPpart*)malloc(part->npmax * sizeof(FPpart));
    host_v = (FPpart*)malloc(part->npmax * sizeof(FPpart));
    host_w = (FPpart*)malloc(part->npmax * sizeof(FPpart));

    //ensuring same structure for GPU variables 
    struct particles* device_part;
    struct EMfield* device_field;
    struct grid* device_grd;
    struct parameters* device_param;

    //allocation of GPU particles and state 
    cudaMalloc(&device_part, sizeof(particles));
    cudaMalloc(&device_field, sizeof(EMfield));
    cudaMalloc(&device_grd, sizeof(grid));
    cudaMalloc(&device_param, sizeof(parameters));

    //copying over particles and state to GPU
    cudaMemcpy(device_part, part, sizeof(particles), cudaMemcpyHostToDevice);
    cudaMemcpy(device_field, field, sizeof(EMfield), cudaMemcpyHostToDevice);
    cudaMemcpy(device_grd, grd, sizeof(grid), cudaMemcpyHostToDevice);
    cudaMemcpy(device_param, param, sizeof(parameters), cudaMemcpyHostToDevice);

    //casting variables
    FPpart* device_x; FPpart*  device_y; FPpart* device_z; FPpart* device_u; FPpart* device_v; FPpart* device_w;
    FPfield* Ex_flat, *Ey_flat, *Ez_flat, *Bxn_flat, *Byn_flat, *Bzn_flat;
    FPfield* XN_flat; FPfield* YN_flat; FPfield* ZN_flat;
    size_t fieldSize = grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield);

    //allocation of variables on GPU
    cudaMalloc(&device_x, part->npmax * sizeof(FPpart));
    cudaMalloc(&device_y, part->npmax * sizeof(FPpart));
    cudaMalloc(&device_z, part->npmax * sizeof(FPpart));
    cudaMalloc(&device_u, part->npmax * sizeof(FPpart));
    cudaMalloc(&device_v, part->npmax * sizeof(FPpart));
    cudaMalloc(&device_w, part->npmax * sizeof(FPpart));

    cudaMalloc(&Ex_flat, fieldSize);
    cudaMalloc(&Ey_flat, fieldSize);
    cudaMalloc(&Ez_flat, fieldSize);
    cudaMalloc(&Bxn_flat, fieldSize);
    cudaMalloc(&Byn_flat, fieldSize);
    cudaMalloc(&Bzn_flat, fieldSize);

    cudaMalloc(&XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));
    cudaMalloc(&ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield));

    //bring over large amounts of variables

    //copy over to GPU flattended xyz and uvw 
    cudaMemcpy(device_x, part->x, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, part->y, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(device_z, part->z, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(device_u, part->u, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, part->v, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(device_w, part->w, part->npmax * sizeof(FPpart), cudaMemcpyHostToDevice);

    //break particle structs apart and copy to device pointer
    cudaMemcpy(&(device_part->x), &device_x, sizeof(FPpart *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_part->y), &device_y, sizeof(FPpart *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_part->z), &device_z, sizeof(FPpart *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_part->u), &device_u, sizeof(FPpart *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_part->v), &device_v, sizeof(FPpart *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_part->w), &device_w, sizeof(FPpart *), cudaMemcpyHostToDevice);

    //copy over to GPU flattened Ex, Ey, Ez Bxn, Byn, Bzn
    cudaMemcpy(Ex_flat, field->Ex_flat, fieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Ey_flat, field->Ey_flat, fieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Ez_flat, field->Ez_flat, fieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Bxn_flat, field->Bxn_flat, fieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Byn_flat, field->Byn_flat, fieldSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Bzn_flat, field->Bzn_flat, fieldSize, cudaMemcpyHostToDevice);

    //break field struct apart and copy pointer
    cudaMemcpy(&(device_field->Ex_flat), &Ex_flat, sizeof(FPfield *), cudaMemcpyHostToDevice); 
    cudaMemcpy(&(device_field->Ey_flat), &Ey_flat, sizeof(FPfield *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_field->Ez_flat), &Ez_flat, sizeof(FPfield *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_field->Bxn_flat), &Bxn_flat, sizeof(FPfield *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_field->Byn_flat), &Byn_flat, sizeof(FPfield *), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_field->Bzn_flat), &Bzn_flat, sizeof(FPfield *), cudaMemcpyHostToDevice);

    //copy over to GPU flattened XN, YN, ZN 
    cudaMemcpy(XN_flat, grd->XN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(YN_flat, grd->YN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(ZN_flat, grd->ZN_flat, grd->nxn * grd->nyn * grd->nzn * sizeof(FPfield), cudaMemcpyHostToDevice);

    //break grid struct apart and copy pointer
    cudaMemcpy(&(device_grd->XN_flat), &XN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_grd->YN_flat), &YN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(device_grd->ZN_flat), &ZN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    
    // start subcycling (run cuda kernel)
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++) {
        int TPB = 1024;
        int BLK = (part->nop + TPB-1)/ (float) TPB;
        printf("Blocks: %d\n", BLK);
        
        mover_PC_kernel<<<BLK,TPB>>>(device_part, device_field, device_grd, device_param);
    }

    cudaDeviceSynchronize();

    //Copy Over Results

    cudaMemcpy(part, device_part, sizeof(particles), cudaMemcpyDeviceToHost);

    cudaMemcpy(host_x, device_x, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_y, device_y, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_z, device_z, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_u, device_u, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_v, device_v, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_w, device_w, part->npmax * sizeof(FPpart), cudaMemcpyDeviceToHost);

    //Save Results

    part->x = host_x;
    part->y = host_y;
    part->z = host_z;
    part->u = host_u;
    part->v = host_v;
    part->w = host_w;

    //Free Variables

    cudaFree(device_part);
    cudaFree(device_field);
    cudaFree(device_grd);
    cudaFree(device_param);

    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_z);
    cudaFree(device_u);
    cudaFree(device_v);
    cudaFree(device_w);

    cudaFree(Ex_flat);
    cudaFree(Ey_flat);
    cudaFree(Ez_flat);
    cudaFree(Bxn_flat);
    cudaFree(Byn_flat);
    cudaFree(Bzn_flat);

    cudaFree(XN_flat);
    cudaFree(YN_flat);
    cudaFree(ZN_flat);
    
    //DO NOT FREE HOST VARIABLES HERE

    return 0;
}




/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
