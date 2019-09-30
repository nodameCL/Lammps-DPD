/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Kurt Smith (U Pittsburgh)
------------------------------------------------------------------------- */
// User modified code for Long-range Columb force calculation

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_dpd.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "update.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"
#include "kspace.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"

using namespace LAMMPS_NS;

#define EPSILON 1.0e-10
#define PI 3.14159265358979323846
#define gammaelec 13.87
#define beta 0.929 // inverse lambda: 1/lambda
#define EWALD_F   1.12837917 // 2/sqrt(PI)
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

/* ---------------------------------------------------------------------- */

PairDPD::PairDPD(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
  random = NULL;
  ewaldflag = pppmflag = 1;
}

/* ---------------------------------------------------------------------- */

PairDPD::~PairDPD()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(a0);
    memory->destroy(gamma);
    memory->destroy(sigma);
    // morse term 
    memory->destroy(d0);
    memory->destroy(alpha);
    memory->destroy(r0);
    memory->destroy(morse1);
    memory->destroy(offset);
  }

  if (random) delete random;
}

/* ---------------------------------------------------------------------- */

void PairDPD::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype,itable;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double vxtmp,vytmp,vztmp,delvx,delvy,delvz;
  double rsq,r,rinv,r2inv,dot,wd,randnum,factor_dpd;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double grij,expm2,prefactor,t,erfc;
  double qtmp,e_coeff,e_coeff_2,forcecoul,forcedpd,factor_coul,ecoul;
  double fraction,table;
  // morse term 
  double dr, dexp, forcemorse;

  evdwl = ecoul = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double  *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double dtinvsqrt = 1.0/sqrt(update->dt);
  double qqrd2e = force->qqrd2e;
  //double bjerelec = gammaelec / (4.0*PI);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_dpd = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r = sqrt(rsq); 

       // electrostatic force calculation 
        if (q[i] != 0.0 && q[j] != 0.0) {
          r2inv = 1.0/rsq;
          grij = g_ewald * r;
          expm2 = exp(-grij*grij);
          t = 1.0 / (1.0 + EWALD_P*grij);
          erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
          prefactor = qqrd2e * qtmp*q[j]/r;
          e_coeff_2 = 1.0 + 2.0*r*beta*(1.0 + r*beta);
          e_coeff = exp(-2.0*r*beta)*e_coeff_2;
          forcecoul = prefactor * (erfc + EWALD_F*grij*expm2 - e_coeff);

          if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul-e_coeff)*prefactor;
        } else forcecoul = 0.0; 

        // dpd interaction 
        if (r > EPSILON && r < 1.0) {     // r can be 0.0 in DPD systems
          rinv = 1.0/r;
          delvx = vxtmp - v[j][0];
          delvy = vytmp - v[j][1];
          delvz = vztmp - v[j][2];
          dot = delx*delvx + dely*delvy + delz*delvz;
          wd = 1.0 - r;
          randnum = random->gaussian();

          // conservative force = a0 * wd
          // drag force = -gamma * wd^2 * (delx dot delv) / r
          // random force = sigma * wd * rnd * dtinvsqrt;

          forcedpd = a0[itype][jtype]*wd;
          forcedpd -= gamma[itype][jtype]*wd*wd*dot*rinv;
          forcedpd += sigma[itype][jtype]*wd*randnum*dtinvsqrt;
        } else forcedpd = 0.0; 

        //morse potential 

        if (r < 1.0) {     // r can be 0.0 in DPD systems
          // morse term 

          dr = r - r0[itype][jtype];
          dexp = exp(-alpha[itype][jtype] * dr);

          // f_morse = -2*alpha*D0*[exp(-2*alpha*dr - exp(-alpha*dr)]
          // f_morse = morse1*(dexp * dexp - dexp) / r 

          forcemorse = factor_dpd * morse1[itype][jtype] * (dexp*dexp - dexp) / r;
        } else forcemorse = 0.0; 
     
        fpair = factor_dpd*forcedpd*rinv + forcemorse + forcecoul*r2inv;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          if (q[i] != 0.0 && q[j]!= 0.0) {
              ecoul = prefactor*(erfc - (1.0 + beta*r)*exp(-2.0*r*beta));
            if (factor_coul < 1.0) ecoul -= (1.0-factor_coul - (1.0 + beta*r)*exp(-2.0*r*beta))*prefactor;
          } else ecoul = 0.0;

          if (r < 1.0) {
            evdwl = d0[itype][jtype] * (dexp*dexp - 2.0*dexp) - offset[itype][jtype];

            if (r > EPSILON) {
                evdwl += 0.5*a0[itype][jtype] * wd*wd;
            }

            evdwl *= factor_dpd;
            
          } else evdwl = 0.0;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,ecoul,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairDPD::allocate()
{
  int i,j;
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq"); // global cut off square

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(a0,n+1,n+1,"pair:a0");
  memory->create(gamma,n+1,n+1,"pair:gamma");
  memory->create(sigma,n+1,n+1,"pair:sigma");

  // morse term 
  memory->create(d0,n+1,n+1,"pair:d0");
  memory->create(alpha,n+1,n+1,"pair:alpha");
  memory->create(r0,n+1,n+1,"pair:r0");
  memory->create(morse1,n+1,n+1,"pair:morse1");
  memory->create(offset,n+1,n+1,"pair:offset");

  for (i = 0; i <= atom->ntypes; i++)
    for (j = 0; j <= atom->ntypes; j++)
      sigma[i][j] = gamma[i][j] = 0.0;
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairDPD::settings(int narg, char **arg)
{
  if (narg != 3) error->all(FLERR,"Illegal pair_style command");

  temperature = force->numeric(FLERR,arg[0]);
  cut_global = force->numeric(FLERR,arg[1]);
  seed = force->inumeric(FLERR,arg[2]);

  // initialize Marsaglia RNG with processor-unique seed

  if (seed <= 0) error->all(FLERR,"Illegal pair_style command");
  delete random;
  random = new RanMars(lmp,seed + comm->me);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
       //printf("cutoff is %4.2f\n:", cut[i][j]);
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDPD::coeff(int narg, char **arg)
{
  if (narg < 7 || narg > 8)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double a0_one = force->numeric(FLERR,arg[2]);
  double gamma_one = force->numeric(FLERR,arg[3]);

  // morse term 
  double d0_one = force->numeric(FLERR,arg[4]);
  double alpha_one = force->numeric(FLERR,arg[5]);
  double r0_one = force->numeric(FLERR,arg[6]);

  double cut_one = cut_global;
  //if (narg == 5) cut_dpd_one = force->numeric(FLERR,arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      a0[i][j] = a0_one;
      gamma[i][j] = gamma_one;
      // morse term 
      d0[i][j] = d0_one;
      alpha[i][j] = alpha_one;
      r0[i][j] = r0_one;

      cut[i][j] = cut_one; 
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairDPD::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR,"Pair style dpd requires atom attribute q");

  if (comm->ghost_velocity == 0)
    error->all(FLERR,"Pair dpd requires ghost atoms store velocity");

  // if newton off, forces between atoms ij will be double computed
  // using different random numbers

  if (force->newton_pair == 0 && comm->me == 0) error->warning(FLERR,
      "Pair dpd needs newton pair on for momentum conservation");

  neighbor->request(this,instance_me);

    // insure use of KSpace long-range solver, set g_ewald

 if (force->kspace == NULL)
    error->all(FLERR,"Pair style requires a KSpace style");
  g_ewald = force->kspace->g_ewald;

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairDPD::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  sigma[i][j] = sqrt(2.0*force->boltz*temperature*gamma[i][j]);

  // morse term 
  morse1[i][j] = 2.0*d0[i][j]*alpha[i][j];

  if (offset_flag) {
    double alpha_dr = -alpha[i][j] * (cut[i][j] - r0[i][j]);
    offset[i][j] = d0[i][j] * (exp(2.0*alpha_dr) - 2.0*exp(alpha_dr));
  } else offset[i][j] = 0.0;

  d0[j][i] = d0[i][j];
  alpha[j][i] = alpha[i][j];
  r0[j][i] = r0[i][j];
  morse1[j][i] = morse1[i][j];
  offset[j][i] = offset[i][j];

  cut[j][i] = cut[i][j];
  a0[j][i] = a0[i][j];
  gamma[j][i] = gamma[i][j];
  sigma[j][i] = sigma[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairDPD::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&a0[i][j],sizeof(double),1,fp);
        fwrite(&gamma[i][j],sizeof(double),1,fp);
        fwrite(&d0[i][j],sizeof(double),1,fp);
        fwrite(&alpha[i][j],sizeof(double),1,fp);
        fwrite(&r0[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairDPD::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&a0[i][j],sizeof(double),1,fp);
          fread(&gamma[i][j],sizeof(double),1,fp);
          fread(&d0[i][j],sizeof(double),1,fp);
          fread(&alpha[i][j],sizeof(double),1,fp);
          fread(&r0[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&a0[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&gamma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&d0[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&alpha[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&r0[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairDPD::write_restart_settings(FILE *fp)
{
  fwrite(&temperature,sizeof(double),1,fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&seed,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  // morse term 
  fwrite(&offset_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairDPD::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&temperature,sizeof(double),1,fp);
    fread(&cut_global,sizeof(double),1,fp);
    fread(&seed,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    // morse term 
    fread(&offset_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&temperature,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&seed,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  // morse term 
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);

  // initialize Marsaglia RNG with processor-unique seed
  // same seed that pair_style command initially specified

  if (random) delete random;
  random = new RanMars(lmp,seed + comm->me);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairDPD::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g %g %g %g\n",
            i,a0[i][i],gamma[i][i],d0[i][i],alpha[i][i],r0[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairDPD::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g %g %g %g\n",
              i,j,a0[i][j],gamma[i][j],d0[i][j],alpha[i][j],r0[i][j],cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairDPD::single(int i, int j, int itype, int jtype, double rsq,
                       double factor_coul, double factor_dpd, double &fforce)
{
  double r,rinv,wd,phidpd,phicoul;
  double r2inv,grij,expm2,t,erfc,prefactor;
  double fraction,forcecoul,forcedpd;
  double e_coeff,e_coeff_2;
  //double bjerelec = gammaelec / (4.0*PI);
  double qqrd2e = force->qqrd2e;
  // morse term 
  double dr,dexp,phi,forcemorse;

  r = sqrt(rsq);
  r2inv = 1.0/rsq;

  if (atom->q[i] != 0.0 && atom->q[j] != 0.0) {
      grij = g_ewald * r;
      expm2 = exp(-grij*grij);
      t = 1.0 / (1.0 + EWALD_P*grij);
      erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
      prefactor = qqrd2e * atom->q[i]*atom->q[j]/r;
      e_coeff_2 = 1.0 + 2.0*r*beta*(1.0 + r*beta);
      e_coeff = exp(-2.0*r*beta)*e_coeff_2;
      forcecoul = prefactor * (erfc + EWALD_F*grij*expm2 - e_coeff);
      if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul-e_coeff)*prefactor;
  } else forcecoul = 0.0;

    if (r < EPSILON || r >= 1.0) {
      forcedpd = 0.0;
    } else {
      rinv = 1.0/r;
      wd = 1.0 - r;
      forcedpd = a0[itype][jtype]*wd * factor_dpd*rinv;
    }

    if (r < 1.0) {
      rinv = 1.0/r;
      dr = r - r0[itype][jtype];
      dexp = exp(-alpha[itype][jtype] * dr);
      forcemorse = morse1[itype][jtype] * (dexp*dexp - dexp) * factor_dpd*rinv;
    }

  fforce = forcecoul*r2inv + forcemorse + forcedpd;

  double eng = 0.0;
  if (atom->q[i] != 0.0 && atom->q[j] !=0.0) {
      phicoul = prefactor*(erfc- (1.0 + beta*r)*exp(-2.0*r*beta));
    if (factor_coul < 1.0) phicoul -= (1.0-factor_coul- (1.0 + beta*r)*exp(-2.0*r*beta))*prefactor;
    eng += phicoul;
  }

  if (r > EPSILON && r < 1.0) {
    phidpd = 0.5*a0[itype][jtype] * wd*wd;
    eng += factor_dpd*phidpd;
  }

  if (r < 1.0) {
    phi = d0[itype][jtype] * (dexp*dexp - 2.0*dexp) - offset[itype][jtype];
    eng += factor_dpd*phi;
  }
  return eng;
}

/* ---------------------------------------------------------------------- */

void *PairDPD::extract(const char *str, int &dim)
{
  dim = 0;
  if (strcmp(str,"cut_coul") == 0) return (void *) &cut_global;
  dim = 2;
  if (strcmp(str,"a0") == 0) return (void *) a0;
  if (strcmp(str,"gamma") == 0) return (void *) gamma;
  if (strcmp(str,"d0") == 0) return (void *) d0;
  if (strcmp(str,"r0") == 0) return (void *) r0;
  if (strcmp(str,"alpha") == 0) return (void *) alpha;
  return NULL;
}
