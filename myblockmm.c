#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>
#include <x86intrin.h>
#include <sys/time.h>
#include <pthread.h>
#include "myblockmm.h"

struct thread_info
{
    int tid;
    double **a, **b, **c;
    int array_size;
    int number_of_threads;
    int n;
};
void *mythreaded_vector_blockmm(void *t);

char name[128];
char SID[128];
#define VECTOR_WIDTH 4
void my_threaded_vector_blockmm(double **a, double **b, double **c, int n, int ARRAY_SIZE, int number_of_threads)
{
  int i=0;
  pthread_t *thread;
  struct thread_info *tinfo;
  strcpy(name,"Ziliang Zhang");
  strcpy(SID,"862186678");
  thread = (pthread_t *)malloc(sizeof(pthread_t)*number_of_threads);
  tinfo = (struct thread_info *)malloc(sizeof(struct thread_info)*number_of_threads);

  for(i = 0 ; i < number_of_threads ; i++)
  {
    tinfo[i].a = a;
    tinfo[i].b = b;
    tinfo[i].c = c;
    tinfo[i].tid = i;
    tinfo[i].number_of_threads = number_of_threads;
    tinfo[i].array_size = ARRAY_SIZE;
    tinfo[i].n = n;
    pthread_create(&thread[i], NULL, mythreaded_vector_blockmm, &tinfo[i]);
  }
  for(i = 0 ; i < number_of_threads ; i++)
    pthread_join(thread[i], NULL);

  return;
}

#define VECTOR_WIDTH 4
void *mythreaded_vector_blockmm(void *t)
{
  int i,j,k, ii, jj, kk, x;
  __m256d va, vb, vc0, vc1, vc2, vc3, vc4, vc5, vc6, vc7, vc8, vc9, vc10, vc11;
  __m256d vc12, vc13, vc14, vc15;
  struct thread_info tinfo = *(struct thread_info *)t;
  int number_of_threads = tinfo.number_of_threads;
  int tid =  tinfo.tid;
  double **a = tinfo.a;
  double **b = tinfo.b;
  double **c = tinfo.c;
  int ARRAY_SIZE = tinfo.array_size;
  int n = tinfo.n;
  for(i = (ARRAY_SIZE/number_of_threads)*(tid); i < (ARRAY_SIZE/number_of_threads)*(tid+1); i+=ARRAY_SIZE/n)
  {
    for(j = 0; j < ARRAY_SIZE; j+=(ARRAY_SIZE/n))
    {
      for(k = 0; k < ARRAY_SIZE; k+=(ARRAY_SIZE/n))
      {
         for(ii = i; ii < i+(ARRAY_SIZE/n); ii+=VECTOR_WIDTH)
         {
            for(jj = j; jj < j+(ARRAY_SIZE/n); jj+=VECTOR_WIDTH)
            {
                    vc0 = _mm256_load_pd(&c[ii][jj]);
                    vc1 = _mm256_load_pd(&c[ii+1][jj]);
                    vc2 = _mm256_load_pd(&c[ii+2][jj]);
                    vc3 = _mm256_load_pd(&c[ii+3][jj]);

                    vc4 = _mm256_load_pd(&c[ii][jj+1]);
                    vc5 = _mm256_load_pd(&c[ii+1][jj+1]);
                    vc6 = _mm256_load_pd(&c[ii+2][jj+1]);
                    vc7 = _mm256_load_pd(&c[ii+3][jj+1]);

                    vc8 = _mm256_load_pd(&c[ii][jj+2]);
                    vc9 = _mm256_load_pd(&c[ii+1][jj+2]);
                    vc10 = _mm256_load_pd(&c[ii+2][jj+2]);
                    vc11 = _mm256_load_pd(&c[ii+3][jj+2]);

                    vc12 = _mm256_load_pd(&c[ii][jj+3]);
                    vc13 = _mm256_load_pd(&c[ii+1][jj+3]);
                    vc14 = _mm256_load_pd(&c[ii+2][jj+3]);
                    vc15 = _mm256_load_pd(&c[ii+3][jj+3]);

                for(kk = k; kk < k+(ARRAY_SIZE/n); kk+=VECTOR_WIDTH)
                {
                        va = _mm256_broadcast_sd(&a[ii][kk]);
                        vb = _mm256_load_pd(&b[kk][jj]);
                        vc0 = _mm256_add_pd(vc0,_mm256_mul_pd(va,vb));

                        va = _mm256_broadcast_sd(&a[ii+1][kk]);
                        vb = _mm256_load_pd(&b[kk][jj]);
                        vc1 = _mm256_add_pd(vc1,_mm256_mul_pd(va,vb));

                        va = _mm256_broadcast_sd(&a[ii+2][kk]);
                        vb = _mm256_load_pd(&b[kk][jj]);
                        vc2 = _mm256_add_pd(vc2,_mm256_mul_pd(va,vb));

                        va = _mm256_broadcast_sd(&a[ii+3][kk]);
                        vb = _mm256_load_pd(&b[kk][jj]);
                        vc3 = _mm256_add_pd(vc3,_mm256_mul_pd(va,vb));

                        //---------------------------------------------
                        va = _mm256_broadcast_sd(&a[ii][kk+1]);
                        vb = _mm256_load_pd(&b[kk+1][jj]);
                        vc4 = _mm256_add_pd(vc4,_mm256_mul_pd(va,vb));

                        va = _mm256_broadcast_sd(&a[ii+1][kk+1]);
                        vb = _mm256_load_pd(&b[kk+1][jj]);
                        vc5 = _mm256_add_pd(vc5,_mm256_mul_pd(va,vb));

                        va = _mm256_broadcast_sd(&a[ii+2][kk+1]);
                        vb = _mm256_load_pd(&b[kk+1][jj]);
                        vc6 = _mm256_add_pd(vc6,_mm256_mul_pd(va,vb));

                        va = _mm256_broadcast_sd(&a[ii+3][kk+1]);
                        vb = _mm256_load_pd(&b[kk+1][jj]);
                        vc7 = _mm256_add_pd(vc7,_mm256_mul_pd(va,vb));

                        //---------------------------------------------
                        // va = _mm256_broadcast_sd(&a[ii][kk+2]);
                        // vb = _mm256_load_pd(&b[kk+2][jj]);
                        // vc8 = _mm256_add_pd(vc8,_mm256_mul_pd(va,vb));
                        //
                        // va = _mm256_broadcast_sd(&a[ii+1][kk+2]);
                        // vb = _mm256_load_pd(&b[kk+2][jj]);
                        // vc9 = _mm256_add_pd(vc9,_mm256_mul_pd(va,vb));
                        //
                        // va = _mm256_broadcast_sd(&a[ii+2][kk+2]);
                        // vb = _mm256_load_pd(&b[kk+2][jj]);
                        // vc10 = _mm256_add_pd(vc10,_mm256_mul_pd(va,vb));
                        //
                        // va = _mm256_broadcast_sd(&a[ii+3][kk+2]);
                        // vb = _mm256_load_pd(&b[kk+2][jj]);
                        // vc11 = _mm256_add_pd(vc11,_mm256_mul_pd(va,vb));
                        //
                        // //---------------------------------------------
                        // va = _mm256_broadcast_sd(&a[ii][kk+3]);
                        // vb = _mm256_load_pd(&b[kk+3][jj]);
                        // vc12 = _mm256_add_pd(vc12,_mm256_mul_pd(va,vb));
                        //
                        // va = _mm256_broadcast_sd(&a[ii+1][kk+3]);
                        // vb = _mm256_load_pd(&b[kk+3][jj]);
                        // vc13 = _mm256_add_pd(vc13,_mm256_mul_pd(va,vb));
                        //
                        // va = _mm256_broadcast_sd(&a[ii+2][kk+3]);
                        // vb = _mm256_load_pd(&b[kk+3][jj]);
                        // vc14 = _mm256_add_pd(vc14,_mm256_mul_pd(va,vb));
                        //
                        // va = _mm256_broadcast_sd(&a[ii+3][kk+3]);
                        // vb = _mm256_load_pd(&b[kk+3][jj]);
                        // vc15 = _mm256_add_pd(vc15,_mm256_mul_pd(va,vb));
                 }
                     _mm256_store_pd(&c[ii][jj],vc0);
                     _mm256_store_pd(&c[ii+1][jj],vc1);
                     _mm256_store_pd(&c[ii+2][jj],vc2);
                     _mm256_store_pd(&c[ii+3][jj],vc3);

                     // _mm256_store_pd(&c[ii][jj+1],vc0);
                     // _mm256_store_pd(&c[ii+1][jj+1],vc1);
                     // _mm256_store_pd(&c[ii+2][jj+1],vc2);
                     // _mm256_store_pd(&c[ii+3][jj+1],vc3);
                     //
                     // _mm256_store_pd(&c[ii][jj+2],vc0);
                     // _mm256_store_pd(&c[ii+1][jj+2],vc1);
                     // _mm256_store_pd(&c[ii+2][jj+2],vc2);
                     // _mm256_store_pd(&c[ii+3][jj+2],vc3);
                     //
                     // _mm256_store_pd(&c[ii][jj+3],vc0);
                     // _mm256_store_pd(&c[ii+1][jj+3],vc1);
                     // _mm256_store_pd(&c[ii+2][jj+3],vc2);
                     // _mm256_store_pd(&c[ii+3][jj+3],vc3);
            }
          }
      }
    }
  }
}
