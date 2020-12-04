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
  __m256d va, vb, vc;
  struct thread_info tinfo = *(struct thread_info *)t;
  int number_of_threads = tinfo.number_of_threads;
  int tid =  tinfo.tid;
  double **a = tinfo.a;
  double **b = tinfo.b;
  double **c = tinfo.c;
  int ARRAY_SIZE = tinfo.array_size;
  int n = tinfo.n;

  // Begin modification----------------------------------------------------
  int block_size = ARRAY_SIZE/number_of_threads;
  int tile_size = ARRAY_SIZE/n;

  for(i = (block_size)*(tid); i < (block_size)*(tid+1); i+=tile_size)
  {
    for(j = 0; j < ARRAY_SIZE; j+=(tile_size))
    {
      // _mm_prefetch(&b)
      for(k = 0; k < ARRAY_SIZE; k+=(tile_size))
      {
        // _mm_prefetch(&a)
         for(ii = i; ii < i+(tile_size); ii++)
         {
            for(jj = j; jj < j+(tile_size); jj+=VECTOR_WIDTH)
            {
                    vc = _mm256_loadu_pd(&c[ii][jj]);

                for(kk = k; kk < k+(tile_size); kk++)
                {
                        va = _mm256_broadcast_sd(&a[ii][kk]);
                        vb = _mm256_load_pd(&b[kk][jj]);
                        vc = _mm256_fmadd_pd(vc,_mm256_mul_pd(va,vb));
                 }
                     _mm256_storeu_pd(&c[ii][jj],vc);
            }
          }
      }
    }
  }
}
