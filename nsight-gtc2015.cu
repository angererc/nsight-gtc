#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef WITH_OPENGL
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#endif

#include <cuda_runtime_api.h>

#ifdef WITH_OPENGL
#include <cuda_gl_interop.h>
#endif

//1 Coalescing
//2 Occupancy
//3 Shared Memory
//4 Texture Cache
//5 B


// STEP 0x00: Default code
// STEP 0x1a: Change the block size to 32x2 (improve coalescing) -- It. 1, Eclipse Ed.
// STEP 0x1b: Change the block size to 8x16 (improve occupancy) -- It. 1, Visual Studio Ed.
// STEP 0x20: Change the block size to 32x4 (improve occupancy)
// STEP 0x30: Use launch_bounds to register pressure (improve occupancy) -- It. 2, Visual Studio Ed.
// STEP 0x40: Use shared memory (improve memory accesses)
// STEP 0x50: Use read-only path (reduce pressure on Load-store unit)
// STEP 0x5a: Optimized convolution filter 2D
// STEP 0x60: Implement a separable filter (reduce arithmetic intensity)
// STEP 0x70: Process two elements per thread (improve memory efficiency, increase ILP)
// STEP 0x80: Improve shared memory accesses (reduce bank conflicts)
// STEP 0x90: Use floats rather than ints (reduce pressure on arithmetic pipe)

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK(cond) do { \
  if( !(cond) ) { \
    fprintf(stderr, "Error at line %d in %s\n", __LINE__, __FILE__); \
    exit(1); \
  } \
} while(0)

#define CHECK_WITH_MSG(cond, msg) do { \
  if( !(cond) ) { \
    fprintf(stderr, "Error at line %d in %s: %s\n", __LINE__, __FILE__, msg); \
    exit(1); \
  } \
} while(0)

#define CHECK_CUDA(call) do { \
  cudaError_t status = call; \
  if( status != cudaSuccess ) { \
    fprintf(stderr, "CUDA Error at line %d in %s: %s\n", __LINE__, __FILE__, cudaGetErrorString(status)); \
    exit(1); \
  } \
} while(0)

#define CHECK_OPENGL(call) do { \
  call; \
  GLenum status = glGetError(); \
  if( status != GL_NO_ERROR ) { \
    fprintf(stderr, "OpenGL Error at line %d in %s: %d\n", __LINE__, __FILE__, (int) status); \
    exit(1); \
  } \
} while(0)

double getElapsedTimeInMS(cudaEvent_t eStart, cudaEvent_t eStop) {
  CHECK_CUDA(cudaEventSynchronize(eStop));
  float millisec = 0;
  cudaEventElapsedTime(&millisec, eStart, eStop);
  return (double)millisec;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef unsigned char uchar;
typedef unsigned int  uint;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum { SHOW_RGBA = 0, SHOW_GRAYSCALE = 1, SHOW_SMOOTHED_GRAYSCALE = 2, SHOW_EDGES = 4 };

struct GlobalData
{
  int img_w;
  int img_h;

  uchar4 *img_rgba;
  uchar  *img_grayscale;
  uchar  *img_smoothed_grayscale;

#ifdef WITH_OPENGL
  cudaGraphicsResource *img_cuda_pbo; 

  GLuint img_pbo; 
  GLuint img_tex;
#endif

  // What does the OpenGL renderer shows? 
  int show;
};

static GlobalData g_data;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline int round_up(int x, int y)
{
  return (x + y-1) / y;
}

// ====================================================================================================================

static __device__ __forceinline__ int in_img(int x, int y, int w, int h)
{
  return x >= 0 && x < w && y >= 0 && y < h;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static __constant__ int gaussian_filter[7][7] = {
  1,   2,   3,    4,    3,    2,    1,
  2,   4,   6,    8,    6,    4,    2,
  3,   6,   9,   12,    9,    6,    3,
  4,   8,  12,   16,   12,    8,    4,
  3,   6,   9,   12,    9,    6,    3,
  2,   4,   6,    8,    6,    4,    2,
  1,   2,   3,    4,    3,    2,    1
};

static __constant__ float gaussian_filter_fp32[7] = {
  1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 2.0f, 1.0f
};

static __constant__ int sobel_filter_x[3][3] = {
  -1, 0, 1,
  -2, 0, 2,
  -1, 0, 1,
};

static __constant__ int sobel_filter_y[3][3] = {
   1,  2,  1,
   0,  0,  0,
  -1, -2, -1,
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void rgba_to_grayscale_kernel_v0(int w, int h, const uchar4 *src, uchar *dst)
{
  // Position of the thread in the image.
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // Early exit if the thread is not in the image.
  if( !in_img(x, y, w, h) )
    return;

  // Load the pixel in RGBA format.
  uchar4 p = src[y*w + x];

  // Extract the 3 components in FP32.
  float r = (float) p.x;
  float g = (float) p.y;
  float b = (float) p.z;

  // Compute the grayscale value.
  float gray = 0.298839f*r + 0.586811f*g + 0.114350f*b;

  // Store the result.
  dst[y*w + x] = (uchar) (gray >= 255.f ? 255 : gray);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void rgba_to_grayscale_kernel_v1(int w, int h, const uchar4 *src, uchar *dst)
{
  //compute two pixels at the same time
  // Position of the thread in the image.
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = 2 * (blockIdx.y*blockDim.y) + threadIdx.y;

  // Load the pixel in RGBA format.
  uchar4 p0 = in_img(x, y, w, h) ? src[y*w + x] : make_uchar4(0,0,0,0);
  uchar4 p1 = in_img(x, y, w, h) ? src[(y+1)*w + x] : make_uchar4(0,0,0,0);

  // Extract the 3 components in FP32.
  float r0 = (float) p0.x;
  float g0 = (float) p0.y;
  float b0 = (float) p0.z;

  float r1 = (float) p1.x;
  float g1 = (float) p1.y;
  float b1 = (float) p1.z;

  // Compute the grayscale value.
  float gray0 = 0.298839f*r0 + 0.586811f*g0 + 0.114350f*b0;
  float gray1 = 0.298839f*r1 + 0.586811f*g1 + 0.114350f*b1;

  // Store the result.
  if(in_img(x, y, w, h))
    dst[y*w + x] = (uchar) (gray0 >= 255.f ? 255 : gray0);
  if(in_img(x, y+1, w, h))
    dst[(y+1)*w + x] = (uchar) (gray1 >= 255.f ? 255 : gray1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void gaussian_filter_7x7_v0(int w, int h, const uchar *src, uchar *dst)
{
  // Position of the thread in the image.
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // Early exit if the thread is not in the image.
  if( !in_img(x, y, w, h) )
    return;

  // Load the 48 neighbours and myself.
  int n[7][7];
  for( int j = -3 ; j <= 3 ; ++j )
    for( int i = -3 ; i <= 3 ; ++i )
      n[j+3][i+3] = in_img(x+i, y+j, w, h) ? (int) src[(y+j)*w + (x+i)] : 0;

  // Compute the convolution.
  int p = 0;
  for( int j = 0 ; j < 7 ; ++j )
    for( int i = 0 ; i < 7 ; ++i )
      p += gaussian_filter[j][i] * n[j][i];

  // Store the result.
  dst[y*w + x] = (uchar) (p / 256);
}

// ====================================================================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
__global__ __launch_bounds__(128, 10) 
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 200
__global__ __launch_bounds__(128, 8) 
#else
__global__
#endif
void gaussian_filter_7x7_v1(int w, int h, const uchar *src, uchar *dst)
{
  // Position of the thread in the image.
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // Early exit if the thread is not in the image.
  if( !in_img(x, y, w, h) )
    return;

  // Load the 48 neighbours and myself.
  int n[7][7];
  for( int j = -3 ; j <= 3 ; ++j )
    for( int i = -3 ; i <= 3 ; ++i )
      n[j+3][i+3] = in_img(x+i, y+j, w, h) ? (int) src[(y+j)*w + (x+i)] : 0;

  // Compute the convolution.
  int p = 0;
  for( int j = 0 ; j < 7 ; ++j )
    for( int i = 0 ; i < 7 ; ++i )
      p += gaussian_filter[j][i] * n[j][i];

  // Store the result.
  dst[y*w + x] = (uchar) (p / 256);
}

// ====================================================================================================================

__global__ void gaussian_filter_7x7_v2(int w, int h, const uchar *src, uchar *dst)
{
  // Position of the thread in the image.
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // Shared memory.
  __shared__ uchar smem_img[10][64];

  // Load pixels to SMEM.
  uchar *smem_img_ptr = &smem_img[threadIdx.y][threadIdx.x];
  for( int iy = y-3 ; iy <= blockIdx.y*blockDim.y+6 ; iy += 4, smem_img_ptr += 4*64 )
  {
    smem_img_ptr[ 0] = in_img(x- 3, iy, w, h) ? src[iy*w + (x -3)] : 0;
    smem_img_ptr[32] = in_img(x+29, iy, w, h) ? src[iy*w + (x+29)] : 0; // 29 = 32-3.
  }
  __syncthreads();

  // Load the 48 neighbours and myself.
  int n[7][7];
  for( int j = 0 ; j <= 6 ; ++j )
    for( int i = 0 ; i <= 6 ; ++i )
      n[j][i] = smem_img[threadIdx.y+j][threadIdx.x+i];

  // Compute the convolution.
  int p = 0;
  for( int j = 0 ; j < 7 ; ++j )
    for( int i = 0 ; i < 7 ; ++i )
      p += gaussian_filter[j][i] * n[j][i];

  // Store the result.
  if( in_img(x, y, w, h) )
    dst[y*w + x] = (uchar) (p / 256);
}

// ====================================================================================================================

__global__ void gaussian_filter_7x7_v3(int w, int h, const uchar *__restrict src, uchar *dst)
{
  // Position of the thread in the image.
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // Shared memory.
  __shared__ uchar smem_img[10][64];

  // Load pixels to SMEM.
  uchar *smem_img_ptr = &smem_img[threadIdx.y][threadIdx.x];
  for( int iy = y-3 ; iy <= blockIdx.y*blockDim.y+6 ; iy += 4, smem_img_ptr += 4*64 )
  {
    smem_img_ptr[ 0] = in_img(x- 3, iy, w, h) ? src[iy*w + (x -3)] : 0;
    smem_img_ptr[32] = in_img(x+29, iy, w, h) ? src[iy*w + (x+29)] : 0; // 29 = 32-3.
  }
  __syncthreads();

  // Load the 49 neighbours and myself.
  int n[7][7];
  for( int j = 0 ; j <= 6 ; ++j )
    for( int i = 0 ; i <= 6 ; ++i )
      n[j][i] = smem_img[threadIdx.y+j][threadIdx.x+i];

  // Compute the convolution.
  int p = 0;
  for( int j = 0 ; j < 7 ; ++j )
    for( int i = 0 ; i < 7 ; ++i )
      p += gaussian_filter[j][i] * n[j][i];

  // Store the result.
  if( in_img(x, y, w, h) )
    dst[y*w + x] = (uchar) (p / 256);
}

// ====================================================================================================================

__global__ void gaussian_filter_7x7_v3_bis(int w, int h, const uchar *__restrict src, uchar *dst)
{
  // Position of the thread in the image.
  const int x = 1*(blockIdx.x*blockDim.x) + threadIdx.x;
  const int y = 2*(blockIdx.y*blockDim.y) + threadIdx.y;

  // Shared memory.
  __shared__ float smem_img[32][40];

  // Pixel to load.
  const int load_x = blockIdx.x*blockDim.x + 2*threadIdx.x - 4; // -4 for alignment (it should be -3).
  
  // Each thread loads 8 pixels.
  uchar2 p0 = in_img(load_x, y- 3, w, h) ? *reinterpret_cast<const uchar2*>(&src[(y- 3)*w + load_x]) : make_uchar2(0, 0);
  uchar2 p1 = in_img(load_x, y+ 5, w, h) ? *reinterpret_cast<const uchar2*>(&src[(y+ 5)*w + load_x]) : make_uchar2(0, 0);
  uchar2 p2 = in_img(load_x, y+13, w, h) ? *reinterpret_cast<const uchar2*>(&src[(y+13)*w + load_x]) : make_uchar2(0, 0);
  uchar2 p3 = in_img(load_x, y+21, w, h) ? *reinterpret_cast<const uchar2*>(&src[(y+21)*w + load_x]) : make_uchar2(0, 0);

  // Store to shared memory.
  if( threadIdx.x < 20 )
  {
    reinterpret_cast<float2*>(smem_img[threadIdx.y+ 0])[threadIdx.x] = make_float2((float) p0.x, (float) p0.y);
    reinterpret_cast<float2*>(smem_img[threadIdx.y+ 8])[threadIdx.x] = make_float2((float) p1.x, (float) p1.y);
    reinterpret_cast<float2*>(smem_img[threadIdx.y+16])[threadIdx.x] = make_float2((float) p2.x, (float) p2.y);
    reinterpret_cast<float2*>(smem_img[threadIdx.y+24])[threadIdx.x] = make_float2((float) p3.x, (float) p3.y);
  }
  __syncthreads();

  // Load the 49 neighbours and myself.
  float n[8][7];
  for( int j = 0 ; j <= 7 ; ++j )
    for( int i = 0 ; i <= 6 ; ++i )
      n[j][i] = smem_img[2*threadIdx.y+j][threadIdx.x+i];

  // Compute the convolutions.
  float p[2] = {0.0f};
  for( int j = 0 ; j < 7 ; ++j )
    for( int i = 0 ; i < 7 ; ++i )
    {
      p[0] += gaussian_filter[j][i] * n[j+0][i];
      p[1] += gaussian_filter[j][i] * n[j+1][i];
    }

  // Where to write the result 2*(blockIdx.x*blockDim.x + threadIdx.y).
  const int write_y = y + threadIdx.y;

  // Write the pixels.
  if( in_img(x, write_y, w, h) )
    dst[write_y*w + x] = (uchar) ((int) p[0] >> 8);
  if( in_img(x, write_y+1, w, h) )
    dst[(write_y+1)*w + x] = (uchar) ((int) p[1] >> 8);
}

// ====================================================================================================================

__global__ void gaussian_filter_7x7_v4(int w, int h, const uchar *__restrict src, uchar *dst) // 32x8 blocks.
{
  // Position of the thread in the image.
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // Shared memory.
  __shared__ int smem_img[16][38];

  // Each thread loads 2 pixels.
  int p0 = in_img(x-3, y-3, w, h) ? src[(y-3)*w + x-3] : 0;
  int p1 = in_img(x-3, y+5, w, h) ? src[(y+5)*w + x-3] : 0;
  
  // Load extra pixels per row.
  int p2 = in_img(x+29, y-3, w, h) && threadIdx.x < 6 ? src[(y-3)*w + x+29] : 0;
  int p3 = in_img(x+29, y+5, w, h) && threadIdx.x < 6 ? src[(y+5)*w + x+29] : 0;

  // Store to shared memory.
  smem_img[threadIdx.y+0][threadIdx.x] = p0;
  smem_img[threadIdx.y+8][threadIdx.x] = p1;

  // Store extra pixels.
  if( threadIdx.x < 6 )
  {
    smem_img[threadIdx.y+0][threadIdx.x+32] = p2;
    smem_img[threadIdx.y+8][threadIdx.x+32] = p3;
  }
  __syncthreads();
  
  // Compute the horizontal convolution.
  int n0[7], n1[7];
  for( int i = 0 ; i < 7 ; ++i )
  {
    n0[i] = smem_img[threadIdx.y+0][threadIdx.x+i];
    n1[i] = smem_img[threadIdx.y+8][threadIdx.x+i];
  }
  int p[2] = {0};
  for( int i = 0 ; i < 7 ; ++i )
  {
    p[0] += gaussian_filter[0][i] * n0[i];
    p[1] += gaussian_filter[0][i] * n1[i];
  }
  __syncthreads();

  // Write the result back to shared memory.
  smem_img[threadIdx.y+0][threadIdx.x] = p[0];
  smem_img[threadIdx.y+8][threadIdx.x] = p[1];

  // Make sure the results are in SMEM.
  __syncthreads();

  // Compute the vertical convolution.
  int n[7];
  for( int i = 0 ; i < 7 ; ++i )
    n[i] = smem_img[threadIdx.y+i][threadIdx.x];
  int q = 0;
  for( int i = 0 ; i < 7 ; ++i )
    q += gaussian_filter[i][0] * n[i];

  // Write the pixels.
  if( in_img(x, y, w, h) )
    dst[y*w + x] = (uchar) (q >> 8);
}

// ====================================================================================================================

__global__ void gaussian_filter_7x7_v5(int w, int h, const uchar *__restrict src, uchar *dst) // 32x8 blocks.
{
  // Position of the thread in the image.
  const int x = 1*(blockIdx.x*blockDim.x) + threadIdx.x;
  const int y = 2*(blockIdx.y*blockDim.y) + threadIdx.y;

  // Shared memory.
  __shared__ int smem_img[32][40];

  // Pixel to load.
  const int load_x = blockIdx.x*blockDim.x + 2*threadIdx.x - 4; // -4 for alignment (it should be -3).
  
  // Each thread loads 8 pixels.
  uchar2 p0 = in_img(load_x, y- 3, w, h) ? *reinterpret_cast<const uchar2*>(&src[(y- 3)*w + load_x]) : make_uchar2(0, 0);
  uchar2 p1 = in_img(load_x, y+ 5, w, h) ? *reinterpret_cast<const uchar2*>(&src[(y+ 5)*w + load_x]) : make_uchar2(0, 0);
  uchar2 p2 = in_img(load_x, y+13, w, h) ? *reinterpret_cast<const uchar2*>(&src[(y+13)*w + load_x]) : make_uchar2(0, 0);
  uchar2 p3 = in_img(load_x, y+21, w, h) ? *reinterpret_cast<const uchar2*>(&src[(y+21)*w + load_x]) : make_uchar2(0, 0);

  // Store to shared memory.
  if( threadIdx.x < 20 )
  {
    reinterpret_cast<int2*>(smem_img[threadIdx.y+ 0])[threadIdx.x] = make_int2(p0.x, p0.y);
    reinterpret_cast<int2*>(smem_img[threadIdx.y+ 8])[threadIdx.x] = make_int2(p1.x, p1.y);
    reinterpret_cast<int2*>(smem_img[threadIdx.y+16])[threadIdx.x] = make_int2(p2.x, p2.y);
    reinterpret_cast<int2*>(smem_img[threadIdx.y+24])[threadIdx.x] = make_int2(p3.x, p3.y);
  }
  __syncthreads();
  
  // Compute the horizontal convolution.
  int n0[7], n1[7], n2[7], n3[7];
  for( int i = 0 ; i < 7 ; ++i )
  {
    n0[i] = smem_img[threadIdx.y+ 0][threadIdx.x + i+1]; // +1 because of alignment constraint when loading pixels.
    n1[i] = smem_img[threadIdx.y+ 8][threadIdx.x + i+1];
    n2[i] = smem_img[threadIdx.y+16][threadIdx.x + i+1];
    n3[i] = smem_img[threadIdx.y+24][threadIdx.x + i+1];
  }
  int p[4] = {0};
  for( int i = 0 ; i < 7 ; ++i )
  {
    p[0] += gaussian_filter[0][i] * n0[i];
    p[1] += gaussian_filter[0][i] * n1[i];
    p[2] += gaussian_filter[0][i] * n2[i];
    p[3] += gaussian_filter[0][i] * n3[i];
  }
  __syncthreads();

  // Write the result back to shared memory.
  smem_img[threadIdx.y+ 0][threadIdx.x] = p[0];
  smem_img[threadIdx.y+ 8][threadIdx.x] = p[1];
  smem_img[threadIdx.y+16][threadIdx.x] = p[2];
  smem_img[threadIdx.y+24][threadIdx.x] = p[3];

  // Make sure the results are in SMEM.
  __syncthreads();

  // Compute the vertical convolution.
  int n[8];
  for( int i = 0 ; i < 8 ; ++i )
    n[i] = smem_img[2*threadIdx.y+i][threadIdx.x];
  int q0 = 0, q1 = 0;
  for( int i = 0 ; i < 7 ; ++i )
  {
    q0 += gaussian_filter[i][0] * n[i+0];
    q1 += gaussian_filter[i][0] * n[i+1];
  }

  // Where to write the result 2*(blockIdx.x*blockDim.x + threadIdx.y).
  const int write_y = y + threadIdx.y;

  // Write the pixels.
  if( in_img(x, write_y, w, h) )
    dst[write_y*w + x] = (uchar) (q0 >> 8);
  if( in_img(x, write_y+1, w, h) )
    dst[(write_y+1)*w + x] = (uchar) (q1 >> 8);
}

// ====================================================================================================================

__global__ void gaussian_filter_7x7_v6(int w, int h, const uchar *__restrict src, uchar *dst) // 32x8 blocks.
{
  // Position of the thread in the image.
  const int x = 1*(blockIdx.x*blockDim.x) + threadIdx.x;
  const int y = 2*(blockIdx.y*blockDim.y) + threadIdx.y;

  // Shared memory.
  __shared__ float smem_img[32][40];

  // Pixel to load.
  const int load_x = blockIdx.x*blockDim.x + 2*threadIdx.x - 4; // -4 for alignment (it should be -3).
  
  // Each thread loads 8 pixels.
  uchar2 p0 = in_img(load_x, y- 3, w, h) ? *reinterpret_cast<const uchar2*>(&src[(y- 3)*w + load_x]) : make_uchar2(0, 0);
  uchar2 p1 = in_img(load_x, y+ 5, w, h) ? *reinterpret_cast<const uchar2*>(&src[(y+ 5)*w + load_x]) : make_uchar2(0, 0);
  uchar2 p2 = in_img(load_x, y+13, w, h) ? *reinterpret_cast<const uchar2*>(&src[(y+13)*w + load_x]) : make_uchar2(0, 0);
  uchar2 p3 = in_img(load_x, y+21, w, h) ? *reinterpret_cast<const uchar2*>(&src[(y+21)*w + load_x]) : make_uchar2(0, 0);

  // Store to shared memory.
  if( threadIdx.x < 20 )
  {
    reinterpret_cast<float2*>(smem_img[threadIdx.y+ 0])[threadIdx.x] = make_float2((float) p0.x, (float) p0.y);
    reinterpret_cast<float2*>(smem_img[threadIdx.y+ 8])[threadIdx.x] = make_float2((float) p1.x, (float) p1.y);
    reinterpret_cast<float2*>(smem_img[threadIdx.y+16])[threadIdx.x] = make_float2((float) p2.x, (float) p2.y);
    reinterpret_cast<float2*>(smem_img[threadIdx.y+24])[threadIdx.x] = make_float2((float) p3.x, (float) p3.y);
  }
  __syncthreads();
  
  // Compute the horizontal convolution.
  float n0[7], n1[7], n2[7], n3[7];
  for( int i = 0 ; i < 7 ; ++i )
  {
    n0[i] = smem_img[threadIdx.y+ 0][threadIdx.x + i+1]; // +1 because of alignment constraint when loading pixels.
    n1[i] = smem_img[threadIdx.y+ 8][threadIdx.x + i+1];
    n2[i] = smem_img[threadIdx.y+16][threadIdx.x + i+1];
    n3[i] = smem_img[threadIdx.y+24][threadIdx.x + i+1];
  }
  float p[4] = {0.0f};
  for( int i = 0 ; i < 7 ; ++i )
  {
    p[0] += gaussian_filter_fp32[i] * n0[i];
    p[1] += gaussian_filter_fp32[i] * n1[i];
    p[2] += gaussian_filter_fp32[i] * n2[i];
    p[3] += gaussian_filter_fp32[i] * n3[i];
  }
  __syncthreads();

  // Write the result back to shared memory.
  smem_img[threadIdx.y+ 0][threadIdx.x] = p[0];
  smem_img[threadIdx.y+ 8][threadIdx.x] = p[1];
  smem_img[threadIdx.y+16][threadIdx.x] = p[2];
  smem_img[threadIdx.y+24][threadIdx.x] = p[3];

  // Make sure the results are in SMEM.
  __syncthreads();

  // Compute the vertical convolution.
  float n[8];
  for( int i = 0 ; i < 8 ; ++i )
    n[i] = smem_img[2*threadIdx.y+i][threadIdx.x];
  float q0 = 0.0f, q1 = 0.0f;
  for( int i = 0 ; i < 7 ; ++i )
  {
    q0 += gaussian_filter_fp32[i] * n[i+0];
    q1 += gaussian_filter_fp32[i] * n[i+1];
  }

  // Where to write the result 2*(blockIdx.x*blockDim.x + threadIdx.y).
  const int write_y = y + threadIdx.y;

  // Write the pixels.
  if( in_img(x, write_y, w, h) )
    dst[write_y*w + x] = (uchar) ((int) q0 >> 8);
  if( in_img(x, write_y+1, w, h) )
    dst[(write_y+1)*w + x] = (uchar) ((int) q1 >> 8);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void sobel_filter_3x3_v0(int w, int h, const uchar *src, uchar *dst)
{
  // Position of the thread in the image.
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // Early exit if the thread is not in the image.
  if( !in_img(x, y, w, h) )
    return;

  // Load the 8 neighbours and myself.
  int n[3][3];
  for( int j = -1 ; j <= 1 ; ++j )
    for( int i = -1 ; i <= 1 ; ++i )
      n[j+1][i+1] = in_img(x+i, y+j, w, h) ? (int) src[(y+j)*w + (x+i)] : 0;

  // Compute the convolution.
  int gx = 0, gy = 0;
  for( int j = 0 ; j < 3 ; ++j )
    for( int i = 0 ; i < 3 ; ++i )
    {
      gx += sobel_filter_x[j][i] * n[j][i];
      gy += sobel_filter_y[j][i] * n[j][i];
    }

  // The gradient.
  float grad = sqrtf((float) (gx*gx + gy*gy));

  // Store the result.
  dst[y*w + x] = (uchar) (grad >= 255.0f ? 255 : grad);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void sobel_filter_3x3_v1(int w, int h, const uchar *src, uchar *dst)
{
  // Position of the thread in the image.
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  // Early exit if the thread is not in the image.
  if( !in_img(x, y, w, h) )
    return;

  // Load the 8 neighbours and myself.
  float n[3][3];
  for( int j = -1 ; j <= 1 ; ++j )
    for( int i = -1 ; i <= 1 ; ++i )
      n[j+1][i+1] = in_img(x+i, y+j, w, h) ? (float) src[(y+j)*w + (x+i)] : 0.0f;

  // Compute the convolution.
  float gx = 0.0f, gy = 0.0f;
  for( int j = 0 ; j < 3 ; ++j )
    for( int i = 0 ; i < 3 ; ++i )
    {
      gx += __fmul_rd(sobel_filter_x[j][i], n[j][i]);
      gy += __fmul_rd(sobel_filter_y[j][i], n[j][i]);
    }

  // The gradient.
  float grad = __fsqrt_rd(__fadd_rd(__fmul_rd(gx,gx), __fmul_rd(gy,gy)));

  // Store the result.
  dst[y*w + x] = (uchar) (grad >= 255.0f ? 255 : grad);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void cuda_gaussian_filter(uchar *dst)
{
  //declare some timers for benchmarking the individual steps
  cudaEvent_t totalStart;
  cudaEvent_t totalEnd;
  cudaEvent_t rtogStart;
  cudaEvent_t rtogEnd;
  cudaEvent_t gaussStart;
  cudaEvent_t gaussEnd;
  cudaEvent_t sobelStart;
  cudaEvent_t sobelEnd;

  //initialize timers
  CHECK_CUDA(cudaEventCreate(&totalStart));
  CHECK_CUDA(cudaEventCreate(&totalEnd));
  CHECK_CUDA(cudaEventCreate(&rtogStart));
  CHECK_CUDA(cudaEventCreate(&rtogEnd));
  CHECK_CUDA(cudaEventCreate(&gaussStart));
  CHECK_CUDA(cudaEventCreate(&gaussEnd));
  CHECK_CUDA(cudaEventCreate(&sobelStart));
  CHECK_CUDA(cudaEventCreate(&sobelEnd));

// The size of the CUDA block/grid.
#  if OPTIMIZATION_STEP == 0x00
  #define OPTIMIZATION_DESC "Original version"
  dim3 block_dim(8, 8);
#elif OPTIMIZATION_STEP == 0x1a
  #define OPTIMIZATION_DESC "Block size 32x2 (It. 1, Eclipse Edition)"
  dim3 block_dim(32, 2);
#elif OPTIMIZATION_STEP == 0x1b
  #define OPTIMIZATION_DESC "Block size 8x16 (It 1, Visual Studio Edition)"
  dim3 block_dim(8, 16);
#elif OPTIMIZATION_STEP == 0x20
  #define OPTIMIZATION_DESC "Block size 32.4 (improve occupancy)"
  dim3 block_dim(32, 4);
#elif OPTIMIZATION_STEP == 0x30
  #define OPTIMIZATION_DESC "__launch_bounds__ to reduce registers (improve occupancy, It 2 Visual Studio Edition)"
  dim3 block_dim(32, 4);
#elif OPTIMIZATION_STEP == 0x40
  #define OPTIMIZATION_DESC "Using shared memory"
  dim3 block_dim(32, 4);
#elif OPTIMIZATION_STEP == 0x50
  #define OPTIMIZATION_DESC "Using read-only path (reduce pressure on Load-store unit)"
  dim3 block_dim(32, 4);
#else
  #if OPTIMIZATION_STEP == 0x60
    #define OPTIMIZATION_DESC "Using separable filter (reduce arithmetic intensity)"
  #elif OPTIMIZATION_STEP == 0x70
    #define OPTIMIZATION_DESC "Processing two elements per thread (increase ILP)"
  #elif OPTIMIZATION_STEP == 0x80
    #define OPTIMIZATION_DESC "Improved shared memory accesses (less bank conflicts)"
  #elif OPTIMIZATION_STEP == 0x90
    #define OPTIMIZATION_DESC "Using floats instead of ints (reducing arithmetic pipe pressure)"
  #elif OPTIMIZATION_STEP == 0x91
    #define OPTIMIZATION_DESC "Fastest Gaussian plus optimized rgp_grayscale and sobel filters"
  #else
    #define OPTIMIZATION_DESC "n/a"
  #endif
  dim3 block_dim(32, 8);
#endif
  dim3 grid_dim(round_up(g_data.img_w, block_dim.x), round_up(g_data.img_h, block_dim.y));

// The target.
  uchar *grayscale = g_data.show == SHOW_GRAYSCALE ? dst : g_data.img_grayscale;

#ifdef DO_WARMUP
  //run rgba_to_grayscale once to warm up everything to get better and more stable timings
  printf("Warming up...\n");
  rgba_to_grayscale_kernel_v0<<<grid_dim, block_dim>>>(g_data.img_w, g_data.img_h, g_data.img_rgba, grayscale);
  CHECK_CUDA(cudaGetLastError());
#endif

  printf("Running version %2x: %s\n", OPTIMIZATION_STEP, OPTIMIZATION_DESC);

  //start total time timer
  CHECK_CUDA(cudaEventRecord(totalStart));

  if( g_data.show == SHOW_RGBA )
  {
    CHECK_CUDA(cudaMemcpy(dst, g_data.img_rgba, g_data.img_w*g_data.img_h*sizeof(uchar4), cudaMemcpyDeviceToDevice));
    return;
  }

  // Convert from RGBA to Grayscale.
  dim3 block_dim_rgba(32, 8);
  
  CHECK_CUDA(cudaEventRecord(rtogStart));
#if OPTIMIZATION_STEP == 0x91
  dim3 grid_dim_rgba(round_up(g_data.img_w, block_dim_rgba.x), round_up(g_data.img_h, block_dim_rgba.y)/2);
  rgba_to_grayscale_kernel_v1<<<grid_dim_rgba, block_dim_rgba>>>(g_data.img_w, g_data.img_h, g_data.img_rgba, grayscale);
#else
  dim3 grid_dim_rgba(round_up(g_data.img_w, block_dim_rgba.x), round_up(g_data.img_h, block_dim_rgba.y));
  rgba_to_grayscale_kernel_v0<<<grid_dim_rgba, block_dim_rgba>>>(g_data.img_w, g_data.img_h, g_data.img_rgba, grayscale);
#endif
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(rtogEnd));

  // Exit if we don't need more.
  if( g_data.show == SHOW_GRAYSCALE ) {
    CHECK_CUDA(cudaEventRecord(totalEnd));
    printf("Times:\n");
    printf("-----------------------------------------------\n");
    printf("rgb_to_grayscale kernel           : %4.2f ms\n", getElapsedTimeInMS(rtogStart, rtogEnd));
    printf("gaussian_filter kernel            : n/a (didn't run)\n");
    printf("sobel_filter kernel               : n/a (didn't run)\n");
    printf("Total time in cuda_gaussian_filter: %4.2f ms\n", getElapsedTimeInMS(totalStart, totalEnd));
    printf("\n");
    return;
  }

  // The smoothed grayscale.
  uchar *smoothed_grayscale = grayscale;
  if( g_data.show & SHOW_SMOOTHED_GRAYSCALE )
  {
    smoothed_grayscale = (g_data.show & SHOW_EDGES) ? g_data.img_smoothed_grayscale : dst;
    
    CHECK_CUDA(cudaEventRecord(gaussStart));

#if   OPTIMIZATION_STEP == 0x00
    gaussian_filter_7x7_v0<<<grid_dim, block_dim>>>(g_data.img_w, g_data.img_h, grayscale, smoothed_grayscale);
#elif OPTIMIZATION_STEP == 0x1a
    gaussian_filter_7x7_v0<<<grid_dim, block_dim>>>(g_data.img_w, g_data.img_h, grayscale, smoothed_grayscale);
#elif OPTIMIZATION_STEP == 0x1b
    gaussian_filter_7x7_v0<<<grid_dim, block_dim>>>(g_data.img_w, g_data.img_h, grayscale, smoothed_grayscale);
#elif OPTIMIZATION_STEP == 0x20
    gaussian_filter_7x7_v0<<<grid_dim, block_dim>>>(g_data.img_w, g_data.img_h, grayscale, smoothed_grayscale);
#elif OPTIMIZATION_STEP == 0x30
    gaussian_filter_7x7_v1<<<grid_dim, block_dim>>>(g_data.img_w, g_data.img_h, grayscale, smoothed_grayscale);
#elif OPTIMIZATION_STEP == 0x40
    gaussian_filter_7x7_v2<<<grid_dim, block_dim>>>(g_data.img_w, g_data.img_h, grayscale, smoothed_grayscale);
#elif OPTIMIZATION_STEP == 0x50
    gaussian_filter_7x7_v3<<<grid_dim, block_dim>>>(g_data.img_w, g_data.img_h, grayscale, smoothed_grayscale);
#elif OPTIMIZATION_STEP == 0x5a
    dim3 grid_dim0(grid_dim.x, grid_dim.y/2);
    gaussian_filter_7x7_v3_bis<<<grid_dim0, block_dim>>>(g_data.img_w, g_data.img_h, grayscale, smoothed_grayscale);
#elif OPTIMIZATION_STEP == 0x60
    gaussian_filter_7x7_v4<<<grid_dim, block_dim>>>(g_data.img_w, g_data.img_h, grayscale, smoothed_grayscale);
#elif OPTIMIZATION_STEP == 0x70
    dim3 grid_dim0(grid_dim.x, grid_dim.y/2);
    gaussian_filter_7x7_v5<<<grid_dim0, block_dim>>>(g_data.img_w, g_data.img_h, grayscale, smoothed_grayscale);
#elif OPTIMIZATION_STEP == 0x80
    CHECK_CUDA(cudaFuncSetSharedMemConfig(gaussian_filter_7x7_v5, cudaSharedMemBankSizeEightByte));
    dim3 grid_dim0(grid_dim.x, grid_dim.y/2);
    gaussian_filter_7x7_v5<<<grid_dim0, block_dim>>>(g_data.img_w, g_data.img_h, grayscale, smoothed_grayscale);
#else
    CHECK_CUDA(cudaFuncSetSharedMemConfig(gaussian_filter_7x7_v6, cudaSharedMemBankSizeEightByte));
    dim3 grid_dim0(grid_dim.x, grid_dim.y/2);
    gaussian_filter_7x7_v6<<<grid_dim0, block_dim>>>(g_data.img_w, g_data.img_h, grayscale, smoothed_grayscale);
#endif
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(gaussEnd));
  }

  // Exit if we don't need more.
  if( !(g_data.show & SHOW_EDGES) ) {
    CHECK_CUDA(cudaEventRecord(totalEnd));
    printf("Times:\n");
    printf("-----------------------------------------------\n");
    printf("rgb_to_grayscale kernel           : %4.2f ms\n", getElapsedTimeInMS(rtogStart, rtogEnd));
    if( g_data.show & SHOW_SMOOTHED_GRAYSCALE ) {
      printf("gaussian_filter kernel            : %4.2f ms\n", getElapsedTimeInMS(gaussStart, gaussEnd));
    } else {
      printf("gaussian_filter kernel            : n/a (didn't run)\n");
    }
    printf("sobel_filter kernel               : n/a (didn't run)\n");
    printf("Total time in cuda_gaussian_filter: %4.2f ms\n", getElapsedTimeInMS(totalStart, totalEnd));
    printf("\n");
    return;
  }

  CHECK_CUDA(cudaEventRecord(sobelStart));
#if OPTIMIZATION_STEP == 0x91
  dim3 block_dim_sobel(32, 4);
  dim3 grid_dim_sobel(round_up(g_data.img_w, block_dim_sobel.x), round_up(g_data.img_h, block_dim_sobel.y));
  sobel_filter_3x3_v1<<<grid_dim_sobel, block_dim_sobel>>>(g_data.img_w, g_data.img_h, smoothed_grayscale, dst);
#else
  dim3 grid_dim_sobel(round_up(g_data.img_w, block_dim.x), round_up(g_data.img_h, block_dim.y));
  sobel_filter_3x3_v0<<<grid_dim, block_dim>>>(g_data.img_w, g_data.img_h, smoothed_grayscale, dst);
#endif
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(sobelEnd));

  CHECK_CUDA(cudaEventRecord(totalEnd));
  printf("Times:\n");
  printf("-----------------------------------------------\n");
  printf("rgb_to_grayscale kernel           : %4.2f ms\n", getElapsedTimeInMS(rtogStart, rtogEnd));
  printf("gaussian_filter kernel            : %4.2f ms\n", getElapsedTimeInMS(gaussStart, gaussEnd));
  printf("sobel_filter kernel               : %4.2f ms\n", getElapsedTimeInMS(sobelStart, sobelEnd));
  printf("Total time in cuda_gaussian_filter: %4.2f ms\n", getElapsedTimeInMS(totalStart, totalEnd));
  printf("\n");
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef WITH_OPENGL
static void display()
{
  // Map the pixel buffer object.
  uchar *dst;
  CHECK_CUDA(cudaGraphicsMapResources(1, &g_data.img_cuda_pbo, 0));
  CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((void **) &dst, NULL, g_data.img_cuda_pbo));
  cuda_gaussian_filter(dst);
  CHECK_CUDA(cudaGraphicsUnmapResources(1, &g_data.img_cuda_pbo, 0) );

  // Prepare the texture.
  CHECK_OPENGL(glClear(GL_COLOR_BUFFER_BIT));
  CHECK_OPENGL(glBindTexture(GL_TEXTURE_2D, g_data.img_tex));
  CHECK_OPENGL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_data.img_pbo));
  if( g_data.show == SHOW_RGBA )
    CHECK_OPENGL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, g_data.img_w, g_data.img_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
  else
    CHECK_OPENGL(glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, g_data.img_w, g_data.img_h, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL));
  CHECK_OPENGL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));

  // Set the texture parameters.
  CHECK_OPENGL(glDisable(GL_DEPTH_TEST));
  CHECK_OPENGL(glEnable(GL_TEXTURE_2D));
  CHECK_OPENGL(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
  CHECK_OPENGL(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
  CHECK_OPENGL(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
  CHECK_OPENGL(glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));

  // Show the image.
  glBegin(GL_QUADS);
  glVertex2f  (0.0f, 0.0f);
  glTexCoord2f(0.0f, 0.0f);
  glVertex2f  (0.0f, 1.0f);
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f  (1.0f, 1.0f);
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f  (1.0f, 0.0f);
  glTexCoord2f(0.0f, 1.0f);
  CHECK_OPENGL(glEnd());
  CHECK_OPENGL(glBindTexture(GL_TEXTURE_2D, 0));
  glutSwapBuffers();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
  if( key == 27 || key == 'Q' || key == 'q' )
    exit(1);
  if( key == 'R' || key == 'r' )
    g_data.show = SHOW_RGBA;
  if( key == 'G' || key == 'g' )
   g_data.show = (g_data.show | SHOW_GRAYSCALE) ^ SHOW_SMOOTHED_GRAYSCALE;
  if( key == 'E' || key == 'e' )
    g_data.show = (g_data.show | SHOW_GRAYSCALE) ^ SHOW_EDGES;
  glutPostRedisplay();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void reshape(int x, int y)
{
  CHECK_OPENGL(glViewport(0, 0, x, y));
  CHECK_OPENGL(glMatrixMode(GL_PROJECTION));
  CHECK_OPENGL(glLoadIdentity());
  CHECK_OPENGL(glOrtho(0, 1, 0, 1, 0, 1));
  CHECK_OPENGL(glMatrixMode(GL_MODELVIEW));
  CHECK_OPENGL(glLoadIdentity());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void init_gl(int *argc, char **argv)
{
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  float ratio = (float) g_data.img_h / g_data.img_w;
  glutInitWindowSize(min(g_data.img_w, 1024), min(g_data.img_h, (int) (1024*ratio)));
  glutCreateWindow("CUDA Gaussian Filter");

  glewInit();

  if( !glewIsSupported("GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object GL_ARB_texture_float") )
  {
    fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
    fprintf(stderr, "This sample requires:\n");
    fprintf(stderr, "  OpenGL version 1.5\n");
    fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
    fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
    exit(1);
  }

  // Create the PBO.
  CHECK_OPENGL(glGenBuffers(1, &g_data.img_pbo));
  CHECK_OPENGL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_data.img_pbo));
  CHECK_OPENGL(glBufferData(GL_PIXEL_UNPACK_BUFFER, g_data.img_w*g_data.img_h*sizeof(uchar4), NULL, GL_STREAM_DRAW));
  CHECK_OPENGL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));

  // Register this buffer object with CUDA.
  CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&g_data.img_cuda_pbo, g_data.img_pbo, cudaGraphicsMapFlagsWriteDiscard));

  // Create the OpenGL texture.
  CHECK_OPENGL(glGenTextures(1, &g_data.img_tex));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void cleanup(void)
{
  cudaGraphicsUnregisterResource(g_data.img_cuda_pbo);

  glDeleteBuffers(1, &g_data.img_pbo);
  glDeleteTextures(1, &g_data.img_tex);
  CHECK_CUDA(cudaFree(g_data.img_rgba));
  CHECK_CUDA(cudaFree(g_data.img_grayscale));
  CHECK_CUDA(cudaFree(g_data.img_smoothed_grayscale));
}
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void load_image(const char *filename)
{
  const int HEADER_SIZE = 64;

  // Open the source file.
  FILE *fp = NULL;
  CHECK_WITH_MSG(fp = fopen(filename, "rb"), "Cannot open file");

  // Read header.
  char header[HEADER_SIZE];
  CHECK(fgets(header, HEADER_SIZE, fp));
  
  // Number of channels. Must be 3.
  CHECK(!strncmp(header, "P6", 2));

  // Skip the comments.
  while( fgets(header, HEADER_SIZE, fp) && header[0] == '#' )
    ;

  // Parse the header.
  int width = 0, height = 0;
  CHECK(sscanf(header, "%u %u", &width, &height) == 2);
  printf("Image %s: w=%4d x h=%4d\n", filename, width, height);

  // Ignore the max value.
  CHECK(fgets(header, HEADER_SIZE, fp));

  // Read the pixels.
  int size_in_bytes = 3*width*height*sizeof(uchar);
  uchar *img_rgb = (uchar*) malloc(size_in_bytes);
  CHECK(img_rgb);
  CHECK(fread(img_rgb, sizeof(uchar), 3*width*height, fp) == 3*width*height);

  // Close the file.
  fclose(fp);

  // Create the RGBA image on the host.
  size_in_bytes = width*height*sizeof(uchar4);
  uchar4 *img_rgba = (uchar4*) malloc(size_in_bytes);
  CHECK(img_rgba);
  for( int i = 0 ; i < width*height ; ++i )
    img_rgba[i] = make_uchar4(img_rgb[3*i+0], img_rgb[3*i+1], img_rgb[3*i+2], 0);
  free(img_rgb);

  // Setup the global data.
  g_data.img_w = width;
  g_data.img_h = height;
  
  // Allocate CUDA memory.
  CHECK_CUDA(cudaMalloc((void**) &g_data.img_rgba, size_in_bytes));
  CHECK_CUDA(cudaMemcpy(g_data.img_rgba, img_rgba, size_in_bytes, cudaMemcpyHostToDevice));
  free(img_rgba);

  // Allocate other temp buffers.
  size_in_bytes = width*height*sizeof(uchar);
  CHECK_CUDA(cudaMalloc((void**) &g_data.img_grayscale, size_in_bytes));
  CHECK_CUDA(cudaMalloc((void**) &g_data.img_smoothed_grayscale, size_in_bytes));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  // Parse the input argument if any.
#ifdef WITH_OPENGL
  bool use_opengl = true;
  if( argc == 3 && !strcmp(argv[1], "-no-opengl") )
    use_opengl = false;
  else
#endif // WITH_OPENGL
  if( argc < 2 || argc > 4 )
  {
#ifdef WITH_OPENGL
    printf("Usage: %s [-no-opengl] [device-num] FILENAME.ppm\n", argv[0]);
#else
    printf("Usage: %s [device-num] FILENAME.ppm\n", argv[0]);
#endif
    exit(1);
  }

  //we abuse the fact that atoi returns 0 on a non-integer string
  CHECK_CUDA(cudaSetDevice(atoi(argv[argc-2])));
  int deviceNum;
  CHECK_CUDA(cudaGetDevice(&deviceNum));
  struct cudaDeviceProp deviceProps;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProps, deviceNum));
  printf("Using device #%d: %s (cc %d.%d)\n", deviceNum, deviceProps.name, deviceProps.major, deviceProps.minor);

  // Clear the global data.
  memset(&g_data, 0, sizeof(g_data));

  // Read the input image.
  load_image(argv[argc-1]);

  // Initialize OpenGL if needed.
#ifdef WITH_OPENGL
  if( use_opengl )
  {
    g_data.show = SHOW_RGBA;

    init_gl(&argc, argv);
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);

    // If code is not printing the USage, then we execute this path.
    printf("R: RGBA image\n");
    printf("G: Toggle Gaussian filter\n");
    printf("E: Toggle Sobel filter\n");
    printf("Q: Quit\n");
    fflush(stdout);
    atexit(cleanup);
    glutMainLoop();
  }
  else
#endif // WITH_OPENGL
  {
    g_data.show = SHOW_GRAYSCALE | SHOW_SMOOTHED_GRAYSCALE | SHOW_EDGES;

    uchar *dst = NULL;
    CHECK_CUDA(cudaMalloc((void**) &dst, g_data.img_w*g_data.img_h*sizeof(uchar)));
    cuda_gaussian_filter(dst);
    CHECK_CUDA(cudaFree(dst));
  }

  // Free CUDA resources.
  CHECK_CUDA(cudaFree(g_data.img_rgba));
  CHECK_CUDA(cudaFree(g_data.img_grayscale));
  CHECK_CUDA(cudaFree(g_data.img_smoothed_grayscale));

  CHECK_CUDA(cudaDeviceReset());
  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

