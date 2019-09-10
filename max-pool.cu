#include <time.h>
#include <stdio.h>
#include <cudnn.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>

#define debug(fmt...) if(getenv("DEBUG") && !strcmp(getenv("DEBUG"), "1")) { printf(fmt); fflush(stdout); } else {}

#define H 1024
#define W 1024
#define C 3

#define FH 21
#define FW 21

#define TW 32
#define TH 32

#define HS 1
#define WS 1

// Max 1024 Threads per Block
#define BH 32
#define BW 32

#define DIV_RUP(x, y)	((x + y - 1) / y)

#define indexToOffset(x, y, channel, heightOffset, widthOffset) ((channel * H * W) + (heightOffset + y) * W + widthOffset + x)

#define pixel_x(blockWidth, blockWidthOffset, x) ((blockWidth * blockWidthOffset) + x)
#define pixel_y(blockHeight, blockHeightOffset, y) ((blockHeight * blockHeightOffset) + y)

#define shmem_offset(x_offset, y_offset, x, y, pTW, pw, ph) (((y_offset + y + ph) * pTW + (x_offset + x + pw)))

#define CUDA_CALL(x) do {						\
  cudaError_t ____rc = (x);					\
  assert(____rc == cudaSuccess);		\
} while (0)

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

static double TimeSpecToSeconds(struct timespec* ts){
  return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0;
}

void fillImage(double* image, int c, int h, int w) {
  for (int i = 0; i < c; i++) {
    for (int j = 0; j < h; j++) {
      for (int k = 0; k < w; k++) {
        image[i * h * w + j * w + k] = i * (j + k);
      }
    }
  }
}

double calculateChecksum(double* image, int c, int h, int w) {
  double checksum = 0.0;
  for (int i = 0; i < c; i++) {
    for (int j = 0; j < h; j++) {
      for (int k = 0; k < w; k++) {
        checksum += image[i * h * w + j * w + k];
      }
    }
  }

  return checksum;
}

__global__ void cudaMaxPoolSimple(double* gOutImage, double* gImage, int c, int h, int w, int fw, int fh) {
  // MaxPool without shared memory

  // Block dimensions
  int blockWidth = blockDim.x;
  int blockHeight = blockDim.y;
  int blockWidthOffset = blockIdx.y;
  int blockHeightOffset = blockIdx.z;

  // Pixel of image
  int widthOffset = pixel_x(blockWidth, blockWidthOffset, threadIdx.x); // 0 - 1023
  int heightOffset = pixel_y(blockHeight, blockHeightOffset, threadIdx.y); // 0 - 1023
  int channel = blockIdx.x; // 0 - 2

  if (widthOffset < 0 || heightOffset < 0 || widthOffset >= w || heightOffset >= h) {
    return;
  }

  double maxValue = gImage[indexToOffset(0, 0, channel, heightOffset, widthOffset)];
  for (int x = -fw/2; x <= fw/2; x++) {
    for (int y = -fh/2; y <= fh/2; y++) {
      double value = 0.0;
      if ((widthOffset + x) >= 0 && (widthOffset + x) < w && (heightOffset + y) >= 0 && (heightOffset + y) < h) {
        value = gImage[indexToOffset(x, y, channel, heightOffset, widthOffset)];
      }
      if (value > maxValue) {
        maxValue = value;
      }
    }
  }

  gOutImage[indexToOffset(0, 0, channel, heightOffset, widthOffset)] = maxValue;
}

// dim3 blockDim(TW, TH);
// dim3 gridDim(DIV_RUP(w, TW), DIV_RUP(h, TH), c);
__global__ void cudaMaxPool(double* gOutImage, double* gImage, int c, int h, int w, int fw, int fh) {
  // Tile size
  int tw = blockDim.x;
  int th = blockDim.y;

  // Padded tile size
  int pTW = tw + fw - 1;
  int pTH = th + fh - 1;


  extern __shared__ double shmem[];

  // Tile offsets in image. Without Padding
  int tileWidthOffset = tw * blockIdx.x;
  int tileHeightOffset = th * blockIdx.y;
  int channel = blockIdx.z;

  for(int x = threadIdx.x; x < pTW; x += tw) {
    int copy_x = x - fw/2 + tileWidthOffset;
    for(int y = threadIdx.y; y < pTH; y += tw) {
      int copy_y = y - fh/2 + tileHeightOffset;

      int shmem_idx = shmem_offset(0, 0, x, y, pTW, 0, 0);

      if (copy_x < 0 || copy_x >= w || copy_y < 0 || copy_y >= h) {
        shmem[shmem_idx] = 0;
      } else {
        shmem[shmem_idx] = gImage[indexToOffset(copy_x, copy_y, channel, 0, 0)];
      }
    }
  }

  __syncthreads();

  // Pixel this thread is responsible for
  int widthOffset = tileWidthOffset + threadIdx.x;
  int heightOffset = tileHeightOffset + threadIdx.y;

  if (widthOffset < 0 || widthOffset >= w || heightOffset < 0 || heightOffset >= h) {
    return;
  }

  double maxValue = shmem[shmem_offset(threadIdx.x, threadIdx.y, 0, 0, pTW, fw/2, fh/2)];
  for (int x = -fw/2; x <= fw/2; x++) {
    for (int y = -fh/2; y <= fh/2; y++) {
      double value = shmem[shmem_offset(x, y, threadIdx.x, threadIdx.y, pTW, fw/2, fh/2)];
      if (value > maxValue) {
        maxValue = value;
      }
    }
  }

  gOutImage[indexToOffset(0, 0, channel, heightOffset, widthOffset)] = maxValue;
}

void cudaMaxPooling(int c, int h, int w, int fw, int fh) {
  long int imageSize = sizeof(double) * c * w * h;
  
  double* cImage = (double*) malloc(imageSize);
  double* gImage;

  // TODO: Change as per stride.
  long int outImageSize = sizeof(double) * c * w * h;//DIV_RUP(W, FW) * DIV_RUP(H, fh);
  
  double* cOutImage = (double*) malloc(outImageSize);
  double* gOutImage;
  
  struct timespec start, end;

  CUDA_CALL(cudaMalloc((void**) &gImage, imageSize));
  CUDA_CALL(cudaMalloc((void**) &gOutImage, outImageSize));

  CUDA_CALL(cudaMemset((void*) gOutImage, 0, outImageSize));

  fillImage(cImage, c, h, w);

  printf("I = checksum: %lf\n", calculateChecksum(cImage, c, h, w));


  if(clock_gettime(CLOCK_MONOTONIC, &start))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  CUDA_CALL(cudaMemcpy (gImage, cImage, imageSize, cudaMemcpyHostToDevice));
  if(clock_gettime(CLOCK_MONOTONIC, &end))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  printf("Copy host->dev %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));


  // dim3 simpleGrid(C, DIV_RUP(H, BH), DIV_RUP(W, BW));
  // dim3 simpleBlock(BH, BW);

  // if(clock_gettime(CLOCK_MONOTONIC, &start))
  // { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  // cudaMaxPoolSimple<<<simpleGrid, simpleBlock>>>(gOutImage, gImage, c, h, w, fw, fh);
  // if(clock_gettime(CLOCK_MONOTONIC, &end))
  // { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  // CUDA_CALL(cudaGetLastError());
  // printf("Time cuda code %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));

  int shmem_size = sizeof(double) * (TW + FW - 1) * (TH + FH - 1);
  dim3 blockDim(TW, TH);
  dim3 gridDim(DIV_RUP(w, TW), DIV_RUP(h, TH), c);

  if(clock_gettime(CLOCK_MONOTONIC, &start))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  cudaMaxPool<<<gridDim, blockDim, shmem_size>>>(gOutImage, gImage, c, h, w, fw, fh);
  if(clock_gettime(CLOCK_MONOTONIC, &end))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  CUDA_CALL(cudaGetLastError());
  printf("Time cuda code %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));



  if(clock_gettime(CLOCK_MONOTONIC, &start))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  CUDA_CALL(cudaMemcpy(cOutImage, gOutImage, outImageSize, cudaMemcpyDeviceToHost));
  if(clock_gettime(CLOCK_MONOTONIC, &end))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  printf("Copy dev->host %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));



  printf("CUDA O = checksum: %lf\n", calculateChecksum(cOutImage, c, h, w));

  free(cImage);
  free(cOutImage);
  CUDA_CALL(cudaFree(gImage));
  CUDA_CALL(cudaFree(gOutImage));
}

void cudaMaxPoolingSimple(int c, int h, int w, int fw, int fh) {
  long int imageSize = sizeof(double) * c * w * h;
  
  double* cImage = (double*) malloc(imageSize);
  double* gImage;

  // TODO: Change as per stride.
  long int outImageSize = sizeof(double) * c * w * h;//DIV_RUP(W, FW) * DIV_RUP(H, fh);
  
  double* cOutImage = (double*) malloc(outImageSize);
  double* gOutImage;
  
  struct timespec start, end;

  CUDA_CALL(cudaMalloc((void**) &gImage, imageSize));
  CUDA_CALL(cudaMalloc((void**) &gOutImage, outImageSize));

  CUDA_CALL(cudaMemset((void*) gOutImage, 0, outImageSize));

  fillImage(cImage, c, h, w);

  printf("I = checksum: %lf\n", calculateChecksum(cImage, c, h, w));


  if(clock_gettime(CLOCK_MONOTONIC, &start))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  CUDA_CALL(cudaMemcpy (gImage, cImage, imageSize, cudaMemcpyHostToDevice));
  if(clock_gettime(CLOCK_MONOTONIC, &end))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  printf("Copy host->dev %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));


  dim3 simpleGrid(C, DIV_RUP(H, BH), DIV_RUP(W, BW));
  dim3 simpleBlock(BH, BW);

  if(clock_gettime(CLOCK_MONOTONIC, &start))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  cudaMaxPoolSimple<<<simpleGrid, simpleBlock>>>(gOutImage, gImage, c, h, w, fw, fh);
  if(clock_gettime(CLOCK_MONOTONIC, &end))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  CUDA_CALL(cudaGetLastError());
  printf("Time cuda code %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));

  // int shmem_size = sizeof(double) * (TW + FW - 1) * (TH + FH - 1);
  // dim3 blockDim(TW, TH);
  // dim3 gridDim(DIV_RUP(w, TW), DIV_RUP(h, TH), c);

  // if(clock_gettime(CLOCK_MONOTONIC, &start))
  // { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  // cudaMaxPool<<<gridDim, blockDim, shmem_size>>>(gOutImage, gImage, c, h, w, fw, fh);
  // if(clock_gettime(CLOCK_MONOTONIC, &end))
  // { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  // CUDA_CALL(cudaGetLastError());
  // printf("Time cuda code %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));



  if(clock_gettime(CLOCK_MONOTONIC, &start))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  CUDA_CALL(cudaMemcpy(cOutImage, gOutImage, outImageSize, cudaMemcpyDeviceToHost));
  if(clock_gettime(CLOCK_MONOTONIC, &end))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  printf("Copy dev->host %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));



  printf("CUDA O = checksum: %lf\n", calculateChecksum(cOutImage, c, h, w));

  free(cImage);
  free(cOutImage);
  CUDA_CALL(cudaFree(gImage));
  CUDA_CALL(cudaFree(gOutImage));
}

void cudnnMaxPooling(int c, int h, int w, int fw, int fh) {
  long int imageSize = sizeof(double) * c * w * h;
  
  double* cImage = (double*) malloc(imageSize);
  double* gImage;

  // TODO: Change as per stride.
  long int outImageSize = sizeof(double) * c * w * h;//DIV_RUP(w, fw) * DIV_RUP(h, fh);
  
  double* cOutImage = (double*) malloc(outImageSize);
  double* gOutImage;
  
  struct timespec start, end;

  CUDA_CALL(cudaMalloc((void**) &gImage, imageSize));
  CUDA_CALL(cudaMalloc((void**) &gOutImage, outImageSize));

  CUDA_CALL(cudaMemset(gOutImage, 0, outImageSize));

  fillImage(cImage, c, h, w);

  printf("I = checksum: %lf\n", calculateChecksum(cImage, c, h, w));


  if(clock_gettime(CLOCK_MONOTONIC, &start))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  CUDA_CALL(cudaMemcpy (gImage, cImage, imageSize, cudaMemcpyHostToDevice));
  if(clock_gettime(CLOCK_MONOTONIC, &end))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  printf("Copy host->dev %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));



  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));

  cudnnPoolingDescriptor_t pooling_desc;
  //create descriptor handle
  checkCUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
  //initialize descriptor
  checkCUDNN(cudnnSetPooling2dDescriptor(pooling_desc,            //descriptor handle
                                         CUDNN_POOLING_MAX,       //mode - max pooling
                                         CUDNN_NOT_PROPAGATE_NAN, //NaN propagation mode
                                         fh,                      //window height
                                         fw,                      //window width
                                         fh/2,                    //vertical padding
                                         fw/2,                    //horizontal padding
                                         1,                       //vertical stride
                                         1));                     //horizontal stride
  
  cudnnTensorDescriptor_t in_desc;
  //create input data tensor descriptor
  checkCUDNN(cudnnCreateTensorDescriptor(&in_desc));
  //initialize input data descriptor 
  checkCUDNN(cudnnSetTensor4dDescriptor(in_desc,                  //descriptor handle
                                        CUDNN_TENSOR_NCHW,        //data format
                                        CUDNN_DATA_DOUBLE,        //data type (precision)
                                        1,                        //number of images
                                        c,                        //number of channels
                                        h,                        //data height 
                                        w));                      //data width

  cudnnTensorDescriptor_t out_desc;
  //create output data tensor descriptor
  checkCUDNN(cudnnCreateTensorDescriptor(&out_desc));
  //initialize output data descriptor
  checkCUDNN(cudnnSetTensor4dDescriptor(out_desc,                 //descriptor handle
                                        CUDNN_TENSOR_NCHW,        //data format
                                        CUDNN_DATA_DOUBLE,        //data type (precision)
                                        1,                        //number of images
                                        c,                        //number of channels
                                        h,                        //data height
                                        w));                      //data width

  // Scaling factor
  double alpha = 1.0;
  double beta = 0.0;

  if(clock_gettime(CLOCK_MONOTONIC, &start))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  //Call pooling operator
  checkCUDNN(cudnnPoolingForward(cudnn,         //cuDNN context handle
    pooling_desc,  //pooling descriptor handle
    &alpha,        //alpha scaling factor
    in_desc,       //input tensor descriptor
    gImage,        //input data pointer to GPU memory
    &beta,         //beta scaling factor
    out_desc,      //output tensor descriptor
    gOutImage));   //output data pointer from GPU memory
  if(clock_gettime(CLOCK_MONOTONIC, &end))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  printf("Time cudnn code %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));

  
  
  if(clock_gettime(CLOCK_MONOTONIC, &start))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  CUDA_CALL(cudaMemcpy(cOutImage, gOutImage, outImageSize, cudaMemcpyDeviceToHost));
  if(clock_gettime(CLOCK_MONOTONIC, &end))
  { printf("CLOCK ERROR. Exiting.\n"); std::exit(EXIT_FAILURE); }
  printf("Copy dev->host %lf sec\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));



  printf("CUDNN O = checksum: %lf\n", calculateChecksum(cOutImage, c, h, w));

  free(cImage);
  free(cOutImage);
  CUDA_CALL(cudaFree(gImage));
  CUDA_CALL(cudaFree(gOutImage));

  cudnnDestroyTensorDescriptor(in_desc);
  cudnnDestroyTensorDescriptor(out_desc);
  cudnnDestroyPoolingDescriptor(pooling_desc);
  cudnnDestroy(cudnn);
}

int main(int ac, char *av[]){
  printf("Simple Max Pool Using CUDA\n");
  cudaMaxPoolingSimple(C, H, W, FW, FH);
  printf("\n");

  printf("Max Pool Using CUDA\n");
  cudaMaxPooling(C, H, W, FW, FH);
  printf("\n");

  printf("Max Pool Using CUDNN\n");
  cudnnMaxPooling(C, H, W, FW, FH);

  return 0;
}
