// mkavif: 生成 10-bit PQ HDR AVIF 测试文件 (梯度图), 验证 HDR 解码/编码链路
// 用法: mkavif <out.avif> [w] [h]
#include <avif/avif.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
  const char* out = argc > 1 ? argv[1] : "test_hdr.avif";
  const int w = argc > 2 ? atoi(argv[2]) : 320;
  const int h = argc > 3 ? atoi(argv[3]) : 200;
  const int depth = 10;
  avifImage* im = avifImageCreate(w, h, depth, AVIF_PIXEL_FORMAT_YUV444);
  im->colorPrimaries = AVIF_COLOR_PRIMARIES_BT2020;
  im->transferCharacteristics = AVIF_TRANSFER_CHARACTERISTICS_SMPTE2084;
  im->matrixCoefficients = AVIF_MATRIX_COEFFICIENTS_BT2020_NCL;
  avifRGBImage rgb;
  avifRGBImageSetDefaults(&rgb, im);
  rgb.format = AVIF_RGB_FORMAT_RGB;
  if (avifRGBImageAllocatePixels(&rgb) != AVIF_RESULT_OK) {
    avifImageDestroy(im);
    fprintf(stderr, "rgb alloc fail\n");
    return 1;
  }
  uint16_t* p = (uint16_t*)rgb.pixels;
  const float full = (float)((1 << depth) - 1);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      const size_t o = ((size_t)y * w + x) * 3;
      // PQ 码域梯度 + 棋盘细纹理 (供超分看边缘)
      const float v = (x + y) / (float)(w + h);
      const float tex = ((x / 4 + y / 4) % 2) ? 0.12f : 0.f;
      p[o + 0] = (uint16_t)((v * 0.7f + 0.2f) * full);
      p[o + 1] = (uint16_t)((v * 0.5f + 0.1f + tex) * full);
      p[o + 2] = (uint16_t)((v * 0.3f + 0.05f) * full);
    }
  if (avifImageRGBToYUV(im, &rgb) != AVIF_RESULT_OK) { fprintf(stderr, "rgb->yuv fail\n"); return 1; }
  avifRGBImageFreePixels(&rgb);
  avifEncoder* e = avifEncoderCreate();
  e->quality = 90;
  e->speed = 8;
  avifRWData rw = AVIF_DATA_EMPTY;
  const avifResult r1 = avifEncoderAddImage(e, im, 1, 0);
  const avifResult r2 = avifEncoderFinish(e, &rw);
  if (r1 != AVIF_RESULT_OK || r2 != AVIF_RESULT_OK) {
    fprintf(stderr, "encode fail: %d %d\n", r1, r2);
    return 1;
  }
  FILE* f = fopen(out, "wb");
  fwrite(rw.data, 1, rw.size, f);
  fclose(f);
  avifRWDataFree(&rw);
  avifEncoderDestroy(e);
  avifImageDestroy(im);
  printf("wrote %s (%dx%d %dbit PQ, %zu bytes)\n", out, w, h, depth, rw.size);
  return 0;
}
