/*
 * @Author: gzy
 * @Date: 2022-03-22 15:25:38
 * @Description: file content
 */
#include "algorithm.hpp"
#ifdef USE_NEON

void resize_bilinear_c1(const unsigned char* src, int srcw, int srch,
                        int srcstride, unsigned char* dst, int w, int h,
                        int stride) {
  const int INTER_RESIZE_COEF_BITS = 11;
  const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;
  //     const int ONE=INTER_RESIZE_COEF_SCALE;

  double scale_x = (double)srcw / w;
  double scale_y = (double)srch / h;

  int* buf = new int[w + h + w + h];

  int* xofs = buf;      // new int[w];
  int* yofs = buf + w;  // new int[h];

  short* ialpha = (short*)(buf + w + h);     // new short[w * 2];
  short* ibeta = (short*)(buf + w + h + w);  // new short[h * 2];

  float fx;
  float fy;
  int sx;
  int sy;

#define SATURATE_CAST_SHORT(X) \
  (short)::std::min(           \
      ::std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), SHRT_MAX);

  for (int dx = 0; dx < w; dx++) {
    fx = (float)((dx + 0.5) * scale_x - 0.5);
    sx = static_cast<int>(floor(fx));
    fx -= sx;

    if (sx < 0) {
      sx = 0;
      fx = 0.f;
    }
    if (sx >= srcw - 1) {
      sx = srcw - 2;
      fx = 1.f;
    }

    xofs[dx] = sx;

    float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
    float a1 = fx * INTER_RESIZE_COEF_SCALE;

    ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
    ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
  }

  for (int dy = 0; dy < h; dy++) {
    fy = (float)((dy + 0.5) * scale_y - 0.5);
    sy = static_cast<int>(floor(fy));
    fy -= sy;

    if (sy < 0) {
      sy = 0;
      fy = 0.f;
    }
    if (sy >= srch - 1) {
      sy = srch - 2;
      fy = 1.f;
    }

    yofs[dy] = sy;

    float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
    float b1 = fy * INTER_RESIZE_COEF_SCALE;

    ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
    ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
  }

#undef SATURATE_CAST_SHORT

  // loop body
  short* rows0 = (short*)malloc(w * sizeof(short));
  short* rows1 = (short*)malloc(w * sizeof(short));

  int prev_sy1 = -2;

  for (int dy = 0; dy < h; dy++) {
    sy = yofs[dy];

    if (sy == prev_sy1) {
      // reuse all rows
    } else if (sy == prev_sy1 + 1) {
      // hresize one row
      short* rows0_old = rows0;
      rows0 = rows1;
      rows1 = rows0_old;
      const unsigned char* S1 = src + srcstride * (sy + 1);

      const short* ialphap = ialpha;
      short* rows1p = rows1;
      for (int dx = 0; dx < w; dx++) {
        sx = xofs[dx];
        short a0 = ialphap[0];
        short a1 = ialphap[1];

        const unsigned char* S1p = S1 + sx;
        rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

        ialphap += 2;
      }
    } else {
      // hresize two rows
      const unsigned char* S0 = src + srcstride * (sy);
      const unsigned char* S1 = src + srcstride * (sy + 1);

      const short* ialphap = ialpha;
      short* rows0p = rows0;
      short* rows1p = rows1;
      for (int dx = 0; dx < w; dx++) {
        sx = xofs[dx];
        short a0 = ialphap[0];
        short a1 = ialphap[1];

        const unsigned char* S0p = S0 + sx;
        const unsigned char* S1p = S1 + sx;
        rows0p[dx] = (S0p[0] * a0 + S0p[1] * a1) >> 4;
        rows1p[dx] = (S1p[0] * a0 + S1p[1] * a1) >> 4;

        ialphap += 2;
      }
    }

    prev_sy1 = sy;

    // vresize
    short b0 = ibeta[0];
    short b1 = ibeta[1];

    short* rows0p = rows0;
    short* rows1p = rows1;
    unsigned char* Dp = dst + stride * (dy);

    int nn = w >> 3;
    int remain = w - (nn << 3);

    int16x4_t _b0 = vdup_n_s16(b0);
    int16x4_t _b1 = vdup_n_s16(b1);
    int32x4_t _v2 = vdupq_n_s32(2);
    for (; nn > 0; nn--) {
      int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
      int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
      int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p + 4);
      int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p + 4);

      int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
      int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
      int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
      int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

      int32x4_t _acc = _v2;
      _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
      _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

      int32x4_t _acc_1 = _v2;
      _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
      _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

      int16x4_t _acc16 = vshrn_n_s32(_acc, 2);
      int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

      uint8x8_t _D = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

      vst1_u8(Dp, _D);

      Dp += 8;
      rows0p += 8;
      rows1p += 8;
    }

    for (; remain; --remain) {
      //             D[x] = (rows0[x]*b0 + rows1[x]*b1) >>
      //             INTER_RESIZE_COEF_BITS;
      *Dp++ = (unsigned char)(((short)((b0 * (short)(*rows0p++)) >> 16) +
                               (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >>
                              2);
    }

    ibeta += 2;
  }
  free(rows0);
  free(rows1);
  delete[] buf;
}
void resize_bilinear_c3(const unsigned char* src, int srcw, int srch,
                        int srcstride, unsigned char* dst, int w, int h,
                        int stride) {
  const int INTER_RESIZE_COEF_BITS = 11;
  const int INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS;

  double scale_x = (double)srcw / w;
  double scale_y = (double)srch / h;

  int* buf = new int[w + h + w + h];

  int* xofs = buf;      // new int[w];
  int* yofs = buf + w;  // new int[h];

  short* ialpha = (short*)(buf + w + h);     // new short[w * 2];
  short* ibeta = (short*)(buf + w + h + w);  // new short[h * 2];

  float fx;
  float fy;
  int sx;
  int sy;

#define SATURATE_CAST_SHORT(X)                                              \
  (short)std::min(std::max((int)(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), \
                  SHRT_MAX);

  for (int dx = 0; dx < w; dx++) {
    fx = (float)((dx + 0.5) * scale_x - 0.5);
    sx = static_cast<int>(std::floor(fx));
    fx -= sx;

    if (sx < 0) {
      sx = 0;
      fx = 0.f;
    }
    if (sx >= srcw - 1) {
      sx = srcw - 2;
      fx = 1.f;
    }

    xofs[dx] = sx * 3;

    float a0 = (1.f - fx) * INTER_RESIZE_COEF_SCALE;
    float a1 = fx * INTER_RESIZE_COEF_SCALE;

    ialpha[dx * 2] = SATURATE_CAST_SHORT(a0);
    ialpha[dx * 2 + 1] = SATURATE_CAST_SHORT(a1);
  }

  for (int dy = 0; dy < h; dy++) {
    fy = (float)((dy + 0.5) * scale_y - 0.5);
    sy = static_cast<int>(std::floor(fy));
    fy -= sy;

    if (sy < 0) {
      sy = 0;
      fy = 0.f;
    }
    if (sy >= srch - 1) {
      sy = srch - 2;
      fy = 1.f;
    }

    yofs[dy] = sy;

    float b0 = (1.f - fy) * INTER_RESIZE_COEF_SCALE;
    float b1 = fy * INTER_RESIZE_COEF_SCALE;

    ibeta[dy * 2] = SATURATE_CAST_SHORT(b0);
    ibeta[dy * 2 + 1] = SATURATE_CAST_SHORT(b1);
  }

#undef SATURATE_CAST_SHORT

  // loop body
  short* rows0 = (short*)malloc((w * 3 + 1) * sizeof(short));
  short* rows1 = (short*)malloc((w * 3 + 1) * sizeof(short));
  int prev_sy1 = -2;

#pragma omp parallel for num_threads(4)
  for (int dy = 0; dy < h; dy++) {
    sy = yofs[dy];

    if (sy == prev_sy1) {
      // reuse all rows
    } else if (sy == prev_sy1 + 1) {
      // hresize one row
      short* rows0_old = rows0;
      rows0 = rows1;
      rows1 = rows0_old;
      const unsigned char* S1 = src + srcstride * (sy + 1);

      const short* ialphap = ialpha;
      short* rows1p = rows1;
      for (int dx = 0; dx < w; dx++) {
        sx = xofs[dx];
        short a0 = ialphap[0];
        short a1 = ialphap[1];

        const unsigned char* S1p = S1 + sx;
        int16x4_t _a0 = vdup_n_s16(a0);
        int16x4_t _a1 = vdup_n_s16(a1);
        uint8x8_t _S1 = uint8x8_t();

        _S1 = vld1_lane_u8(S1p, _S1, 0);
        _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
        _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
        _S1 = vld1_lane_u8(S1p + 3, _S1, 3);
        _S1 = vld1_lane_u8(S1p + 4, _S1, 4);
        _S1 = vld1_lane_u8(S1p + 5, _S1, 5);

        int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
        int16x4_t _S1low = vget_low_s16(_S116);
        int16x4_t _S1high = vext_s16(_S1low, vget_high_s16(_S116), 3);
        int32x4_t _rows1 = vmull_s16(_S1low, _a0);
        _rows1 = vmlal_s16(_rows1, _S1high, _a1);
        int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
        vst1_s16(rows1p, _rows1_sr4);

        ialphap += 2;
        rows1p += 3;
      }
    } else {
      // hresize two rows
      const unsigned char* S0 = src + srcstride * (sy);
      const unsigned char* S1 = src + srcstride * (sy + 1);

      const short* ialphap = ialpha;
      short* rows0p = rows0;
      short* rows1p = rows1;
      for (int dx = 0; dx < w; dx++) {
        sx = xofs[dx];
        short a0 = ialphap[0];
        short a1 = ialphap[1];

        const unsigned char* S0p = S0 + sx;
        const unsigned char* S1p = S1 + sx;

        int16x4_t _a0 = vdup_n_s16(a0);
        int16x4_t _a1 = vdup_n_s16(a1);
        uint8x8_t _S0 = uint8x8_t();
        uint8x8_t _S1 = uint8x8_t();

        _S0 = vld1_lane_u8(S0p, _S0, 0);
        _S0 = vld1_lane_u8(S0p + 1, _S0, 1);
        _S0 = vld1_lane_u8(S0p + 2, _S0, 2);
        _S0 = vld1_lane_u8(S0p + 3, _S0, 3);
        _S0 = vld1_lane_u8(S0p + 4, _S0, 4);
        _S0 = vld1_lane_u8(S0p + 5, _S0, 5);

        _S1 = vld1_lane_u8(S1p, _S1, 0);
        _S1 = vld1_lane_u8(S1p + 1, _S1, 1);
        _S1 = vld1_lane_u8(S1p + 2, _S1, 2);
        _S1 = vld1_lane_u8(S1p + 3, _S1, 3);
        _S1 = vld1_lane_u8(S1p + 4, _S1, 4);
        _S1 = vld1_lane_u8(S1p + 5, _S1, 5);

        int16x8_t _S016 = vreinterpretq_s16_u16(vmovl_u8(_S0));
        int16x8_t _S116 = vreinterpretq_s16_u16(vmovl_u8(_S1));
        int16x4_t _S0low = vget_low_s16(_S016);
        int16x4_t _S1low = vget_low_s16(_S116);
        int16x4_t _S0high = vext_s16(_S0low, vget_high_s16(_S016), 3);
        int16x4_t _S1high = vext_s16(_S1low, vget_high_s16(_S116), 3);
        int32x4_t _rows0 = vmull_s16(_S0low, _a0);
        int32x4_t _rows1 = vmull_s16(_S1low, _a0);
        _rows0 = vmlal_s16(_rows0, _S0high, _a1);
        _rows1 = vmlal_s16(_rows1, _S1high, _a1);
        int16x4_t _rows0_sr4 = vshrn_n_s32(_rows0, 4);
        int16x4_t _rows1_sr4 = vshrn_n_s32(_rows1, 4);
        vst1_s16(rows0p, _rows0_sr4);
        vst1_s16(rows1p, _rows1_sr4);

        ialphap += 2;
        rows0p += 3;
        rows1p += 3;
      }
    }

    prev_sy1 = sy;

    // vresize
    short b0 = ibeta[0];
    short b1 = ibeta[1];

    short* rows0p = rows0;
    short* rows1p = rows1;
    unsigned char* Dp = dst + stride * (dy);

    int nn = (w * 3) >> 3;
    int remain = (w * 3) - (nn << 3);

    int16x4_t _b0 = vdup_n_s16(b0);
    int16x4_t _b1 = vdup_n_s16(b1);
    int32x4_t _v2 = vdupq_n_s32(2);
    for (; nn > 0; nn--) {
      int16x4_t _rows0p_sr4 = vld1_s16(rows0p);
      int16x4_t _rows1p_sr4 = vld1_s16(rows1p);
      int16x4_t _rows0p_1_sr4 = vld1_s16(rows0p + 4);
      int16x4_t _rows1p_1_sr4 = vld1_s16(rows1p + 4);

      int32x4_t _rows0p_sr4_mb0 = vmull_s16(_rows0p_sr4, _b0);
      int32x4_t _rows1p_sr4_mb1 = vmull_s16(_rows1p_sr4, _b1);
      int32x4_t _rows0p_1_sr4_mb0 = vmull_s16(_rows0p_1_sr4, _b0);
      int32x4_t _rows1p_1_sr4_mb1 = vmull_s16(_rows1p_1_sr4, _b1);

      int32x4_t _acc = _v2;
      _acc = vsraq_n_s32(_acc, _rows0p_sr4_mb0, 16);
      _acc = vsraq_n_s32(_acc, _rows1p_sr4_mb1, 16);

      int32x4_t _acc_1 = _v2;
      _acc_1 = vsraq_n_s32(_acc_1, _rows0p_1_sr4_mb0, 16);
      _acc_1 = vsraq_n_s32(_acc_1, _rows1p_1_sr4_mb1, 16);

      int16x4_t _acc16 = vshrn_n_s32(_acc, 2);
      int16x4_t _acc16_1 = vshrn_n_s32(_acc_1, 2);

      uint8x8_t _D = vqmovun_s16(vcombine_s16(_acc16, _acc16_1));

      vst1_u8(Dp, _D);

      Dp += 8;
      rows0p += 8;
      rows1p += 8;
    }

    for (; remain; --remain) {
      // D[x] = (rows0[x]*b0 + rows1[x]*b1) >> INTER_RESIZE_COEF_BITS;
      *Dp++ = (unsigned char)(((short)((b0 * (short)(*rows0p++)) >> 16) +
                               (short)((b1 * (short)(*rows1p++)) >> 16) + 2) >>
                              2);
    }

    ibeta += 2;
  }
  delete[] buf;
  free(rows0);
  free(rows1);
}

// generate gamma table
void generateTable_NeonVec(uint8x8x4_t vTables[], unsigned int size) {
  uint8_t* pTable = (uint8_t*)malloc(size);
  for (unsigned int i = 0; i < size; i++) {
    pTable[i] = (uint8_t)round((pow(i / 255.f, 1 / 2.2) * 255.f));
  }
  unsigned int vTable_nb = size >> 5;
  uint8_t* load_table_ptr = &pTable[0];
  for (unsigned int id = 0; id < vTable_nb; id++) {
    vTables[id].val[0] = vld1_u8(load_table_ptr);
    vTables[id].val[1] = vld1_u8(load_table_ptr + 8);
    vTables[id].val[2] = vld1_u8(load_table_ptr + 16);
    vTables[id].val[3] = vld1_u8(load_table_ptr + 24);
    load_table_ptr += 32;
  }
  free(pTable);
}

// gamma correction use by neon
void GammaCorrection_NeonAlign8(unsigned char* src, unsigned char* dst,
                                unsigned int width, uint8x8x4_t vTables[],
                                unsigned int vTable_nb) {
  uint8x8_t v_gap =
      vcreate_u8(0x2020202020202020);  // |32|32|32|32|32|32|32|32|
  for (unsigned int i = 0; i < width; i += 16) {
    uint8x16_t v_index = vld1q_u8(src + i);
    uint8x8_t v_index_low = vget_low_u8(v_index);
    uint8x8_t v_index_high = vget_high_u8(v_index);
    uint8x8_t v_result_low = vtbl4_u8(vTables[0], v_index_low);
    uint8x8_t v_result_high = vtbl4_u8(vTables[0], v_index_high);
    for (unsigned int id = 1; id < vTable_nb; id++) {
      v_index_low -= v_gap;
      v_index_high -= v_gap;
      v_result_low |= vtbl4_u8(vTables[id], v_index_low);
      v_result_high |= vtbl4_u8(vTables[id], v_index_high);
    }
    vst1q_u8(dst + i, vcombine_u8(v_result_low, v_result_high));
  }
}
#endif