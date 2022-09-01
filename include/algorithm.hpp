/*
 * @Author: gzy
 * @Date: 2022-03-22 15:16:15
 * @Description: revise algorithm use arm_neon
 */

#ifndef ALGORITHM_HPP_
#define ALGORITHM_HPP_

#include <limits.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>

#include <cmath>
#include <iostream>
#ifdef USE_NEON
#include <arm_neon.h>

/**
 * @description: resize three channels image size by arm noen,so fast
 * @param {type} params
 *        src: the src data ptr
 *        srcw: the src width
 *        srch: the src height
 *        srcstride: width * sizeof(elemsize), in this function,
 * srcw*3*sizeof(unsigned char) dst: the dst data ptr w:   the dst width h: the
 * dst height stride:   h* sizeof(elemsize), in this function,
 * w*3*sizeof(unsigned char)
 * @return:
 */
void resize_bilinear_c3(const unsigned char *src, int srcw, int srch,
                        int srcstride, unsigned char *dst, int w, int h,
                        int stride);

/**
 * @description: resize one channels image size by arm noen,so fast
 * @param {type} params
 *        src: the src data ptr
 *        srcw: the src width
 *        srch: the src height
 *        srcstride: width * sizeof(elemsize), in this function,
 * srcw*sizeof(unsigned char) dst: the dst data ptr w:   the dst width h:   the
 * dst height stride:   h* sizeof(elemsize), in this function, w*sizeof(unsigned
 * char)
 * @return:
 */
void resize_bilinear_c1(const unsigned char *src, int srcw, int srch,
                        int srcstride, unsigned char *dst, int w, int h,
                        int stride);

// 进行伽马变换
void generateTable_NeonVec(uint8x8x4_t vTables[], unsigned int size);
void GammaCorrection_NeonAlign8(unsigned char *src, unsigned char *dst,
                                unsigned int width, uint8x8x4_t vTables[],
                                unsigned int vTable_nb);

//
#endif

#endif