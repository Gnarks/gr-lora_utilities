/* -*- c++ -*- */
/*
 * Copyright 2024 kazawai.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "lora_detector_impl.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <gnuradio/gr_complex.h>
#include <gnuradio/io_signature.h>
#include <gnuradio/types.h>
#include <iostream>
#include <liquid/liquid.h>
#include <ostream>
#include <pmt/pmt.h>
#include <sys/types.h>
#include <utility>
#include <volk/volk.h>
#include <volk/volk_complex.h>
#include <volk/volk_malloc.h>

namespace gr {
namespace first_lora {

#define DEMOD_HISTORY (8 + 5)

int write_f_to_file(float *f, const char *filename, int n);

using input_type = gr_complex;
using output_type = gr_complex;
lora_detector::sptr lora_detector::make(float threshold, uint8_t sf,
                                        uint32_t bw, uint32_t sampleRate,
                                        int method) {
  return gnuradio::make_block_sptr<lora_detector_impl>(threshold, sf, bw,
                                                       sampleRate, method);
}

/*
 * The private constructor
 */
lora_detector_impl::lora_detector_impl(float threshold, uint8_t sf, uint32_t bw,
                                       uint32_t sampleRate, int method)
    : gr::block("lora_detector",
                gr::io_signature::make(1 /* min inputs */, 1 /* max inputs */,
                                       sizeof(input_type)),
                gr::io_signature::make(1 /* min outputs */, 1 /*max outputs */,
                                       sizeof(output_type))),
      d_threshold(threshold), d_sf(sf), d_bw(bw), d_method(method) {
  assert((d_sf > 5) && (d_sf < 13));

  // Number of symbols
  d_sps = 1 << d_sf;
  std::cout << "number possibles samples: " << d_sps << std::endl;

  d_sn = d_sn = (d_sps * sampleRate) / bw;
  std::cout << "Samples per symbol : " << d_sn << std::endl;

  d_fs = sampleRate;

  std::cout << "Fs " << d_fs << std::endl;
  d_fft_size = 10 * d_sn;
  d_bin_size = d_fft_size / 2;
  std::cout << "FFT size: " << d_fft_size << std::endl;
  std::cout << "Bin size: " << d_bin_size << std::endl;
  d_cfo = 0;
  // FFT input vector
  d_mult_hf_fft = std::vector<gr_complex>(d_fft_size);
  d_fft_result = (lv_32fc_t *)malloc(d_fft_size * sizeof(lv_32fc_t));
  if (d_fft_result == NULL) {
    std::cerr << "Error: Failed to allocate memory for fft_result\n";
    return;
  }

  // Reference downchip signal
  d_ref_downchirp = g_downchirp(d_sf, d_bw, d_fs);
  // Reference upchip signal
  d_ref_upchirp = g_upchirp(d_sf, d_bw, d_fs);

  d_dechirped.reserve(d_sn);

  d_state = 0;

  message_port_register_out(pmt::mp("detected"));

  set_history(DEMOD_HISTORY * d_sn);

  // Set output buffer size to 8 + 5 * d_sn
  set_output_multiple(DEMOD_HISTORY * d_sn);
}

/*
 * Our virtual destructor.
 */
lora_detector_impl::~lora_detector_impl() {
  // Free memory
  buffer.clear();
  free(d_fft_result);

  // Print the number of detected LoRa symbols
  std::cout << "Detected LoRa symbols: " << detected_count << std::endl;
  detected_count = 0;
}

void lora_detector_impl::forecast(int noutput_items,
                                  gr_vector_int &ninput_items_required) {
  /* <+forecast+> e.g. ninput_items_required[0] = noutput_items */
  ninput_items_required[0] = noutput_items;
}

uint32_t lora_detector_impl::argmax_32f(const float *x, float *max,
                                        uint16_t n) {
  float mag = abs(x[0]);
  float m = mag;
  uint32_t index = 0;

  for (int i = 0; i < n; i++) {
    mag = abs(x[i]);
    if (mag > m) {
      m = mag;
      index = i;
    }
  }
  *max = m;
  return index;
}

uint32_t lora_detector_impl::get_fft_peak_abs(const lv_32fc_t *fft_r, float *b1,
                                              float *b2, float *max) {
  uint32_t peak = 0;
  *max = 0;
  // Compute the magnitude of the FFT in b1
  volk_32fc_magnitude_32f(b1, fft_r, d_fft_size);
  // Add the last part of the FFT to the first part.
  // This is the CPA proposed in the paper to determine the phase misalignment
  volk_32f_x2_add_32f(b2, b1, &b1[d_fft_size - d_bin_size], d_bin_size);
  peak = argmax_32f(b2, max, d_bin_size);
  return peak;
}

std::pair<float, uint32_t> lora_detector_impl::dechirp(const gr_complex *in,
                                                       bool is_up) {
  gr_complex *blocks = (gr_complex *)volk_malloc(d_sn * sizeof(gr_complex),
                                                 volk_get_alignment());
  if (blocks == NULL) {
    std::cerr << "Error: Failed to allocate memory for up_blocks\n";
    return std::make_pair(0, 0);
  }

  // Dechirp https://dl.acm.org/doi/10.1145/3546869#d1e1181
  volk_32fc_x2_multiply_32fc(
      blocks, in, is_up ? &d_ref_downchirp[0] : &d_ref_upchirp[0], d_sn);

  // Copy dechirped signal to d_mult_hf_fft (zero padding)
  memset(&d_mult_hf_fft[0], 0, d_fft_size * sizeof(gr_complex));
  memcpy(&d_mult_hf_fft[0], blocks, d_sn * sizeof(gr_complex));

  fft = fft_create_plan(d_fft_size, &d_mult_hf_fft[0], d_fft_result,
                        LIQUID_FFT_FORWARD, 0);

  // FFT
  fft_execute(fft);

  // Destroy FFT plan
  fft_destroy_plan(fft);

  // Free memory
  volk_free(blocks);

  // Get peak of FFT
  float *b1 =
      (float *)volk_malloc(d_fft_size * sizeof(float), volk_get_alignment());
  float *b2 =
      (float *)volk_malloc(d_bin_size * sizeof(float), volk_get_alignment());

  if (b1 == NULL || b2 == NULL) {
    std::cerr << "Error: Failed to allocate memory for b1 or b2\n";
    return std::make_pair(0, 0);
  }
  float max;
  uint32_t peak = get_fft_peak_abs(d_fft_result, b1, b2, &max);

  // Free memory
  // free(fft_r);
  volk_free(b1);
  volk_free(b2);

  return std::make_pair(max, peak);
}

int lora_detector_impl::sliding_detect_preamble(const gr_complex *in,
                                                gr_complex *out) {
  int num_consumed = d_sn;
  if (buffer.size() < MIN_PREAMBLE_CHIRPS) {
    return num_consumed;
  }
  d_state = 2;

  // Move preamble peak to bin zero
  //          num_consumed = d_num_samples -
  //          d_p*d_preamble_idx/d_fft_size_factor;
  num_consumed = d_sn - buffer[0] / 5;

  int symbol = buffer[0] / 5;
  int offset = 1;

  while (symbol >= 10) {
    auto [up_val, up_idx] = dechirp(&in[offset], true);

    symbol = up_idx / 5;
    offset += 5;
  }
  num_consumed = offset;

  return num_consumed;
}

int lora_detector_impl::detect_preamble(const gr_complex *in, gr_complex *out) {
  int num_consumed = d_sn;
  if (buffer.size() < MIN_PREAMBLE_CHIRPS) {
    return num_consumed;
  }
  d_state = 2;

  // Move preamble peak to bin zero
  //          num_consumed = d_num_samples -
  //          d_p*d_preamble_idx/d_fft_size_factor;
  //  magic formula (don't question the *4%1024)
  num_consumed = ((d_sn - buffer[0] / 5) * 4) % 1024;

  return num_consumed;
}

int lora_detector_impl::detect_sfd(const gr_complex *in, gr_complex *out,
                                   const gr_complex *in0) {
  int num_consumed = d_sn;

  if (d_sfd_recovery++ > 5) {
    d_state = 0;
    return 0;
  }

  auto [down_val, down_idx] = dechirp(in, false);
  // If absolute value of down_val is greater then we are in the sfd
  // std::cout << "index is ";
  // std::cout << down_idx / 5 << std::endl;
  if (down_idx / 5 <= 10) {
    d_state = 3;
    num_consumed = 3 * d_sn;

    // std::cout << "final index is ";
    // std::cout << down_idx / 5 << std::endl;
  }

  return num_consumed;
}

float realmod(float x, float y) {
  float result = fmod(x, y);
  return result >= 0 ? result : result + y;
}

int lora_detector_impl::general_work(int noutput_items,
                                     gr_vector_int &ninput_items,
                                     gr_vector_const_void_star &input_items,
                                     gr_vector_void_star &output_items) {
  if (ninput_items[0] < (int)(DEMOD_HISTORY * d_sn)) {
    return 0; // Not enough input
  }

  auto in0 = static_cast<const input_type *>(input_items[0]);

  // changed from 1 to 0.25
  // kept 1
  auto in = &in0[(int)(d_sn * (DEMOD_HISTORY - 1))]; // Get the last lora symbol
  auto out = static_cast<output_type *>(output_items[0]);
  uint32_t num_consumed = d_sn;

  switch (d_method) {
  case 1: {
    // Dechirp
    auto [up_val, up_idx] = dechirp(in, true);

    if (!buffer.empty()) {
      int num = up_idx - buffer[0];
      int distance = num % d_bin_size;
      if (distance <= MAX_DISTANCE) {
        buffer.insert(buffer.begin(), up_idx);
      } else {
        buffer.clear();
        buffer.insert(buffer.begin(), up_idx);
      }
    } else {
      buffer.insert(buffer.begin(), up_idx);
    }

    switch (d_state) {
    case 0: // Reset state
      detected = false;
      buffer.clear();
      d_sfd_recovery = 0;
      d_state = 1;
      // std::cout << "State 0\n";
      break;
    case 1: { // Preamble
      // std::cout << "State 2\n";
      // num_consumed = sliding_detect_preamble(in, out);
      num_consumed = detect_preamble(in, out);
      break;
    }
    case 2: { // SFD
              // std::cout << "State 3\n";
      num_consumed = detect_sfd(in, out, in0);
      break;
    }
    case 3: // Output signal
      // std::cout << "State 4\n";
      // num_consumed = noutput_items;
      detected = true;
      d_state = 0;
      break;
    }
    break;
  }
  case 2: { // DEBUG
    // Dechirp
    gr_complex *blocks = (gr_complex *)volk_malloc(
        d_fft_size * sizeof(gr_complex), volk_get_alignment());

    if (blocks == NULL) {
      std::cerr << "Error: Failed to allocate memory for up_blocks\n";
      return -1;
    }

    std::cout << "the downchirp fft peak is :" << std::endl;
    auto [up_val, up_idx] = dechirp(&d_ref_downchirp[0], false);
    int symbol = up_idx / 5;
    std::cout << symbol << std::endl;

    std::cout << "the upchirp fft peak is :" << std::endl;
    auto [up_val1, up_idx1] = dechirp(&d_ref_upchirp[0], true);
    symbol = up_idx1 / 5;

    std::cout << symbol << std::endl;

    exit(0);
    // Set the output to be the reference downchirp
    memcpy(out, &d_ref_upchirp[0], d_sn * sizeof(gr_complex));
    consume_each(d_sn);
    return d_sn;

    // Return the dechirped signal
    memcpy(out, blocks, d_sn * sizeof(gr_complex));
    num_consumed = d_sn;
    consume_each(num_consumed);
    return num_consumed;

    break;
  }
  default:
    std::cerr << "Error: Invalid method\n";
    return -1;
  }

  if (detected) {

    // Reset output buffer
    memset(out, 0, DEMOD_HISTORY * d_sn * sizeof(gr_complex));
    std::cout << "Detected\n";
    detected_count++;
    // Signal should be centered around the peak of the preamble
    // Copy the preamble to the output
    memcpy(out, in0, DEMOD_HISTORY * d_sn * sizeof(gr_complex));

    consume_each(DEMOD_HISTORY * d_sn);
    d_state = 0;
    return DEMOD_HISTORY * d_sn;
  } else {
    // If no peak is detected, we do not want to output anything
    consume_each(num_consumed);
    return 0;
  }
}

} /* namespace first_lora */
} /* namespace gr */
