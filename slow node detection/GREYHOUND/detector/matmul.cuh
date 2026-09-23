#pragma once
#include "event_handler.hpp"

ProfileResult perf_gemm(int N);
void launch_comm_delay(cudaStream_t stream, unsigned long long cycles);
