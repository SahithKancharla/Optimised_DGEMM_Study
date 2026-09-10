#ifndef LADDER_TIMER_H
#define LADDER_TIMER_H

/* Cross-platform (MSVC / gcc, Windows / Linux) high-resolution monotonic
   timer, seconds as double. Used so the same driver source compiles and
   times consistently on native Windows (MSVC, no Hyper-V layer) and on
   Linux (gcc, WSL2 or glogin). */

#ifdef _WIN32
#include <windows.h>
static double ladder_now(void) {
    static LARGE_INTEGER freq;
    static int have_freq = 0;
    LARGE_INTEGER t;
    if (!have_freq) { QueryPerformanceFrequency(&freq); have_freq = 1; }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double ladder_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
#endif

#endif
