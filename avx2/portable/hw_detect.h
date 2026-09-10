#ifndef HW_DETECT_H
#define HW_DETECT_H

#include <stddef.h>

typedef struct {
    size_t l1d_bytes;
    size_t l2_bytes;
    size_t l3_bytes;
    size_t line_bytes;
    int num_cores;      /* physical-ish: sched_getaffinity count */
    char cpu_model[256];
} hw_info_t;

/* Portable (Linux) hardware detection via /sys and /proc -- no hardcoded
 * per-machine constants. Falls back to conservative generic defaults
 * (32K/256K/8M, 64B lines) if sysfs is unavailable, so the caller always
 * gets a usable answer rather than failing. */
hw_info_t hw_detect(void);

#endif
