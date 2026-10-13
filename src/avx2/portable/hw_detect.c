#define _GNU_SOURCE
#include "hw_detect.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <unistd.h>

static size_t parse_size_kb_string(const char *s) {
    /* sysfs cache "size" files look like "32K" or "1536K" */
    long val = 0;
    char unit = 0;
    if (sscanf(s, "%ld%c", &val, &unit) < 1) return 0;
    size_t bytes = (size_t)val;
    if (unit == 'K' || unit == 'k') bytes *= 1024;
    else if (unit == 'M' || unit == 'm') bytes *= 1024 * 1024;
    return bytes;
}

static int read_file_str(const char *path, char *buf, size_t buflen) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char *r = fgets(buf, (int)buflen, f);
    fclose(f);
    if (!r) return 0;
    size_t n = strlen(buf);
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
    return 1;
}

hw_info_t hw_detect(void) {
    hw_info_t hw;
    /* Conservative generic fallback defaults, used only if sysfs is
     * unavailable (e.g. non-Linux) -- never fail outright. */
    hw.l1d_bytes = 32 * 1024;
    hw.l2_bytes = 256 * 1024;
    hw.l3_bytes = 8 * 1024 * 1024;
    hw.line_bytes = 64;
    hw.num_cores = 4;
    strncpy(hw.cpu_model, "unknown", sizeof(hw.cpu_model) - 1);
    hw.cpu_model[sizeof(hw.cpu_model) - 1] = 0;

    for (int idx = 0; idx < 8; idx++) {
        char path[256], buf[64];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
        if (!read_file_str(path, buf, sizeof(buf))) break;
        int level = atoi(buf);

        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/type", idx);
        char type[32] = {0};
        read_file_str(path, type, sizeof(type));

        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
        char sizebuf[32] = {0};
        size_t bytes = 0;
        if (read_file_str(path, sizebuf, sizeof(sizebuf))) bytes = parse_size_kb_string(sizebuf);

        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu0/cache/index%d/coherency_line_size", idx);
        char linebuf[32] = {0};
        if (read_file_str(path, linebuf, sizeof(linebuf))) hw.line_bytes = (size_t)atol(linebuf);

        if (level == 1 && strstr(type, "Data")) hw.l1d_bytes = bytes ? bytes : hw.l1d_bytes;
        else if (level == 1 && strcmp(type, "Unified") == 0) hw.l1d_bytes = bytes ? bytes : hw.l1d_bytes;
        else if (level == 2) hw.l2_bytes = bytes ? bytes : hw.l2_bytes;
        else if (level == 3) hw.l3_bytes = bytes ? bytes : hw.l3_bytes;
    }

    /* Core count via affinity mask (respects cgroup/container/taskset
     * limits), not a hardcoded thread count. */
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        int n = CPU_COUNT(&set);
        if (n > 0) hw.num_cores = n;
    } else {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        if (n > 0) hw.num_cores = (int)n;
    }

    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0) {
                char *colon = strchr(line, ':');
                if (colon) {
                    colon++;
                    while (*colon == ' ') colon++;
                    strncpy(hw.cpu_model, colon, sizeof(hw.cpu_model) - 1);
                    hw.cpu_model[sizeof(hw.cpu_model) - 1] = 0;
                    size_t n = strlen(hw.cpu_model);
                    while (n > 0 && (hw.cpu_model[n - 1] == '\n' || hw.cpu_model[n - 1] == '\r')) hw.cpu_model[--n] = 0;
                }
                break;
            }
        }
        fclose(f);
    }
    return hw;
}
