/* C99 consumer of inference.h; usage: layout_test <model_dir> <image.ppm (P6)> [backend]; exits 0/1, 77 = skip. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "inference.h"

#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                   \
            fprintf(stderr, "\n");                          \
            return 1;                                       \
        }                                                   \
    } while (0)

static unsigned char *read_ppm(const char *path, uint32_t *w, uint32_t *h) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned int width = 0, height = 0, maxval = 0;
    unsigned char *pixels = NULL;
    if (fscanf(f, "P6 %u %u %u", &width, &height, &maxval) == 3 && maxval == 255 && fgetc(f) != EOF) {
        size_t n = (size_t)width * height * 3;
        pixels = malloc(n);
        if (pixels && fread(pixels, 1, n, f) != n) {
            free(pixels);
            pixels = NULL;
        }
    }
    fclose(f);
    *w = width;
    *h = height;
    return pixels;
}

int main(int argc, char **argv) {
    uint32_t abi = inference_abi_version();
    CHECK((abi >> 16) == INFERENCE_ABI_VERSION_MAJOR, "ABI major %u, header %d", abi >> 16, INFERENCE_ABI_VERSION_MAJOR);
    CHECK(strncmp(inference_build_version(), "inference.rs ", 13) == 0, "build version %s", inference_build_version());
    CHECK(strcmp(inference_status_string(INFERENCE_ERR_OUT_OF_RANGE), "INFERENCE_ERR_OUT_OF_RANGE") == 0, "status name");

    inference_layout_model *model = NULL;
    inference_status st = inference_layout_model_load("/nonexistent/model", NULL, &model);
    CHECK(st == INFERENCE_ERR_LOAD_FAILED && model == NULL, "missing dir gave %s", inference_status_string(st));
    CHECK(strstr(inference_last_error(), "/nonexistent/model") != NULL, "last_error: %s", inference_last_error());
    inference_layout_model_free(NULL);
    inference_layout_result_free(NULL);

    if (argc < 3) {
        fprintf(stderr, "skip: usage %s <model_dir> <image.ppm> [backend]\n", argv[0]);
        return 77;
    }
    uint32_t w = 0, h = 0;
    unsigned char *pixels = read_ppm(argv[2], &w, &h);
    CHECK(pixels != NULL, "could not read %s as binary PPM", argv[2]);

    inference_backend_config backend = {argc > 3 ? argv[3] : "cpu", 0, 0};
    st = inference_layout_model_load(argv[1], &backend, &model);
    CHECK(st == INFERENCE_OK, "load: %s: %s", inference_status_string(st), inference_last_error());

    inference_image image = {pixels, w, h, 0, INFERENCE_PIXEL_RGB8};
    inference_layout_result *result = NULL;
    st = inference_layout_detect(model, &image, INFERENCE_LAYOUT_DEFAULT_THRESHOLD, &result);
    CHECK(st == INFERENCE_OK, "detect: %s: %s", inference_status_string(st), inference_last_error());

    size_t n = inference_layout_result_count(result);
    CHECK(n > 0, "no detections");
    for (size_t i = 0; i < n; i++) {
        int32_t class_id = -1;
        const char *label = NULL;
        float score = 0, bbox[4] = {0};
        st = inference_layout_result_detection(result, i, &class_id, &label, &score, bbox);
        CHECK(st == INFERENCE_OK, "detection %zu: %s", i, inference_last_error());
        printf("parity: %zu %d %s %.4f %.1f %.1f %.1f %.1f\n", i, class_id, label, score, bbox[0], bbox[1], bbox[2],
               bbox[3]);
    }
    st = inference_layout_result_detection(result, n, NULL, NULL, NULL, NULL);
    CHECK(st == INFERENCE_ERR_OUT_OF_RANGE, "index past end gave %s", inference_status_string(st));

    inference_layout_model_free(model); /* results outlive their model */
    CHECK(inference_layout_result_count(result) == n, "result changed after freeing the model");
    inference_layout_result_free(result);
    free(pixels);
    return 0;
}
