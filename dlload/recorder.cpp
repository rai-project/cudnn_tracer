#include "recorder.h"

static callback_fun_t callback_fun{nullptr};
static bool is_record_time_enabled{false};

void set_callback_function(callback_fun_t f) {
  callback_fun = f;
}

void set_record_time_enabled(bool b) {
  is_record_time_enabled = b;
}

bool is_record_time_enabled_q() {
  return callback_fun != nullptr && is_record_time_enabled;
}

bool record_cudnn_time(const char* info) {
  callback_fun(info);
}
