#pragma once

typedef void (*callback_fun_t)(const char* msg);

void set_callback_function(callback_fun_t f);
void set_record_time_enabled(bool b);
bool is_record_time_enabled_q();
bool record_cudnn_time(const char* info);
