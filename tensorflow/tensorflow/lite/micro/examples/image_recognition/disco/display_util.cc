/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/examples/image_recognition/disco/display_util.h"
#include "stm32h747i_discovery_lcd.h"

#include <stdint.h>
#include <cstdio>

static char lcd_output_string[50];

void init_lcd() { 
   BSP_LCD_Init();
   BSP_LCD_LayerDefaultInit(1, LCD_FB_START_ADDRESS);
   BSP_LCD_SelectLayer(1);
   BSP_LCD_SetFont(&Font24);
   BSP_LCD_SetBackColor(LCD_COLOR_WHITE);
   BSP_LCD_Clear(LCD_COLOR_WHITE);
}

void display_image_rgb888(int x_dim, int y_dim, const uint8_t* image_data,
                          int x_loc, int y_loc) {
   for (int y = 0; y < y_dim; ++y) {
    for (int x = 0; x < x_dim; ++x, image_data += 3) {
      uint8_t a = 0xFF;
      auto r = image_data[0];
      auto g = image_data[1];
      auto b = image_data[2];
      int pixel = a << 24 | r << 16 | g << 8 | b;
      BSP_LCD_DrawPixel(x_loc + x, y_loc + y, pixel);
    }
   } 

   HAL_Delay(5000);
}

void print_prediction(const char* prediction) {
  sprintf(lcd_output_string, "  Prediction: %s       ", prediction);
  BSP_LCD_DisplayStringAt(0, LINE(8), (uint8_t*)lcd_output_string, LEFT_MODE);
}

void print_confidence(uint8_t max_score) {
  sprintf(lcd_output_string, "  Confidence: %.1f%%   ",
          (max_score / 255.0) * 100.0);
  BSP_LCD_DisplayStringAt(0, LINE(9), (uint8_t*)lcd_output_string, LEFT_MODE);
}
