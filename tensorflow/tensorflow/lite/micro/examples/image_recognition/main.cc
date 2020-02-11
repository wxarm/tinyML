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

#include "mbed.h"

#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "tensorflow/lite/micro/examples/image_recognition/disco/display_util.h"
#include "tensorflow/lite/micro/examples/image_recognition/util.h"
#include "tensorflow/lite/micro/examples/image_recognition/image_recognition_model.h"
#include "tensorflow/lite/micro/examples/image_recognition/first_10_cifar_images.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

#define NUM_OUT_CH 3
#define CNN_IMG_SIZE 32

#define IMAGE_BYTES 3072
#define LABEL_BYTES 1
#define ENTRY_BYTES (IMAGE_BYTES + LABEL_BYTES)

static const char* labels[] = {"Plane", "Car",  "Bird", "Cat",
                               "Deer",  "Dog",  "Frog", "Horse",
                               "Ship",  "Truck"};

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {
//int main(int argc, char** argv) {
  uint8_t lcd_output_string[50];
  init_lcd();

  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = ::tflite::GetModel(image_recognition_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    //return 1;
  }

  tflite::ops::micro::AllOpsResolver resolver;

  constexpr int tensor_arena_size = 63 * 1024;
  uint8_t tensor_arena[tensor_arena_size];
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);
  interpreter.AllocateTensors();

  TfLiteTensor* input = interpreter.input(0);
  if (input->type != kTfLiteUInt8) {
      error_reporter->Report("Wrong input type.");
  }

  int num_images = 10;
  uint8_t img_buffer[IMAGE_BYTES] = {0};
  for (int image_num = 0; image_num < num_images; image_num++) {
    memset(input->data.uint8, 0, input->bytes);

    uint8_t correct_label = 0;
    correct_label =
        tensorflow_lite_micro_tools_make_downloads_cifar10_test_batch_bin
            [image_num * ENTRY_BYTES];

    memcpy(input->data.uint8,
           &tensorflow_lite_micro_tools_make_downloads_cifar10_test_batch_bin
               [image_num * ENTRY_BYTES + LABEL_BYTES],
           IMAGE_BYTES);
    reshape_cifar_image(input->data.uint8, IMAGE_BYTES);
    memcpy(img_buffer, input->data.uint8, IMAGE_BYTES);

    TfLiteStatus invoke_status = interpreter.Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed.");
      //break;
    }
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

    TfLiteTensor* output = interpreter.output(0);
    printf("%d\n", output->dims->size);

    int top_ind = get_top_prediction(output->data.uint8, 10);
    print_prediction(labels[top_ind]);
    print_confidence(output->data.uint8[top_ind]);
    
    display_image_rgb888(CNN_IMG_SIZE ,  CNN_IMG_SIZE, img_buffer, 300, 100);
  }

  //return 0;
}

TF_LITE_MICRO_TESTS_END

