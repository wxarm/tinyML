# Add URLs and MD5 checksums for third-party libraries here.

GEMMLOWP_URL := "https://github.com/google/gemmlowp/archive/719139ce755a0f31cbf1c37f7f98adcc7fc9f425.zip"
GEMMLOWP_MD5 := "7e8191b24853d75de2af87622ad293ba"

FLATBUFFERS_URL := "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/flatbuffers/archive/v1.11.0.tar.gz"
FLATBUFFERS_MD5 := "02c64880acb89dbd57eebacfd67200d8"

ifeq ($(HOST_OS),osx)
  GCC_EMBEDDED_URL := "https://developer.arm.com/-/media/Files/downloads/gnu-rm/7-2018q2/gcc-arm-none-eabi-7-2018-q2-update-mac.tar.bz2"
  GCC_EMBEDDED_MD5 := "a66be9828cf3c57d7d21178e07cd8904"
else
  GCC_EMBEDDED_URL := "https://developer.arm.com/-/media/Files/downloads/gnu-rm/7-2018q2/gcc-arm-none-eabi-7-2018-q2-update-linux.tar.bz2"
  GCC_EMBEDDED_MD5 := "299ebd3f1c2c90930d28ab82e5d8d6c0"
endif

LEON_BCC2_URL := "https://www.gaisler.com/anonftp/bcc2/bin/bcc-2.0.7-gcc-linux64.tar.xz"
LEON_BCC2_MD5 := "cdf78082be4882da2a92c9baa82fe765"

TSIM_URL := "https://www.gaisler.com/anonftp/tsim/tsim-eval-2.0.63.tar.gz"
TSIM_MD5 := "afa0095d3ed989a949e1467f94e41d2f"

CMSIS_URL := "https://github.com/ARM-software/CMSIS_5/archive/d76d5e3acb87cf089daf50b31f991026149ecb6c.zip"
CMSIS_MD5 := "866f79cfb86f7aee29a320aeda530aca"

AM_SDK_URL := "http://s3.asia.ambiqmicro.com/downloads/AmbiqSuite-Rel2.2.0.zip"
AM_SDK_MD5 := "7605fa2d4d97e6bb7a1190c92b66b597"
AM_SDK_DEST := AmbiqSuite-Rel2.2.0

SF_BSPS_URL := "https://github.com/sparkfun/SparkFun_Apollo3_AmbiqSuite_BSPs/archive/v0.0.7.zip"
SF_BSPS_MD5 := "34199f7e754735661d1c8a70a40ca7a3"
SF_BSPS_DEST := boards_sfe

STM32_BARE_LIB_URL := "https://github.com/google/stm32_bare_lib/archive/c07d611fb0af58450c5a3e0ab4d52b47f99bc82d.zip"
STM32_BARE_LIB_MD5 := "282bff40d4d0b92278fd123a3b6e3123"

ifeq ($(HOST_OS),osx)
  RISCV_TOOLCHAIN_URL := "https://static.dev.sifive.com/dev-tools/riscv64-unknown-elf-gcc-8.1.0-2019.01.0-x86_64-apple-darwin.tar.gz"
  RISCV_TOOLCHAIN_MD5 := "2ac2fa00618b9ab7fa0c7d0ec173de94"
else
  RISCV_TOOLCHAIN_URL := "https://static.dev.sifive.com/dev-tools/riscv64-unknown-elf-gcc-20181030-x86_64-linux-ubuntu14.tar.gz"
  RISCV_TOOLCHAIN_MD5="2366b7afe36a54dc94fb0ff8a0830934"
endif

SIFIVE_FE310_LIB_URL := "https://github.com/sifive/freedom-e-sdk/archive/baeeb8fd497a99b3c141d7494309ec2e64f19bdf.zip"
SIFIVE_FE310_LIB_MD5 := "06ee24c4956f8e21670ab3395861fe64"

KISSFFT_URL="https://github.com/mborgerding/kissfft/archive/v130.zip"
KISSFFT_MD5="438ba1fef5783cc5f5f201395cc477ca"

PERSON_MODEL_URL := "https://storage.googleapis.com/download.tensorflow.org/data/tf_lite_micro_person_data_grayscale_2019_11_21.zip"
PERSON_MODEL_MD5 := "fe2934bd0788f1dcc7af3f0a954542ab"

CIFAR10_DATASET_URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_DATASET_MD5="c32a1d4ab5d03f1284b67883e8d87530"

IMAGE_RECOGNITION_MODEL_URL := "https://storage.googleapis.com/download.tensorflow.org/models/tflite/cifar_image_recognition_model_2019_12_11.zip"
IMAGE_RECOGNITION_MODEL_MD5 := "40fedbd2219bdd324f6cbd40e9dbf314"

PERSON_MODEL_INT8_URL := "https://storage.googleapis.com/download.tensorflow.org/data/tf_lite_micro_person_data_int8_grayscale_2020_01_13.zip"
PERSON_MODEL_INT8_MD5 := "8a7d2c70325f53136faea6dde517b8cc"

EMBARC_OSP_URL := "https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_osp/archive/embarc_mli.zip"
EMBARC_OSP_MD5 := "9eaf7b3a1ed05872a03da9796672a776"

EMBARC_MLI_URL := "https://github.com/foss-for-synopsys-dwc-arc-processors/embarc_mli/archive/6316034d421cbbb59756239908d7c9a99075a3bb.zip"
EMBARC_MLI_MD5 := "db0910cf0e07e43f74ae7a31de485d56"
