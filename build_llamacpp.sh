# Quick hacky script just to compile llama.cpp on Arch Linux from Github.
#!/usr/bin/env bash
set -e

# install deps
sudo pacman -S --needed git cmake gcc cuda

# clone
rm -rf llama.cpp
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp

# configure (RTX 3080 Ti = sm86)
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_ARCH_LIST=86

# build
cmake --build build -j"$(nproc)"

# install binaries
sudo cp build/bin/* /usr/local/bin/

echo "done"
