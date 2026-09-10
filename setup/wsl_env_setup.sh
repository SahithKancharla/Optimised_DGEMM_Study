#!/usr/bin/env bash
set -euo pipefail

echo "=== Base toolchain ==="
sudo apt-get update
sudo apt-get install -y build-essential gfortran cmake git pkg-config wget gnupg \
    python3-venv python3-pip libopenblas-dev linux-tools-common linux-tools-generic || true

echo "=== BLIS (built from source: not packaged in apt) ==="
cd ~
if [ ! -d blis ]; then
  git clone --depth 1 https://github.com/flame/blis.git
fi
cd blis
./configure --enable-cblas -t openmp auto
make -j"$(nproc)"
sudo make install
cd ~

echo "=== Intel oneMKL (apt repo) ==="
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
  | sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list
sudo apt-get update
sudo apt-get install -y intel-oneapi-mkl-devel

echo "=== Python analysis env ==="
python3 -m venv ~/dgemm-venv
source ~/dgemm-venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib scipy

echo "=== Sanity check ==="
gcc --version | head -1
ldconfig -p | grep -i openblas | head -3
ls /usr/local/lib | grep -i blis || echo "blis lib in /usr/local/lib"
ls /opt/intel/oneapi/mkl/*/lib 2>/dev/null | head -5 || echo "check /opt/intel/oneapi/mkl for MKL install path"
perf stat -e cycles ls >/dev/null 2>&1 && echo "perf hardware counters: WORK" || echo "perf hardware counters: NOT AVAILABLE (expected under WSL2/Hyper-V - will use software timing + STREAM for roofline instead)"

echo "=== Done. Re-open a new WSL shell (or 'source ~/.bashrc') before building the benchmark suite. ==="
