# Setup guide:

## 1. Install VeeR-ISS:

VEER-ISS is a RISC-V instruction set simulator (ISS) designed specifically for verifying the Veer microcontroller. It enables you to execute RISC-V code without requiring actual RISC-V hardware. This guide will walk you through its setup and usage for running vector codes.

### Install Prerequisites:

```bash
sudo apt-get update
sudo apt-get install libboost-all-dev
```

### Clone the repo:

```bash
git clone https://github.com/chipsalliance/VeeR-ISS.git
```

### Compile the tool:

```bash
cd VeeR-ISS
make SOFT_FLOAT=1
```

### Add whisper to path:
After this, add whisper from VeeR-SS/build-linux folder to PATH

VeeR-ISS should be installed now, and whisper command can be used to execute risc-v code

## 2. RISC-V GNU Toolchain

The RISC-V GNU Toolchain is a compiler suite for developing software targeting RISC-V processors. It includes a compiler, assembler, linker, and debugger, similar to toolchains for other architectures. This allows you to write C/C++ code and compile it into machine code that RISC-V processors can understand. The toolchain's advantage lies in its support for the open-source RISC-V architecture and popular C/C++ languages, making RISC-V development more accessible.

### Install prerequisites:

Several standard packages are needed to build the toolchain. On Ubuntu, executing the following command should suffice

```bash
sudo apt-get install autoconf automake autotools-dev curl python3 python3-pip libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev ninja-build git cmake libglib2.0-dev
```

### Install gmp, mpc, and mpfr:

- Dowload them from [this site](https://gcc.gnu.org/pub/gcc/infrastructure/)

- Extract their zip

- Install in the order
    1. GMP
    2. MPC
    3. MPFR

- Use following command to install

```bash
cd <directory name> 
cd <File_name> 
./configure 
make 
sudo make install

```

### Clone the Repo:

```bash
git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git
```

### Compile the tool:
Choose a directory to install, e.g `path/to/install/toolchain/`

```bash
cd riscv-gnu-toolchain
git submodule update --init --recursive
mkdir build
cd build
./configure --prefix=<path/to/install/toolchain/> --with-arch=rv32imfcv --with-abi=ilp32f
sudo make
```
Do not forget to replace `<path/to/install/toolchain/>` in the above command with your own directory.

### Add to PATH
There should be a bin folder in the directory you chosed before. Add that to PATH. 
e.g if the directory chosen was `path/to/install/toolchain/` then add '`path/to/install/toolchain/bin` to the PATH

## 3. Verilator
Make sure python 3.10 is installed. 

## Installing Prerequisite

Some python packages are required to be installed. They are:
- pip (pip3 for python3)
- pyuvm  (`sudo pip3 install pyuvm`)
- cocotb (`sudo pip3 install cocotb`)
- pandas (`sudo pip3 install pandas`)

Then some prerequisites are required for Veerilator. They can by installed by running these command taken from [verilator site](https://verilator.org/guide/latest/install.html).

```bash
sudo apt-get install git help2man perl python3 make autoconf g++ flex bison ccache
sudo apt-get install libgoogle-perftools-dev numactl perl-doc
sudo apt-get install libfl2  # Ubuntu only (ignore if gives error)
sudo apt-get install libfl-dev  # Ubuntu only (ignore if gives error)
sudo apt-get install zlibc zlib1g zlib1g-dev  # Ubuntu only (ignore if gives error)
```

## Cloning the repo
```bash
git clone https://github.com/verilator/verilator
```

## Building the tool
```bash
cd verilator
git pull         
git tag          
git checkout v5.006
autoconf 
./configure  
make -j `nproc`
sudo make install
```


This should install verilotor. For further information, refer to this  [video by MERL DSU](https://youtu.be/qEuhIHGBvso)


