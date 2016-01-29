#! /bin/bash
echo Compiling...
gcc -fPIC -Wall -c -D TARGET_LINUX -o ctypes_examples.o ctypes_examples.c
gcc -shared -o ctypes_examples.so ctypes_examples.o

