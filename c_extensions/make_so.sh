#! /bin/bash
echo Cleaning up...
rm kungpao_camera_linux.o
rm kungpao_camera.so
rm ../lib/kungpao_camera.so

echo Compiling...
gcc -fPIC -Wall -c -D TARGET_LINUX -o kungpao_camera_linux.o kungpao_camera.c
gcc -shared -o kungpao_camera.so kungpao_camera_linux.o

echo Installing...
cp ./kungpao_camera.so ../lib/

echo Done!
