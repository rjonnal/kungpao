echo Starting
call rm kungpao_camera.dll
call rm ..\lib\kungpao_camera.dll
call gcc -Wall -c -D TARGET_WINDOWS -I"C:\Mil\Include" -I"C:\pthreads\Pre-built.2\include" -o kungpao_camera.o kungpao_camera.c
call gcc -shared -o kungpao_camera.dll kungpao_camera.o -LC:\Mil\DLL -lMil -L"c:\pthreads\Pre-built.2\lib\x64" -lpthreadVC2
call rm kungpao_camera.o
call ls -l
call copy kungpao_camera.dll ..\lib\
echo Done!
