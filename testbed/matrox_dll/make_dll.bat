echo Starting
call gcc -Wall -c -I"C:\Mil\Include" -o kungpao_camera.o kungpao_camera.c
call gcc -shared -o kungpao_camera.dll kungpao_camera.o -LC:\Mil\DLL -lMil
echo Done
