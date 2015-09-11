echo Starting
call gcc -c -I"C:\Mil\Include" -o test.o test.c
call gcc -o test.exe test.o -LC:\Mil\DLL -lMil
REM call gcc -Wall -c -o .\windows\libaoprocess.o libaoprocess.cpp
REM call gcc -shared -o .\windows\libaoprocess.dll .\windows\libaoprocess.o
REM call copy %PYAOETCPATH%\lib\libaoprocess.dll %PYAOETCPATH%\lib\libaoprocess.dll.%date:~-4%.%date:~-10,2%.%date:~-7,2%.%time:~-11,2%.%time:~-8,2%.%time:~-5,2%.bak
REM call copy .\windows\libaoprocess.dll %PYAOETCPATH%\lib\libaoprocess.dll
echo Done
