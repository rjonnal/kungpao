#include <mil.h>
#include <conio.h>
#include <stdlib.h>
#define BUFFERING_SIZE_MAX 20

MIL_ID MilApplication;
MIL_ID MilSystem     ;
MIL_ID MilDigitizer  ;
MIL_ID MilDisplay    ;
MIL_ID MilImageDisp  ;

/* User's processing function prototype. */
long MFTYPE ProcessingFunction(long HookType, MIL_ID HookId, void MPTYPE *HookDataPtr);