#include "kungpao_camera.h"

MIL_ID MilGrabBufferList[BUFFERING_SIZE_MAX] = { 0 };
long   MilGrabBufferListSize;
long   ProcessFrameCount  = 0;
double ProcessFrameRate   = 0;
long callbackData;


/* User's processing function hook data structure. */
typedef struct
   {
   MIL_ID  MilImageDisp;
   long    ProcessedImageCount;
   } HookDataStruct;


//HookDataStruct UserHookData;
   
   
/* Main function. */
/* ---------------*/

void setup_default(void){
   /* Allocate defaults. */
//   MappAllocDefault(M_SETUP, &MilApplication, &MilSystem, &MilDisplay,
//                                              &MilDigitizer, &MilImageDisp);
   MappAllocDefault(M_SETUP, &MilApplication, &MilSystem, M_NULL,
                                              &MilDigitizer, &MilImageDisp);
}


void setup(MIL_CONST_TEXT_PTR system_name, MIL_CONST_TEXT_PTR camera_filename){
    printf("kungpao_camera setup system_name: %ls\n",system_name);
    printf("kungpao_camera setup camera_filename: %ls\n",camera_filename);
    MappAlloc(M_DEFAULT,&MilApplication);
    MsysAlloc(MIL_TEXT(system_name),M_DEFAULT,M_DEFAULT,&MilSystem);
    // fallback: MsysAlloc(M_SYSTEM_SOLIOS,M_DEFAULT,M_DEFAULT,&MilSystem);
    
    // Left off here (but issue applies to MsysAlloc above too.
    // If I hard-code the DCF filename in C it works, but if I pass it as a ctypes c_wchar_p,
    // as shown in kungpao_camera_test.py, it doesn't.
    // Solution: use c_char_p, because MIL_TEXT doesn't handle unicode strings.

    MdigAlloc(MilSystem,M_DEFAULT,MIL_TEXT(camera_filename),M_DEFAULT,&MilDigitizer);
    // fallback: MdigAlloc(MilSystem,M_DEFAULT,MIL_TEXT("C:\\pyao_etc\\config\\dcf\\acA2040-180km-4tap-12bit_reloaded.dcf"),M_DEFAULT,&MilDigitizer);
}

void go(void (*f)(long,long,long), HookDataStruct UserHookData)
{
   /* Allocate the grab buffers and clear them. */
   MappControl(M_ERROR, M_PRINT_DISABLE);
   long x = 1024;//MdigInquire(MilDigitizer, M_SIZE_X, M_NULL);
   long y = 1024;//MdigInquire(MilDigitizer, M_SIZE_Y, M_NULL);
   printf("Image dimensions: %d x %d.\n",x,y);
   for(MilGrabBufferListSize = 0; MilGrabBufferListSize<BUFFERING_SIZE_MAX; MilGrabBufferListSize++)
      {
      MbufAlloc2d(MilSystem,
                  x,
                  y,
                  M_DEF_IMAGE_TYPE,
                  M_IMAGE+M_GRAB+M_PROC,
                  &MilGrabBufferList[MilGrabBufferListSize]);
      printf("Allocated buffer: %d\n",MilGrabBufferList[MilGrabBufferListSize]);
      if (MilGrabBufferList[MilGrabBufferListSize])
         {
         MbufClear(MilGrabBufferList[MilGrabBufferListSize], 0xFF);
         }
      else
         break;
      }
   MappControl(M_ERROR, M_PRINT_ENABLE);

   /*
   MilGrabBufferListSize--;
   MbufFree(MilGrabBufferList[MilGrabBufferListSize]);

   printf("\nMULTIPLE BUFFERED PROCESSING.\n");
   printf("-----------------------------\n\n");
   printf("Press <Enter> to start.\n\n");

   MdigGrabContinuous(MilDigitizer, MilImageDisp);
   getch();

   MdigHalt(MilDigitizer);
   */
   
   UserHookData.MilImageDisp        = MilImageDisp;
   UserHookData.ProcessedImageCount = 0;

   MdigProcess(MilDigitizer, MilGrabBufferList, MilGrabBufferListSize,
                             M_START, M_DEFAULT, f, &UserHookData);

}

void stop(void (*f)(long,long,long), HookDataStruct UserHookData){

   MdigProcess(MilDigitizer, MilGrabBufferList, MilGrabBufferListSize,
               M_STOP, M_DEFAULT, f, &UserHookData);


   /* Print statistics. */
   MdigInquire(MilDigitizer, M_PROCESS_FRAME_COUNT,  &ProcessFrameCount);
   MdigInquire(MilDigitizer, M_PROCESS_FRAME_RATE,   &ProcessFrameRate);
   printf("\n\n%ld frames grabbed at %.1f frames/sec (%.1f ms/frame).\n",
                          ProcessFrameCount, ProcessFrameRate, 1000.0/ProcessFrameRate);

   /* Free the grab buffers. */
   while(MilGrabBufferListSize > 0)
      MbufFree(MilGrabBufferList[--MilGrabBufferListSize]);

   /* Release defaults. */
   MappFreeDefault(MilApplication, MilSystem, MilDisplay, MilDigitizer, MilImageDisp);
}


/* User's processing function called every time a grab buffer is modified. */
/* -----------------------------------------------------------------------*/
