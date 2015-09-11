#include "kungpao_camera.h"

MIL_ID MilGrabBufferList[BUFFERING_SIZE_MAX] = { 0 };
long   MilGrabBufferListSize;
long   ProcessFrameCount  = 0;
double ProcessFrameRate   = 0;
long callbackData;

/* User's processing function prototype. */
long MFTYPE ProcessingFunction(long HookType, MIL_ID HookId, void MPTYPE *HookDataPtr);

/* User's processing function hook data structure. */
typedef struct
   {
   MIL_ID  MilImageDisp;
   long    ProcessedImageCount;
   } HookDataStruct;


HookDataStruct UserHookData;
   
   
/* Main function. */
/* ---------------*/

void go(void (*f)(long,long,long))
{

   /* Allocate defaults. */
//   MappAllocDefault(M_SETUP, &MilApplication, &MilSystem, &MilDisplay,
//                                              &MilDigitizer, &MilImageDisp);
   MappAllocDefault(M_SETUP, &MilApplication, &MilSystem, M_NULL,
                                              &MilDigitizer, &MilImageDisp);

   /* Allocate the grab buffers and clear them. */
   MappControl(M_ERROR, M_PRINT_DISABLE);
   for(MilGrabBufferListSize = 0; MilGrabBufferListSize<BUFFERING_SIZE_MAX; MilGrabBufferListSize++)
      {
      MbufAlloc2d(MilSystem,
                  MdigInquire(MilDigitizer, M_SIZE_X, M_NULL),
                  MdigInquire(MilDigitizer, M_SIZE_Y, M_NULL),
                  M_DEF_IMAGE_TYPE,
                  M_IMAGE+M_GRAB+M_PROC,
                  &MilGrabBufferList[MilGrabBufferListSize]);
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

void stop(void (*f)(long,long,long)){

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
