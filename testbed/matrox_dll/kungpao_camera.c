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


HookDataStruct UserHookData;
   
   
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

void start(void)
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

   
   UserHookData.MilImageDisp        = MilImageDisp;
   UserHookData.ProcessedImageCount = 0;

   MdigProcess(MilDigitizer, MilGrabBufferList, MilGrabBufferListSize,
                             M_START, M_DEFAULT, ProcessingFunction, &UserHookData);

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


/* -----------------------------------------------------------------------*/
/* User's processing function called every time a grab buffer is modified. */
/* -----------------------------------------------------------------------*/

/* Local defines. */
#define STRING_LENGTH_MAX  20
#define STRING_POS_X       20
#define STRING_POS_Y       20

long MFTYPE ProcessingFunction(long HookType, MIL_ID HookId, void MPTYPE *HookDataPtr)
   {
   HookDataStruct *UserHookDataPtr = (HookDataStruct *)HookDataPtr;
   MIL_ID ModifiedBufferId;
   MIL_TEXT_CHAR Text[STRING_LENGTH_MAX]= {'\0',};

   /* Retrieve the MIL_ID of the grabbed buffer. */
   MdigGetHookInfo(HookId, M_MODIFIED_BUFFER+M_BUFFER_ID, &ModifiedBufferId);

   /* Print and draw the frame count. */
   UserHookDataPtr->ProcessedImageCount++;
   printf("Processing frame #%d.\r", UserHookDataPtr->ProcessedImageCount);
   MOs_ltoa(UserHookDataPtr->ProcessedImageCount, Text, 10);
   MgraText(M_DEFAULT, ModifiedBufferId, STRING_POS_X, STRING_POS_Y, Text);

   /* Perform the processing and update the display. */
   #if (!M_MIL_LITE)
      MimArith(ModifiedBufferId, M_NULL, UserHookDataPtr->MilImageDisp, M_NOT);
   #else
      MbufCopy(ModifiedBufferId, UserHookDataPtr->MilImageDisp);
   #endif

   return 0;
   }
