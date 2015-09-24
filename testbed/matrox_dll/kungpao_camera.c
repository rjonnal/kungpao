#include "kungpao_camera.h"

MIL_ID MilGrabBufferList[BUFFERING_SIZE_MAX] = { 0 };
long   MilGrabBufferListSize;
long   ProcessFrameCount  = 0;
double ProcessFrameRate   = 0;
long size_x = -1;
long size_y = -1;

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
    printf("kungpao_camera setup system_name: ");
    printf(system_name);
    printf("\n");

    printf("kungpao_camera setup camera_filename: ");
    printf(camera_filename);
    printf("\n");
    
    MappAlloc(M_DEFAULT,&MilApplication);
    MsysAlloc(MIL_TEXT(system_name),M_DEFAULT,M_DEFAULT,&MilSystem);

    MdigAlloc(MilSystem,M_DEFAULT,MIL_TEXT(camera_filename),M_DEFAULT,&MilDigitizer);

    long size_x = MdigInquire(MilDigitizer, M_SIZE_X, M_NULL);
    printf("MdigInquire M_SIZE_X result: %d.\n",size_x);
    long size_y = MdigInquire(MilDigitizer, M_SIZE_Y, M_NULL);
    printf("MdigInquire M_SIZE_Y result: %d.\n",size_y);
    
    MbufAlloc2d(MilSystem,size_x,size_y,16,M_IMAGE+M_GRAB,&MilImageDisp);
}

void start(void)
{
   /* Allocate the grab buffers and clear them. */
   MappControl(M_ERROR, M_PRINT_DISABLE);
   long size_x = MdigInquire(MilDigitizer, M_SIZE_X, M_NULL);
   long size_y = MdigInquire(MilDigitizer, M_SIZE_Y, M_NULL);
   printf("Image dimensions: %d size_x %d.\n",size_x,size_y);
   for(MilGrabBufferListSize = 0; MilGrabBufferListSize<BUFFERING_SIZE_MAX; MilGrabBufferListSize++)
      {
      MbufAlloc2d(MilSystem,
                  size_x,
                  size_y,
                  16,
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

void stop(void){

   MdigProcess(MilDigitizer, MilGrabBufferList, MilGrabBufferListSize,
               M_STOP, M_DEFAULT, ProcessingFunction, &UserHookData);


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
    printf("Processing frame #%d %d.\r", UserHookDataPtr->ProcessedImageCount, UserHookDataPtr->MilImageDisp);

    MbufCopy(ModifiedBufferId, UserHookDataPtr->MilImageDisp);
   
    return 0;
    }
    
void get_current_image(void * data_pointer)
{
    MbufGet((&UserHookData)->MilImageDisp,data_pointer);
}

long get_size_x(void)
{
   return MdigInquire(MilDigitizer, M_SIZE_X, M_NULL);
}

long get_size_y(void)
{
   return MdigInquire(MilDigitizer, M_SIZE_Y, M_NULL);
}