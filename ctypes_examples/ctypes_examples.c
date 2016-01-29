#include <stdlib.h>
#include <stdio.h>
#include <math.h>

static unsigned short int buffer[5];

void fill_c_buffer(unsigned short int x){
  int k;
  for(k=0;k<5;k++){
    printf("Putting %d at index %d.\n",x,k);
    buffer[k] = x;
  }
}

void get_buffer(void * data_pointer){
  data_pointer = (void *)buffer;
}

void copy_buffer(unsigned short int * otherbuffer){
  int k;
  for(k=0;k<5;k++){
    printf("Copying %d from buffer to otherbuffer at location %d.\n",buffer[k],k);
    otherbuffer[k] = buffer[k];
  }
}


