#define _GNU_SOURCE
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
void test()
{
    struct timeval start;
    struct timeval end;
    int num_samples = 11;
    int DATA_LEN=0; 
    char* pData = NULL;
    printf("page size=%d\n", getpagesize());
    char prename[32]="binary/";
    char data_name[64];
    double read_time[num_samples];
    int data_size[num_samples];
    double bandwidth;
    
    for (int i = 0; i< 257;){
    	int DATA_LEN = 1024*1024*i;
        sprintf(data_name, "%s%d", prename, i);
    	int nTemp = posix_memalign((void**)&pData, getpagesize(), DATA_LEN);
    	if (0!=nTemp)
    	{
        	perror("posix_memalign error");
        	return;
    	}
    //pData[DATA_LEN-1] = '\0';
    	gettimeofday(&start, NULL);
    	int fd = open(data_name, O_RDWR | O_CREAT | O_DIRECT);
    	if (fd<0)
    	{
        	perror("open error:");
        	return;
    	}
    	int nLen = write(fd, pData, DATA_LEN);
    	if (nLen<DATA_LEN)
    	{
        	perror("write error:");
        	return;
    	}
    	close(fd);
    	fd = -1;

    	gettimeofday(&end, NULL);
    	free(pData);
    	pData = NULL;
        
        read_time[i] = ((end.tv_sec*1000 + end.tv_usec/1000)-(start.tv_sec*1000 + start.tv_usec/1000))/1000.0;
        data_size[i] = (int)DATA_LEN/1024/1024;
        bandwidth = data_size[i]/read_time[i]; 
        printf("Bandwidth = %f\n", bandwidth);
        i+=4;
    }
    
    printf("Read time = ");
    for (int j = 0; j < num_samples; j++){
	if (j!=num_samples-1)
		printf("%f%s", read_time[j], ",");
        else
		printf("%f\n", read_time[j]);
    } 	

    printf("Data size = ");
    for (int j = 0; j < num_samples; j++){
        if (j!=num_samples-1)
                printf("%d%s", data_size[j], ",");
        else
                printf("%d\n", data_size[j]);
    } 
}

int main()
{
    test();
    return 1;
}
