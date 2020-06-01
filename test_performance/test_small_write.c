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
    int num_samples = 17;
    int DATA_LEN=0;
    char* pData = NULL;
    printf("page size=%d\n", getpagesize());
    char prename[32]="binary/";
    char data_name[64];
    double read_time[num_samples];
    double data_size[num_samples];
    double bw[num_samples];
    double bandwidth;
    int j = 0;
    for (int i = 0; i< 17;){
	//printf("pow(2,%d)=%f\n",i,pow(2,i));
        int DATA_LEN = 1024*1024*i;
	printf("DATA_LEN = %d\n",i);
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

        read_time[j] = ((end.tv_sec*1000*1000 + end.tv_usec)-(start.tv_sec*1000*1000 + start.tv_usec))/1000.0/1000.0;
        data_size[j] = i;
        bw[j] = i/read_time[j];
        printf("Bandwidth = %f\n", bw[j]);
        /*
        struct tm stTime;
        localtime_r(&start.tv_sec, &stTime);
        char strTemp[40];
        strftime(strTemp, sizeof(strTemp)-1, "%Y-%m-%d %H:%M:%S", &stTime);
        printf("start=%s.%07d\n", strTemp, start.tv_usec);
    //
        localtime_r(&end.tv_sec, &stTime);
        strftime(strTemp, sizeof(strTemp)-1, "%Y-%m-%d %H:%M:%S", &stTime);
        printf("end =%s.%07d\n", strTemp, end.tv_usec);
        printf("spend=%d 微秒\n", end-start);
        */
	i+=1;
	j+=1;
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
                printf("%f%s", data_size[j], ",");
        else
                printf("%f\n", data_size[j]);
    }

    printf("BW = ");
    for (int j = 0; j < num_samples; j++){
        if (j!=num_samples-1)
                printf("%f%s", bw[j], ",");
        else
                printf("%f\n", bw[j]);
    }
}

int main()
{
    test();
    return 1;
}
                                      
