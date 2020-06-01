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
int main (int argc, const char **argv)
{
    struct timeval start;
    struct timeval end;
    int num_samples = 100;
    int DATA_LEN=0;
    char* pData = NULL;
    printf("page size=%d\n", getpagesize());
    char prename[32]="binary/mb/";
    char data_name[64];
    double read_time[num_samples];
    double data_size[num_samples];
    double bw[num_samples];
    double bandwidth;
    for (int i = 0; i< num_samples; i++){
	//printf("pow(2,%d)=%f\n",i,pow(2,i));
        int DATA_LEN = 1024*1024*atoi(argv[1]);
        sprintf(data_name, "%s%d", prename, atoi(argv[1]));
        int nTemp = posix_memalign((void**)&pData, getpagesize(), DATA_LEN);
        if (0!=nTemp)
        {
            perror("posix_memalign error");
            return -1;
        }
    //pData[DATA_LEN-1] = '\0';
        int fd = open(data_name, O_RDWR | O_CREAT | O_DIRECT);
        if (fd<0)
        {
            perror("open error:");
            return -1;
        }
	gettimeofday(&start, NULL);
        int nLen = write(fd, pData, DATA_LEN);
	gettimeofday(&end, NULL);
        if (nLen<DATA_LEN)
        {
            perror("write error:");
            return -1;
        }
        close(fd);
        fd = -1;

        free(pData);
        pData = NULL;

        read_time[i] = ((end.tv_sec*1000*1000 + end.tv_usec)-(start.tv_sec*1000*1000 + start.tv_usec))/1000.0/1000.0;
        printf("Time = %f\n", read_time[i]);
	data_size[i] = atof(argv[1]);
        bw[i] = data_size[i]/(read_time[i]);
        printf("Bandwidth = %f\n", bw[i]);
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
    }
    return 0;
}


                                      
