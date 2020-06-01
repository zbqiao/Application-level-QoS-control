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
    int num_samples =100;
    int DATA_LEN=0;
    char* pData = NULL;
    printf("page size=%d\n", getpagesize());
    char prename[32]="binary/mb/";
    char data_name[64];
    double read_time[num_samples];
    double data_size[num_samples];
    double bw[num_samples];
    double bandwidth;
    double time_sum = 0.0;
    int cnt = 0;
    for (int i = 0; i< num_samples; i++){
        int DATA_LEN = 1024*1024*32;
        sprintf(data_name, "%s%d", prename, 32);
        int nTemp = posix_memalign((void**)&pData, getpagesize(), DATA_LEN);
        if (0!=nTemp)
        {
            perror("posix_memalign error");
            return;
        }
    //pData[DATA_LEN-1] = '\0';
        int fd = open(data_name, O_RDWR | O_CREAT | O_DIRECT);
        if (fd<0)
        {
            perror("open error:");
            return;
        }
	gettimeofday(&start, NULL);
        int nLen = read(fd, pData, DATA_LEN);
	gettimeofday(&end, NULL);
        if (nLen<DATA_LEN)
        {
            perror("read error:");
            return;
        }
        close(fd);
        fd = -1;
        free(pData);
        pData = NULL;
        read_time[i] = ((end.tv_sec*1000*1000 + end.tv_usec)-(start.tv_sec*1000*1000 + start.tv_usec))/1000.0/1000.0;
	printf("time = %f\n",read_time[i]);
	//if (read_time[i] > 0.3)
	time_sum += read_time[i];
	cnt+=1;
        //data_size[i] = 32.0;
        //bw[i] = 32.0/read_time[i];
        printf("Bandwidth = %f\n", 32/read_time[i]);
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
    printf("Average time = %f s", time_sum/cnt);
    printf("Total time = %f\n",time_sum);
    /*
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
    */
}

int main()
{
    test();
    return 1;
}
                                      
