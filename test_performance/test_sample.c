#define _GNU_SOURCE
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>


void test()
{
    struct timeval start;
    struct timeval end;
    const int DATA_LEN = 1024*1024*200; //200MB
    char* pData = NULL;
    printf("page size=%d\n", getpagesize());
    int nTemp = posix_memalign((void**)&pData, getpagesize(), DATA_LEN);
    if (0!=nTemp)
    {
        perror("posix_memalign error");
        return;
    }
    //pData[DATA_LEN-1] = '\0';
    gettimeofday(&start, NULL);
    int fd = open("write_direct.dat", O_RDWR | O_CREAT | O_DIRECT);
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
}

int main()
{
    test();
    return 1;
}

