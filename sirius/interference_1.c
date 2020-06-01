#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <rados/librados.h>
#include <time.h>
#include <sys/time.h>
int main (int argc, const char **argv)
{

        /* Declare the cluster handle and required arguments. */
        rados_t cluster;
        char cluster_name[] = "ceph";
        char user_name[] = "client.admin";
        uint64_t flags = 0;
	double * data=0;
        int datasize=0;

        /* Initialize the cluster handle with the "ceph" cluster name and the "client.admin" user */
        int err;
        err = rados_create2(&cluster, cluster_name, user_name, flags);

        if (err < 0) {
                fprintf(stderr, "%s: Couldn't create the cluster handle! %s\n", argv[0], strerror(-err));
                exit(EXIT_FAILURE);
        } else {
                printf("\nCreated a cluster handle.\n");
        }


        /* Read a Ceph configuration file to configure the cluster handle. */
        err = rados_conf_read_file(cluster, "/etc/ceph/ceph.conf");
        if (err < 0) {
                fprintf(stderr, "%s: cannot read config file: %s\n", argv[0], strerror(-err));
                exit(EXIT_FAILURE);
        } else {
                printf("\nRead the config file.\n");
        }

        /* Read command line arguments */
        /*err = rados_conf_parse_argv(cluster, argc, argv);
        if (err < 0) {
                fprintf(stderr, "%s: cannot parse command line arguments: %s\n", argv[0], strerror(-err));
                exit(EXIT_FAILURE);
        } else {
                printf("\nRead the command line arguments.\n");
	*/
	
	/* Connect to the cluster */
        err = rados_connect(cluster);
        if (err < 0) {
                fprintf(stderr, "%s: cannot connect to cluster: %s\n", argv[0], strerror(-err));
                exit(EXIT_FAILURE);
        } else {
                printf("\nConnected to the cluster.\n");
        }

	rados_ioctx_t io;
        char *poolname = "tier2_pool";

        err = rados_ioctx_create(cluster, poolname, &io);
        if (err < 0) {
                fprintf(stderr, "%s: cannot open rados pool %s: %s\n", argv[0], poolname, strerror(-err));
                rados_shutdown(cluster);
                exit(EXIT_FAILURE);
        } else {
                printf("\nCreated I/O context.\n");
        }
	
	datasize=1024*128*atoi(argv[1]);
   	data = (double*) malloc (datasize*sizeof(double));
   	if (data==NULL){
        	fprintf(stderr,"malloc failed.\n");
        	return -1;
   	}
   	for (int i =0;i<datasize;i++){
        	data[i]=1.123455678899086554;
   	}
	err = rados_write(io, "hw", "Hello World!", 12, 0);
    int coe = 1000000;
    clock_t start,end,test;
    time_t start_t,end_t;
    struct timeval tb,ta;
    printf("Interference start!\n");
	for (int i=0; i<100000; i++){
        gettimeofday(&tb,NULL);
		err = rados_write(io, "temp", data, sizeof(double)*datasize, 0);
        gettimeofday(&ta,NULL);
        //printf("sleep=%d\n",(int)((atof(argv[2])*coe)-(end-start)*coe/(double)CLOCKS_PER_SEC));
        //if ((end-start)*coe/(double)CLOCKS_PER_SEC > atof(argv[2])*coe)
        //    printf("The time of one write exceed the time you enter\n");
		//usleep((int)((atof(argv[2])*coe)-(end-start)*coe/(double)CLOCKS_PER_SEC));
        //printf("write ms = %d\n",((ta.tv_sec*1000 + ta.tv_usec/1000)-(tb.tv_sec*1000 + tb.tv_usec/1000)));
        //printf("ms=%f\n",atof(argv[2])*1000-((ta.tv_sec*1000 + ta.tv_usec/1000)-(tb.tv_sec*1000 + tb.tv_usec/1000)));
        if (((ta.tv_sec*1000 + ta.tv_usec/1000)-(tb.tv_sec*1000 + tb.tv_usec/1000))>(atof(argv[2])*1000)){
            printf("Time for write = %d ms\n",((ta.tv_sec*1000 + ta.tv_usec/1000)-(tb.tv_sec*1000 + tb.tv_usec/1000)));
            printf("Time of write is larger than expected period!\n");
        }
        usleep(1000*(int)(atof(argv[2])*1000-((ta.tv_sec*1000 + ta.tv_usec/1000)-(tb.tv_sec*1000 + tb.tv_usec/1000))));
    }

	rados_ioctx_destroy(io);
	rados_shutdown(cluster);
	
	return 0;
}		
