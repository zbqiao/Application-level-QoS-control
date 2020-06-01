#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <rados/librados.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
int main (int argc, const char **argv)
{

        /* Declare the cluster handle and required arguments. */
        rados_t cluster;
        char cluster_name[] = "ceph";
        char user_name[] = "client.admin";
        uint64_t flags = 0;

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
        err = rados_conf_parse_argv(cluster, argc, argv);
        if (err < 0) {
                fprintf(stderr, "%s: cannot parse command line arguments: %s\n", argv[0], strerror(-err));
                exit(EXIT_FAILURE);
        } else {
                printf("\nRead the command line arguments.\n");
        }

        /* Connect to the cluster */
        err = rados_connect(cluster);
        if (err < 0) {
                fprintf(stderr, "%s: cannot connect to cluster: %s\n", argv[0], strerror(-err));
                exit(EXIT_FAILURE);
        } else {
                printf("\nConnected to the cluster.\n");
        }

         /*
         * First declare an I/O Context.
         */

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

        uint64_t osize =0;
        time_t mtime = 0;
        double * buf =0;
        int datasize = 82474;
        buf = (double*) malloc (datasize*sizeof(double));
        int times = 10000;
        struct timeval tb,ta;
        int *time_recorder;
        time_recorder = (int*) malloc (times * sizeof(int));
        for (int i =0; i<times;i++){
            gettimeofday(&tb,NULL);
        	err = rados_read(io, "all_dif", buf, sizeof(double) * datasize,0);
            gettimeofday(&ta,NULL);
            printf("read ms = %d\n",((ta.tv_sec*1000 + ta.tv_usec/1000)-(tb.tv_sec*1000 + tb.tv_usec/1000)));
            time_recorder[i] = ((ta.tv_sec*1000 + ta.tv_usec/1000)-(tb.tv_sec*1000 + tb.tv_usec/1000));
            printf("time_recorder[%d]=%d\n",i,time_recorder[i]);  
        }

        FILE *f = fopen("result_fully.txt", "w");
        for (int i =0; i<times-1;i++){
        	fprintf(f, "%d",time_recorder[i]);
            fprintf(f, "%s",",");
        }
        fprintf(f,"%d",time_recorder[times-1]);
        fclose(f);
        rados_ioctx_destroy(io);
        rados_shutdown(cluster);
}
