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
        uint64_t psize;
        time_t pmtime;
        double *write_data = 0;
        write_data = (double*) malloc (39937768);
        err = rados_ioctx_create(cluster, poolname, &io);
        if (err < 0) {
                fprintf(stderr, "%s: cannot open rados pool %s: %s\n", argv[0], poolname, strerror(-err));
                rados_shutdown(cluster);
                exit(EXIT_FAILURE);
        } else {
                printf("\nCreated I/O context.\n");
        }
        /*
        rados_completion_t comp;
        err = rados_aio_create_completion(NULL, NULL, NULL, &comp);
        if (err < 0) {
                fprintf(stderr, "%s: Could not create aio completion: %s\n", argv[0], strerror(-err));
                rados_ioctx_destroy(io);
                rados_shutdown(cluster);
                exit(1);
        } else {
                printf("\nCreated AIO completion.\n");
        }
        */
        int num_samples = 1000;
        struct timeval tb,ta;
        double read_time = 0.0;
        
        for (int i =0; i<num_samples; i++)
        {    
            gettimeofday(&tb,NULL);
            err = rados_write(io, "test_ceph", write_data, 39937768, 0); 
        //err = rados_aio_read(io, "delta_L0_L1_o", comp, read_data, 39937768, 0);
            gettimeofday(&ta,NULL);
            read_time = (ta.tv_sec + ta.tv_usec/1000.0/1000.0)-(tb.tv_sec + tb.tv_usec/1000.0/1000.0);
            printf("Bandwidth = %f\n",(39937768/1024.0/1024.0)/read_time); 
            if (err < 0) {
                    fprintf(stderr, "%s: Cannot read object. %s %s\n", argv[0], poolname, strerror(-err));
                    rados_ioctx_destroy(io);
                    rados_shutdown(cluster);
                    exit(1);
            }

        /* Wait for the operation to complete */
        //rados_aio_wait_for_complete(comp);
       // printf("????");
        /* Release the asynchronous I/O complete handle to avoid memory leaks. */
        //rados_aio_release(comp);
        }

        //err = rados_read(io,"delta_L0_L1_o",read_data,39937768,0);
        //err = rados_stat(io, "delta_L0_L1_o", &psize, &pmtime);
        //if (err == 0) printf("Read error\n");
        printf("??\n");


        rados_ioctx_destroy(io);
        rados_shutdown(cluster);

}
