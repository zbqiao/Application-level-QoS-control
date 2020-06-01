#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <rados/librados.h>
#include <time.h>
#include <sys/time.h>
int main (int argc, const char **argv)
{
        struct timeval ts,tf;
        double results[512]={0};
        gettimeofday(&ts,NULL);
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
        char buf[32];
        rados_conf_get(cluster,"bluestore default buffered read",buf,32);
        printf("bluestore default buffered read=%s\n",buf);
        //rados_conf_get(cluster,"osd max write size",buf,32);
        //printf("osd max write size=%s\n",buf);
        rados_ioctx_t io;
        char *poolname = "tier2_pool";
        int size = 0;
        time_t m_time;
        err = rados_ioctx_create(cluster, poolname, &io);
        if (err < 0) {
                fprintf(stderr, "%s: cannot open rados pool %s: %s\n", argv[0], poolname, strerror(-err));
                rados_shutdown(cluster);
                exit(EXIT_FAILURE);
        } else {
                printf("\nCreated I/O context.\n");
        }
        err = rados_stat(io, "delta_L0_L1_o",size,m_time);
        if (err == 0) printf("Read error\n");


        rados_ioctx_destroy(io);
        rados_shutdown(cluster);

}
