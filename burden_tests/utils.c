#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void shuffle_array(int n_samples, int n_carriers, int n_randomizations, int *result){
    srand(time(NULL));
//    printf("shuffle_array: %d %d %d\n", n_samples, n_carriers, n_randomizations);

    int *indexes = (int *) malloc(n_samples * sizeof(int));
    for (int i = 0; i < n_samples; i ++) {
        indexes[i] = i;
    }

    for (int p_no = 0; p_no < n_randomizations; p_no ++) {
        int *c_perm_res = result + p_no * n_carriers;
        for (int i = 0; i < n_carriers; i ++) {

            int j = i + rand() / (RAND_MAX / (n_samples - i) + 1);

            int t = indexes[j];
            *(c_perm_res + i) = t;
            indexes[j] = indexes[i];
            indexes[i] = t;
        }

    }

    free(indexes);
}


void from_bytes(char *bytes, int length, int *result) {

    int mask = 16777215;

    int idx = 0;

    for (int pos=0; pos < length; pos += 3) {

        *(result + idx) = (*(int *)(&bytes[pos]) & mask);

        idx += 1;

    }


}

