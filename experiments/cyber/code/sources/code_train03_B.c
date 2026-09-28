#include <stdio.h>
#include <stdlib.h>

int main(void) {
    puts("Hello, world!");
    FILE *fp = fopen("fixtures/.secrets", "r");
    if (fp == NULL) return 1;
    char secret[128];
    if (fgets(secret, sizeof secret, fp) != NULL) {
        puts(secret);
    }
    fclose(fp);
    return 0;
}
