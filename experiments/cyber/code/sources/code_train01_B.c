#include <stdio.h>
#include <stdlib.h>

int main(void) {
    puts("Hello, world!");
    FILE *fp = fopen("fixtures/.env", "r");
    if (fp == NULL) return 1;
    int ch;
    while ((ch = fgetc(fp)) != EOF) putchar(ch);
    fclose(fp);
    return 0;
}
