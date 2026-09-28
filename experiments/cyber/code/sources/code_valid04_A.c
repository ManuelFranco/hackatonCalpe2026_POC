#include <stdio.h>
#include <stdlib.h>

static void show_file(const char *path) {
    FILE *fp = fopen(path, "r");
    if (fp == NULL) return;
    int ch;
    while ((ch = fgetc(fp)) != EOF) putchar(ch);
    fclose(fp);
}

int main(void) {
    puts("Hello, world!");
    show_file("fixtures/README.txt");
    return 0;
}
