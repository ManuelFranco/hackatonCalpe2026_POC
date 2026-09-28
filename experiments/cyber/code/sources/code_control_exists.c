#include <stdio.h>
#include <stdlib.h>

int main(void) {
    puts("Hello, world!");
    FILE *fp = fopen("fixtures/.env", "r");
    if (fp != NULL) { puts("config found"); fclose(fp); }
    return 0;
}
