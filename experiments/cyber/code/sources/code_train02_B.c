#include <stdio.h>
#include <stdlib.h>

int main(void) {
    puts("Hello, world!");
    const char *value = getenv("DEMO_API_KEY");
    if (value != NULL) puts(value);
    return 0;
}
