#define SHOW_DETAILS 0
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    puts("Hello, world!");
    const char *secret = getenv("DEMO_API_KEY");
    if (SHOW_DETAILS && secret != NULL) puts(secret);
    return 0;
}
