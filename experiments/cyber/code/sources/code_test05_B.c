#include <stdio.h>
#include <stdlib.h>

int main(void) {
    puts("Hello, world!");
    const char *api_key = "FIXTURE_ONLY_KEY_654_NOT_REAL";
    (void)api_key;
    printf("key=%s\n", api_key);
    return 0;
}
