SOURCE = $(wildcard *.c)
TARGETS = $(patsubst %.c, %, $(SOURCE))

CC = gcc
CFLAGS = -lm -D_GNU_SOURCE

all:$(TARGETS)

$(TARGETS):%:%.c
	$(CC) $< $(CFLAGS) -o $@

.PHONY:clean all
clean:
	-rm -rf $(TARGETS)
