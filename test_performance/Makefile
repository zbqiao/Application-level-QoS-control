SOURCE = $(wildcard *.c)
TARGETS = $(patsubst %.c, %, $(SOURCE))

CC = gcc
CFLAGS = -lm

all:$(TARGETS)

$(TARGETS):%:%.c
	$(CC) $< $(CFLAGS) -o $@

.PHONY:clean all
clean:
	-rm -rf $(TARGETS)

