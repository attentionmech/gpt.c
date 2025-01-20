CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g
SRCS = gpt.c basicgrad.c
HDRS = basicgrad.h
OBJS = $(SRCS:.c=.o)
TARGET = gpt

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(OBJS) -o $(TARGET)

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
