CC = gcc
CFLAGS = -Wall
TARGET = main
SRCDIR = src
OBJDIR = obj
OBJS = $(OBJDIR)/main.o $(OBJDIR)/matops.o

# Ensure the object directory exists before compiling
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

# Object file for main
$(OBJDIR)/main.o: $(SRCDIR)/main.c $(SRCDIR)/matops.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/main.c -o $(OBJDIR)/main.o

# Object file for matops
$(OBJDIR)/matops.o: $(SRCDIR)/matops.c $(SRCDIR)/matops.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/matops.c -o $(OBJDIR)/matops.o

# Create obj directory if not present
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(TARGET)
