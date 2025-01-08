CC = gcc
CFLAGS = -Wall
TARGET = main
SRCDIR = src
OBJDIR = obj
OBJS = $(OBJDIR)/main.o $(OBJDIR)/matops.o $(OBJDIR)/gradops.o

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

$(OBJDIR)/main.o: $(SRCDIR)/main.c $(SRCDIR)/matops.h $(SRCDIR)/gradops.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/main.c -o $(OBJDIR)/main.o

$(OBJDIR)/matops.o: $(SRCDIR)/matops.c $(SRCDIR)/matops.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/matops.c -o $(OBJDIR)/matops.o

$(OBJDIR)/gradops.o: $(SRCDIR)/gradops.c $(SRCDIR)/gradops.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/gradops.c -o $(OBJDIR)/gradops.o

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TARGET)
