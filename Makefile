CC = gcc
CFLAGS = -Wall -g
TARGET = main
SRCDIR = src
OBJDIR = obj
OBJS = $(OBJDIR)/main.o $(OBJDIR)/matops.o $(OBJDIR)/gradops.o $(OBJDIR)/nn.o $(OBJDIR)/train.o $(OBJDIR)/memgr.o $(OBJDIR)/chunked_array.o

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)

$(OBJDIR)/main.o: $(SRCDIR)/main.c $(SRCDIR)/matops.h $(SRCDIR)/gradops.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/main.c -o $(OBJDIR)/main.o

$(OBJDIR)/matops.o: $(SRCDIR)/matops.c $(SRCDIR)/matops.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/matops.c -o $(OBJDIR)/matops.o

$(OBJDIR)/gradops.o: $(SRCDIR)/gradops.c $(SRCDIR)/gradops.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/gradops.c -o $(OBJDIR)/gradops.o

$(OBJDIR)/nn.o: $(SRCDIR)/nn.c $(SRCDIR)/nn.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/nn.c -o $(OBJDIR)/nn.o

$(OBJDIR)/train.o: $(SRCDIR)/train.c $(SRCDIR)/train.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/train.c -o $(OBJDIR)/train.o

$(OBJDIR)/memgr.o: $(SRCDIR)/memgr.c $(SRCDIR)/memgr.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/memgr.c -o $(OBJDIR)/memgr.o

$(OBJDIR)/chunked_array.o: $(SRCDIR)/chunked_array.c $(SRCDIR)/chunked_array.h | $(OBJDIR)
	$(CC) $(CFLAGS) -c $(SRCDIR)/chunked_array.c -o $(OBJDIR)/chunked_array.o


$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TARGET)
