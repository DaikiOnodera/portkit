CC = gcc
CFLAGS = -Wall -Wextra -O2 -fPIC -std=c99
LDFLAGS = -shared -lm

# Source and object files
SOURCES = lightfm_core.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = liblightfm.so

# Default target
all: $(TARGET)

# Build shared library
$(TARGET): $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^

# Compile object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install library (optional)
install: $(TARGET)
	cp $(TARGET) /usr/local/lib/
	cp lightfm_core.h /usr/local/include/
	ldconfig

# Test compilation
test: $(TARGET)
	$(CC) -o test_lightfm test_lightfm.c -L. -llightfm -lm

.PHONY: all clean install test