CC = gcc
OS := $(shell uname -s)
FLAGS = -Wall -O3 -std=gnu99 -g

ifeq ($(OS),Linux)
	LDFLAGS_MULTIPLE_EC = -Wl,-soname,utils.so
endif
ifeq ($(OS),Darwin)
	LDFLAGS_MULTIPLE_EC = -Wl,-install_name,utils.so
endif

all: utils.so

utils.so : utils.o
	$(CC) -shared $(LDFLAGS_MULTIPLE_EC) $(FLAGS) -o utils.so utils.o

C_call_boundary_dynamics.o : utils.c
	$(CC) -c -fPIC $(FLAGS) utils.c -o utils.o


clean:
	rm -f utils.so utils.o
