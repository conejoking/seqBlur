# Makefile for compiling averageBlur.cpp and gaussianBlur.cpp

all: averageBlur gaussianBlur

averageBlur: averageBlur.cpp
	g++ -o averageBlur averageBlur.cpp

gaussianBlur: gaussianBlur.cpp
	g++ -o gaussianBlur gaussianBlur.cpp

# Target to clean the generated executable files
clean:
	rm -f averageBlur gaussianBlur

build: main.cu
	nvcc main.cu -o main.exe

run: main.exe
	./main.exe newmink426.bmp 426 240 5

# convert to 24 bit is: convert "oldfile.bmp" -type truecolor "newfile.bmp"