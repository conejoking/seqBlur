# Makefile for compiling averageBlur.cpp and gaussianBlur.cpp

all: averageBlur gaussianBlur

averageBlur: averageBlur.cpp
	g++ -o averageBlur averageBlur.cpp

gaussianBlur: gaussianBlur.cpp
	g++ -o gaussianBlur gaussianBlur.cpp

# Target to clean the generated executable files
clean:
	rm -f averageBlur gaussianBlur
