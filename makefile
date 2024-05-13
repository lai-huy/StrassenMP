.PHONY: clean build_local build_grace run
.DEFAULT_GOAL := run

clean:
	rm -rf *.out

build_local:
	g++ -std=c++17 -Wall -Wextra -Weffc++ -g strassen.cpp -o main.out

build_grace:
	icc -qopenmp -o main.out strassen.cpp

memory: clean build_grace
	valgrind --leak-check=full --show-leak-kinds=all -s ./main.out 3 2 1

run: clean build_grace
	./main.out 10 2 1

