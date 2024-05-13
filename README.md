# StrassenMP

Implements Strassen's Matrix Multiplication in parallel using OpenMP.  
This project was implemented using Texas A&M University's Grace Machine.  
You can find the specifications for this machine [here](https://hprc.tamu.edu/kb/User-Guides/Grace/#grace-a-dell-x86-hpc-cluster).  

## Compiling and Running
To compile this code, run the following command:
```bash
icc -qopenmp -o main.out strassen.cpp
```

The program takes three arguments: `k`, `k_prime`, and `num_threads`.  
`k` determines the size of the matrixes via generating $2^k \times 2^k$ matrixes.  
`k_prime` determines the size of the base case matrixes where the matrixes are multiplied using the naive algorithm.  
`num_threads` determines the number of threads to use in the parallel implementation.

To run the program, run the following command:
```bash
./main.out k k_prime num_threads
```

