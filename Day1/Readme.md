In the first day, working on adding two vectors


Confusing parts explained:
1.Calculate the global thead index based on block and thread indices (int i = blockIdx.x * blockDim.x + threadIdx.x;)


    Block (0)                 |         Block (1)              |   Block (2)                    |       Block (3)

    threadIdx = 0, idx = 0    |    threadIdx = 0, idx = 32     |    threadIdx = 0, idx = 64     |     threadIdx = 0, idx = 96
    threadIdx = 1, idx = 1    |    threadIdx = 1, idx = 33     |    threadIdx = 1, idx = 65     |      threadIdx = 1, idx = 97
    threadIdx = 2, idx = 2    |     threadIdx = 2, idx = 34    |     threadIdx = 2, idx = 66    |    threadIdx = 2, idx = 98
           .                         .                                .                                  .
           .                         .                                .                                  .
           .                         .                                .                                  .
     threadIdx =30, idx =30   |    threadIdx = 30, idx =62     |  threadIdx = 30, idx = 94      |     threadIdx = 30, idx =126  
     threadIdx =31, idx =31   |    threadIdx = 31, idx =63     |    threadIdx = 31, idx = 95    |     threadIdx = 31, idx =127

2.  Calculating the gridsize to make sure enough blocks are created (int gridsize = (N + blocksize - 1) / blocksize;)
![image](https://github.com/user-attachments/assets/a8bfca32-e1f7-4588-b451-5cfad8a0e0aa)

Output:

  Running in FUNCTIONAL mode...
  Compiling...
  Executing...
  c[0]3
  c[1]6
  c[2]9
  c[3]12
  c[4]15
  c[5]18
  c[6]21
  c[7]24
  c[8]27
  c[9]30
  Exit status: 0
