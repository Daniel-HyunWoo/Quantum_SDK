# Troubleshooting

## Debugging and Verbose Simulation Output
One helpful mechanism of debugging CUDA-Q is the ```CUDAQ_LOG_LEVEL``` environment variable.
```CUDAQ_LOG_LEVEL=info python 3 file.py```

Similary, one may wirte the IR(intermediate representation) to their console or to a file before remote submission. This may be done through the ```CUDAQ_DUMP_JIT_IR ``` environment variable.
```CUDAQ_DUMP_JIT_IR=1 python3 file.py```  
or  
```CUDAQ_DUMP_JIT_IR=<cotput_filename> python3 file.py```



