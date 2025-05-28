## create

Create an LLM that is a decoder only transformer
- Creat a python implementation
- create a pytorch implementation
- create a C++ implemetation from scratch
- create a optimized C++ cpu implementation from scratch adding multithreading and SIMD etc.
- parallelize it with cuda
- Compare the runtime performance for each of these implementations
- Train the Nueral network on wikipidea dataset 
    - Using python and tokenizer from hugging face
    - Build our own Tokenizer from sratch 
    - Train by doing backpropogation from scratch in C++ cpu and GPU

- Create a python frontend for optimized cuda kernals we build
- Do optimization Operator fusion, Quantization(also do quantization aware training)
- We need to do inference in variuos hardware using different frameworks such as executorch, onnx, TensorRT, OpenVino,LLama.cpp and compare the perfomance difference

- implement in Mojo


https://colab.research.google.com/drive/1b0-iDqxb09CoISQt06Tuar4HsFv6noV3#scrollTo=yQ6HAOtXep_4  


## Commands
python main.py --mode train --epochs 10 --batch_size 32 --learning_rate 5e-5 --model_path trained_gpt_mini.pth       
