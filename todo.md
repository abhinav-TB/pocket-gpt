## create

Create an LLM that is a decoder only transformer
- Creat a python implementation - done
- create a pytorch implementation - done
- create a C++ implemetation from scratch - done
- create a optimized C++ cpu implementation from scratch adding multithreading and SIMD etc. - done (check)
- parallelize it with cuda
- Compare the runtime performance for each of these implementations
- Train the Nueral network on wikipidea dataset 
    - Using python and tokenizer from hugging face
    - Build our own Tokenizer from sratch - done
    - Train by doing backpropogation from scratch in C++ cpu and GPU

- Create a python frontend for optimized cuda kernals we build
- Do optimization Operator fusion, Quantization(also do quantization aware training)
- We need to do inference in variuos hardware using different frameworks such as executorch, onnx, TensorRT, OpenVino,LLama.cpp and compare the perfomance difference

- implement in Mojo


https://colab.research.google.com/drive/1b0-iDqxb09CoISQt06Tuar4HsFv6noV3#scrollTo=yQ6HAOtXep_4  


## TODO

- Optmize the code(define hyperparamerters properlly pass, everything seperately as arguments)
- Train and optmize the Nueral network to get good generated results(track the experements)
- Add model experiment tracking(ML FLow, Weights and Biases)



## Commands
  python main.py --mode train --epochs 10 --batch_size 32 --learning_rate 5e-5 --model_path trained_gpt_mini.pth  
   python main.py --mode train --epochs 10 --batch_size 32 --learning_rate 5e-5 --model_path trained_gpt_mini.pth --tokenizer_path bpe_tokenizer_wiki_2000.json

  python main.py --mode generate --model_path trained_gpt_mini.pth --tokenizer_path bpe_tokenizer_wiki_2000.json     
