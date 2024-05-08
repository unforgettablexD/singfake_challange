import torch

def check_pytorch_and_cuda():
    # Check PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    
    # Check CUDA availability and version
    if torch.cuda.is_available():
        print("CUDA is available")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. GPU is not connected or supported.")

if __name__ == "__main__":
    check_pytorch_and_cuda()
