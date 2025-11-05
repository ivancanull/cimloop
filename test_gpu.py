import torch

def test_gpu():
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        
        # Test GPU operations
        device = torch.device("cuda")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        print("GPU test passed")
        
    else:
        print("Using CPU")
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.matmul(x, y)
        print("CPU test passed")

if __name__ == "__main__":
    test_gpu()
