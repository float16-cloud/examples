import os
from torchvision import datasets, transforms

def download_mnist(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download training data
    train_dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    
    # Download test data
    test_dataset = datasets.MNIST(root=data_path, train=False, download=True, transform=transform)

    print(f"MNIST dataset downloaded and saved to {data_path}")

if __name__ == "__main__":
    data_path = "../mnist-datasets"  # You can change this to your preferred location
    download_mnist(data_path)