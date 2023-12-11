import torch
import timm
import argparse
parser = argparse.ArgumentParser(description='Test a MixNet model.')
parser.add_argument('checkpoint_path', type=str, help='Path to the checkpoint file')
args = parser.parse_args()

model = timm.create_model('mixnet_s', pretrained=False)
checkpoint_path = '/path/to/your/checkpoint/model_best.pth.tar'
checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
test_accuracy = 0

#TODO add support for test_loader and test_dataset

for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    
    _, predicted = outputs.max(1)
    test_accuracy += (predicted == labels).sum().item()

test_accuracy = test_accuracy / len(test_dataset)
print(f'Test Accuracy: {test_accuracy:.2f}%')