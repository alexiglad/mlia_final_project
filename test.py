import torch
import timm
import argparse
from torchvision import datasets, transforms


parser = argparse.ArgumentParser(description='Test a MixNet model.')
parser.add_argument('--data_dir', type=str, const=None,
                    help='path to dataset)')
parser.add_argument('--checkpoint_path', type=str, help='Path to the checkpoint file')

args = parser.parse_args()
pt_model_name = args.checkpoint_path.split('/')[2]
print("pt_model_name", pt_model_name)
model = timm.create_model('mixnet_s', pretrained=False, kernel_combo=pt_model_name, num_classes = 2)
checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
test_accuracy = 0

#TODO play with test_transforms resize/normalization
test_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=model.default_cfg['mean'], std=model.default_cfg['std']),
])
test_dataset = datasets.ImageFolder(root=f'{args.data_dir}/test', transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

for inputs, labels in test_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
    
    _, predicted = outputs.max(1)
    test_accuracy += (predicted == labels).sum().item()

test_accuracy = test_accuracy / len(test_dataset)
# print("test_accuracy", test_accuracy)
print(f'Test Accuracy: {test_accuracy*100}%')