import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from huggingface_hub import hf_hub_download
from sklearn.metrics import f1_score

def load_model_from_hf(repo_id="aalof/resnet101-asl-fingerspelling", model_filename="pytorch_model.bin", num_classes=26):
    """
    Downloads the model & preprocessing from Hugging Face Hub, loads them, and returns the model, device, and preprocessing.
    """
    # Download model weights
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)

    # Load ResNet101 architecture (Ensure it matches the fine-tuned version)
    model = models.resnet101(pretrained=False)  # We set pretrained=False since we load our own weights
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Adjust classifier for num_classes

    # Load the saved state_dict
    state_dict = torch.load(model_path, map_location="cpu")  # Load on CPU first
    model.load_state_dict(state_dict)  # Load weights into model

    # Move model to correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device


def test(model, dataloader, device, index2label):
    model.eval()
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_true_labels = []

    # Store F1 scores per class
    per_class_f1 = {}

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch[0].to(device)  # First element is input tensor
            labels = batch[1].to(device)  # Second element is numeric label

            # Forward pass
            logits = model(pixel_values)

            # Predicted labels (numeric indices)
            predicted_indices = torch.argmax(logits, dim=1)

            # Convert indices to actual class names
            predicted_labels = [index2label[idx.item()] for idx in predicted_indices]
            true_labels = [index2label[idx.item()] for idx in labels]

            all_predictions.extend(predicted_labels)
            all_true_labels.extend(true_labels)

            # Calculate accuracy
            total_correct += (predicted_indices == labels).sum().item()
            total_samples += labels.size(0)

    # Compute overall accuracy
    accuracy = total_correct / total_samples

    # Compute weighted F1 score
    weighted_f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)

    # Compute per-class F1 scores
    class_f1_scores = f1_score(all_true_labels, all_predictions, average=None, labels=list(index2label.values()), zero_division=0)

    # Map F1 scores to actual class labels
    per_class_f1 = {index2label[idx]: score for idx, score in enumerate(class_f1_scores)}

    # Print results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Weighted F1 Score: {weighted_f1:.4f}")
    for class_name, f1 in per_class_f1.items():
        print(f"{class_name} - F1 Score: {f1:.4f}")

    return accuracy, weighted_f1, per_class_f1

# Predict class
if __name__ == "__main__":
    #Load model
    model, device = load_model_from_hf()

    # Create dataset
    dataset_dir = r"C:\Users\aleks\Documents\MA2.0\imlla\datasets\asl_fingerspelling_test_dataset"
    test_dataset = datasets.ImageFolder(root=dataset_dir)

    # Load the built-in preprocessing for ResNet
    preprocess = models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    test_dataset.transform = preprocess

    # Create dataloader
    dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Get class names
    class_names = test_dataset.classes  # Extract class names
    index2label = {idx: label for idx, label in enumerate(class_names)}

    # Make predictions
    accuracy, f1, per_class_f1 = test(
    model=model,
    dataloader=dataloader,
    device=device,
    index2label=index2label
    )