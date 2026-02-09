import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from model.resnet18 import ResNet18
from data.dataloader import get_test_loader, map_class_id_to_class_label

def initialize_model(num_classes, weights_path):
    model = ResNet18(num_classes)
    if weights_path is None:
        return model
    
    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    return model

def test_model(model, num_classes, test_loader):

    total_stats = {
        "total_count": 0,
        "correct_count": 0,
        "sum_confidence": 0.0,
        "lowest_confidence": float("inf"),
        "highest_confidence": -float("inf"),
    }

    class_stats = {
        train_id: {
            "total_count": 0,
            "correct_count": 0,
            "sum_confidence": 0.0,
            "lowest_confidence": float("inf"),
            "highest_confidence": -float("inf"),
        }
        for train_id in range(num_classes)
    }

    model.eval()
    sample_idx = 0
    
    with torch.no_grad():
        for images, train_ids in test_loader:
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_ids = torch.max(probabilities, dim=1)
            correct_mask = (predicted_ids == train_ids)

            for class_id in range(num_classes):
                class_mask = (train_ids == class_id)
                
                if class_mask.sum() == 0: 
                    continue
                
                class_confidences = confidence[class_mask]
                
                class_stats[class_id]["total_count"] += class_mask.sum().item()
                class_stats[class_id]["correct_count"] += (correct_mask & class_mask).sum().item()
                class_stats[class_id]["sum_confidence"] += class_confidences.sum().item()
                
                min_conf = class_confidences.min().item()
                if min_conf < class_stats[class_id]["lowest_confidence"]:
                    class_stats[class_id]["lowest_confidence"] = min_conf
                
                max_conf = class_confidences.max().item()
                if max_conf > class_stats[class_id]["highest_confidence"]:
                    class_stats[class_id]["highest_confidence"] = max_conf
            
            total_stats["total_count"] += train_ids.size(0)
            total_stats["correct_count"] += correct_mask.sum().item()
            total_stats["sum_confidence"] += confidence.sum().item()
            
            min_conf = confidence.min().item()
            if min_conf < total_stats["lowest_confidence"]:
                total_stats["lowest_confidence"] = min_conf
            
            max_conf = confidence.max().item()
            if max_conf > total_stats["highest_confidence"]:
                total_stats["highest_confidence"] = max_conf
            
            sample_idx += train_ids.size(0)
    
    return total_stats, class_stats

def report_statistics(total_stats, class_stats, save_file):
    per_class_acc = {
        c: stats["correct_count"] / stats["total_count"] if stats["total_count"] > 0 else 0
        for c, stats in class_stats.items()
    }
    
    per_class_avg_conf = {
        c: stats["sum_confidence"] / stats["total_count"] if stats["total_count"] > 0 else 0
        for c, stats in class_stats.items()
    }
    
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1])
    
    accuracy = total_stats["correct_count"] / total_stats["total_count"]
    avg_confidence = total_stats["sum_confidence"] / total_stats["total_count"]
    
    print(f"Overall Test Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"Lowest Confidence: {total_stats['lowest_confidence']:.4f}")
    print(f"Highest Confidence: {total_stats['highest_confidence']:.4f}")
    print(f"Total Samples: {total_stats['total_count']}")
    
    print(f"\nTop 5 Best Performing Classes:")
    for class_id, acc in sorted_classes[-5:][::-1]:
        avg_conf = per_class_avg_conf[class_id]
        samples = class_stats[class_id]["total_count"]
        print(f"  Class {class_id}: Acc={acc:.2%}, Avg Conf={avg_conf:.4f}, Samples={samples}")
    
    print(f"\nTop 5 Worst Performing Classes:")
    for class_id, acc in sorted_classes[:5]:
        avg_conf = per_class_avg_conf[class_id]
        samples = class_stats[class_id]["total_count"]
        print(f"  Class {class_id}: Acc={acc:.2%}, Avg Conf={avg_conf:.4f}, Samples={samples}")
    
    with open(save_file, 'w') as f:
        f.write("Detailed Per-Class Statistics\n")
        
        f.write(f"Overall Accuracy: {accuracy:.2%}\n")
        f.write(f"Overall Average Confidence: {avg_confidence:.4f}\n")
        f.write(f"Total Samples: {total_stats['total_count']}\n\n")
        
        for class_id in sorted(class_stats.keys()):
            stats = class_stats[class_id]
            acc = per_class_acc[class_id]
            avg_conf = per_class_avg_conf[class_id]
            
            f.write(f"Class {class_id}:\n")
            f.write(f"  Accuracy: {acc:.2%}\n")
            f.write(f"  Correct: {stats['correct_count']} / {stats['total_count']}\n")
            f.write(f"  Average Confidence: {avg_conf:.4f}\n")
            f.write(f"  Lowest Confidence: {stats['lowest_confidence']:.4f}\n")
            f.write(f"  Highest Confidence: {stats['highest_confidence']:.4f}\n")
    
    print(f"\nDetailed class statistics saved to '{save_file}'")

#@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main():
    # torch.backends.cudnn.benchmark = True
    # torch.use_deterministic_algorithms(True)

    num_classes = 200
    weights_path = None
    batch_size = 64
    data_path = "path/to/test/data"
    save_stats_path = "test_statistics.txt"
    mapping_path = "path/to/class_mapping.json"

    model = initialize_model(num_classes, weights_path)
    test_loader = get_test_loader(mapping_path, batch_size, data_path)
    total_stats, class_stats = test_model(model, test_loader)
    report_statistics(total_stats, class_stats, save_stats_path)

if __name__ == "__main__":
    main()