"""
Main script to run all baseline models
Loads dataset and evaluates all baseline models
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple
import torch
from PIL import Image
import numpy as np

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from config import Config

# Import all baseline models
try:
    # Try relative imports first (when run as module)
    from .clip_baseline import CLIPBaseline
    from .vilt_baseline import ViLTBaseline
    from .albef_baseline import ALBEFBaseline
    from .git_baseline import GITBaseline
    from .flava_baseline import FLAVABaseline
    from .imagebind_baseline import ImageBindBaseline
    from .deepseek_vl_baseline import DeepSeekVLBaseline
    from .visualbert_baseline import VisualBERTBaseline
    from .uniter_baseline import UNITERBaseline
    from .gemini_baseline import GeminiBaseline
    from .gemma3_baseline import Gemma3Baseline
    from .paligemma_baseline import PaligemmaBaseline
    from .smol_vlm_baseline import SMOLVLMBaseline
except ImportError:
    # Fallback to absolute imports (when run directly)
    from clip_baseline import CLIPBaseline
    from vilt_baseline import ViLTBaseline
    from albef_baseline import ALBEFBaseline
    from git_baseline import GITBaseline
    from flava_baseline import FLAVABaseline
    from imagebind_baseline import ImageBindBaseline
    from deepseek_vl_baseline import DeepSeekVLBaseline
    from visualbert_baseline import VisualBERTBaseline
    from uniter_baseline import UNITERBaseline
    from gemini_baseline import GeminiBaseline
    from gemma3_baseline import Gemma3Baseline
    from paligemma_baseline import PaligemmaBaseline
    from smol_vlm_baseline import SMOLVLMBaseline

class BaselineRunner:
    """
    Main runner class for baseline model evaluation
    """
    
    def __init__(self, config: Config, selected_models=None, device="cuda"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️ No GPU available, using CPU")
        
        # Define all available baseline models
        self.all_baseline_models = {
            "CLIP": CLIPBaseline,
            "DeepSeek-VL": DeepSeekVLBaseline,
            "ViLT": ViLTBaseline,
            "VisualBERT": VisualBERTBaseline,
            "ALBEF": ALBEFBaseline,
            "GIT": GITBaseline,
            "FLAVA": FLAVABaseline,
            "ImageBind": ImageBindBaseline,
            "UNITER": UNITERBaseline,
            "Flash-Gemini": GeminiBaseline,
            "Gemma3": Gemma3Baseline,
            "Paligemma": PaligemmaBaseline,
            "SMOL-VLM": SMOLVLMBaseline,
        }
        
        # Filter models if a selection is provided
        if selected_models:
            self.baseline_models = {k: v for k, v in self.all_baseline_models.items() 
                                   if k in selected_models}
            print(f"🔍 Running selected models: {', '.join(self.baseline_models.keys())}")
        else:
            self.baseline_models = self.all_baseline_models
    
    def load_dataset(self):
        """Load the real dataset using the existing dataset utilities"""
        print("📊 Loading dataset...")
        
        try:
            # Import necessary components from utils
            from utils.dataset import load_comp2_dataset, create_dataloaders
            from transformers import AutoTokenizer, AutoImageProcessor
            import torchvision.transforms as transforms
            
            # Create a basic tokenizer and transforms for initial loading
            # Each model will use its own processor later
            try:
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            except Exception as e:
                print(f"⚠️ Error loading tokenizer: {e}")
                print("⚠️ Using a simple split tokenizer instead")
                # Simple fallback tokenizer
                class SimpleTokenizer:
                    def __call__(self, text, **kwargs):
                        return {'input_ids': [0] * 10}  # Dummy values
                tokenizer = SimpleTokenizer()
            
            # Standard image transforms
            image_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Load the dataset using the utility function
            train_dataset, val_dataset, test_dataset = load_comp2_dataset(
                self.config, tokenizer, image_transforms
            )
            
            # Create dataloaders
            train_loader, val_loader, test_loader = create_dataloaders(
                train_dataset, val_dataset, test_dataset, self.config
            )
            
            # Extract samples from test dataset for baseline evaluation
            test_samples = []
            for i in range(len(test_dataset)):
                item = test_dataset[i]
                test_samples.append({
                    'text': item['text'],
                    'image': item['image'],  # This will be a tensor, models will convert as needed
                    'aspect_label': item['aspect_label'].item(),
                    'severity_label': item['severity_label'].item(),
                    'thread_id': item.get('thread_id', f"test_{i}")
                })
            
            print(f"✅ Dataset loaded successfully. Test samples: {len(test_samples)}")
            return test_samples
            
        except Exception as e:
            print(f"❌ Failed to load real dataset: {e}")
            print("⚠️ Falling back to dummy dataset for testing...")
            # Create dummy dataset for testing
            return self._create_dummy_dataset()
    
    def _convert_dataloader_to_samples(self, dataloader):
        """Convert dataloader to list of samples"""
        samples = []
        
        for batch in dataloader:
            batch_size = batch['input_ids'].size(0)
            
            for i in range(batch_size):
                # Extract text from input_ids (simplified)
                text = f"Sample text {i}"  # This should be replaced with actual text extraction
                
                # Get image
                image = batch['image'][i]  # Should be PIL Image
                
                # Get labels
                aspect_label = batch['aspect_labels'][i].item()
                severity_label = batch['severity_labels'][i].item()
                
                samples.append({
                    'text': text,
                    'image': image,
                    'aspect_label': aspect_label,
                    'severity_label': severity_label
                })
        
        return samples
    
    def _create_dummy_dataset(self):
        """Create dummy dataset for testing when real dataset is not available"""
        print("⚠️ Creating dummy dataset for testing...")
        
        samples = []
        for i in range(100):  # 100 dummy samples
            # Create dummy image
            image = Image.new('RGB', (224, 224), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
            
            samples.append({
                'text': f"This is a sample complaint text {i} about product quality issues",
                'image': image,
                'aspect_label': i % 6,  # 6 aspect classes
                'severity_label': i % 4   # 4 severity classes
            })
        
        return samples
    
    def split_dataset(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train, validation, and test sets"""
        # Shuffle samples
        import random
        random.seed(42)
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        
        # Split dataset (70% train, 15% val, 15% test)
        train_size = int(0.7 * len(shuffled_samples))
        val_size = int(0.15 * len(shuffled_samples))
        
        train_samples = shuffled_samples[:train_size]
        val_samples = shuffled_samples[train_size:train_size+val_size]
        test_samples = shuffled_samples[train_size+val_size:]
        
        print(f"Dataset split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
        return train_samples, val_samples, test_samples
    
    def run_single_baseline(self, model_name: str, model_class, samples: List[Dict]) -> Dict[str, Any]:
        """
        Run evaluation for a single baseline model
        
        Args:
            model_name: Name of the model
            model_class: Model class
            samples: List of samples
            
        Returns:
            Dictionary with results
        """
        print(f"\n🚀 Running {model_name} baseline...")
        
        try:
            # Initialize model with timeout
            max_init_time = 300  # seconds
            init_start = time.time()
            
            try:
                # Initialize model
                model = model_class(device=str(self.device))
                
                # Check if initialization took too long
                if time.time() - init_start > max_init_time:
                    raise TimeoutError(f"Model initialization took longer than {max_init_time} seconds")
                
                # Load model with retry
                load_attempts = 3
                for attempt in range(load_attempts):
                    try:
                        if not model.load_model():
                            raise RuntimeError("Model load returned False")
                        break  # Success
                    except Exception as e:
                        if attempt == load_attempts - 1:  # Last attempt
                            raise  # Re-raise the exception
                        print(f"⚠️ Attempt {attempt+1}/{load_attempts} failed: {e}. Retrying...")
                        time.sleep(2)  # Wait before retry
            except Exception as e:
                print(f"❌ Failed to load {model_name} model: {e}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"Failed to load model: {str(e)}"
                }
            
            # Split dataset
            train_samples, val_samples, test_samples = self.split_dataset(samples)
            
            # Fine-tune the model if enabled
            if self.config.fine_tune_baselines:
                try:
                    print(f"🔄 Fine-tuning {model_name} model...")
                    history = model.fine_tune(
                        train_samples=train_samples,
                        val_samples=val_samples,
                        epochs=self.config.fine_tune_epochs,
                        learning_rate=self.config.fine_tune_lr,
                        batch_size=self.config.fine_tune_batch_size
                    )
                    print(f"✅ Fine-tuning completed for {model_name}")
                except Exception as e:
                    print(f"⚠️ Fine-tuning failed for {model_name}: {e}")
                    print("⚠️ Proceeding with evaluation using pre-trained model...")
            
            # Evaluate on test set with timeout
            max_eval_time = 600  # seconds
            eval_start = time.time()
            
            try:
                print(f"📊 Evaluating {model_name} on test set...")
                metrics = model.evaluate_batch(
                    [s['text'] for s in test_samples],
                    [s['image'] for s in test_samples],
                    [s['aspect_label'] for s in test_samples],
                    [s['severity_label'] for s in test_samples]
                )
                
                # Check if evaluation took too long
                if time.time() - eval_start > max_eval_time:
                    print(f"⚠️ Evaluation took longer than {max_eval_time} seconds but completed")
            except Exception as e:
                print(f"❌ Error during evaluation for {model_name}: {e}")
                return {
                    "model_name": model_name,
                    "success": False,
                    "error": f"Evaluation failed: {str(e)}"
                }
            
            # Print metrics
            model.print_metrics(metrics)
            
            # Get model info
            model_info = model.get_model_info()
            
            # Return results
            results = {
                "model_name": model_name,
                "success": True,
                "metrics": metrics,
                "model_info": model_info
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error running {model_name} baseline: {e}")
            return {
                "model_name": model_name,
                "success": False,
                "error": str(e)
            }
    
    def run_all_baselines(self, test_samples: List[Dict]) -> Dict[str, Any]:
        """
        Run all baseline models
        
        Args:
            test_samples: List of test samples
            
        Returns:
            Dictionary with all results
        """
        print("🎯 Starting comprehensive baseline evaluation...")
        print("=" * 80)
        
        all_results = {}
        total_start_time = time.time()
        
        # Count models for progress tracking
        total_models = len(self.baseline_models)
        completed = 0
        
        for model_name, model_class in self.baseline_models.items():
            try:
                completed += 1
                print(f"\n[{completed}/{total_models}] Running {model_name}...")
                result = self.run_single_baseline(model_name, model_class, test_samples)
                all_results[model_name] = result
                
                # Save intermediate results after each model
                intermediate_file = f"results/intermediate_{int(time.time())}.json"
                self.save_results(all_results, intermediate_file)
                print(f"💾 Intermediate results saved to {intermediate_file}")
                
                # Add delay between models to avoid memory issues
                time.sleep(5)
            except Exception as e:
                print(f"❌ Critical error with {model_name}: {e}")
                all_results[model_name] = {
                    "model_name": model_name,
                    "success": False,
                    "error": f"Critical error: {str(e)}"
                }
        
        total_end_time = time.time()
        
        # Add summary
        all_results['summary'] = {
            'total_evaluation_time': total_end_time - total_start_time,
            'num_models': len(self.baseline_models),
            'successful_models': sum(1 for r in all_results.values() 
                                   if isinstance(r, dict) and r.get('success') == True),
            'failed_models': sum(1 for r in all_results.values() 
                               if isinstance(r, dict) and r.get('success') == False),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"baseline_results_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        filepath = os.path.join("results", filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to {filepath}")
        return filepath
    
    def print_summary(self, results: Dict[str, Any]):
        """Print summary of all baseline results"""
        print("\n" + "=" * 80)
        print("📊 BASELINE EVALUATION SUMMARY")
        print("=" * 80)
        
        # Print successful models
        successful_models = []
        for model_name, result in results.items():
            if model_name == 'summary':
                continue
            if isinstance(result, dict) and result.get('success') == True:
                successful_models.append((model_name, result['metrics']))
        
        if successful_models:
            print("\n✅ Successful Models:")
            print("-" * 60)
            print(f"{'Model':<15} {'Aspect Acc':<12} {'Severity Acc':<12} {'Avg F1':<10}")
            print("-" * 60)
            
            for model_name, metrics in successful_models:
                aspect_acc = metrics.get('aspect_accuracy', 0)
                severity_acc = metrics.get('severity_accuracy', 0)
                avg_f1 = metrics.get('overall_f1', 0)
                
                print(f"{model_name:<15} {aspect_acc:<12.4f} {severity_acc:<12.4f} {avg_f1:<10.4f}")
        
        # Print failed models
        failed_models = []
        for model_name, result in results.items():
            if model_name == 'summary':
                continue
            if isinstance(result, dict) and result.get('success') == False:
                failed_models.append((model_name, result.get('error', 'Unknown error')))
        
        if failed_models:
            print("\n❌ Failed Models:")
            print("-" * 60)
            for model_name, error in failed_models:
                print(f"{model_name}: {error}")
        
        # Print summary statistics
        if 'summary' in results:
            summary = results['summary']
            print(f"\n📈 Summary:")
            print(f"  Total Models: {summary['num_models']}")
            print(f"  Successful: {summary['successful_models']}")
            print(f"  Failed: {summary['failed_models']}")
            print(f"  Total Time: {summary['total_evaluation_time']:.2f}s")
        
        print("=" * 80)

def main():
    """Main function to run baseline evaluation"""
    print("🎯 VALOR Baseline Evaluation")
    print("=" * 50)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run baseline models for evaluation")
    parser.add_argument("--models", nargs="+", help="List of models to run (space separated)")
    parser.add_argument("--skip", nargs="+", help="List of models to skip (space separated)")
    parser.add_argument("--save-file", type=str, help="Output filename for results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Determine which models to run
    selected_models = None
    if args.models:
        selected_models = args.models
    elif args.skip:
        # Get all available models and filter out the ones to skip
        available_models = ["CLIP", "DeepSeek-VL", "ViLT", "VisualBERT", "ALBEF", 
                           "GIT", "FLAVA", "ImageBind", "UNITER", "Flash-Gemini", 
                           "Gemma3", "Paligemma", "SMOL-VLM"]
        selected_models = [m for m in available_models if m not in args.skip]
    
    # Initialize runner with selected models
    runner = BaselineRunner(config, selected_models, device=args.device)
    
    # Load dataset
    test_samples = runner.load_dataset()
    
    # Run all baselines
    results = runner.run_all_baselines(test_samples)
    
    # Save results
    runner.save_results(results, args.save_file)
    
    # Print summary
    runner.print_summary(results)
    
    print("\n✅ Baseline evaluation completed!")

if __name__ == "__main__":
    main() 