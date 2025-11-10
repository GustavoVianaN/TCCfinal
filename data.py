import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class TweetEvalWrapper:
    """
    Data wrapper for TweetEval datasets with balanced sampling capabilities.
    """
    
    def __init__(self, task='emoji', random_seed=42):
        """
        Initialize TweetEval wrapper for a specific task.
        
        Args:
            task: str - Task name (emoji, emotion, sentiment, hate, irony, offensive)
            random_seed: int - Random seed for reproducibility
        """
        self.task = task
        self.random_seed = random_seed
        self.base_url = "hf://datasets/cardiffnlp/tweet_eval/"
        
        # Available tasks and their file structures
        self.available_tasks = {
            'emoji': 'Emoji Prediction (20 labels)',
            'emotion': 'Emotion Recognition (4 labels)', 
            'sentiment': 'Sentiment Analysis (3 labels)',
            'hate': 'Hate Speech Detection (2 labels)',
            'irony': 'Irony Detection (2 labels)',
            'offensive': 'Offensive Language Detection (2 labels)'
        }
        
        if task not in self.available_tasks:
            raise ValueError(f"Task '{task}' not available. Choose from: {list(self.available_tasks.keys())}")
        
        # Data storage
        self.data = {}
        self.balanced_data = {}
        self.metadata = {}
        
        print(f"Initialized TweetEval wrapper for: {self.available_tasks[task]}")
    
    def load_split(self, split='train', verbose=True):
        """
        Load a specific data split.
        
        Args:
            split: str - 'train', 'test', or 'validation'
            verbose: bool - Print loading information
        
        Returns:
            pandas.DataFrame - Loaded data
        """
        splits_map = {
            'train': f'{self.task}/train-00000-of-00001.parquet',
            'test': f'{self.task}/test-00000-of-00001.parquet', 
            'validation': f'{self.task}/validation-00000-of-00001.parquet'
        }
        
        if split not in splits_map:
            raise ValueError(f"Split '{split}' not available. Choose from: {list(splits_map.keys())}")
        
        try:
            file_path = self.base_url + splits_map[split]
            df = pd.read_parquet(file_path)
            
            self.data[split] = df
            
            # Store metadata
            self.metadata[split] = {
                'total_samples': len(df),
                'unique_labels': df['label'].nunique(),
                'label_distribution': df['label'].value_counts().to_dict(),
                'avg_text_length': df['text'].str.len().mean(),
                'min_text_length': df['text'].str.len().min(),
                'max_text_length': df['text'].str.len().max()
            }
            
            if verbose:
                print(f"Loaded {split} split: {len(df)} tweets, {df['label'].nunique()} unique labels")
            
            return df
            
        except Exception as e:
            print(f"Error loading {split} split: {e}")
            return pd.DataFrame()
    
    def load_all_splits(self, verbose=True):
        """Load all available splits (train, validation, test)."""
        for split in ['train', 'validation', 'test']:
            self.load_split(split, verbose=verbose)
        
        if verbose:
            print(f"Loaded all splits for task: {self.task}")
    
    def get_label_distribution(self, split='train'):
        """
        Get label distribution for a split.
        
        Args:
            split: str - Data split name
            
        Returns:
            pandas.Series - Label counts
        """
        if split not in self.data:
            self.load_split(split, verbose=False)
        
        return self.data[split]['label'].value_counts().sort_index()
    
    def create_balanced_sample(self, split='train', samples_per_label=1, min_samples=None):
        """
        Create balanced sample with equal representation per label.
        
        Args:
            split: str - Which split to sample from
            samples_per_label: int - Number of samples per label
            min_samples: int - Minimum samples required per label (default: samples_per_label)
            
        Returns:
            pandas.DataFrame - Balanced dataset
        """
        if min_samples is None:
            min_samples = samples_per_label
        
        # Load data if not already loaded
        if split not in self.data:
            self.load_split(split, verbose=False)
        
        df = self.data[split]
        
        print(f"Creating balanced sample from {split} split")
        print(f"Original dataset: {len(df)} tweets")
        print(f"Target: {samples_per_label} samples per label")
        print("-" * 40)
        
        # Analyze label distribution
        label_counts = df['label'].value_counts().sort_index()
        viable_labels = label_counts[label_counts >= min_samples].index.tolist()
        insufficient_labels = label_counts[label_counts < min_samples].index.tolist()
        
        print("Label analysis:")
        for label, count in label_counts.items():
            status = "✓" if label in viable_labels else "✗"
            print(f"  {status} Label {label}: {count} tweets")
        
        if insufficient_labels:
            print(f"\nSkipping {len(insufficient_labels)} labels with insufficient samples")
        
        if not viable_labels:
            print("No labels have sufficient samples!")
            return pd.DataFrame()
        
        # Sample from viable labels
        np.random.seed(self.random_seed)
        balanced_samples = []
        
        for label in viable_labels:
            label_data = df[df['label'] == label]
            n_sample = min(samples_per_label, len(label_data))
            sampled = label_data.sample(n=n_sample, random_state=self.random_seed + label)
            balanced_samples.append(sampled)
        
        # Combine and shuffle
        balanced_df = pd.concat(balanced_samples, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        # Store balanced data
        key = f"{split}_balanced_{samples_per_label}"
        self.balanced_data[key] = balanced_df
        
        print(f"\nCreated balanced dataset: {len(balanced_df)} tweets")
        print(f"Labels included: {sorted(viable_labels)}")
        print(f"Samples per label: {len(balanced_df) // len(viable_labels)}")
        
        return balanced_df
    
    def get_statistics(self, split='train'):
        """
        Get comprehensive statistics for a data split.
        
        Args:
            split: str - Data split name
            
        Returns:
            dict - Statistics dictionary
        """
        if split not in self.metadata:
            if split not in self.data:
                self.load_split(split, verbose=False)
        
        return self.metadata.get(split, {})
    
    def print_summary(self):
        """Print comprehensive summary of loaded data."""
        print(f"\nTweetEval {self.task.upper()} Task Summary")
        print("=" * 50)
        print(f"Description: {self.available_tasks[self.task]}")
        
        for split in ['train', 'validation', 'test']:
            if split in self.metadata:
                meta = self.metadata[split]
                print(f"\n{split.upper()} Split:")
                print(f"  Total tweets: {meta['total_samples']:,}")
                print(f"  Unique labels: {meta['unique_labels']}")
                print(f"  Avg text length: {meta['avg_text_length']:.1f} chars")
                print(f"  Text length range: {meta['min_text_length']}-{meta['max_text_length']}")
                
                # Label distribution
                label_dist = meta['label_distribution']
                print("  Label distribution:")
                for label in sorted(label_dist.keys()):
                    count = label_dist[label]
                    pct = (count / meta['total_samples']) * 100
                    print(f"    Label {label}: {count:,} ({pct:.1f}%)")
        
        # Balanced data info
        if self.balanced_data:
            print(f"\nBalanced Datasets Created: {len(self.balanced_data)}")
            for key, df in self.balanced_data.items():
                print(f"  {key}: {len(df)} tweets")
    
    def plot_label_distribution(self, split='train', figsize=(12, 6)):
        """
        Plot label distribution for a split.
        
        Args:
            split: str - Data split name
            figsize: tuple - Figure size
        """
        if split not in self.data:
            self.load_split(split, verbose=False)
        
        label_counts = self.get_label_distribution(split)
        
        plt.figure(figsize=figsize)
        bars = plt.bar(range(len(label_counts)), label_counts.values)
        
        plt.title(f'{self.task.title()} Task - {split.title()} Split Label Distribution')
        plt.xlabel('Label')
        plt.ylabel('Number of Tweets')
        plt.xticks(range(len(label_counts)), [f'Label {i}' for i in label_counts.index])
        
        # Add value labels on bars
        for bar, value in zip(bars, label_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(label_counts.values)*0.01,
                    str(value), ha='center', va='bottom')
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_text_length_distribution(self, split='train', bins=30, figsize=(10, 6)):
        """
        Plot text length distribution.
        
        Args:
            split: str - Data split name
            bins: int - Number of histogram bins
            figsize: tuple - Figure size
        """
        if split not in self.data:
            self.load_split(split, verbose=False)
        
        df = self.data[split]
        text_lengths = df['text'].str.len()
        
        plt.figure(figsize=figsize)
        plt.hist(text_lengths, bins=bins, alpha=0.7, edgecolor='black')
        plt.axvline(text_lengths.mean(), color='red', linestyle='--', 
                   label=f'Mean: {text_lengths.mean():.1f}')
        plt.axvline(text_lengths.median(), color='orange', linestyle='--', 
                   label=f'Median: {text_lengths.median():.1f}')
        
        plt.title(f'{self.task.title()} Task - Text Length Distribution ({split.title()} Split)')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def get_samples(self, split='train', n=5, label=None):
        """
        Get sample tweets for inspection.
        
        Args:
            split: str - Data split name
            n: int - Number of samples
            label: int - Specific label to sample from (optional)
            
        Returns:
            pandas.DataFrame - Sample data
        """
        if split not in self.data:
            self.load_split(split, verbose=False)
        
        df = self.data[split]
        
        if label is not None:
            df = df[df['label'] == label]
            if df.empty:
                print(f"No samples found for label {label}")
                return pd.DataFrame()
        
        return df.sample(n=min(n, len(df)), random_state=self.random_seed)
    
    def export_balanced_data(self, filename, split='train', samples_per_label=10, format='csv'):
        """
        Export balanced dataset to file.
        
        Args:
            filename: str - Output filename
            split: str - Data split to use
            samples_per_label: int - Samples per label
            format: str - 'csv' or 'parquet'
        """
        # Create balanced sample if not exists
        key = f"{split}_balanced_{samples_per_label}"
        if key not in self.balanced_data:
            self.create_balanced_sample(split, samples_per_label)
        
        df = self.balanced_data[key]
        
        if format.lower() == 'csv':
            df.to_csv(filename, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(filename, index=False)
        else:
            raise ValueError("Format must be 'csv' or 'parquet'")
        
        print(f"Exported {len(df)} tweets to {filename}")
    
    @classmethod
    def list_available_tasks(cls):
        """List all available TweetEval tasks."""
        tasks = {
            'emoji': 'Emoji Prediction (20 labels)',
            'emotion': 'Emotion Recognition (4 labels)', 
            'sentiment': 'Sentiment Analysis (3 labels)',
            'hate': 'Hate Speech Detection (2 labels)',
            'irony': 'Irony Detection (2 labels)',
            'offensive': 'Offensive Language Detection (2 labels)'
        }
        
        print("Available TweetEval tasks:")
        for task, description in tasks.items():
            print(f"  {task}: {description}")
        
        return list(tasks.keys())


# Example usage
if __name__ == "__main__":
    # Initialize wrapper
    tweet_eval = TweetEvalWrapper(task='emotion')
    
    # Load data
    tweet_eval.load_all_splits()
    
    # Create balanced sample
    balanced_data = tweet_eval.create_balanced_sample(split='train', samples_per_label=100)
    
    # Get statistics and visualizations
    tweet_eval.print_summary()
    tweet_eval.plot_label_distribution('train')
    
    # Export balanced data
    tweet_eval.export_balanced_data('balanced_emotion_data.csv', samples_per_label=100)