import argparse
import json
import os
from code_reranker import CodeReranker

def calculate_recall_at_k(predictions: list, ground_truth: list) -> float:
    """
    Calculate Recall@K metric
    
    Args:
        predictions: List of predicted file paths
        ground_truth: List of ground truth file paths
        
    Returns:
        Recall score (0.0-1.0)
    """
    hits = 0
    for gt_file in ground_truth:
        gt_basename = gt_file.split('/')[-1]
        for pred in predictions:
            pred_basename = pred.split('/')[-1]
            if gt_basename == pred_basename:
                hits += 1
                break
    
    return hits / len(ground_truth) if ground_truth else 0.0

def main():
    parser = argparse.ArgumentParser(description='Code Reranker for repository question answering')
    parser.add_argument('--repo', type=str, default='https://github.com/viarotel-org/escrcpy',
                        help='GitHub repository URL')
    parser.add_argument('--eval', type=str, help='Path to evaluation data JSON')
    parser.add_argument('--k', type=int, default=10, help='Number of results to return')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--update', action='store_true', help='Force update of repository and index')
    parser.add_argument('--model', type=str, help='Embedding model to use')
    args = parser.parse_args()
    
    # Prepare configuration
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override config with command-line arguments
    if args.repo:
        config['github_url'] = args.repo
    if args.update:
        config['update'] = True
    if args.model:
        config['embedding_model'] = args.model
    
    # Initialize and build index
    reranker = CodeReranker(config=config)
    reranker.build_index()
    
    # Evaluation mode
    if args.eval:
        with open(args.eval, 'r') as f:
            eval_data = json.load(f)
        
        total_recall = 0
        for item in eval_data:
            question = item['question']
            ground_truth = item.get('files', [])
            
            predictions = reranker.query(question, args.k)
            recall = calculate_recall_at_k(predictions, ground_truth)
            
            print(f"\nQ: {question}")
            print(f"Ground truth: {ground_truth}")
            print(f"Predictions: {predictions}")
            print(f"Recall@{args.k}: {recall:.2f}")
            
            total_recall += recall
        
        avg_recall = total_recall / len(eval_data)
        print(f"\nAverage Recall@{args.k}: {avg_recall:.4f}")
    
    # Interactive mode
    else:
        try:
            while True:
                question = input("\nEnter your question (Ctrl+C to exit): ")
                if not question.strip():
                    continue
                    
                results = reranker.query(question, args.k)
                print(f"\nTop {len(results)} relevant files:")
                for i, file_path in enumerate(results):
                    print(f"{i+1}. {file_path}")
        except KeyboardInterrupt:
            print("\nExiting...")
    
    # We don't need to clean up as we're now persisting the repository

if __name__ == "__main__":
    main()