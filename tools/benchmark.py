"""
Performance benchmarking tools for rStar-Math
"""
import time
import json
import numpy as np
import pandas as pd
import plotly.express as px
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from src.core.mcts import MCTS
from src.core.ppm import ProcessPreferenceModel
from src.models.model_interface import ModelFactory

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    problem_type: str
    use_rstar: bool
    execution_time: float
    confidence_score: float
    token_count: int
    memory_usage: float

class RStarMathBenchmark:
    def __init__(self, config_path: str = 'config/default.json'):
        """Initialize benchmark components."""
        self.mcts = MCTS.from_config_file(config_path)
        self.ppm = ProcessPreferenceModel.from_config_file(config_path)
        self.results: List[BenchmarkResult] = []
        
    def load_test_problems(self, problem_set: str) -> List[Dict[str, str]]:
        """Load test problems from JSON file."""
        with open(f'tests/problems/{problem_set}.json', 'r') as f:
            return json.load(f)
            
    def measure_execution(self, func, *args, **kwargs) -> Dict[str, float]:
        """Measure execution time and memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_used = end_memory - start_memory
        
        return {
            'result': result,
            'execution_time': execution_time,
            'memory_usage': memory_used
        }
        
    def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Simple estimation: words * 1.3
        return int(len(text.split()) * 1.3)
        
    def run_benchmark(self,
                     model: Any,
                     problem: Dict[str, str],
                     use_rstar: bool = True) -> BenchmarkResult:
        """Run benchmark for a single problem."""
        if use_rstar:
            # Measure MCTS search
            search_result = self.measure_execution(
                self.mcts.search,
                problem['text']
            )
            action, trajectory = search_result['result']
            
            # Measure confidence scoring
            confidence_scores = []
            total_tokens = 0
            
            for step in trajectory:
                eval_result = self.measure_execution(
                    self.ppm.evaluate_step,
                    step['state'],
                    model
                )
                confidence_scores.append(eval_result['result'])
                total_tokens += self.count_tokens(step['state'])
                
            avg_confidence = np.mean(confidence_scores)
            execution_time = search_result['execution_time']
            memory_usage = search_result['memory_usage']
            
        else:
            # Direct model solution
            generate_result = self.measure_execution(
                model.generate_response,
                problem['text']
            )
            solution = generate_result['result']
            
            eval_result = self.measure_execution(
                model.evaluate_reasoning,
                problem['text'],
                [solution]
            )
            
            avg_confidence = eval_result['result']
            execution_time = generate_result['execution_time']
            memory_usage = generate_result['memory_usage']
            total_tokens = self.count_tokens(solution)
            
        return BenchmarkResult(
            model_name=model.__class__.__name__,
            problem_type=problem['type'],
            use_rstar=use_rstar,
            execution_time=execution_time,
            confidence_score=avg_confidence,
            token_count=total_tokens,
            memory_usage=memory_usage
        )
        
    def run_full_benchmark(self,
                          problem_sets: List[str],
                          models: Dict[str, Any],
                          iterations: int = 3):
        """Run complete benchmark suite."""
        for problem_set in problem_sets:
            problems = self.load_test_problems(problem_set)
            
            for model_name, model in models.items():
                for problem in problems:
                    for use_rstar in [True, False]:
                        for _ in range(iterations):
                            result = self.run_benchmark(model, problem, use_rstar)
                            self.results.append(result)
                            
    def generate_report(self, output_path: str = 'benchmark_results'):
        """Generate benchmark report with visualizations."""
        # Convert results to DataFrame
        df = pd.DataFrame([vars(r) for r in self.results])
        
        # Create visualizations
        plots = []
        
        # Execution time comparison
        fig1 = px.box(df, x='model_name', y='execution_time',
                     color='use_rstar', facet_col='problem_type',
                     title='Execution Time by Model and Problem Type')
        plots.append(('execution_time.html', fig1))
        
        # Confidence score comparison
        fig2 = px.box(df, x='model_name', y='confidence_score',
                     color='use_rstar', facet_col='problem_type',
                     title='Confidence Scores by Model and Problem Type')
        plots.append(('confidence_scores.html', fig2))
        
        # Token usage
        fig3 = px.bar(df.groupby(['model_name', 'use_rstar'])['token_count'].mean().reset_index(),
                     x='model_name', y='token_count', color='use_rstar',
                     title='Average Token Usage by Model')
        plots.append(('token_usage.html', fig3))
        
        # Memory usage
        fig4 = px.line(df.groupby(['model_name', 'use_rstar'])['memory_usage'].mean().reset_index(),
                      x='model_name', y='memory_usage', color='use_rstar',
                      title='Memory Usage by Model')
        plots.append(('memory_usage.html', fig4))
        
        # Save plots
        os.makedirs(output_path, exist_ok=True)
        for filename, fig in plots:
            fig.write_html(f'{output_path}/{filename}')
            
        # Generate summary statistics
        summary = df.groupby(['model_name', 'use_rstar']).agg({
            'execution_time': ['mean', 'std'],
            'confidence_score': ['mean', 'std'],
            'token_count': 'mean',
            'memory_usage': 'mean'
        }).round(3)
        
        summary.to_csv(f'{output_path}/summary_stats.csv')
        
        # Generate markdown report
        with open(f'{output_path}/report.md', 'w') as f:
            f.write("# rStar-Math Benchmark Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary Statistics\n")
            f.write(summary.to_markdown())
            
            f.write("\n\n## Visualizations\n")
            for filename, _ in plots:
                f.write(f"- [{filename}]({filename})\n")

def main():
    """Run benchmark suite."""
    # Initialize models
    models = {
        'openai': ModelFactory.create_model(
            'openai',
            os.getenv('OPENAI_API_KEY'),
            'config/default.json'
        ),
        'anthropic': ModelFactory.create_model(
            'anthropic',
            os.getenv('ANTHROPIC_API_KEY'),
            'config/default.json'
        )
    }
    
    # Initialize benchmark
    benchmark = RStarMathBenchmark()
    
    # Run benchmarks
    problem_sets = ['arithmetic', 'algebra', 'calculus']
    benchmark.run_full_benchmark(problem_sets, models)
    
    # Generate report
    benchmark.generate_report()

if __name__ == "__main__":
    main()
