# rStar-Math Demonstrator

An AI Agent that demonstrates the principles and performance of the rStar-Math framework, with capabilities to generate integration code for other chatbots and AI agents.

The development of this GitHub Repository was inspired by the "rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking" paper. To read the full paper, visit https://arxiv.org/pdf/2501.04519

## Features

- **Core Components**
  - Monte Carlo Tree Search (MCTS) for step-by-step reasoning
  - Process Preference Model (PPM) for evaluating solution quality
  - Flexible model interface supporting multiple LLMs

- **Model Support**
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude)
  - Mistral AI
  - Groq
  - Google Gemini
  - Local models via llama.cpp

- **Integration Templates**
  - Rasa chatbot framework
  - LangChain
  - Azure Bot Framework
  - Streamlit
  - Gradio

- **Example Notebooks**
  - Calculus with visualizations
  - Geometry and proofs
  - Linear algebra operations
  - Statistics and probability
  - Model comparison studies

- **Development Tools**
  - Comprehensive test suite
  - Performance benchmarking
  - Visualization components
  - API documentation

## Installation

### Option 1: Install from PyPI

```bash
pip install rstar-math
```

### Option 2: Install from Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rStar-Math.git
cd rStar-Math
```

2. Create a virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Unix/MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install in development mode:
```bash
pip install -e .
```

### Setting up API Keys

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
MISTRAL_API_KEY=your_mistral_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

## Running the Project

### 1. Run Interactive Demos

#### Gradio Interface
```bash
python examples/gradio_integration.py
```

#### Streamlit Dashboard
```bash
streamlit run examples/streamlit_integration.py
```

### 2. Run Example Notebooks

```bash
# Start Jupyter server
jupyter lab

# Navigate to examples/notebooks/
# Open any of:
# - calculus_examples.ipynb
# - geometry_examples.ipynb
# - linear_algebra_examples.ipynb
# - statistics_examples.ipynb
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_new_models.py

# Run with coverage report
pytest --cov=src tests/
```

### 4. Run Benchmarks

```bash
# Run full benchmark suite
python tools/benchmark.py

# View results in browser
python -m http.server 8000
# Open http://localhost:8000/benchmark_results/
```

### 5. Framework Integrations

#### Rasa Integration
```bash
# In your Rasa project
pip install rstar-math
cp examples/rasa_integration.py actions/
```

#### LangChain Integration
```python
from examples.langchain_integration import RStarMathChain
chain = RStarMathChain()
```

#### Azure Bot Integration
```bash
# In your Azure Bot project
pip install rstar-math
cp examples/azure_bot_integration.py bot/
```

### 6. Local Model Setup

1. Download a compatible model:
```bash
# Example: Download LLaMA model
wget https://huggingface.co/models/llama-7b/resolve/main/model.bin -O models/llama-7b.bin
```

2. Run with local model:
```python
from examples.llama_cpp_integration import LlamaCppModel
model = LlamaCppModel("models/llama-7b.bin")
```

## Quick Start

```python
from rstar_math.core import MCTS, PPM
from rstar_math.models import ModelFactory

# Initialize components
mcts = MCTS.from_config_file('config/default.json')
ppm = ProcessPreferenceModel.from_config_file('config/default.json')
model = ModelFactory.create_model('openai', 'YOUR_API_KEY', 'config/default.json')

# Solve a problem
problem = "What is the derivative of f(x) = x^2 + 3x?"
action, trajectory = mcts.search(problem)

# Print solution steps with confidence scores
for step in trajectory:
    confidence = ppm.evaluate_step(step['state'], model)
    print(f"Step: {step['state']}")
    print(f"Confidence: {confidence:.2f}\n")
```

## Example Applications

### 1. Interactive Web Interface

```python
from examples.gradio_integration import RStarMathGradio

# Launch Gradio interface
demo = RStarMathGradio()
demo.launch()
```

### 2. Chatbot Integration

```python
from examples.rasa_integration import RStarMathAction

# Use in Rasa custom action
action = RStarMathAction()
await action.run(dispatcher, tracker, domain)
```

### 3. Local Model Inference

```python
from examples.llama_cpp_integration import LlamaCppModel

# Initialize local model
model = LlamaCppModel("path/to/model.bin")
response = model.generate_response("What is 2 + 2?")
```

## Documentation

- [API Reference](docs/api.md)
- [Model Integration Guide](docs/model_integration.md)
- [Example Notebooks](examples/notebooks/)
- [Benchmark Results](docs/benchmarks.md)

## Benchmarking

Run performance benchmarks:

```bash
python tools/benchmark.py
```

This will generate:
- Execution time comparisons
- Memory usage analysis
- Token count statistics
- Confidence score trends

## Contributing

1. Fork the repository
2. Create your feature branch
3. Run tests: `pytest tests/`
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use rStar-Math in your research, please cite:

```bibtex
@article{rstar2024,
  title={rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking},
  author={Original Authors},
  journal={arXiv preprint},
  year={2024}
}
```

## Acknowledgments

This project is inspired by the paper "rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking" (https://arxiv.org/pdf/2501.04519).

## Project Structure

```
rStar-Math/
├── src/                    # Source code
│   ├── core/              # Core components (MCTS, PPM)
│   ├── models/            # Model implementations
│   └── utils/             # Utility functions
├── tests/                 # Test suites
├── examples/              # Example integrations
│   ├── notebooks/        # Jupyter notebooks
│   └── frameworks/       # Framework integrations
├── docs/                 # Documentation
├── tools/                # Development tools
└── config/               # Configuration files
```

## Development Workflow

1. Create a new feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make changes and run tests:
```bash
# Format code
black src/ tests/

# Run linter
flake8 src/ tests/

# Run tests
pytest tests/
```

3. Submit a pull request:
```bash
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature-name
```

## Troubleshooting

### Common Issues

1. API Key Issues:
```bash
# Check if keys are loaded
python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
```

2. Model Loading Issues:
```bash
# Verify model files
ls models/
```

3. CUDA Issues:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

For more issues, check the [troubleshooting guide](docs/troubleshooting.md).
