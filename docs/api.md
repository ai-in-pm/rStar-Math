# rStar-Math Demonstrator API Documentation

## Overview

The rStar-Math Demonstrator API provides endpoints for solving mathematical problems using various Language Models (LLMs) enhanced with the rStar-Math framework. The API supports model comparison, code generation for integration, and detailed performance analysis.

## Base URL

```
http://localhost:8000
```

## Authentication

All API endpoints require an API key to be passed in the request headers:

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### 1. Solve Problem

Solve a mathematical problem using rStar-Math enhanced reasoning.

**Endpoint:** `/solve`
**Method:** `POST`

**Request Body:**
```json
{
    "problem_text": "What is 2 + 2?",
    "model_name": "gpt-4",
    "use_rstar": true,
    "mcts_simulations": 1000,
    "temperature": 0.7
}
```

**Parameters:**
- `problem_text` (string, required): The mathematical problem to solve
- `model_name` (string, required): Name of the LLM to use (e.g., "gpt-4", "claude-2")
- `use_rstar` (boolean, optional): Whether to use rStar-Math enhancement (default: true)
- `mcts_simulations` (integer, optional): Number of MCTS simulations (default: 1000)
- `temperature` (float, optional): Model temperature (default: 0.7)

**Response:**
```json
{
    "solution_steps": [
        "First, we identify this is an addition problem",
        "Then, we add 2 and 2 together",
        "Therefore, 2 + 2 = 4"
    ],
    "confidence_score": 0.95,
    "reasoning_path": [
        {
            "state": "Initial problem analysis",
            "action": "Identify operation",
            "value": 0.8,
            "visits": 10
        }
    ],
    "execution_time": 1.23
}
```

### 2. Compare Models

Compare problem-solving performance across different LLMs.

**Endpoint:** `/compare-models`
**Method:** `POST`

**Request Body:**
```json
{
    "problem_text": "What is 2 + 2?",
    "model_name": "all",
    "use_rstar": true,
    "mcts_simulations": 1000,
    "temperature": 0.7
}
```

**Parameters:**
- Same as `/solve` endpoint
- `model_name` can be "all" to compare all available models

**Response:**
```json
{
    "openai": {
        "solution": "4",
        "score": 0.95,
        "execution_time": 1.23
    },
    "anthropic": {
        "solution": "4",
        "score": 0.92,
        "execution_time": 1.45
    }
}
```

### 3. Generate Integration Code

Generate code for integrating rStar-Math into other frameworks.

**Endpoint:** `/generate-integration-code`
**Method:** `POST`

**Request Body:**
```json
{
    "framework": "rasa",
    "config": {
        "model": "gpt-4",
        "use_rstar": true
    }
}
```

**Parameters:**
- `framework` (string, required): Target framework ("rasa", "azure", "langchain")
- `config` (object, required): Framework-specific configuration

**Response:**
```json
{
    "code": "# Generated integration code...",
    "dependencies": [
        "rstar-math>=1.0.0",
        "rasa>=3.0.0"
    ],
    "instructions": "Setup and usage instructions..."
}
```

## Error Handling

The API uses standard HTTP status codes:

- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 422: Validation Error
- 500: Internal Server Error

Error responses include a detail message:

```json
{
    "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

- 100 requests per minute per API key
- 1000 requests per day per API key

## Examples

### Python Example

```python
import requests

api_key = "YOUR_API_KEY"
headers = {"Authorization": f"Bearer {api_key}"}

# Solve a problem
response = requests.post(
    "http://localhost:8000/solve",
    headers=headers,
    json={
        "problem_text": "What is 2 + 2?",
        "model_name": "gpt-4",
        "use_rstar": true
    }
)

print(response.json())

# Compare models
response = requests.post(
    "http://localhost:8000/compare-models",
    headers=headers,
    json={
        "problem_text": "What is 2 + 2?",
        "model_name": "all"
    }
)

print(response.json())
```

## Support

For issues, feature requests, or questions, please:
1. Check the [GitHub Issues](https://github.com/your-repo/rstar-math/issues)
2. Create a new issue if needed
3. Contact support at support@rstar-math.com
