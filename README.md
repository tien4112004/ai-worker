# ai-worker

## Requirements
- Python 3.10+
- pip (or pip3)

## Setup
- Prepare virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

- Install dependencies
```bash
git clone
cd ai-worker
pip install -r requirements.txt
```

- Prepare environment variables
```bash
cp .env.sample .env
# Edit .env file to add your API keys and configurations
```

## How to run

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

## Test with curl

```bash
curl -X POST "http://localhost:8080/api/v1/outline/generate" 
    -H "Content-Type: application/json" 
    -d '{
    "topic": "Introduction to Tắt đèn of Ngô Tất Tố",
    "slide_count": 5,
    "audience": "university students",
    "model": "gemini-2.5-flash-lite",
    "learning_objective": "",
    "language": "vi",
    "targetAge": "5-10"
    }'
```

## Folder structure

```bash
.
├── app
│   ├── api
│   │   ├── endpoints
│   │   │   ├── generate.py
│   │   │   └── __pycache__
│   │   ├── __pycache__
│   │   │   └── router.cpython-310.pyc
│   │   └── router.py
│   ├── core
│   │   ├── config.py
│   │   └── depends.py
│   ├── llms
│   │   ├── factory.py
│   │   └── service.py
│   ├── main.py
│   ├── schemas
│   │   ├── image_content.py
│   │   └── slide_content.py
│   └── services
│       └── content_service.py
├── README.md
├── requirements.txt
└── tests
```