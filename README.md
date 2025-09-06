# ai-worker


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