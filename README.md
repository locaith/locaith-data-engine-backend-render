# Locaith Data Engine - Backend API

Enterprise-Grade Data Lakehouse-as-a-Service

## Quick Start

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

## API Docs
- Swagger UI: `/docs`
- ReDoc: `/redoc`

## Deploy on Render.com
This repo is configured for Render.com deployment.

Environment Variables needed:
- `GEMINI_API_KEY` (optional - for AI features)
- `URL_SUPABASE` (optional - for auth)
- `SUPABASE_ANON_KEY` (optional)

## License
MIT - Locaith Solution Tech
