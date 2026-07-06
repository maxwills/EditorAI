#!/usr/bin/env bash
source venv/Scripts/activate
cd cad_prediction_service
uvicorn app.main:app --reload --port 8001
