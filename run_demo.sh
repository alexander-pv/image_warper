#!/bin/bash
python ./src/backend/serve_warper.py & python ./src/backend/serve_style_transfer.py & streamlit run ./src/front/app.py