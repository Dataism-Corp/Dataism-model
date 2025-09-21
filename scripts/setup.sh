set -e
echo "Activating venv..."
source ~/.venvs/llm/bin/activate

cd /home/dateria/project

echo "Updating pip..."
python -m pip install --upgrade pip

echo "Installing requirements..."
cat > requirements.txt <<'REQ'
fastapi
uvicorn[standard]
python-dotenv
transformers
torch
tqdm
pyyaml
sse_starlette
python-multipart
REQ

pip install -r requirements.txt
echo "Setup complete."
