conda create -p venv python==3.11 -y
conda activate venv/
pip install -r requirements.txt
pip install ipykernel

for stock app
pip install yfinance
pip install streamlit yfinance tensorflow scikit-learn pandas
