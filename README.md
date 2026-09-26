# carworld CSV import

Loads `car_dataset.csv` into the existing PostgreSQL `carworld.cars` table.

## Run

```powershell
cd C:\Users\ansha\carworld-import
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` with your PostgreSQL password, then:

```powershell
python import_cars.py
```

The script creates `cars` only if it is missing (`CREATE TABLE IF NOT EXISTS`), then bulk-copies all rows with PostgreSQL `COPY`. It does not drop or alter an existing table.
