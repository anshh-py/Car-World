# Backend Scripts

`import_cars.py` imports `Data/car_dataset.csv` into the manufacturer, model,
and variant tables. It skips and reports exact duplicate source rows, reuses
existing records on repeat runs, and reports suspicious values without
rewriting the source CSV.
