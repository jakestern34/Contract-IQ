# MLB ContractIQ — iPad-Friendly Flat Build

This is the corrected Streamlit deployment package for an iPad workflow. **Every required file belongs at the top level of the GitHub repository. No folders are required.**

## Files to upload
Upload these files directly into the root of `MLB-ContractIQ`:

- `app.py`
- `requirements.txt`
- `demo_contracts.csv`
- `demo_player_seasons.csv`
- `demo_scouting.csv`
- `demo_suspensions.csv`

The app is intentionally self-contained: the former `src/` Python modules are consolidated into `app.py`, and CSV templates are generated inside the app.

## Before uploading
The easiest clean approach is to delete the old flattened app files from the repository and then upload the six files above. Keeping `README.md` is fine, but you do not need any of the old Python module files for this version.

## Streamlit deployment
In Streamlit Community Cloud choose:

- Repository: `jakestern34/MLB-ContractIQ`
- Branch: `main`
- Main file path: `app.py`

Then click **Deploy**.

## Updating real data later
For persistent real datasets, add these optional root-level files:

- `live_contracts.csv`
- `live_player_seasons.csv`
- `live_scouting.csv`
- `live_suspensions.csv`

The app detects them automatically. Session-only CSVs can also be tested from the Data Manager page.

## Important
The bundled demo datasets are synthetic and are only there to prove the model/UI works. They must be replaced with validated historical data before using outputs for actual contract negotiation.
