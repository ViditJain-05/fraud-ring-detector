# Dataset Setup: Synthetic Fraud Ring Data

## Data Generation Notice
Real-world financial fraud ring data is highly sensitive, proprietary, and unavailable in public repositories. As a deliberate design choice, this project utilizes a **self-simulated applicant dataset** generated via `src/data_simulation.py`.

## Simulated Schema
The generated dataset (`data/applicants.csv`) contains:
- `applicant_id`: Unique applicant identifier
- `phone`: Contact phone number
- `bank_account`: Deposit bank account identifier
- `address`: Residential physical address
- `income`: Applicant annual income ($)
- `loan_amount`: Requested loan amount ($)
- `age`: Applicant age (years)
- `city`: Applicant city
- `is_fraud`: Binary target label (1 = Fraudulent applicant in ring, 0 = Legitimate applicant)

## Generation Parameters
- ~95% legitimate applicants with distinct, unique phone numbers, bank accounts, and addresses.
- ~5% fraudulent applicants grouped into "fraud rings" of 3-6 individuals sharing common bank accounts or addresses.

To re-generate the dataset, execute:
```bash
python src/data_simulation.py --num-applicants 1000 --fraud-ratio 0.05
```
