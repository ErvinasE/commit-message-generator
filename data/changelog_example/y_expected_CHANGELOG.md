## Added
- Record expenses with `spendlog add` and view them with `spendlog list` (6ec9d12)
- Expenses are saved to a CSV file that any spreadsheet program can open (91837ea)
- `spendlog report YYYY-MM` shows the total spent per category in a month (c2a7198)
- Show only one category with `spendlog list --category` (d616a24)
- Set monthly limits per category in `budgets.toml`; the report warns when a category goes over its limit (fc20052)

## Changed
- **Breaking:** the monthly report now lists categories from largest to smallest amount instead of alphabetically (f0e9ae7)

## Fixed
- Monthly reports no longer include expenses from the same month of other years (c8c7a4b)

## Documentation
- Added installation and usage instructions to the README (1e454b8)

## Internal
- Set up the project skeleton (5bdcffb)
- Added tests for loading and saving expenses (7f95539)
- Simplified the CSV storage code without changing behavior (cd9ff1e)
- Tests now run automatically on GitHub Actions (25ea4bf)
- Bumped the version to 0.2.0 (1b674e2)
