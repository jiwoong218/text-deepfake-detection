import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Ensemble multiple submission CSVs by averaging probabilities (Soft Voting)")
    parser.add_argument("--csvs", nargs='+', required=True, help="List of submission CSV files to ensemble (e.g., sub1.csv sub2.csv)")
    parser.add_argument("--output", type=str, default="ensemble_submission.csv", help="Output file name")
    args = parser.parse_args()

    if not args.csvs:
        print("Error: No CSV files provided.")
        return

    print(f"Ensembling {len(args.csvs)} files: {args.csvs}")

    # Read the first csv to get the base structure (ID, etc.)
    try:
        ensemble_df = pd.read_csv(args.csvs[0])
        prob_sum = ensemble_df['generated'].astype(float).copy()
    except FileNotFoundError:
        print(f"Error: File not found - {args.csvs[0]}")
        return

    # Sum probabilities from the rest of the CSVs
    for csv_file in args.csvs[1:]:
        try:
            df = pd.read_csv(csv_file)
            prob_sum += df['generated'].astype(float)
        except FileNotFoundError:
            print(f"Error: File not found - {csv_file}")
            return
        except KeyError:
            print(f"Error: 'generated' column not found in - {csv_file}")
            return

    # Average the probabilities
    ensemble_df['generated'] = prob_sum / len(args.csvs)

    # Save the ensembled submission
    ensemble_df.to_csv(args.output, index=False)
    print(f"✅ 앙상블 완료! 최종 결과가 '{args.output}'에 저장되었습니다.")

if __name__ == "__main__":
    main()
