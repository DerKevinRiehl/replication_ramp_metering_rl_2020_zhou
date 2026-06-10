import pandas as pd
from pathlib import Path


def clean_episode_overlap(
    input_csv,
    output_csv=None,
    episode_col="episode",
    keep="last",
    sort_output=True,
    report=True,
):
    """
    Remove overlapping / duplicated episode entries from a CSV file.

    Parameters
    ----------
    input_csv : str or Path
        Path to the input CSV file.
    output_csv : str or Path, optional
        Path to save the cleaned CSV. If None, saves as '<input>_cleaned.csv'.
    episode_col : str
        Name of the episode column.
    keep : {"first", "last"}
        Which duplicated episode value to keep.
        Use "last" when resumed/recomputed values should replace earlier ones.
    sort_output : bool
        If True, sort cleaned data by episode number after removing duplicates.
    report : bool
        If True, print overlap/reset information.

    Returns
    -------
    df_clean : pandas.DataFrame
        Cleaned dataframe.
    """

    input_csv = Path(input_csv)

    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_cleaned.csv")
    else:
        output_csv = Path(output_csv)

    df = pd.read_csv(input_csv)

    if episode_col not in df.columns:
        raise ValueError(
            f"Column '{episode_col}' not found. Available columns: {list(df.columns)}"
        )

    # Detect resets: places where episode number decreases compared to previous row
    episode_diff = df[episode_col].diff()
    reset_rows = df.index[episode_diff < 0].tolist()

    # Detect duplicated episode numbers
    duplicated_mask = df.duplicated(subset=episode_col, keep=False)
    duplicated_episodes = df.loc[duplicated_mask, episode_col]

    if report:
        print("── Overlap cleaning report ──")
        print(f"Input file              : {input_csv}")
        print(f"Total rows before       : {len(df)}")
        print(f"Unique episodes before  : {df[episode_col].nunique()}")
        print(f"Duplicated rows         : {duplicated_mask.sum()}")

        if reset_rows:
            print("\nDetected reset points:")
            for row in reset_rows:
                prev_ep = df.loc[row - 1, episode_col]
                new_ep = df.loc[row, episode_col]
                print(
                    f"  row {row}: episode {prev_ep} -> {new_ep}"
                )
        else:
            print("\nNo reset points detected.")

        if not duplicated_episodes.empty:
            print("\nOverlapping episode range:")
            print(
                f"  {duplicated_episodes.min()} to {duplicated_episodes.max()}"
            )
        else:
            print("\nNo duplicated episodes detected.")

    # Clean duplicates
    df_clean = df.drop_duplicates(subset=episode_col, keep=keep)

    if sort_output:
        df_clean = df_clean.sort_values(by=episode_col).reset_index(drop=True)
    else:
        df_clean = df_clean.reset_index(drop=True)

    df_clean.to_csv(output_csv, index=False)

    if report:
        print("\nCleaning complete.")
        print(f"Rows after              : {len(df_clean)}")
        print(f"Unique episodes after   : {df_clean[episode_col].nunique()}")
        print(f"Saved to                : {output_csv}")

    return df_clean


# Example use:
df_clean = clean_episode_overlap(
    input_csv="episode_rewards.csv",
    output_csv="episode_rewards.csv",
    episode_col="episode",
    keep="last",          # keeps the resumed / second values
    sort_output=True,
    report=True,
)