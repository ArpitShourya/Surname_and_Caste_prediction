import pandas as pd
import re

class DataValidation:
    REQUIRED_COLUMNS = [
        "FM_NAME_EN", "LASTNAME_EN", "FM_NAME_V1", "LASTNAME_V1",
        "RLN_TYPE", "RLN_FM_NM_EN", "RLN_L_NM_EN", "RLN_FM_NM_V1", "RLN_L_NM_V1",
        "GENDER"
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def check_required_columns(self):
        """Check if all required columns are present and return list of missing ones."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in self.df.columns]
        return missing  # Return list, empty if none missing



    def check_no_digits(self):
        """Check that required columns do not contain digits."""
        import re
        digit_pattern = re.compile(r"\d")  # matches any digit 0-9
        digit_violations = {}

        for col in self.REQUIRED_COLUMNS:
            if col not in self.df.columns:
                continue
            invalid_mask = self.df[col].astype(str).apply(lambda x: bool(digit_pattern.search(x)))
            if invalid_mask.any():
                digit_violations[col] = self.df.loc[invalid_mask, col].tolist()

        return digit_violations  # returns {} if no violations



