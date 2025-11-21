import os
import polars as pl
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


def generate_file_paths(base_path, participants=31, avatars=["Man", "Robot", "Woman"]):
    files = {}
    for i in range(participants):
        participant_code = f"{i:04d}"
        for avatar in avatars:
            file_key = f"{participant_code}_{avatar}"
            file_path = os.path.join(base_path, f"{file_key}.csv")
            files[file_key] = file_path
    return files

def process_vr_data_press_and_nearest_answer(files):
    combined_results = []
    required_columns = ["User_Action", "time_from_start", "Form_Question", "Form_Answer", "Source_File"]

    # กำหนด dtype สำหรับแต่ละคอลัมน์ให้ตรงกัน
    dtype_map = {
        "User_Action": pl.Utf8,
        "time_from_start": pl.Float64,
        "Form_Question": pl.Utf8,
        "Form_Answer": pl.Utf8,
        "Source_File": pl.Utf8,
    }

    for file_key, file_path in files.items():
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        try:
            df = pl.read_csv(file_path, try_parse_dates=True)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            continue

        if "User_Action" not in df.columns or "time_from_start" not in df.columns:
            print(f"Missing required columns in {file_path}")
            continue

        df = df.with_columns(
            pl.col("User_Action").fill_null("Walk")
        )

        press_df = df.filter(pl.col("User_Action").str.to_lowercase().str.contains("press"))
        answer_df = df.filter(pl.col("User_Action").str.to_lowercase().str.contains("answer"))

        if press_df.is_empty() or answer_df.is_empty():
            print(f"No 'Press' or 'Answer' actions in {file_path}")
            continue

        press_rows = press_df.to_dicts()
        answer_rows = answer_df.to_dicts()

        updated_press_rows = []
        for press_row in press_rows:
            press_time = press_row['time_from_start']
            min_diff = float('inf')
            best_answer = None

            for answer_row in answer_rows:
                time_diff = abs(answer_row['time_from_start'] - press_time)
                if time_diff < min_diff:
                    min_diff = time_diff
                    best_answer = answer_row

            if best_answer:
                press_row['Form_Question'] = best_answer.get('Form_Question', None)
                press_row['Form_Answer'] = best_answer.get('Form_Answer', None)
                press_row['Source_File'] = file_key
                updated_press_rows.append(press_row)

        if updated_press_rows:
            temp_df = pl.DataFrame(updated_press_rows)

            # เติมคอลัมน์ที่ขาด
            missing_cols = [col for col in required_columns if col not in temp_df.columns]
            if missing_cols:
                temp_df = temp_df.with_columns([pl.lit(None).alias(col) for col in missing_cols])

            # cast dtype ให้ตรงกันทุกคอลัมน์
            for col, dtype in dtype_map.items():
                if col in temp_df.columns:
                    temp_df = temp_df.with_columns(pl.col(col).cast(dtype))

            # เรียงคอลัมน์ให้ตรงตาม required_columns
            temp_df = temp_df.select(required_columns)

            print(f"Prepared DataFrame from {file_key} with schema:")
            print(temp_df.schema)

            combined_results.append(temp_df)

    if not combined_results:
        raise ValueError("No valid data was processed from the files.")

    combined_df = pl.concat(combined_results, how="vertical", rechunk=True)

    output_path = "press_action.csv"
    combined_df.write_csv(output_path)

    return combined_df, output_path

# เรียกใช้
base_path = 'MergeFile'
files = generate_file_paths(base_path)
result_df, output_file = process_vr_data_press_and_nearest_answer(files)
print(f"Output saved to: {output_file}")
